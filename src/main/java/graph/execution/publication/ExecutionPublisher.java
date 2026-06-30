package graph.execution.publication;

import runtime.contract.CpuMaterializationReason;
import runtime.contract.ExecutionMode;
import graph.model.AliasViewPolicy;
import graph.model.CompiledGradientBinding;
import graph.compile.publication.PublicationPlan;
import graph.execution.PublicationPolicy;
import runtime.execution.ExecutionState;
import tensor.Tensor;
import tensor.TensorInternalAccess;

/**
 * Publishes run-scoped runtime values back to user-visible tensors.
 */
public final class ExecutionPublisher {
    private ExecutionPublisher() {
    }

    public static void publishAfterExecution(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            PublicationPlan publicationPlan
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, publicationPlan);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState, publicationPlan);
        }
        if (mode == ExecutionMode.FORWARD_BACKWARD) {
            if (publication.publishesGradients()) {
                publishCompiledGradients(executionState, publicationPlan);
            } else {
                clearPublishedGradients(publicationPlan);
            }
        }
    }

    public static void publishAfterOptimizerStep(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            PublicationPlan publicationPlan
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, publicationPlan);
        }
        if (publication.publishesGradients()) {
            publishCompiledGradients(executionState, publicationPlan);
        } else {
            clearPublishedGradients(publicationPlan);
        }
    }

    public static void publishForwardOnly(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            PublicationPlan publicationPlan
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, publicationPlan);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState, publicationPlan);
        }
    }

    public static void syncRootData(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPlan publicationPlan
    ) {
        PublicationPlan.ForwardPublicationBinding rootOutput = publicationPlan.rootOutput();
        Tensor rootTensor = rootOutput.targetTensor();
        int actualRootNodeId = rootOutput.sourceNodeId();
        Tensor publishTarget = resolvePublicationTarget(rootTensor);
        Integer publishNodeId = publicationPlan.nodeIdsByPublicationTarget().get(publishTarget);
        if (publishNodeId != null) {
            if (publishNodeId != actualRootNodeId
                    && shouldPublishActualRootForAlias(executionState, publishNodeId, actualRootNodeId)) {
                publishRuntimeTensor(
                        mode,
                        executionState,
                        rootTensor,
                        actualRootNodeId,
                        CpuMaterializationReason.GRAPH_OUTPUT
                );
                repairPublicationAliasChain(rootOutput);
                return;
            }
            executionState.requireCpuReadable(publishNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
            Tensor runtimePublished = executionState.runtimeTensorForNodeId(publishNodeId);
            if (TensorInternalAccess.storage(publishTarget) == TensorInternalAccess.storage(runtimePublished)) {
                repairPublicationAliasChain(rootOutput);
                return;
            }
            if (mode == ExecutionMode.FORWARD_BACKWARD || runtimePublished != publishTarget) {
                publishTarget.copyDataFrom(runtimePublished);
            }
            repairPublicationAliasChain(rootOutput);
            return;
        }

        publishRuntimeTensor(mode, executionState, rootTensor, actualRootNodeId);
        repairPublicationAliasChain(rootOutput);
    }

    private static void publishAllForwardValues(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPlan publicationPlan
    ) {
        syncRootData(mode, executionState, publicationPlan);
        Tensor rootTensor = publicationPlan.rootOutput().targetTensor();
        Tensor rootPublishTarget = resolvePublicationTarget(rootTensor);
        for (PublicationPlan.ForwardPublicationBinding binding : publicationPlan.forwardValuePublications()) {
            Tensor target = binding.targetTensor();
            if (target == rootTensor || target == rootPublishTarget) {
                continue;
            }
            publishRuntimeTensor(
                    mode,
                    executionState,
                    target,
                    binding.sourceNodeId(),
                    CpuMaterializationReason.GRAPH_VALUE_PUBLICATION
            );
            repairPublicationAliasChain(binding);
        }
        repairPublicationAliasChain(publicationPlan.rootOutput());
    }

    private static boolean shouldPublishActualRootForAlias(
            ExecutionState executionState,
            int publishNodeId,
            int actualRootNodeId
    ) {
        var publishState = executionState.residencyForNodeId(publishNodeId);
        var actualRootState = executionState.residencyForNodeId(actualRootNodeId);
        return !publishState.cpuCurrent()
                && (actualRootState.cpuCurrent() || actualRootState.requiresCpuMaterialization());
    }

    private static void publishRuntimeTensor(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor publishTarget,
            int nodeId
    ) {
        publishRuntimeTensor(mode, executionState, publishTarget, nodeId, CpuMaterializationReason.GRAPH_OUTPUT);
    }

    private static void publishRuntimeTensor(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor publishTarget,
            int nodeId,
            CpuMaterializationReason reason
    ) {
        executionState.requireCpuReadable(nodeId, reason);
        Tensor runtimeTensor = executionState.runtimeTensorForNodeId(nodeId);
        if (mode == ExecutionMode.FORWARD_BACKWARD || runtimeTensor != publishTarget) {
            publishTarget.copyDataFrom(runtimeTensor);
        }
    }

    private static Tensor resolvePublicationTarget(Tensor tensor) {
        Tensor current = tensor;
        while (isAliasViewOp(current) && current.getPrevTensors() != null && !current.getPrevTensors().isEmpty()) {
            current = current.getPrevTensors().getFirst();
        }
        return current;
    }

    private static boolean isAliasViewOp(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        return AliasViewPolicy.aliasesInput0AtRuntime(tensor);
    }

    private static void repairPublicationAliasChain(PublicationPlan.ForwardPublicationBinding binding) {
        for (PublicationPlan.AliasRepairStep step : binding.aliasRepairChain()) {
            TensorInternalAccess.aliasRuntimeFrom(step.aliasTensor(), step.sourceTensor());
        }
    }

    private static void publishCompiledGradients(
            ExecutionState executionState,
            PublicationPlan publicationPlan
    ) {
        clearPublishedGradients(publicationPlan);
        for (PublicationPlan.GradientPublicationBinding publication : publicationPlan.gradientPublications()) {
            Tensor published;
            CompiledGradientBinding binding = publication.binding();
            if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                executionState.requireCpuReadable(nodeBinding.nodeId(), CpuMaterializationReason.GRADIENT_PUBLICATION);
                published = detachedCopy(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
            } else if (binding instanceof CompiledGradientBinding.ConstantBinding constantBinding) {
                published = constantBinding.value().toTensor("gradient");
            } else {
                throw new IllegalStateException("Unsupported gradient binding type: " + binding.getClass().getName());
            }
            TensorInternalAccess.setGradient(publication.targetTensor(), published);
        }
    }

    private static void clearPublishedGradients(PublicationPlan publicationPlan) {
        for (Tensor tensor : publicationPlan.gradientClearTargets()) {
            TensorInternalAccess.setGradient(tensor, null);
        }
    }

    private static Tensor detachedCopy(Tensor source) {
        Tensor copy = new Tensor(source.getShape(), null, source.getLabel(), source.getDataType());
        copy.copyDataFrom(source);
        return copy;
    }
}
