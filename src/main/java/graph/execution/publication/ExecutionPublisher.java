package graph.execution.publication;

import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionMode;
import graph.AliasViewPolicy;
import graph.CompiledGradientBinding;
import graph.CompiledNode;
import graph.execution.PublicationPolicy;
import graph.execution.state.ExecutionState;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;
import java.util.Map;

/**
 * Publishes run-scoped runtime values back to user-visible semantic tensors.
 */
public final class ExecutionPublisher {
    private ExecutionPublisher() {
    }

    public static void publishAfterExecution(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            Tensor rootTensor,
            List<CompiledNode> allNodes,
            CompiledNode forwardOutputNode,
            Map<Tensor, CompiledGradientBinding> compiledGradients
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        }
        if (mode == ExecutionMode.FORWARD_BACKWARD) {
            if (publication.publishesGradients()) {
                publishCompiledGradients(executionState, allNodes, compiledGradients);
            } else {
                clearPublishedGradients(allNodes);
            }
        }
    }

    public static void publishAfterOptimizerStep(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            Tensor rootTensor,
            List<CompiledNode> allNodes,
            CompiledNode forwardOutputNode,
            Map<Tensor, CompiledGradientBinding> compiledGradients
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        }
        if (publication.publishesGradients()) {
            publishCompiledGradients(executionState, allNodes, compiledGradients);
        } else {
            clearPublishedGradients(allNodes);
        }
    }

    public static void publishForwardOnly(
            ExecutionMode mode,
            ExecutionState executionState,
            PublicationPolicy publication,
            Tensor rootTensor,
            List<CompiledNode> allNodes,
            CompiledNode forwardOutputNode
    ) {
        if (publication.publishesAllForwardValues()) {
            publishAllForwardValues(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        } else if (publication.publishesOutputValue()) {
            syncRootData(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        }
    }

    public static void syncRootData(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor rootTensor,
            List<CompiledNode> allNodes,
            CompiledNode forwardOutputNode
    ) {
        Integer semanticRootNodeId = nodeIdForSemanticTensor(rootTensor, allNodes);
        int actualRootNodeId = semanticRootNodeId == null
                ? resolveForwardRuntimeRootNodeId(forwardOutputNode)
                : semanticRootNodeId;
        Tensor publishTarget = resolveSemanticPublishTarget(rootTensor);
        Integer publishNodeId = nodeIdForSemanticTensor(publishTarget, allNodes);
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
                repairSemanticAliasChain(rootTensor);
                return;
            }
            executionState.requireCpuReadable(publishNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
            Tensor runtimePublished = executionState.runtimeTensorForNodeId(publishNodeId);
            if (TensorInternalAccess.storage(publishTarget) == TensorInternalAccess.storage(runtimePublished)) {
                repairSemanticAliasChain(rootTensor);
                return;
            }
            if (mode == ExecutionMode.FORWARD_BACKWARD || runtimePublished != publishTarget) {
                publishTarget.copyDataFrom(runtimePublished);
            }
            repairSemanticAliasChain(rootTensor);
            return;
        }

        publishRuntimeTensor(mode, executionState, rootTensor, actualRootNodeId);
        repairSemanticAliasChain(rootTensor);
    }

    private static void publishAllForwardValues(
            ExecutionMode mode,
            ExecutionState executionState,
            Tensor rootTensor,
            List<CompiledNode> allNodes,
            CompiledNode forwardOutputNode
    ) {
        syncRootData(mode, executionState, rootTensor, allNodes, forwardOutputNode);
        Tensor rootPublishTarget = resolveSemanticPublishTarget(rootTensor);
        for (CompiledNode node : allNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor target = node.sourceTensor();
            if (target == null || target == rootTensor || target == rootPublishTarget) {
                continue;
            }
            publishRuntimeTensor(
                    mode,
                    executionState,
                    target,
                    node.id(),
                    CpuMaterializationReason.GRAPH_VALUE_PUBLICATION
            );
            repairSemanticAliasChain(target);
        }
        repairSemanticAliasChain(rootTensor);
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

    private static Tensor resolveSemanticPublishTarget(Tensor tensor) {
        Tensor current = tensor;
        while (isAliasViewOp(current) && current.getPrevTensors() != null && !current.getPrevTensors().isEmpty()) {
            current = current.getPrevTensors().getFirst();
        }
        return current;
    }

    private static Integer nodeIdForSemanticTensor(Tensor tensor, List<CompiledNode> allNodes) {
        if (tensor == null) {
            return null;
        }
        for (CompiledNode node : allNodes) {
            if (node.semanticTensor() == tensor || node.sourceTensor() == tensor) {
                return node.id();
            }
        }
        return null;
    }

    private static int resolveForwardRuntimeRootNodeId(CompiledNode forwardOutputNode) {
        if (forwardOutputNode.operation() != null
                && forwardOutputNode.operation().opType() == operations.Operation.OpType.NOOP
                && Tensor.SYSTEM_FORWARD_OUTPUT_LABEL.equals(forwardOutputNode.label())
                && !forwardOutputNode.inputIds().isEmpty()) {
            return forwardOutputNode.inputIds().getFirst();
        }
        return forwardOutputNode.id();
    }

    private static boolean isAliasViewOp(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        return AliasViewPolicy.aliasesInput0AtRuntime(tensor);
    }

    private static void repairSemanticAliasChain(Tensor tensor) {
        if (!isAliasViewOp(tensor) || tensor.getPrevTensors() == null || tensor.getPrevTensors().isEmpty()) {
            return;
        }
        Tensor source = tensor.getPrevTensors().getFirst();
        repairSemanticAliasChain(source);
        TensorInternalAccess.aliasRuntimeFrom(tensor, source);
    }

    private static void publishCompiledGradients(
            ExecutionState executionState,
            List<CompiledNode> allNodes,
            Map<Tensor, CompiledGradientBinding> compiledGradients
    ) {
        for (CompiledNode node : allNodes) {
            if (node.backwardNode()) {
                continue;
            }
            Tensor tensor = node.sourceTensor();
            CompiledGradientBinding binding = compiledGradients.get(tensor);
            if (binding == null) {
                TensorInternalAccess.setGradient(tensor, null);
                continue;
            }
            Tensor published;
            if (binding instanceof CompiledGradientBinding.NodeBinding nodeBinding) {
                executionState.requireCpuReadable(nodeBinding.nodeId(), CpuMaterializationReason.GRADIENT_PUBLICATION);
                published = detachedCopy(executionState.runtimeTensorForNodeId(nodeBinding.nodeId()));
            } else if (binding instanceof CompiledGradientBinding.ConstantBinding constantBinding) {
                published = detachedCopy(constantBinding.template());
            } else {
                throw new IllegalStateException("Unsupported gradient binding type: " + binding.getClass().getName());
            }
            TensorInternalAccess.setGradient(tensor, published);
        }
    }

    private static void clearPublishedGradients(List<CompiledNode> allNodes) {
        for (CompiledNode node : allNodes) {
            if (!node.backwardNode()) {
                TensorInternalAccess.setGradient(node.sourceTensor(), null);
            }
        }
    }

    private static Tensor detachedCopy(Tensor source) {
        Tensor copy = new Tensor(source.getShape(), null, source.getLabel(), source.getDataType());
        copy.copyDataFrom(source);
        return copy;
    }
}
