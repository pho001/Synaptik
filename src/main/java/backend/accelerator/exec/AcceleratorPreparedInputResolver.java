package backend.accelerator.exec;

import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu.CpuNodeExecutionArtifact;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.plan.CpuPreparedInput;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import tensor.Tensor;
import tensor.layout.TensorRemap;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Resolves accelerator external inputs through the same prepared-input machinery used by CPU execution.
 */
public final class AcceleratorPreparedInputResolver {
    private AcceleratorPreparedInputResolver() {
    }

    public static ResolvedAcceleratorInputs resolve(
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            List<Integer> externalInputNodeIds,
            ExecutionContext context
    ) {
        Objects.requireNonNull(context, "context cannot be null");
        List<Integer> externalIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        if (externalIds.isEmpty()) {
            return new ResolvedAcceleratorInputs(List.of(), List.of(), List.of(), List.of(), List.of());
        }

        List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> steps = List.copyOf(
                cpuFallbackSteps == null ? List.of() : cpuFallbackSteps
        );

        Map<Integer, List<Tensor>> resolvedInputsByConsumer = new HashMap<>();
        List<Tensor> originals = new ArrayList<>(externalIds.size());
        List<Tensor> executions = new ArrayList<>(externalIds.size());
        List<Boolean> prepared = new ArrayList<>(externalIds.size());
        List<AcceleratorPreparedInputSite> sites = new ArrayList<>(externalIds.size());

        for (int externalInputNodeId : externalIds) {
            Tensor original = context.runtimeTensorForNodeId(externalInputNodeId);
            originals.add(original);
            InputSite site = findInputSite(steps, externalInputNodeId);
            if (site == null) {
                executions.add(original);
                prepared.add(false);
                sites.add(new AcceleratorPreparedInputSite(externalInputNodeId, externalInputNodeId, -1, false));
                continue;
            }

            List<Tensor> resolvedConsumerInputs = resolvedInputsByConsumer.computeIfAbsent(
                    site.consumerNode().id(),
                    ignored -> new ArrayList<>(PreparedAcceleratorExecutionSupport.resolveRuntimeInputs(site.consumerNode(), context))
            );
            Tensor execution = resolveConsumerInput(site, resolvedConsumerInputs, original, context);
            boolean preparedUsed = execution != original;
            executions.add(execution);
            prepared.add(preparedUsed);
            sites.add(new AcceleratorPreparedInputSite(
                    externalInputNodeId,
                    site.consumerNode().id(),
                    site.consumerInputIndex(),
                    preparedUsed
            ));
        }

        return new ResolvedAcceleratorInputs(externalIds, originals, executions, prepared, sites);
    }

    /**
     * Resolves external inputs for native buffer binding without applying CPU fallback prepared-input plans.
     *
     * <p>Supported accelerator interiors must stay device-owned through {@code ExecutionState} bindings.
     * CPU fallback prepared inputs remain handled by {@link #resolve(List, List, ExecutionContext)}.</p>
     */
    public static ResolvedAcceleratorInputs resolveForNativeBufferBinding(
            List<Integer> externalInputNodeIds,
            ExecutionContext context
    ) {
        Objects.requireNonNull(context, "context cannot be null");
        List<Integer> externalIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        if (externalIds.isEmpty()) {
            return new ResolvedAcceleratorInputs(List.of(), List.of(), List.of(), List.of(), List.of());
        }
        List<Tensor> originals = new ArrayList<>(externalIds.size());
        List<Boolean> prepared = new ArrayList<>(externalIds.size());
        List<AcceleratorPreparedInputSite> sites = new ArrayList<>(externalIds.size());
        for (int externalInputNodeId : externalIds) {
            Tensor original = context.runtimeTensorForNodeId(externalInputNodeId);
            originals.add(original);
            prepared.add(false);
            sites.add(new AcceleratorPreparedInputSite(externalInputNodeId, externalInputNodeId, -1, false));
        }
        return new ResolvedAcceleratorInputs(externalIds, originals, originals, prepared, sites);
    }

    private static Tensor resolveConsumerInput(
            InputSite site,
            List<Tensor> resolvedConsumerInputs,
            Tensor original,
            ExecutionContext context
    ) {
        if (site == null || resolvedConsumerInputs == null || site.consumerInputIndex() >= resolvedConsumerInputs.size()) {
            return original;
        }
        CpuNodeExecutionPlan cpuPlan = cpuPlan(site.metadata());
        if (cpuPlan == null || cpuPlan.layoutPlan().preparedInputs().isEmpty()) {
            return resolvedConsumerInputs.get(site.consumerInputIndex());
        }
        for (CpuPreparedInput preparedInput : cpuPlan.layoutPlan().preparedInputs()) {
            if (preparedInput.inputIndex() != site.consumerInputIndex()) {
                continue;
            }
            context.requireCpuReadable(site.externalInputNodeId(), CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT);
            Tensor runtimePrepared = context.preparedInputTensorFor(site.consumerNode().id(), preparedInput.inputIndex());
            TensorRemap.applyTrusted(
                    original,
                    runtimePrepared,
                    preparedInput.remapPlan(),
                    cpuPlan.layoutPlan().materializeThreshold()
            );
            context.mirrorRuntimeState(original, runtimePrepared);
            resolvedConsumerInputs.set(preparedInput.inputIndex(), runtimePrepared);
            return runtimePrepared;
        }
        return resolvedConsumerInputs.get(site.consumerInputIndex());
    }

    private static InputSite findInputSite(
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> steps,
            int externalInputNodeId
    ) {
        for (PreparedAcceleratorExecutionSupport.CpuFallbackStep step : steps) {
            CompiledNode node = step.node();
            int inputIndex = node.inputIds().indexOf(externalInputNodeId);
            if (inputIndex >= 0) {
                return new InputSite(externalInputNodeId, node, step.metadata(), inputIndex);
            }
        }
        return null;
    }

    private record InputSite(
            int externalInputNodeId,
            CompiledNode consumerNode,
            CompiledNodeExecutionMetadata metadata,
            int consumerInputIndex
    ) {
    }

    private static CpuNodeExecutionPlan cpuPlan(CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            return null;
        }
        if (metadata.artifact() instanceof CpuNodeExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        if (metadata.artifact() instanceof CpuFusedExecutionArtifact artifact) {
            return artifact.cpuPlan();
        }
        return null;
    }
}
