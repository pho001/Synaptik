package backend.accelerator.exec;

import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;

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
                    ignored -> resolveConsumerInputs(site.consumerNode(), site.metadata(), context)
            );
            Tensor execution = site.consumerInputIndex() < resolvedConsumerInputs.size()
                    ? resolvedConsumerInputs.get(site.consumerInputIndex())
                    : original;
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

    private static List<Tensor> resolveConsumerInputs(
            CompiledNode consumerNode,
            CompiledNodeExecutionMetadata metadata,
            ExecutionContext context
    ) {
        CpuNodeExecutionPlan cpuPlan = metadata == null ? null : metadata.cpuPlan();
        List<Tensor> originalInputs = PreparedAcceleratorExecutionSupport.resolveRuntimeInputs(consumerNode, context);
        if (cpuPlan == null) {
            return originalInputs;
        }
        for (var preparedInput : cpuPlan.layoutPlan().preparedInputs()) {
            int inputIndex = preparedInput.inputIndex();
            if (inputIndex >= 0 && inputIndex < consumerNode.inputIds().size()) {
                context.requireCpuReadable(
                        consumerNode.inputIds().get(inputIndex),
                        CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT
                );
            }
        }
        return cpuPlan.apply(consumerNode.id(), originalInputs, context);
    }

    private static InputSite findInputSite(
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> steps,
            int externalInputNodeId
    ) {
        for (PreparedAcceleratorExecutionSupport.CpuFallbackStep step : steps) {
            CompiledNode node = step.node();
            int inputIndex = node.inputIds().indexOf(externalInputNodeId);
            if (inputIndex >= 0) {
                return new InputSite(node, step.metadata(), inputIndex);
            }
        }
        return null;
    }

    private record InputSite(
            CompiledNode consumerNode,
            CompiledNodeExecutionMetadata metadata,
            int consumerInputIndex
    ) {
    }
}
