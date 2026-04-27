package backend.accelerator.exec;

import backend.CPUBackend;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PreparedAcceleratorExecutionSupport {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();

    private PreparedAcceleratorExecutionSupport() {
    }

    public record CpuFallbackStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        public CpuFallbackStep {
            Objects.requireNonNull(node, "node cannot be null");
            Objects.requireNonNull(metadata, "metadata cannot be null");
        }
    }

    public static boolean bridgeReady(boolean bridgeAvailable, boolean contextAvailable, boolean executableAvailable) {
        return bridgeAvailable && contextAvailable && executableAvailable;
    }

    public static List<Tensor> resolveRuntimeTensors(List<Integer> nodeIds, ExecutionContext context) {
        if (nodeIds == null || nodeIds.isEmpty()) {
            return List.of();
        }
        List<Tensor> out = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            out.add(context.runtimeTensorForNodeId(nodeId));
        }
        return List.copyOf(out);
    }

    public static List<Tensor> resolveRuntimeInputs(CompiledNode node, ExecutionContext context) {
        if (node == null || node.inputIds().isEmpty()) {
            return List.of();
        }
        return resolveRuntimeTensors(node.inputIds(), context);
    }

    public static void executeCpuFallback(List<CpuFallbackStep> steps, ExecutionContext context) {
        if (steps == null) {
            return;
        }
        for (CpuFallbackStep step : steps) {
            CPU_BACKEND.execute(step.node(), step.metadata(), context);
        }
    }
}
