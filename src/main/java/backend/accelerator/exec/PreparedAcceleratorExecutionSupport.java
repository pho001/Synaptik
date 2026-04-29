package backend.accelerator.exec;

import backend.cpu.CpuBackend;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Shared internal SPI for prepared accelerator executables.
 *
 * <p>The helpers centralize runtime tensor resolution and CPU fallback execution so
 * native CUDA and Metal bridges expose the same availability behavior.</p>
 */
public final class PreparedAcceleratorExecutionSupport {
    private static final CpuBackend CPU_BACKEND = new CpuBackend();

    private PreparedAcceleratorExecutionSupport() {
    }

    /**
     * CPU-prepared step used when an accelerator bridge is unavailable at execution time.
     *
     * @param node compiled node to execute on the CPU backend
     * @param metadata CPU execution metadata prepared for the node
     */
    public record CpuFallbackStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        public CpuFallbackStep {
            Objects.requireNonNull(node, "node cannot be null");
            Objects.requireNonNull(metadata, "metadata cannot be null");
        }
    }

    /**
     * Returns whether all native bridge layers needed for execution are usable.
     */
    public static boolean bridgeReady(boolean bridgeAvailable, boolean contextAvailable, boolean executableAvailable) {
        return bridgeAvailable && contextAvailable && executableAvailable;
    }

    /**
     * Resolves runtime tensors by compiled-node id in list order.
     */
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

    /**
     * Resolves the runtime input tensors for a compiled node.
     */
    public static List<Tensor> resolveRuntimeInputs(CompiledNode node, ExecutionContext context) {
        if (node == null || node.inputIds().isEmpty()) {
            return List.of();
        }
        return resolveRuntimeTensors(node.inputIds(), context);
    }

    /**
     * Executes precomputed CPU fallback steps in partition order.
     *
     * <p>The fallback path executes inside one accelerator wrapper step, so it cannot rely on
     * {@code PreparedExecution}'s outer per-step residency bookkeeping. Each internal CPU step must
     * validate its CPU-readable inputs and publish its output residency itself; otherwise a later
     * consumer can see stale CPU storage with no active device binding after the fallback completes.</p>
     */
    public static void executeCpuFallback(List<CpuFallbackStep> steps, ExecutionContext context) {
        if (steps == null) {
            return;
        }
        for (CpuFallbackStep step : steps) {
            requireCpuReadableInputs(step, context);
            CPU_BACKEND.execute(step.node(), step.metadata(), context);
            context.markCpuCurrent(step.node().id(), "accelerator cpu fallback wrote CPU array");
        }
    }

    private static void requireCpuReadableInputs(CpuFallbackStep step, ExecutionContext context) {
        List<Integer> inputIds = step.metadata().executionInputNodeIds().isEmpty()
                ? step.node().inputIds()
                : step.metadata().executionInputNodeIds();
        for (int inputId : inputIds) {
            context.requireCpuReadable(inputId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }
}
