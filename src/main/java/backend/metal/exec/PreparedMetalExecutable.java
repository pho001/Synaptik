package backend.metal.exec;

import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.metal.lowering.MetalPartitionPlan;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.metal.MetalMpsCapabilities;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Prepared Metal MPS partition executable.
 *
 * <p>The executable owns the lowered Metal plan, native bridge context, compiled
 * bridge executable, and CPU fallback steps. Execution uses Metal only when the
 * bridge is ready and runtime tensors satisfy the current native storage contract;
 * otherwise the partition is replayed on CPU.</p>
 */
public final class PreparedMetalExecutable implements PreparedAcceleratorExecutable {
    private final MetalPartitionPlan plan;
    private final LoweringFamily loweringFamily;
    private final CompiledNode computeNode;
    private final CpuNodeExecutionPlan computeCpuPlan;
    private final MetalMpsGraphBridge bridge;
    private final MetalMpsBridgeContext bridgeContext;
    private final MetalMpsBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;
    private volatile MetalMpsBridgeExecutionStats lastExecutionStats = MetalMpsBridgeExecutionStats.fallback(
            "not executed yet",
            0,
            0,
            0L,
            0L
    );

    /**
     * Creates a prepared Metal executable around a lowered plan and fallback plan.
     */
    public PreparedMetalExecutable(
            MetalPartitionPlan plan,
            LoweringFamily loweringFamily,
            CompiledNode computeNode,
            CpuNodeExecutionPlan computeCpuPlan,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        this.plan = Objects.requireNonNull(plan, "plan cannot be null");
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        this.computeNode = Objects.requireNonNull(computeNode, "computeNode cannot be null");
        this.computeCpuPlan = Objects.requireNonNull(computeCpuPlan, "computeCpuPlan cannot be null");
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = bridge.createContext();
        this.bridgeExecutable = bridge.compile(bridgeContext, plan);
        this.cpuFallbackSteps = List.copyOf(cpuFallbackSteps == null ? List.of() : cpuFallbackSteps);
    }

    /**
     * Returns {@link ComputeBackend#GPU_METAL}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_METAL;
    }

    /**
     * Executes through the Metal bridge when available and compatible, otherwise runs CPU fallback steps.
     */
    @Override
    public void execute(ExecutionContext context) {
        List<Tensor> resolvedExternalInputs = bridgeExecutable.externalInputNodeIds().isEmpty()
                ? List.of()
                : resolveExternalInputs(context);
        List<Tensor> outputs = bridgeExecutable.outputNodeIds().isEmpty()
                ? List.of()
                : PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(bridgeExecutable.outputNodeIds(), context);
        String fallbackReason = metalFallbackReason(context, resolvedExternalInputs, outputs);
        if (fallbackReason.isBlank()) {
            lastExecutionStats = bridge.execute(bridgeContext, bridgeExecutable, resolvedExternalInputs, outputs);
            return;
        }
        lastExecutionStats = MetalMpsBridgeExecutionStats.fallback(
                fallbackReason,
                resolvedExternalInputs.size(),
                outputs.size(),
                byteSize(resolvedExternalInputs),
                byteSize(outputs)
        );
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
    }

    private String metalFallbackReason(ExecutionContext context, List<Tensor> resolvedExternalInputs, List<Tensor> outputs) {
        if (!shouldUseMetalBridge(context)) {
            return "backward pass contains forward SDPA DAG unsupported by current Metal bridge";
        }
        if (!bridge.isAvailable()) {
            return "bridge unavailable: " + bridge.unavailableReason();
        }
        if (!bridgeContext.available()) {
            return "bridge context unavailable: " + bridgeContext.reason();
        }
        if (!bridgeExecutable.available()) {
            return "bridge executable unavailable: " + bridgeExecutable.reason();
        }
        for (int i = 0; i < resolvedExternalInputs.size(); i++) {
            String reason = unsupportedExternalInputReason(resolvedExternalInputs.get(i));
            if (!reason.isBlank()) {
                return "external input " + i + " unsupported: " + reason;
            }
        }
        for (int i = 0; i < outputs.size(); i++) {
            String reason = unsupportedOutputReason(outputs.get(i));
            if (!reason.isBlank()) {
                return "output " + i + " unsupported: " + reason;
            }
        }
        return "";
    }

    private boolean shouldUseMetalBridge(ExecutionContext context) {
        if (context == null || !context.runsBackwardPass()) {
            return true;
        }
        return !containsForwardAttentionDag();
    }

    private boolean containsForwardAttentionDag() {
        return plan.lowering().dagSpec().nodes().stream().anyMatch(node -> switch (node.type()) {
            case SDPA -> true;
            default -> false;
        });
    }

    private List<Tensor> resolveExternalInputs(ExecutionContext context) {
        List<Tensor> computeResolvedInputs = computeCpuPlan.apply(
                computeNode.id(),
                PreparedAcceleratorExecutionSupport.resolveRuntimeInputs(computeNode, context),
                context
        );
        List<Tensor> resolved = new ArrayList<>(bridgeExecutable.externalInputNodeIds().size());
        for (int externalInputNodeId : bridgeExecutable.externalInputNodeIds()) {
            int computeInputIndex = computeNode.inputIds().indexOf(externalInputNodeId);
            if (computeInputIndex >= 0 && computeInputIndex < computeResolvedInputs.size()) {
                resolved.add(computeResolvedInputs.get(computeInputIndex));
            } else {
                resolved.add(context.runtimeTensorForNodeId(externalInputNodeId));
            }
        }
        return List.copyOf(resolved);
    }

    private static String unsupportedExternalInputReason(Tensor tensor) {
        if (tensor == null) {
            return "tensor is null";
        }
        if (!tensor.isContiguous()) {
            return "tensor is not contiguous";
        }
        if (tensor.hasStorageOffset()) {
            return "tensor has storage offset";
        }
        if (!MetalMpsCapabilities.supportsExternalInputDType(tensor.getDataType())) {
            return MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType());
        }
        return switch (tensor.getDataType()) {
            case FLOAT32 -> tensor.getFloat32Data() == null ? "missing direct float[] storage" : "";
            case BOOL -> tensor.getBoolData() == null ? "missing direct bool[] storage" : "";
            default -> MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType());
        };
    }

    private static String unsupportedOutputReason(Tensor tensor) {
        if (tensor == null) {
            return "tensor is null";
        }
        if (!tensor.isContiguous()) {
            return "tensor is not contiguous";
        }
        if (tensor.hasStorageOffset()) {
            return "tensor has storage offset";
        }
        if (!MetalMpsCapabilities.supportsOutputDType(tensor.getDataType())) {
            return MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType());
        }
        return tensor.getFloat32Data() == null ? "missing direct float[] storage" : "";
    }

    private static long byteSize(List<Tensor> tensors) {
        long bytes = 0L;
        if (tensors == null) {
            return 0L;
        }
        for (Tensor tensor : tensors) {
            if (tensor != null) {
                bytes += (long) tensor.getFlatDataSize() * elementByteSize(tensor.getDataType());
            }
        }
        return bytes;
    }

    private static int elementByteSize(DataType dataType) {
        if (dataType == null) {
            return 0;
        }
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
        };
    }

    /**
     * Returns the lowered Metal partition plan compiled for this executable.
     */
    public MetalPartitionPlan plan() {
        return plan;
    }

    /**
     * Returns the lowering family that produced this executable.
     */
    public LoweringFamily loweringFamily() {
        return loweringFamily;
    }

    /**
     * Returns the Metal MPS bridge used for compile and execute calls.
     */
    public MetalMpsGraphBridge bridge() {
        return bridge;
    }

    /**
     * Returns the bridge context created during preparation.
     */
    public MetalMpsBridgeContext bridgeContext() {
        return bridgeContext;
    }

    /**
     * Returns the compiled bridge executable, which may be unavailable with a reason.
     */
    public MetalMpsBridgeExecutable bridgeExecutable() {
        return bridgeExecutable;
    }

    /**
     * Returns diagnostics captured during the most recent execution attempt.
     *
     * <p>The value is updated for both Metal executions and CPU fallbacks, so
     * trace rendering can explain why a selected Metal region did or did not
     * enter the native bridge.</p>
     *
     * @return latest bridge execution diagnostics
     */
    public MetalMpsBridgeExecutionStats lastExecutionStats() {
        return lastExecutionStats;
    }
}
