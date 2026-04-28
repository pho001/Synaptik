package backend.metal.exec;

import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.metal.lowering.MetalPartitionPlan;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.metal.MetalMpsCapabilities;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
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
        if (shouldUseMetalBridge(context)
                && PreparedAcceleratorExecutionSupport.bridgeReady(
                bridge.isAvailable(),
                bridgeContext.available(),
                bridgeExecutable.available())) {
            List<Tensor> resolvedExternalInputs = resolveExternalInputs(context);
            List<Tensor> outputs = PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(
                    bridgeExecutable.outputNodeIds(),
                    context
            );
            if (resolvedExternalInputs.stream().allMatch(PreparedMetalExecutable::isSupportedMetalExternalInput)
                    && outputs.stream().allMatch(PreparedMetalExecutable::isSupportedMetalOutput)) {
                bridge.execute(bridgeContext, bridgeExecutable, resolvedExternalInputs, outputs);
                return;
            }
        }
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
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

    private static boolean isSupportedMetalExternalInput(Tensor tensor) {
        if (tensor == null
                || !tensor.isContiguous()
                || tensor.hasStorageOffset()
                || !MetalMpsCapabilities.supportsExternalInputDType(tensor.getDataType())) {
            return false;
        }
        return switch (tensor.getDataType()) {
            case FLOAT32 -> tensor.getFloat32Data() != null;
            case BOOL -> tensor.getBoolData() != null;
            default -> false;
        };
    }

    private static boolean isSupportedMetalOutput(Tensor tensor) {
        return tensor != null
                && tensor.isContiguous()
                && !tensor.hasStorageOffset()
                && MetalMpsCapabilities.supportsOutputDType(tensor.getDataType())
                && tensor.getFloat32Data() != null;
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
}
