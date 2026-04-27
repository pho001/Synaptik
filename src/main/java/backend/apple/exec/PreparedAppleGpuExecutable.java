package backend.apple.exec;

import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.apple.lowering.AppleGpuPartitionPlan;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.apple.bridge.AppleMpsBridgeContext;
import backend.apple.bridge.AppleMpsBridgeExecutable;
import backend.apple.bridge.AppleMpsGraphBridge;
import backend.runtime.ExecutionContext;
import backend.lowering.LoweringFamily;
import graph.CompiledNode;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PreparedAppleGpuExecutable implements PreparedAcceleratorExecutable {
    private final AppleGpuPartitionPlan plan;
    private final LoweringFamily loweringFamily;
    private final CompiledNode computeNode;
    private final CpuNodeExecutionPlan computeCpuPlan;
    private final AppleMpsGraphBridge bridge;
    private final AppleMpsBridgeContext bridgeContext;
    private final AppleMpsBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;

    public PreparedAppleGpuExecutable(
            AppleGpuPartitionPlan plan,
            LoweringFamily loweringFamily,
            CompiledNode computeNode,
            CpuNodeExecutionPlan computeCpuPlan,
            AppleMpsGraphBridge bridge,
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

    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_METAL;
    }

    @Override
    public void execute(ExecutionContext context) {
        if (shouldUseAppleBridge(context)
                && PreparedAcceleratorExecutionSupport.bridgeReady(
                bridge.isAvailable(),
                bridgeContext.available(),
                bridgeExecutable.available())) {
            List<Tensor> resolvedExternalInputs = resolveExternalInputs(context);
            List<Tensor> outputs = PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(
                    bridgeExecutable.outputNodeIds(),
                    context
            );
            if (resolvedExternalInputs.stream().allMatch(PreparedAppleGpuExecutable::isSupportedAppleInput)
                    && outputs.stream().allMatch(PreparedAppleGpuExecutable::isSupportedAppleInput)) {
                bridge.execute(bridgeContext, bridgeExecutable, resolvedExternalInputs, outputs);
                return;
            }
        }
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
    }

    private boolean shouldUseAppleBridge(ExecutionContext context) {
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

    private static boolean isSupportedAppleInput(Tensor tensor) {
        return tensor != null
                && tensor.isContiguous()
                && !tensor.hasStorageOffset()
                && ((tensor.getDataType() == DataType.FLOAT32 && tensor.getFloat32Data() != null)
                || (tensor.getDataType() == DataType.BOOL && tensor.getBoolData() != null));
    }

    public AppleGpuPartitionPlan plan() {
        return plan;
    }

    public LoweringFamily loweringFamily() {
        return loweringFamily;
    }

    public AppleMpsGraphBridge bridge() {
        return bridge;
    }

    public AppleMpsBridgeContext bridgeContext() {
        return bridgeContext;
    }

    public AppleMpsBridgeExecutable bridgeExecutable() {
        return bridgeExecutable;
    }
}
