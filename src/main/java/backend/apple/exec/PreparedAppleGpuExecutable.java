package backend.apple.exec;

import backend.CPUBackend;
import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.kernels.cpu.CpuNodeExecutionPlan;
import backend.apple.bridge.AppleMpsBridgeContext;
import backend.apple.bridge.AppleMpsBridgeExecutable;
import backend.apple.bridge.AppleMpsGraphBridge;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.apple.AppleGpuPartitionPlan;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PreparedAppleGpuExecutable implements PreparedAcceleratorExecutable {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();

    public record PreparedStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        public PreparedStep {
            Objects.requireNonNull(node, "node cannot be null");
            Objects.requireNonNull(metadata, "metadata cannot be null");
        }
    }

    private final AppleGpuPartitionPlan plan;
    private final CompiledNode computeNode;
    private final CpuNodeExecutionPlan computeCpuPlan;
    private final AppleMpsGraphBridge bridge;
    private final AppleMpsBridgeContext bridgeContext;
    private final AppleMpsBridgeExecutable bridgeExecutable;
    private final List<PreparedStep> cpuFallbackSteps;

    public PreparedAppleGpuExecutable(
            AppleGpuPartitionPlan plan,
            CompiledNode computeNode,
            CpuNodeExecutionPlan computeCpuPlan,
            AppleMpsGraphBridge bridge,
            List<PreparedStep> cpuFallbackSteps
    ) {
        this.plan = Objects.requireNonNull(plan, "plan cannot be null");
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
        if (bridge.isAvailable() && bridgeContext.available() && bridgeExecutable.available()) {
            List<Tensor> resolvedExternalInputs = resolveExternalInputs(context);
            Tensor out = context.runtimeTensorForNodeId(bridgeExecutable.outputNodeId());
            if (resolvedExternalInputs.stream().allMatch(PreparedAppleGpuExecutable::isSupportedAppleInput)
                    && isSupportedAppleInput(out)) {
                bridge.execute(bridgeContext, bridgeExecutable, resolvedExternalInputs, out);
                return;
            }
        }
        for (PreparedStep step : cpuFallbackSteps) {
            CPU_BACKEND.execute(step.node(), step.metadata(), context);
        }
    }

    private List<Tensor> resolveExternalInputs(ExecutionContext context) {
        List<Tensor> computeResolvedInputs = computeCpuPlan.apply(
                computeNode.id(),
                resolveRuntimeInputs(computeNode, context),
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

    private static List<Tensor> resolveRuntimeInputs(CompiledNode node, ExecutionContext context) {
        if (node.inputIds().isEmpty()) {
            return List.of();
        }
        List<Tensor> out = new ArrayList<>(node.inputIds().size());
        for (int inputNodeId : node.inputIds()) {
            out.add(context.runtimeTensorForNodeId(inputNodeId));
        }
        return List.copyOf(out);
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
