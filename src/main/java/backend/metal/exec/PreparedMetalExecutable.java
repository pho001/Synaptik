package backend.metal.exec;

import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.metal.lowering.MetalPartitionPlan;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.metal.MetalMpsCapabilities;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
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
import java.util.Arrays;
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
    private volatile String lastBufferBindingDecision = "not executed yet";
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
     *
     * <p>Buffer-binding execution is evaluated before the legacy tensor-array copy contract. This keeps the
     * future shared-buffer path independent from Java-array layout restrictions such as direct {@code float[]}
     * storage, contiguity, and storage offset. Those checks are applied only when the executable has to use the
     * copy-based bridge path.</p>
     */
    @Override
    public void execute(ExecutionContext context) {
        String bridgeFallbackReason = metalBridgeUnavailableReason(context);
        if (!bridgeFallbackReason.isBlank()) {
            runCpuFallback(context, "not evaluated: " + bridgeFallbackReason, bridgeFallbackReason);
            return;
        }

        BufferBindingResolution externalInputBindings = resolveMetalBufferBindings(
                context,
                bridgeExecutable.externalInputNodeIds(),
                MetalBufferAccess.READ,
                "external input"
        );
        BufferBindingResolution outputBindings = resolveMetalBufferBindings(
                context,
                bridgeExecutable.outputNodeIds(),
                MetalBufferAccess.WRITE,
                "output"
        );

        if (bridge.supportsBufferBindings()
                && externalInputBindings.available()
                && outputBindings.available()) {
            lastBufferBindingDecision = "using native buffer bindings";
            try {
                lastExecutionStats = bridge.executeBuffers(
                        bridgeContext,
                        bridgeExecutable,
                        externalInputBindings.bindings(),
                        outputBindings.bindings()
                );
                if (!lastExecutionStats.usedCpuFallback()) {
                    markBufferOutputsCurrent(context, outputBindings.bindings());
                }
            } catch (RuntimeException ex) {
                runCpuFallback(
                        context,
                        "buffer binding execution failed: " + safeMessage(ex),
                        "buffer binding execution failed: " + safeMessage(ex)
                );
            }
            return;
        }

        lastBufferBindingDecision = bufferBindingDecision(externalInputBindings, outputBindings);
        List<Tensor> resolvedExternalInputs = bridgeExecutable.externalInputNodeIds().isEmpty()
                ? List.of()
                : resolveExternalInputs(context);
        List<Tensor> outputs = bridgeExecutable.outputNodeIds().isEmpty()
                ? List.of()
                : PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(bridgeExecutable.outputNodeIds(), context);
        String tensorArrayFallbackReason = tensorArrayFallbackReason(resolvedExternalInputs, outputs);
        if (tensorArrayFallbackReason.isBlank()) {
            try {
                lastExecutionStats = bridge.execute(bridgeContext, bridgeExecutable, resolvedExternalInputs, outputs);
            } catch (RuntimeException ex) {
                runCpuFallback(
                        context,
                        lastBufferBindingDecision,
                        "tensor-array bridge execution failed: " + safeMessage(ex),
                        resolvedExternalInputs,
                        outputs
                );
            }
            return;
        }

        runCpuFallback(context, lastBufferBindingDecision, tensorArrayFallbackReason, resolvedExternalInputs, outputs);
    }

    private void runCpuFallback(ExecutionContext context, String bufferBindingDecision, String fallbackReason) {
        List<Tensor> resolvedExternalInputs = bridgeExecutable.externalInputNodeIds().isEmpty()
                ? List.of()
                : resolveExternalInputs(context);
        List<Tensor> outputs = bridgeExecutable.outputNodeIds().isEmpty()
                ? List.of()
                : PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(bridgeExecutable.outputNodeIds(), context);
        runCpuFallback(context, bufferBindingDecision, fallbackReason, resolvedExternalInputs, outputs);
    }

    private void runCpuFallback(
            ExecutionContext context,
            String bufferBindingDecision,
            String fallbackReason,
            List<Tensor> resolvedExternalInputs,
            List<Tensor> outputs
    ) {
        lastBufferBindingDecision = bufferBindingDecision;
        lastExecutionStats = MetalMpsBridgeExecutionStats.fallback(
                fallbackReason,
                resolvedExternalInputs.size(),
                outputs.size(),
                byteSize(resolvedExternalInputs),
                byteSize(outputs)
        );
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
    }

    private String bufferBindingDecision(BufferBindingResolution externalInputBindings, BufferBindingResolution outputBindings) {
        if (!bridge.supportsBufferBindings()) {
            return "tensor-array copy path: bridge does not support buffer bindings";
        }
        if (!externalInputBindings.available()) {
            return "tensor-array copy path: " + externalInputBindings.reason();
        }
        if (!outputBindings.available()) {
            return "tensor-array copy path: " + outputBindings.reason();
        }
        return "tensor-array copy path";
    }

    private String metalBridgeUnavailableReason(ExecutionContext context) {
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
        return "";
    }

    private String tensorArrayFallbackReason(List<Tensor> resolvedExternalInputs, List<Tensor> outputs) {
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

    private static BufferBindingResolution resolveMetalBufferBindings(
            ExecutionContext context,
            List<Integer> nodeIds,
            MetalBufferAccess requiredAccess,
            String role
    ) {
        if (nodeIds == null || nodeIds.isEmpty()) {
            return BufferBindingResolution.available(List.of());
        }
        List<MetalBufferBinding> bindings = new ArrayList<>(nodeIds.size());
        for (int nodeId : nodeIds) {
            DeviceBufferBinding binding = requiredAccess == MetalBufferAccess.WRITE
                    ? context.writableDeviceBufferBindingForNodeId(nodeId)
                    : context.deviceBufferBindingForNodeId(nodeId);
            if (binding == null) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId + " has no device buffer binding");
            }
            if (!(binding instanceof MetalBufferBinding metalBinding)) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding is not Metal-compatible: " + binding.getClass().getSimpleName());
            }
            if (!metalBinding.available()) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding is unavailable: " + metalBinding.describe());
            }
            Tensor tensor = context.runtimeTensorForNodeId(nodeId);
            if (metalBinding.dataType() != tensor.getDataType()) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding dtype " + metalBinding.dataType()
                        + " does not match tensor dtype " + tensor.getDataType());
            }
            if (!Arrays.equals(metalBinding.shape(), tensor.getShape())) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding shape " + Arrays.toString(metalBinding.shape())
                        + " does not match tensor shape " + Arrays.toString(tensor.getShape()));
            }
            if (metalBinding.elementCount() != tensor.getFlatDataSize()) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding elementCount " + metalBinding.elementCount()
                        + " does not match tensor elementCount " + tensor.getFlatDataSize());
            }
            if (!accessCompatible(metalBinding.access(), requiredAccess)) {
                return BufferBindingResolution.unavailable(role + " nodeId=" + nodeId
                        + " binding access " + metalBinding.access()
                        + " is incompatible with required " + requiredAccess);
            }
            bindings.add(metalBinding);
        }
        return BufferBindingResolution.available(bindings);
    }

    private static void markBufferOutputsCurrent(ExecutionContext context, List<MetalBufferBinding> outputBindings) {
        if (context == null || outputBindings == null || outputBindings.isEmpty()) {
            return;
        }
        for (MetalBufferBinding binding : outputBindings) {
            context.attachDeviceBufferBinding(
                    binding.nodeId(),
                    binding,
                    residencyForOutputBinding(binding),
                    "metal buffer binding output"
            );
        }
    }

    private static StorageResidency residencyForOutputBinding(MetalBufferBinding binding) {
        if (binding.handle().hostShared()) {
            return StorageResidency.HOST_SHARED_DEVICE_BUFFER;
        }
        return StorageResidency.DEVICE_OWNED;
    }

    private static boolean accessCompatible(MetalBufferAccess actual, MetalBufferAccess required) {
        if (actual == MetalBufferAccess.READ_WRITE) {
            return true;
        }
        return actual == required;
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

    private static String safeMessage(RuntimeException ex) {
        if (ex == null) {
            return "unknown error";
        }
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
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

    /**
     * Returns the buffer-binding path decision from the most recent execution attempt.
     *
     * <p>The current FFM bridge normally reports that it does not support buffer bindings. Future
     * shared-buffer bridges should report when all required bindings were present and
     * {@link MetalMpsGraphBridge#executeBuffers(MetalMpsBridgeContext, MetalMpsBridgeExecutable, List, List)}
     * was selected.</p>
     *
     * @return latest buffer-binding selection diagnostic
     */
    public String lastBufferBindingDecision() {
        return lastBufferBindingDecision;
    }

    private record BufferBindingResolution(List<MetalBufferBinding> bindings, String reason) {
        private BufferBindingResolution {
            bindings = List.copyOf(bindings == null ? List.of() : bindings);
            reason = reason == null ? "" : reason;
        }

        static BufferBindingResolution available(List<MetalBufferBinding> bindings) {
            return new BufferBindingResolution(bindings, "");
        }

        static BufferBindingResolution unavailable(String reason) {
            return new BufferBindingResolution(List.of(), reason);
        }

        boolean available() {
            return reason.isBlank();
        }
    }
}
