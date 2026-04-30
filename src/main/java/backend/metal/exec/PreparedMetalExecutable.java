package backend.metal.exec;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferBindings;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.exec.AcceleratorPreparedInputResolver;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.metal.lowering.MetalPartitionPlan;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.memory.StorageResidency;
import backend.metal.MetalMpsCapabilities;
import backend.metal.buffer.MetalAcceleratorBufferBinder;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutable;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;
import backend.metal.bridge.MetalMpsGraphBridge;
import backend.runtime.ExecutionContext;
import backend.lowering.LoweringFamily;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import tensor.DataType;
import tensor.Tensor;

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
    private final MetalMpsGraphBridge bridge;
    private final MetalMpsBridgeContext bridgeContext;
    private final MetalMpsBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;
    private final AcceleratorBackendConfig backendConfig;
    private final MetalAcceleratorBufferBinder bufferBinder;
    private volatile String lastBufferBindingDecision = "not executed yet";
    private volatile AcceleratorBufferDecision lastAcceleratorBufferDecision =
            AcceleratorBufferDecision.notEvaluated(ComputeBackend.GPU_METAL);
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
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        this(plan, loweringFamily, bridge, cpuFallbackSteps, AcceleratorBackendConfig.defaults());
    }

    /**
     * Creates a prepared Metal executable with an explicit backend runtime policy.
     */
    public PreparedMetalExecutable(
            MetalPartitionPlan plan,
            LoweringFamily loweringFamily,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig
    ) {
        this.plan = Objects.requireNonNull(plan, "plan cannot be null");
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = bridge.createContext();
        this.bridgeExecutable = bridge.compile(bridgeContext, plan);
        this.cpuFallbackSteps = List.copyOf(cpuFallbackSteps == null ? List.of() : cpuFallbackSteps);
        this.backendConfig = backendConfig == null ? AcceleratorBackendConfig.defaults() : backendConfig;
        this.bufferBinder = new MetalAcceleratorBufferBinder(bridge, bridgeContext);
    }

    /**
     * Returns {@link ComputeBackend#GPU_METAL}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_METAL;
    }

    @Override
    public List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps() {
        return cpuFallbackSteps;
    }

    /**
     * Executes through the Metal bridge when available and compatible, otherwise runs CPU fallback steps.
     *
     * <p>Buffer-binding execution is evaluated before the legacy tensor-array copy contract. This keeps the
     * native shared-buffer path independent from Java-array layout restrictions such as direct {@code float[]}
     * storage, contiguity, and storage offset. Those checks are applied only when the executable has to use the
     * copy-based bridge path.</p>
     */
    @Override
    public void execute(ExecutionContext context) {
        String bridgeFallbackReason = metalBridgeUnavailableReason(context);
        if (!bridgeFallbackReason.isBlank()) {
            AcceleratorBufferDecision decision = bridgeUnavailableDecision(bridgeFallbackReason);
            publishDecision(decision);
            requireBufferOrThrow(decision);
            runCpuFallback(context, toLegacyBufferDecision(decision), bridgeFallbackReason);
            return;
        }

        ResolvedAcceleratorInputs resolvedInputs = AcceleratorPreparedInputResolver.resolve(
                cpuFallbackSteps,
                bridgeExecutable.externalInputNodeIds(),
                context
        );
        AcceleratorBufferRequest request = bufferRequest(context);
        AcceleratorBufferDecision decision = bufferBinder.decide(
                request,
                resolvedInputs,
                backendConfig.buffer(),
                context
        );
        publishDecision(decision);
        requireBufferOrThrow(decision);

        if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            try {
                AcceleratorBufferBindings<MetalBufferBinding> bindings = bufferBinder.resolve(request, resolvedInputs, decision, context);
                lastExecutionStats = bridge.executeBuffers(
                        bridgeContext,
                        bridgeExecutable,
                        bindings.inputs(),
                        bindings.outputs()
                );
                if (!lastExecutionStats.usedCpuFallback()) {
                    markBufferOutputsCurrent(context, bindings.outputs());
                }
            } catch (RuntimeException ex) {
                AcceleratorBufferDecision failure = new AcceleratorBufferDecision(
                        ComputeBackend.GPU_METAL,
                        backendConfig.buffer().bindingMode(),
                        backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE
                                ? AcceleratorBufferExecutionPath.UNAVAILABLE
                                : AcceleratorBufferExecutionPath.CPU_FALLBACK,
                        false,
                        backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE,
                        AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                        "buffer binding execution failed: " + safeMessage(ex),
                        decision.inputs(),
                        decision.outputs()
                );
                publishDecision(failure);
                requireBufferOrThrow(failure);
                runCpuFallback(
                        context,
                        toLegacyBufferDecision(failure),
                        failure.reason(),
                        resolvedInputs.executionExternalInputs(),
                        outputTensors(context)
                );
            }
            return;
        }

        ensureTensorArrayInputsCpuReadable(context);
        List<Tensor> resolvedExternalInputs = resolvedInputs.executionExternalInputs();
        List<Tensor> outputs = outputTensors(context);
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

    private void ensureTensorArrayInputsCpuReadable(ExecutionContext context) {
        for (int nodeId : bridgeExecutable.externalInputNodeIds()) {
            context.requireCpuReadable(nodeId, backend.memory.CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private void runCpuFallback(ExecutionContext context, String bufferBindingDecision, String fallbackReason) {
        ResolvedAcceleratorInputs resolvedInputs = AcceleratorPreparedInputResolver.resolve(
                cpuFallbackSteps,
                bridgeExecutable.externalInputNodeIds(),
                context
        );
        List<Tensor> resolvedExternalInputs = resolvedInputs.executionExternalInputs();
        List<Tensor> outputs = outputTensors(context);
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
        ensureTensorArrayInputsCpuReadable(context);
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
    }

    private AcceleratorBufferRequest bufferRequest(ExecutionContext context) {
        return new AcceleratorBufferRequest(
                ComputeBackend.GPU_METAL,
                plan.estimatedWork(),
                bridgeExecutable.externalInputNodeIds(),
                bridgeExecutable.externalInputDataTypes(),
                layoutsForNodeIds(context, bridgeExecutable.externalInputNodeIds()),
                bridgeExecutable.outputNodeIds(),
                bridgeExecutable.outputDataTypes(),
                layoutsForNodeIds(context, bridgeExecutable.outputNodeIds()),
                context != null && context.runsBackwardPass()
        );
    }

    private static List<AcceleratorBufferLayout> layoutsForNodeIds(ExecutionContext context, List<Integer> nodeIds) {
        if (nodeIds == null || nodeIds.isEmpty()) {
            return List.of();
        }
        return nodeIds.stream()
                .map(nodeId -> AcceleratorBufferLayout.fromTensor(context.runtimeTensorForNodeId(nodeId)))
                .toList();
    }

    private List<Tensor> outputTensors(ExecutionContext context) {
        return bridgeExecutable.outputNodeIds().isEmpty()
                ? List.of()
                : PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(bridgeExecutable.outputNodeIds(), context);
    }

    private AcceleratorBufferDecision bridgeUnavailableDecision(String reason) {
        AcceleratorBufferBindingMode mode = backendConfig.buffer().bindingMode();
        return new AcceleratorBufferDecision(
                ComputeBackend.GPU_METAL,
                mode,
                mode == AcceleratorBufferBindingMode.REQUIRE
                        ? AcceleratorBufferExecutionPath.UNAVAILABLE
                        : AcceleratorBufferExecutionPath.CPU_FALLBACK,
                false,
                mode == AcceleratorBufferBindingMode.REQUIRE,
                AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                reason,
                List.of(),
                List.of()
        );
    }

    private void publishDecision(AcceleratorBufferDecision decision) {
        lastAcceleratorBufferDecision = decision == null
                ? AcceleratorBufferDecision.notEvaluated(ComputeBackend.GPU_METAL)
                : decision;
        lastBufferBindingDecision = toLegacyBufferDecision(lastAcceleratorBufferDecision);
    }

    private static String toLegacyBufferDecision(AcceleratorBufferDecision decision) {
        if (decision == null) {
            return "not evaluated";
        }
        if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return "using native buffer bindings";
        }
        if (decision.reason() == null || decision.reason().isBlank()) {
            return decision.path() == AcceleratorBufferExecutionPath.CPU_FALLBACK
                    ? "cpu fallback path"
                    : "tensor-array copy path";
        }
        return decision.path() == AcceleratorBufferExecutionPath.CPU_FALLBACK
                ? "cpu fallback path: " + decision.reason()
                : "tensor-array copy path: " + decision.reason();
    }

    private static void requireBufferOrThrow(AcceleratorBufferDecision decision) {
        if (decision != null && decision.required() && decision.path() != AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            throw new IllegalStateException("Accelerator buffer path is required for "
                    + decision.backend() + " but unavailable: "
                    + decision.reasonCode() + ": " + decision.reason());
        }
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

    private static void markBufferOutputsCurrent(ExecutionContext context, List<MetalBufferBinding> outputBindings) {
        if (context == null || outputBindings == null || outputBindings.isEmpty()) {
            return;
        }
        for (MetalBufferBinding binding : outputBindings) {
            MetalBufferBinding activeBinding = readableAfterWrite(binding);
            context.attachDeviceBufferBinding(
                    activeBinding.nodeId(),
                    activeBinding,
                    residencyForOutputBinding(activeBinding),
                    "metal buffer binding output"
            );
        }
    }

    private static MetalBufferBinding readableAfterWrite(MetalBufferBinding binding) {
        if (binding.access() == MetalBufferAccess.READ_WRITE) {
            return binding;
        }
        return new MetalBufferBinding(
                binding.nodeId(),
                binding.layout(),
                binding.handle(),
                MetalBufferAccess.READ_WRITE
        );
    }

    private static StorageResidency residencyForOutputBinding(MetalBufferBinding binding) {
        // The backing MTLBuffer may be host-shared, but the runtime Tensor's Java float[] storage is still
        // stale until the Metal materializer reads the buffer into that array.
        return StorageResidency.DEVICE_OWNED;
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
     * Returns the compound GPU summary associated with this prepared Metal executable.
     */
    public GpuCompoundRegionSummary compoundSummary() {
        return plan.lowering().compoundSummary();
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

    @Override
    public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
        return lastAcceleratorBufferDecision;
    }

}
