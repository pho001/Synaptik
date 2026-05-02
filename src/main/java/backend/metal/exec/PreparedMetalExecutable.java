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
import backend.accelerator.lowering.GpuLoweredRegionManifest;
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
import backend.metal.kernel.MetalCustomKernelBridge;
import backend.metal.kernel.MetalCustomKernelExecutable;
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
    private final MetalCustomKernelBridge customKernelBridge;
    private final MetalCustomKernelExecutable customKernelExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;
    private final AcceleratorBackendConfig backendConfig;
    private final MetalAcceleratorBufferBinder bufferBinder;
    private final MetalPreparedTransportPlan preparedTransportPlan;
    private final MetalRouteDecision preparedRouteDecision;
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
        this(
                plan,
                loweringFamily,
                bridge,
                cpuFallbackSteps,
                backendConfig,
                MetalCustomKernelBridge.unavailable()
        );
    }

    /**
     * Creates a prepared Metal executable with explicit backend and custom-kernel policies.
     */
    public PreparedMetalExecutable(
            MetalPartitionPlan plan,
            LoweringFamily loweringFamily,
            MetalMpsGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig,
            MetalCustomKernelBridge customKernelBridge
    ) {
        this.plan = Objects.requireNonNull(plan, "plan cannot be null");
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.customKernelBridge = customKernelBridge == null ? MetalCustomKernelBridge.unavailable() : customKernelBridge;
        this.bridgeContext = bridge.createContext();
        this.bridgeExecutable = bridge.compile(bridgeContext, plan);
        this.customKernelExecutable = this.customKernelBridge.compile(plan);
        this.cpuFallbackSteps = List.copyOf(cpuFallbackSteps == null ? List.of() : cpuFallbackSteps);
        this.backendConfig = backendConfig == null ? AcceleratorBackendConfig.defaults() : backendConfig;
        this.bufferBinder = new MetalAcceleratorBufferBinder(bridge, bridgeContext);
        this.preparedTransportPlan = MetalPreparedTransportPlan.prepare(
                plan,
                bridge,
                bridgeContext,
                bridgeExecutable,
                this.backendConfig,
                bufferBinder
        );
        this.preparedRouteDecision = MetalExecutionRouter.decide(
                plan,
                bridge.capabilities(),
                this.backendConfig,
                preparedTransportPlan.toRouteEvidence(),
                this.customKernelBridge.capabilities(),
                customKernelExecutable
        );
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
     * Executes through the Metal bridge when the prepared transport plan and runtime bindings allow it.
     *
     * <p>Run-independent transport gates are fixed in the constructor. Execution only validates facts that can
     * change per run: current device bindings, residency/currentness, concrete tensor layouts, and output buffer
     * allocation/reuse.</p>
     */
    @Override
    public void execute(ExecutionContext context) {
        bufferBinder.registerRuntimeServices(context);
        AcceleratorBufferDecision staticDecision = preparedTransportPlan.toDecision();
        if (preparedTransportPlan.containsForwardAttentionDag() && context != null && context.runsBackwardPass()) {
            AcceleratorBufferDecision decision = preparedTransportPlan.backwardSdpaDecision();
            publishDecision(decision);
            requireBufferOrThrow(decision);
            runCpuFallback(context, toLegacyBufferDecision(decision), decision.reason());
            return;
        }

        if (preparedTransportPlan.preferredPath() == MetalPreparedTransportPath.STATIC_CPU_FALLBACK
                || preparedTransportPlan.preferredPath() == MetalPreparedTransportPath.UNAVAILABLE_REQUIRED) {
            publishDecision(staticDecision);
            requireBufferOrThrow(staticDecision);
            runCpuFallback(context, toLegacyBufferDecision(staticDecision), staticDecision.reason());
            return;
        }

        AcceleratorBufferDecision decision = staticDecision;
        ResolvedAcceleratorInputs nativeBufferInputs = null;
        AcceleratorBufferRequest request = null;
        if (preparedTransportPlan.preferredPath() == MetalPreparedTransportPath.BUFFER_BINDING) {
            nativeBufferInputs = AcceleratorPreparedInputResolver.resolveForNativeBufferBinding(
                    bridgeExecutable.externalInputNodeIds(),
                    context
            );
            request = bufferRequest(context);
            decision = bufferBinder.validateRuntime(
                    request,
                    nativeBufferInputs,
                    backendConfig.buffer(),
                    context
            );
            publishDecision(decision);
            requireBufferOrThrow(decision);
        } else {
            publishDecision(staticDecision);
            requireBufferOrThrow(staticDecision);
        }

        if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            try {
                AcceleratorBufferBindings<MetalBufferBinding> bindings = bufferBinder.resolve(request, nativeBufferInputs, decision, context);
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
                        nativeBufferInputs.executionExternalInputs(),
                        outputTensors(context)
                );
            }
            return;
        }

        ResolvedAcceleratorInputs resolvedInputs = AcceleratorPreparedInputResolver.resolve(
                cpuFallbackSteps,
                bridgeExecutable.externalInputNodeIds(),
                context
        );
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

    private static boolean containsForwardAttentionDag(MetalPartitionPlan plan) {
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
    @Override
    public GpuCompoundRegionSummary compoundSummary() {
        return plan.lowering().compoundSummary();
    }

    @Override
    public GpuLoweredRegionManifest gpuLoweredRegionManifest() {
        return plan.manifest();
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
     * Returns the optional custom-kernel bridge considered during route preparation.
     */
    public MetalCustomKernelBridge customKernelBridge() {
        return customKernelBridge;
    }

    /**
     * Returns the custom-kernel executable descriptor considered during route preparation.
     */
    public MetalCustomKernelExecutable customKernelExecutable() {
        return customKernelExecutable;
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

    /**
     * Returns the static transport plan prepared once for this executable.
     */
    public String preparedTransportPlan() {
        return preparedTransportPlan.describe();
    }

    /**
     * Returns the prepare-time route decision for this Metal executable.
     */
    public MetalRouteDecision routeDecision() {
        return preparedRouteDecision;
    }

    private enum MetalPreparedTransportPath {
        BUFFER_BINDING,
        TENSOR_ARRAY,
        STATIC_CPU_FALLBACK,
        UNAVAILABLE_REQUIRED
    }

    private record MetalPreparedTransportPlan(
            MetalPreparedTransportPath preferredPath,
            AcceleratorBufferBindingMode mode,
            AcceleratorBufferReasonCode reasonCode,
            String reason,
            boolean bridgeAvailable,
            boolean contextAvailable,
            boolean executableAvailable,
            boolean bufferAbiSupported,
            boolean staticDTypeLegal,
            boolean containsForwardAttentionDag,
            long estimatedWork,
            long minimumEstimatedWork
    ) {
        private static MetalPreparedTransportPlan prepare(
                MetalPartitionPlan plan,
                MetalMpsGraphBridge bridge,
                MetalMpsBridgeContext bridgeContext,
                MetalMpsBridgeExecutable bridgeExecutable,
                AcceleratorBackendConfig backendConfig,
                MetalAcceleratorBufferBinder bufferBinder
        ) {
            AcceleratorBufferBindingMode mode = backendConfig.buffer().bindingMode();
            boolean containsForwardAttentionDag = PreparedMetalExecutable.containsForwardAttentionDag(plan);
            long estimatedWork = plan.estimatedWork();
            long minimumEstimatedWork = backendConfig.buffer().minimumEstimatedWork();
            if (!bridge.isAvailable()) {
                return unavailable(mode, AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                        "bridge unavailable: " + bridge.unavailableReason(),
                        false, false, false, false, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }
            if (bridgeContext == null || !bridgeContext.available()) {
                return unavailable(mode, AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                        "bridge context unavailable: " + (bridgeContext == null ? "missing context" : bridgeContext.reason()),
                        true, false, false, false, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }
            if (bridgeExecutable == null || !bridgeExecutable.available()) {
                return unavailable(mode, AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                        "bridge executable unavailable: " + (bridgeExecutable == null ? "missing executable" : bridgeExecutable.reason()),
                        true, true, false, false, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }

            String dtypeReason = staticDTypeUnsupportedReason(bridgeExecutable);
            if (!dtypeReason.isBlank()) {
                return unavailable(mode, AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED,
                        dtypeReason,
                        true, true, true, bridge.supportsBufferBindings(), false, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }

            if (mode == AcceleratorBufferBindingMode.OFF) {
                return new MetalPreparedTransportPlan(
                        MetalPreparedTransportPath.TENSOR_ARRAY,
                        mode,
                        AcceleratorBufferReasonCode.BUFFER_BINDINGS_DISABLED,
                        "buffer bindings disabled",
                        true,
                        true,
                        true,
                        bridge.supportsBufferBindings(),
                        true,
                        containsForwardAttentionDag,
                        estimatedWork,
                        minimumEstimatedWork
                );
            }

            boolean bufferAbiSupported = bridge.supportsBufferBindings();
            if (!bufferAbiSupported) {
                return bufferUnavailable(mode, AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                        "native Metal buffer ABI unavailable: bridge does not support buffer bindings",
                        true, true, true, false, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }

            if (estimatedWork < minimumEstimatedWork) {
                return bufferUnavailable(mode, AcceleratorBufferReasonCode.BELOW_MINIMUM_WORK,
                        "estimated work below buffer minimum",
                        true, true, true, true, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }

            String allocatorReason = bufferBinder.prepareAllocatorUnavailableReason();
            if (!allocatorReason.isBlank()) {
                return bufferUnavailable(mode, AcceleratorBufferReasonCode.BUFFER_ALLOCATOR_UNAVAILABLE,
                        allocatorReason,
                        true, true, true, true, true, containsForwardAttentionDag, estimatedWork, minimumEstimatedWork);
            }

            return new MetalPreparedTransportPlan(
                    MetalPreparedTransportPath.BUFFER_BINDING,
                    mode,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                    "using native buffer bindings",
                    true,
                    true,
                    true,
                    true,
                    true,
                    containsForwardAttentionDag,
                    estimatedWork,
                    minimumEstimatedWork
            );
        }

        private static MetalPreparedTransportPlan unavailable(
                AcceleratorBufferBindingMode mode,
                AcceleratorBufferReasonCode reasonCode,
                String reason,
                boolean bridgeAvailable,
                boolean contextAvailable,
                boolean executableAvailable,
                boolean bufferAbiSupported,
                boolean staticDTypeLegal,
                boolean containsForwardAttentionDag,
                long estimatedWork,
                long minimumEstimatedWork
        ) {
            return new MetalPreparedTransportPlan(
                    mode == AcceleratorBufferBindingMode.REQUIRE
                            ? MetalPreparedTransportPath.UNAVAILABLE_REQUIRED
                            : MetalPreparedTransportPath.STATIC_CPU_FALLBACK,
                    mode,
                    reasonCode,
                    reason,
                    bridgeAvailable,
                    contextAvailable,
                    executableAvailable,
                    bufferAbiSupported,
                    staticDTypeLegal,
                    containsForwardAttentionDag,
                    estimatedWork,
                    minimumEstimatedWork
            );
        }

        private static MetalPreparedTransportPlan bufferUnavailable(
                AcceleratorBufferBindingMode mode,
                AcceleratorBufferReasonCode reasonCode,
                String reason,
                boolean bridgeAvailable,
                boolean contextAvailable,
                boolean executableAvailable,
                boolean bufferAbiSupported,
                boolean staticDTypeLegal,
                boolean containsForwardAttentionDag,
                long estimatedWork,
                long minimumEstimatedWork
        ) {
            return new MetalPreparedTransportPlan(
                    mode == AcceleratorBufferBindingMode.REQUIRE
                            ? MetalPreparedTransportPath.UNAVAILABLE_REQUIRED
                            : MetalPreparedTransportPath.TENSOR_ARRAY,
                    mode,
                    reasonCode,
                    reason,
                    bridgeAvailable,
                    contextAvailable,
                    executableAvailable,
                    bufferAbiSupported,
                    staticDTypeLegal,
                    containsForwardAttentionDag,
                    estimatedWork,
                    minimumEstimatedWork
            );
        }

        private AcceleratorBufferDecision toDecision() {
            AcceleratorBufferExecutionPath executionPath = switch (preferredPath) {
                case BUFFER_BINDING -> AcceleratorBufferExecutionPath.BUFFER_BINDING;
                case TENSOR_ARRAY -> AcceleratorBufferExecutionPath.TENSOR_ARRAY;
                case STATIC_CPU_FALLBACK -> AcceleratorBufferExecutionPath.CPU_FALLBACK;
                case UNAVAILABLE_REQUIRED -> AcceleratorBufferExecutionPath.UNAVAILABLE;
            };
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_METAL,
                    mode,
                    executionPath,
                    preferredPath == MetalPreparedTransportPath.BUFFER_BINDING,
                    mode == AcceleratorBufferBindingMode.REQUIRE,
                    reasonCode,
                    reason,
                    List.of(),
                    List.of()
            );
        }

        private AcceleratorBufferDecision backwardSdpaDecision() {
            AcceleratorBufferExecutionPath path = mode == AcceleratorBufferBindingMode.REQUIRE
                    ? AcceleratorBufferExecutionPath.UNAVAILABLE
                    : AcceleratorBufferExecutionPath.CPU_FALLBACK;
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_METAL,
                    mode,
                    path,
                    false,
                    mode == AcceleratorBufferBindingMode.REQUIRE,
                    AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                    "backward pass contains forward SDPA DAG unsupported by current Metal bridge",
                    List.of(),
                    List.of()
            );
        }

        private String describe() {
            return "preferredPath=" + preferredPath
                    + ", mode=" + mode
                    + ", reasonCode=" + reasonCode
                    + ", bridgeAvailable=" + bridgeAvailable
                    + ", contextAvailable=" + contextAvailable
                    + ", executableAvailable=" + executableAvailable
                    + ", bufferAbiSupported=" + bufferAbiSupported
                    + ", staticDTypeLegal=" + staticDTypeLegal
                    + ", containsForwardAttentionDag=" + containsForwardAttentionDag
                    + ", estimatedWork=" + estimatedWork
                    + ", minimumEstimatedWork=" + minimumEstimatedWork
                    + ", reason=" + reason;
        }

        private MetalExecutionRouter.TransportEvidence toRouteEvidence() {
            return new MetalExecutionRouter.TransportEvidence(
                    switch (preferredPath) {
                        case BUFFER_BINDING -> MetalExecutionRouter.TransportPath.BUFFER_BINDING;
                        case TENSOR_ARRAY -> MetalExecutionRouter.TransportPath.TENSOR_ARRAY;
                        case STATIC_CPU_FALLBACK -> MetalExecutionRouter.TransportPath.STATIC_CPU_FALLBACK;
                        case UNAVAILABLE_REQUIRED -> MetalExecutionRouter.TransportPath.UNAVAILABLE_REQUIRED;
                    },
                    mode,
                    reasonCode,
                    reason,
                    bridgeAvailable,
                    contextAvailable,
                    executableAvailable,
                    bufferAbiSupported,
                    staticDTypeLegal,
                    containsForwardAttentionDag,
                    estimatedWork,
                    minimumEstimatedWork
            );
        }

        private static String staticDTypeUnsupportedReason(MetalMpsBridgeExecutable bridgeExecutable) {
            for (DataType dtype : bridgeExecutable.externalInputDataTypes()) {
                if (!MetalMpsCapabilities.supportsExternalInputDType(dtype)) {
                    return "static Metal external input dtype unsupported: "
                            + MetalMpsCapabilities.unsupportedDTypeMessage(dtype);
                }
            }
            for (DataType dtype : bridgeExecutable.outputDataTypes()) {
                if (!MetalMpsCapabilities.supportsOutputDType(dtype)) {
                    return "static Metal output dtype unsupported: "
                            + MetalMpsCapabilities.unsupportedDTypeMessage(dtype);
                }
            }
            return "";
        }
    }

}
