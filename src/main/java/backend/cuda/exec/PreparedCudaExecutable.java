package backend.cuda.exec;

import backend.contract.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferBindings;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.AcceleratorPreparedInputResolver;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.cuda.buffer.CudaAcceleratorBufferBinder;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferAllocator;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaDeviceLayoutMaterializer;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaBridgeExecutionStats;
import backend.cuda.bridge.CudaGraphBridge;
import backend.cuda.buffer.CudaDeviceToCpuMaterializer;
import backend.lowering.LoweringFamily;
import backend.lowering.region.RegionExecutionPlan;
import runtime.contract.CpuMaterializationReason;
import runtime.contract.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.accelerator.dag.AcceleratorDagSpec;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import graph.execution.device.DeviceLayoutMaterializer;
import tensor.Tensor;
import tensor.DataType;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Prepared CUDA partition executable.
 *
 * <p>The constructor creates the CUDA bridge context and attempts to compile the
 * lowered DAG immediately. {@link #execute(ExecutionContext)} uses the native
 * bridge only when the bridge, context, and executable are all available; otherwise
 * it replays the precomputed CPU fallback steps.</p>
 */
public final class PreparedCudaExecutable implements PreparedAcceleratorExecutable {
    private final AcceleratorDagSpec dagSpec;
    private final GpuCompoundRegionSummary compoundSummary;
    private final GpuLoweredRegionManifest gpuLoweredRegionManifest;
    private final LoweringFamily loweringFamily;
    private final RegionExecutionPlan regionExecutionPlan;
    private final CudaGraphBridge bridge;
    private final CudaBridgeContext bridgeContext;
    private final CudaBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;
    private final AcceleratorBackendConfig backendConfig;
    private final CudaAcceleratorBufferBinder bufferBinder;
    private volatile AcceleratorBufferDecision lastAcceleratorBufferDecision =
            AcceleratorBufferDecision.notEvaluated(ComputeBackend.GPU_CUDA);
    private volatile CudaBridgeExecutionStats lastExecutionStats = CudaBridgeExecutionStats.fallback(
            "not executed yet",
            0,
            0,
            0L,
            0L
    );

    /**
     * Creates a prepared CUDA executable around a lowered DAG and fallback plan.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        this(dagSpec, loweringFamily, bridge, cpuFallbackSteps, AcceleratorBackendConfig.defaults(), null);
    }

    /**
     * Creates a prepared CUDA executable with an explicit backend runtime policy.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig
    ) {
        this(dagSpec, loweringFamily, bridge, cpuFallbackSteps, backendConfig, null);
    }

    /**
     * Creates a prepared CUDA executable with an explicit backend runtime policy and compound summary.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig,
            GpuCompoundRegionSummary compoundSummary
    ) {
        this(dagSpec, loweringFamily, bridge, cpuFallbackSteps, backendConfig, compoundSummary, null);
    }

    /**
     * Creates a prepared CUDA executable with trace/report manifest metadata.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig,
            GpuCompoundRegionSummary compoundSummary,
            GpuLoweredRegionManifest gpuLoweredRegionManifest
    ) {
        this(dagSpec, loweringFamily, null, bridge, cpuFallbackSteps, backendConfig, compoundSummary, gpuLoweredRegionManifest);
    }

    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            RegionExecutionPlan regionExecutionPlan,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps,
            AcceleratorBackendConfig backendConfig,
            GpuCompoundRegionSummary compoundSummary,
            GpuLoweredRegionManifest gpuLoweredRegionManifest
    ) {
        this.dagSpec = Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        this.compoundSummary = compoundSummary == null
                ? GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, dagSpec.outputNodeIds())
                : compoundSummary;
        this.gpuLoweredRegionManifest = gpuLoweredRegionManifest;
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        this.regionExecutionPlan = regionExecutionPlan;
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = bridge.createContext();
        this.bridgeExecutable = bridge.compile(bridgeContext, dagSpec);
        this.cpuFallbackSteps = List.copyOf(cpuFallbackSteps == null ? List.of() : cpuFallbackSteps);
        this.backendConfig = backendConfig == null ? AcceleratorBackendConfig.defaults() : backendConfig;
        this.bufferBinder = new CudaAcceleratorBufferBinder(bridge);
    }

    /**
     * Returns {@link ComputeBackend#GPU_CUDA}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

    @Override
    public List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps() {
        return cpuFallbackSteps;
    }

    /**
     * Executes through the CUDA graph bridge when available, otherwise runs CPU fallback steps.
     */
    @Override
    public void execute(ExecutionContext context) {
        String bridgeUnavailableReason = cudaBridgeUnavailableReason();
        if (!bridgeUnavailableReason.isBlank()) {
            AcceleratorBufferDecision decision = bridgeUnavailableDecision(bridgeUnavailableReason);
            publishDecision(decision);
            requireBufferOrThrow(decision);
            lastExecutionStats = CudaBridgeExecutionStats.fallback(bridgeUnavailableReason, 0, 0, 0L, 0L);
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            return;
        }
        if (context == null) {
            AcceleratorBufferDecision decision = new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    backendConfig.buffer().bindingMode(),
                    backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE
                            ? AcceleratorBufferExecutionPath.UNAVAILABLE
                            : AcceleratorBufferExecutionPath.TENSOR_ARRAY,
                    false,
                    backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE,
                    AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
                    "CUDA buffer execution requires runtime tensor context",
                    List.of(),
                    List.of()
            );
            publishDecision(decision);
            requireBufferOrThrow(decision);
            lastExecutionStats = CudaBridgeExecutionStats.fallback(decision.reason(), 0, 0, 0L, 0L);
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            return;
        }

        CudaBufferAllocator allocator = null;
        if (backendConfig.buffer().bindingMode() != AcceleratorBufferBindingMode.OFF && bridge.supportsBufferBindings()) {
            allocator = bridge.createBufferAllocator(bridgeContext);
            registerRuntimeServices(context, allocator);
        }
        ResolvedAcceleratorInputs nativeBufferInputs = AcceleratorPreparedInputResolver.resolveForNativeBufferBinding(
                bridgeExecutable.externalInputNodeIds(),
                context
        );
        AcceleratorBufferRequest request = bufferRequest(context);
        AcceleratorBufferDecision decision = bufferBinder.decide(
                request,
                nativeBufferInputs,
                backendConfig.buffer(),
                context
        );
        publishDecision(decision);
        requireBufferOrThrow(decision);

        if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            try {
                if (allocator == null) {
                    allocator = bridge.createBufferAllocator(bridgeContext);
                    registerRuntimeServices(context, allocator);
                }
                AcceleratorBufferBindings<CudaBufferBinding> bindings = bufferBinder.resolve(
                        request,
                        nativeBufferInputs,
                        decision,
                        context,
                        allocator
                );
                long startNs = System.nanoTime();
                bridge.executeBuffers(bridgeContext, bridgeExecutable, bindings.inputs(), bindings.outputs());
                long elapsedNs = System.nanoTime() - startNs;
                lastExecutionStats = new CudaBridgeExecutionStats(
                        false,
                        "",
                        AcceleratorBufferExecutionPath.BUFFER_BINDING,
                        bindings.inputs().size(),
                        bindings.outputs().size(),
                        logicalBytes(bindings.inputs()),
                        logicalBytes(bindings.outputs()),
                        0L,
                        elapsedNs,
                        0L,
                        0L,
                        elapsedNs
                );
                markBufferOutputsCurrent(context, bindings.outputs());
            } catch (RuntimeException ex) {
                AcceleratorBufferDecision failure = new AcceleratorBufferDecision(
                        ComputeBackend.GPU_CUDA,
                        backendConfig.buffer().bindingMode(),
                        backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE
                                ? AcceleratorBufferExecutionPath.UNAVAILABLE
                                : AcceleratorBufferExecutionPath.CPU_FALLBACK,
                        false,
                        backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE,
                        AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED,
                        "CUDA buffer binding execution failed: " + safeMessage(ex),
                        decision.inputs(),
                        decision.outputs()
                );
                publishDecision(failure);
                requireBufferOrThrow(failure);
                lastExecutionStats = CudaBridgeExecutionStats.fallback(
                        failure.reason(),
                        failure.inputs().size(),
                        failure.outputs().size(),
                        logicalBytesFromInputs(failure.inputs()),
                        logicalBytesFromOutputs(failure.outputs())
                );
                PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            }
            return;
        }

        ResolvedAcceleratorInputs resolvedInputs = AcceleratorPreparedInputResolver.resolve(
                cpuFallbackSteps,
                bridgeExecutable.externalInputNodeIds(),
                context
        );
        if (decision.path() == AcceleratorBufferExecutionPath.CPU_FALLBACK) {
            List<Tensor> resolvedExternalInputs = resolvedInputs.executionExternalInputs();
            List<Tensor> outputs = outputTensors(context);
            lastExecutionStats = CudaBridgeExecutionStats.fallback(
                    decision.reason(),
                    resolvedExternalInputs.size(),
                    outputs.size(),
                    byteSize(resolvedExternalInputs),
                    byteSize(outputs)
            );
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            return;
        }

        ensureTensorArrayInputsCpuReadable(context);
        List<Tensor> resolvedExternalInputs = resolvedInputs.executionExternalInputs();
        List<Tensor> outputs = outputTensors(context);
        try {
            long startNs = System.nanoTime();
            bridge.execute(
                    bridgeContext,
                    bridgeExecutable,
                    resolvedExternalInputs,
                    outputs
            );
            long elapsedNs = System.nanoTime() - startNs;
            lastExecutionStats = new CudaBridgeExecutionStats(
                    false,
                    "",
                    AcceleratorBufferExecutionPath.TENSOR_ARRAY,
                    resolvedExternalInputs.size(),
                    outputs.size(),
                    byteSize(resolvedExternalInputs),
                    byteSize(outputs),
                    0L,
                    elapsedNs,
                    0L,
                    0L,
                    elapsedNs
            );
        } catch (RuntimeException ex) {
            lastExecutionStats = CudaBridgeExecutionStats.fallback(
                    "tensor-array bridge execution failed: " + safeMessage(ex),
                    resolvedExternalInputs.size(),
                    outputs.size(),
                    byteSize(resolvedExternalInputs),
                    byteSize(outputs)
            );
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
        }
    }

    @Override
    public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
        return lastAcceleratorBufferDecision;
    }

    @Override
    public String outputResidencyReason() {
        return switch (lastExecutionStats.executionPath()) {
            case BUFFER_BINDING -> "cuda buffer binding execution wrote device buffer";
            case CPU_FALLBACK -> "cuda cpu fallback wrote CPU array";
            default -> "cuda bridge copied output to CPU array";
        };
    }

    @Override
    public void contributeRunTraceAttributes(LinkedHashMap<String, Object> attrs) {
        var cudaStats = lastExecutionStats();
        attrs.put("cudaBridgeAvailable", bridge().isAvailable());
        attrs.put("cudaBridgeContextAvailable", bridgeContext().available());
        attrs.put("cudaBridgeExecutableAvailable", bridgeExecutable().available());
        attrs.put("cudaSupportsBufferBindings", bridge().supportsBufferBindings());
        attrs.put("cudaUsedCpuFallback", cudaStats.usedCpuFallback());
        attrs.put("cudaFallbackReason", cudaStats.fallbackReason());
        attrs.put("cudaExecutionPath", cudaStats.executionPath().name());
        attrs.put("cudaExternalInputCount", cudaStats.externalInputCount());
        attrs.put("cudaOutputCount", cudaStats.outputCount());
        attrs.put("cudaInputBytes", cudaStats.inputBytes());
        attrs.put("cudaOutputBytes", cudaStats.outputBytes());
        attrs.put("cudaJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
        attrs.put("cudaNativeExecuteNs", cudaStats.nativeExecuteNs());
        attrs.put("cudaNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
        attrs.put("cudaNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
        attrs.put("cudaBridgeTotalNs", cudaStats.totalNs());
        attrs.put("acceleratorInputBytes", cudaStats.inputBytes());
        attrs.put("acceleratorOutputBytes", cudaStats.outputBytes());
        attrs.put("acceleratorJavaToNativeCopyNs", cudaStats.javaToNativeCopyNs());
        attrs.put("acceleratorNativeToJavaCopyNs", cudaStats.nativeToJavaCopyNs());
        attrs.put("acceleratorNativeDeviceCopyNs", cudaStats.nativeDeviceCopyNs());
    }

    /**
     * Returns the lowered accelerator DAG compiled for this executable.
     */
    public AcceleratorDagSpec dagSpec() {
        return dagSpec;
    }

    /**
     * Returns the compound GPU summary associated with this prepared CUDA executable.
     */
    @Override
    public GpuCompoundRegionSummary compoundSummary() {
        return compoundSummary;
    }

    @Override
    public GpuLoweredRegionManifest gpuLoweredRegionManifest() {
        return gpuLoweredRegionManifest;
    }

    @Override
    public RegionExecutionPlan regionExecutionPlan() {
        return regionExecutionPlan;
    }

    /**
     * Returns the lowering family that produced this executable.
     */
    public LoweringFamily loweringFamily() {
        return loweringFamily;
    }

    /**
     * Returns the CUDA graph bridge used for compile and execute calls.
     */
    public CudaGraphBridge bridge() {
        return bridge;
    }

    /**
     * Returns the bridge context created during preparation.
     */
    public CudaBridgeContext bridgeContext() {
        return bridgeContext;
    }

    /**
     * Returns the compiled bridge executable, which may be unavailable with a reason.
     */
    public CudaBridgeExecutable bridgeExecutable() {
        return bridgeExecutable;
    }

    /**
     * Returns diagnostics captured during the most recent CUDA execution attempt.
     *
     * <p>The value is updated for CUDA buffer execution, tensor-array bridge execution,
     * and CPU fallback paths so trace rendering can explain the actual runtime path.</p>
     *
     * @return latest bridge execution diagnostics
     */
    public CudaBridgeExecutionStats lastExecutionStats() {
        return lastExecutionStats;
    }

    private AcceleratorBufferRequest bufferRequest(ExecutionContext context) {
        return new AcceleratorBufferRequest(
                ComputeBackend.GPU_CUDA,
                dagSpec.nodes().size(),
                bridgeExecutable.externalInputNodeIds(),
                bridgeExecutable.externalInputDataTypes().isEmpty()
                        ? bridgeExecutable.externalInputNodeIds().stream().map(ignored -> tensor.DataType.FLOAT32).toList()
                        : bridgeExecutable.externalInputDataTypes(),
                layoutsForNodeIds(context, bridgeExecutable.externalInputNodeIds()),
                bridgeExecutable.outputNodeIds(),
                bridgeExecutable.outputDataTypes().isEmpty()
                        ? bridgeExecutable.outputNodeIds().stream().map(ignored -> tensor.DataType.FLOAT32).toList()
                        : bridgeExecutable.outputDataTypes(),
                layoutsForNodeIds(context, bridgeExecutable.outputNodeIds()),
                context.runsBackwardPass()
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

    private void ensureTensorArrayInputsCpuReadable(ExecutionContext context) {
        for (int nodeId : bridgeExecutable.externalInputNodeIds()) {
            context.requireCpuReadable(nodeId, CpuMaterializationReason.CPU_CONSUMER);
        }
    }

    private AcceleratorBufferDecision bridgeUnavailableDecision(String reason) {
        AcceleratorBufferBindingMode mode = backendConfig.buffer().bindingMode();
        return new AcceleratorBufferDecision(
                ComputeBackend.GPU_CUDA,
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
                ? AcceleratorBufferDecision.notEvaluated(ComputeBackend.GPU_CUDA)
                : decision;
    }

    private static void requireBufferOrThrow(AcceleratorBufferDecision decision) {
        if (decision != null && decision.required() && decision.path() != AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            throw new IllegalStateException("Accelerator buffer path is required for "
                    + decision.backend() + " but unavailable: "
                    + decision.reasonCode() + ": " + decision.reason());
        }
    }

    private String cudaBridgeUnavailableReason() {
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

    private void registerRuntimeServices(ExecutionContext context, CudaBufferAllocator allocator) {
        if (context == null || allocator == null || !allocator.available()) {
            return;
        }
        context.registerDeviceToCpuMaterializer(
                ComputeBackend.GPU_CUDA.name(),
                new CudaDeviceToCpuMaterializer(allocator)
        );
        if (bridge.supportsLayoutMaterialization()) {
            context.registerRuntimeService(
                    DeviceLayoutMaterializer.class,
                    new CudaDeviceLayoutMaterializer(bridge, bridgeContext, allocator)
            );
        }
    }

    private static void markBufferOutputsCurrent(ExecutionContext context, List<CudaBufferBinding> outputBindings) {
        if (context == null || outputBindings == null || outputBindings.isEmpty()) {
            return;
        }
        for (CudaBufferBinding binding : outputBindings) {
            CudaBufferBinding activeBinding = readableAfterWrite(binding);
            context.attachDeviceBufferBinding(
                    activeBinding.nodeId(),
                    activeBinding,
                    StorageResidency.DEVICE_OWNED,
                    "cuda buffer binding output"
            );
        }
    }

    private static CudaBufferBinding readableAfterWrite(CudaBufferBinding binding) {
        if (binding.access() == CudaBufferAccess.READ_WRITE) {
            return binding;
        }
        return new CudaBufferBinding(
                binding.nodeId(),
                binding.layout(),
                binding.handle(),
                CudaBufferAccess.READ_WRITE
        );
    }

    private static long logicalBytes(List<CudaBufferBinding> bindings) {
        long bytes = 0L;
        if (bindings == null) {
            return 0L;
        }
        for (CudaBufferBinding binding : bindings) {
            if (binding != null) {
                bytes += binding.logicalByteLength();
            }
        }
        return bytes;
    }

    private static long logicalBytesFromInputs(List<backend.accelerator.buffer.AcceleratorBufferInputDecision> inputs) {
        long bytes = 0L;
        if (inputs == null) {
            return 0L;
        }
        for (var input : inputs) {
            if (input != null && input.layout() != null) {
                bytes += input.layout().logicalByteLength();
            }
        }
        return bytes;
    }

    private static long logicalBytesFromOutputs(List<backend.accelerator.buffer.AcceleratorBufferOutputDecision> outputs) {
        long bytes = 0L;
        if (outputs == null) {
            return 0L;
        }
        for (var output : outputs) {
            if (output != null && output.layout() != null) {
                bytes += output.layout().logicalByteLength();
            }
        }
        return bytes;
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
            case INT64 -> Long.BYTES;
        };
    }

    private static String safeMessage(RuntimeException ex) {
        if (ex == null) {
            return "unknown error";
        }
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
    }
}
