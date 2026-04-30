package backend.cuda.exec;

import backend.ComputeBackend;
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
import backend.cuda.buffer.CudaAcceleratorBufferBinder;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferAllocator;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.lowering.LoweringFamily;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.accelerator.dag.AcceleratorDagSpec;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import tensor.Tensor;

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
    private final LoweringFamily loweringFamily;
    private final CudaGraphBridge bridge;
    private final CudaBridgeContext bridgeContext;
    private final CudaBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;
    private final AcceleratorBackendConfig backendConfig;
    private final CudaAcceleratorBufferBinder bufferBinder;
    private volatile AcceleratorBufferDecision lastAcceleratorBufferDecision =
            AcceleratorBufferDecision.notEvaluated(ComputeBackend.GPU_CUDA);

    /**
     * Creates a prepared CUDA executable around a lowered DAG and fallback plan.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        this(dagSpec, loweringFamily, bridge, cpuFallbackSteps, AcceleratorBackendConfig.defaults());
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
        this.dagSpec = Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
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
                CudaBufferAllocator allocator = bridge.createBufferAllocator(bridgeContext);
                AcceleratorBufferBindings<CudaBufferBinding> bindings = bufferBinder.resolve(
                        request,
                        resolvedInputs,
                        decision,
                        context,
                        allocator
                );
                bridge.executeBuffers(bridgeContext, bridgeExecutable, bindings.inputs(), bindings.outputs());
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
                PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            }
            return;
        }

        if (decision.path() == AcceleratorBufferExecutionPath.CPU_FALLBACK) {
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
            return;
        }

        ensureTensorArrayInputsCpuReadable(context);
        try {
            bridge.execute(
                    bridgeContext,
                    bridgeExecutable,
                    resolvedInputs.executionExternalInputs(),
                    outputTensors(context)
            );
        } catch (RuntimeException ex) {
            PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
        }
    }

    @Override
    public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
        return lastAcceleratorBufferDecision;
    }

    /**
     * Returns the lowered accelerator DAG compiled for this executable.
     */
    public AcceleratorDagSpec dagSpec() {
        return dagSpec;
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

    private static String safeMessage(RuntimeException ex) {
        if (ex == null) {
            return "unknown error";
        }
        String message = ex.getMessage();
        return message == null || message.isBlank() ? ex.getClass().getSimpleName() : message;
    }
}
