package backend.cuda.exec;

import backend.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferDecision;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.accelerator.buffer.AcceleratorBufferRequest;
import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.buffer.CudaAcceleratorBufferBinder;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.lowering.LoweringFamily;
import backend.runtime.ExecutionContext;
import backend.accelerator.dag.AcceleratorDagSpec;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;

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
        if (backendConfig.buffer().bindingMode() == AcceleratorBufferBindingMode.REQUIRE) {
            String reason = bridge.supportsBufferBindings()
                    ? "CUDA prepared executable does not implement buffer binding execution"
                    : "CUDA bridge does not support required buffer bindings";
            lastAcceleratorBufferDecision = new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    backendConfig.buffer().bindingMode(),
                    AcceleratorBufferExecutionPath.UNAVAILABLE,
                    false,
                    true,
                    AcceleratorBufferReasonCode.REQUIRED_BUFFER_EXECUTION_UNAVAILABLE,
                    reason,
                    List.of(),
                    List.of()
            );
            throw new IllegalStateException("Accelerator buffer path is required for GPU_CUDA but unavailable: "
                    + lastAcceleratorBufferDecision.reasonCode() + ": " + lastAcceleratorBufferDecision.reason());
        }
        lastAcceleratorBufferDecision = bufferDecision(context);
        if (PreparedAcceleratorExecutionSupport.bridgeReady(
                bridge.isAvailable(),
                bridgeContext.available(),
                bridgeExecutable.available())) {
            bridge.execute(
                    bridgeContext,
                    bridgeExecutable,
                    PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(
                            bridgeExecutable.externalInputNodeIds(),
                            context
                    ),
                    PreparedAcceleratorExecutionSupport.resolveRuntimeTensors(
                            bridgeExecutable.outputNodeIds(),
                            context
                    )
            );
            return;
        }
        PreparedAcceleratorExecutionSupport.executeCpuFallback(cpuFallbackSteps, context);
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

    private AcceleratorBufferDecision bufferDecision(ExecutionContext context) {
        if (context == null
                || bridgeExecutable.externalInputNodeIds().isEmpty()
                || bridgeExecutable.outputNodeIds().isEmpty()) {
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    backendConfig.buffer().bindingMode(),
                    bridge.supportsBufferBindings()
                            ? AcceleratorBufferExecutionPath.UNAVAILABLE
                            : AcceleratorBufferExecutionPath.TENSOR_ARRAY,
                    false,
                    false,
                    bridge.supportsBufferBindings()
                            ? AcceleratorBufferReasonCode.NOT_EVALUATED
                            : AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE,
                    bridge.supportsBufferBindings()
                            ? "CUDA buffer policy not evaluated without runtime tensor context"
                            : "native CUDA buffer ABI unavailable: bridge does not support buffer bindings",
                    List.of(),
                    List.of()
            );
        }
        AcceleratorBufferRequest request = new AcceleratorBufferRequest(
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
        AcceleratorBufferDecision decision = bufferBinder.decide(request, backendConfig.buffer());
        if (decision.path() == AcceleratorBufferExecutionPath.BUFFER_BINDING) {
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    backendConfig.buffer().bindingMode(),
                    AcceleratorBufferExecutionPath.TENSOR_ARRAY,
                    false,
                    false,
                    AcceleratorBufferReasonCode.BACKEND_BUFFER_NOT_IMPLEMENTED,
                    "CUDA dense FLOAT32 buffer metadata accepted; native buffer execution is deferred to Phase 7",
                    decision.inputs(),
                    decision.outputs()
            );
        }
        return decision;
    }

    private static List<AcceleratorBufferLayout> layoutsForNodeIds(ExecutionContext context, List<Integer> nodeIds) {
        return nodeIds.stream()
                .map(nodeId -> AcceleratorBufferLayout.fromTensor(context.runtimeTensorForNodeId(nodeId)))
                .toList();
    }
}
