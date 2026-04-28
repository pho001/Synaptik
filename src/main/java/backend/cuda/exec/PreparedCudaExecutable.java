package backend.cuda.exec;

import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutionSupport;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.lowering.LoweringFamily;
import backend.runtime.ExecutionContext;
import backend.accelerator.dag.AcceleratorDagSpec;

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

    /**
     * Creates a prepared CUDA executable around a lowered DAG and fallback plan.
     */
    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            LoweringFamily loweringFamily,
            CudaGraphBridge bridge,
            List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps
    ) {
        this.dagSpec = Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        this.loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        this.bridge = Objects.requireNonNull(bridge, "bridge cannot be null");
        this.bridgeContext = bridge.createContext();
        this.bridgeExecutable = bridge.compile(bridgeContext, dagSpec);
        this.cpuFallbackSteps = List.copyOf(cpuFallbackSteps == null ? List.of() : cpuFallbackSteps);
    }

    /**
     * Returns {@link ComputeBackend#GPU_CUDA}.
     */
    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

    /**
     * Executes through the CUDA graph bridge when available, otherwise runs CPU fallback steps.
     */
    @Override
    public void execute(ExecutionContext context) {
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
}
