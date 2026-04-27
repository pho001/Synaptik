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

public final class PreparedCudaExecutable implements PreparedAcceleratorExecutable {
    private final AcceleratorDagSpec dagSpec;
    private final LoweringFamily loweringFamily;
    private final CudaGraphBridge bridge;
    private final CudaBridgeContext bridgeContext;
    private final CudaBridgeExecutable bridgeExecutable;
    private final List<PreparedAcceleratorExecutionSupport.CpuFallbackStep> cpuFallbackSteps;

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

    @Override
    public ComputeBackend backend() {
        return ComputeBackend.GPU_CUDA;
    }

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

    public AcceleratorDagSpec dagSpec() {
        return dagSpec;
    }

    public LoweringFamily loweringFamily() {
        return loweringFamily;
    }

    public CudaGraphBridge bridge() {
        return bridge;
    }

    public CudaBridgeContext bridgeContext() {
        return bridgeContext;
    }

    public CudaBridgeExecutable bridgeExecutable() {
        return bridgeExecutable;
    }
}
