package backend.cuda.exec;

import backend.CPUBackend;
import backend.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.cuda.bridge.CudaBridgeContext;
import backend.cuda.bridge.CudaBridgeExecutable;
import backend.cuda.bridge.CudaGraphBridge;
import backend.runtime.ExecutionContext;
import graph.CompiledNode;
import graph.execution.CompiledNodeExecutionMetadata;
import graph.optimizer.partition.model.AcceleratorDagSpec;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class PreparedCudaExecutable implements PreparedAcceleratorExecutable {
    private static final CPUBackend CPU_BACKEND = new CPUBackend();

    public record PreparedStep(CompiledNode node, CompiledNodeExecutionMetadata metadata) {
        public PreparedStep {
            Objects.requireNonNull(node, "node cannot be null");
            Objects.requireNonNull(metadata, "metadata cannot be null");
        }
    }

    private final AcceleratorDagSpec dagSpec;
    private final CudaGraphBridge bridge;
    private final CudaBridgeContext bridgeContext;
    private final CudaBridgeExecutable bridgeExecutable;
    private final List<PreparedStep> cpuFallbackSteps;

    public PreparedCudaExecutable(
            AcceleratorDagSpec dagSpec,
            CudaGraphBridge bridge,
            List<PreparedStep> cpuFallbackSteps
    ) {
        this.dagSpec = Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
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
        if (bridge.isAvailable() && bridgeContext.available() && bridgeExecutable.available()) {
            List<Tensor> inputs = new ArrayList<>(bridgeExecutable.externalInputNodeIds().size());
            for (int inputNodeId : bridgeExecutable.externalInputNodeIds()) {
                inputs.add(context.runtimeTensorForNodeId(inputNodeId));
            }
            Tensor out = context.runtimeTensorForNodeId(bridgeExecutable.outputNodeId());
            bridge.execute(bridgeContext, bridgeExecutable, List.copyOf(inputs), out);
            return;
        }
        for (PreparedStep step : cpuFallbackSteps) {
            CPU_BACKEND.execute(step.node(), step.metadata(), context);
        }
    }

    public AcceleratorDagSpec dagSpec() {
        return dagSpec;
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
