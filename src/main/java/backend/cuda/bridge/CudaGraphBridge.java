package backend.cuda.bridge;

import graph.optimizer.partition.model.AcceleratorDagSpec;
import tensor.Tensor;

import java.util.List;

public interface CudaGraphBridge {
    boolean isAvailable();

    String unavailableReason();

    CudaBridgeContext createContext();

    default void destroyContext(CudaBridgeContext bridgeContext) {
    }

    CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec);

    default void destroyExecutable(CudaBridgeExecutable executable) {
    }

    void execute(
            CudaBridgeContext bridgeContext,
            CudaBridgeExecutable executable,
            List<Tensor> externalInputs,
            Tensor out
    );
}
