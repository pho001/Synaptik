package backend.metal.bridge;

import backend.metal.lowering.MetalPartitionPlan;
import tensor.Tensor;

import java.util.List;

public interface MetalMpsGraphBridge {
    boolean isAvailable();

    String unavailableReason();

    MetalMpsBridgeContext createContext();

    default void destroyContext(MetalMpsBridgeContext bridgeContext) {
    }

    MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan);

    default void destroyExecutable(MetalMpsBridgeExecutable executable) {
    }

    void execute(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            List<Tensor> externalInputs,
            List<Tensor> outputs
    );
}
