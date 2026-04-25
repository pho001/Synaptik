package backend.apple.bridge;

import graph.optimizer.partition.apple.AppleGpuPartitionPlan;
import tensor.Tensor;

import java.util.List;

public interface AppleMpsGraphBridge {
    boolean isAvailable();

    String unavailableReason();

    AppleMpsBridgeContext createContext();

    default void destroyContext(AppleMpsBridgeContext bridgeContext) {
    }

    AppleMpsBridgeExecutable compile(AppleMpsBridgeContext bridgeContext, AppleGpuPartitionPlan plan);

    default void destroyExecutable(AppleMpsBridgeExecutable executable) {
    }

    void execute(
            AppleMpsBridgeContext bridgeContext,
            AppleMpsBridgeExecutable executable,
            List<Tensor> externalInputs,
            Tensor out
    );
}
