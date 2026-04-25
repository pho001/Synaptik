package backend.apple.bridge;

import backend.runtime.ExecutionContext;
import graph.optimizer.partition.apple.AppleGpuPartitionPlan;
import tensor.Tensor;

public final class UnavailableAppleMpsGraphBridge implements AppleMpsGraphBridge {
    private final String reason;

    public UnavailableAppleMpsGraphBridge(String reason) {
        this.reason = reason == null || reason.isBlank() ? "Apple MPSGraph bridge is not implemented yet." : reason;
    }

    @Override
    public boolean isAvailable() {
        return false;
    }

    @Override
    public String unavailableReason() {
        return reason;
    }

    @Override
    public AppleMpsBridgeContext createContext() {
        return AppleMpsBridgeContext.unavailable(unavailableReason());
    }

    @Override
    public AppleMpsBridgeExecutable compile(AppleMpsBridgeContext bridgeContext, AppleGpuPartitionPlan plan) {
        return AppleMpsBridgeExecutable.unavailable(unavailableReason());
    }

    @Override
    public void execute(
            AppleMpsBridgeContext bridgeContext,
            AppleMpsBridgeExecutable executable,
            java.util.List<Tensor> externalInputs,
            Tensor out
    ) {
        throw new UnsupportedOperationException(unavailableReason());
    }
}
