package backend.metal.bridge;

import backend.runtime.ExecutionContext;
import backend.metal.lowering.MetalPartitionPlan;
import tensor.Tensor;

public final class UnavailableMetalMpsGraphBridge implements MetalMpsGraphBridge {
    private final String reason;

    public UnavailableMetalMpsGraphBridge(String reason) {
        this.reason = reason == null || reason.isBlank() ? "Metal MPSGraph bridge is not implemented yet." : reason;
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
    public MetalMpsBridgeContext createContext() {
        return MetalMpsBridgeContext.unavailable(unavailableReason());
    }

    @Override
    public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
        return MetalMpsBridgeExecutable.unavailable(unavailableReason());
    }

    @Override
    public void execute(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<Tensor> externalInputs,
            java.util.List<Tensor> outputs
    ) {
        throw new UnsupportedOperationException(unavailableReason());
    }
}
