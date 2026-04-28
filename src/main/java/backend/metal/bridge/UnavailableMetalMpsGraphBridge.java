package backend.metal.bridge;

import backend.runtime.ExecutionContext;
import backend.metal.lowering.MetalPartitionPlan;
import tensor.Tensor;

/**
 * Metal bridge implementation used when native MPSGraph execution is unavailable.
 */
public final class UnavailableMetalMpsGraphBridge implements MetalMpsGraphBridge {
    private final String reason;

    /**
     * Creates an unavailable Metal bridge with a stable diagnostic reason.
     */
    public UnavailableMetalMpsGraphBridge(String reason) {
        this.reason = reason == null || reason.isBlank() ? "Metal MPSGraph bridge is not implemented yet." : reason;
    }

    /**
     * Always returns {@code false}.
     */
    @Override
    public boolean isAvailable() {
        return false;
    }

    /**
     * Returns the configured unavailable reason.
     */
    @Override
    public String unavailableReason() {
        return reason;
    }

    /**
     * Returns an unavailable Metal context instead of throwing.
     */
    @Override
    public MetalMpsBridgeContext createContext() {
        return MetalMpsBridgeContext.unavailable(unavailableReason());
    }

    /**
     * Returns an unavailable Metal executable instead of throwing.
     */
    @Override
    public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
        return MetalMpsBridgeExecutable.unavailable(unavailableReason());
    }

    /**
     * Throws because unavailable bridges cannot execute.
     */
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
