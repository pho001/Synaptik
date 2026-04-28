package backend.cuda.bridge;

import backend.accelerator.dag.AcceleratorDagSpec;
import tensor.Tensor;

import java.util.List;

/**
 * CUDA bridge implementation used when native graph execution is intentionally unavailable.
 */
public final class UnavailableCudaGraphBridge implements CudaGraphBridge {
    private final String reason;

    /**
     * Creates an unavailable CUDA bridge with a stable diagnostic reason.
     */
    public UnavailableCudaGraphBridge(String reason) {
        this.reason = reason == null || reason.isBlank() ? "CUDA graph bridge is unavailable on this machine." : reason;
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
     * Returns an unavailable CUDA context instead of throwing.
     */
    @Override
    public CudaBridgeContext createContext() {
        return CudaBridgeContext.unavailable(unavailableReason());
    }

    /**
     * Returns an unavailable CUDA executable instead of throwing.
     */
    @Override
    public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
        return CudaBridgeExecutable.unavailable(unavailableReason());
    }

    /**
     * Throws because unavailable bridges cannot execute.
     */
    @Override
    public void execute(
            CudaBridgeContext bridgeContext,
            CudaBridgeExecutable executable,
            List<Tensor> externalInputs,
            List<Tensor> outputs
    ) {
        throw new UnsupportedOperationException(unavailableReason());
    }
}
