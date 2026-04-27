package backend.cuda.bridge;

import backend.accelerator.dag.AcceleratorDagSpec;
import tensor.Tensor;

import java.util.List;

public final class UnavailableCudaGraphBridge implements CudaGraphBridge {
    private final String reason;

    public UnavailableCudaGraphBridge(String reason) {
        this.reason = reason == null || reason.isBlank() ? "CUDA graph bridge is unavailable on this machine." : reason;
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
    public CudaBridgeContext createContext() {
        return CudaBridgeContext.unavailable(unavailableReason());
    }

    @Override
    public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
        return CudaBridgeExecutable.unavailable(unavailableReason());
    }

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
