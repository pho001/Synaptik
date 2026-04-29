package backend.cuda.bridge;

import backend.accelerator.dag.AcceleratorDagSpec;
import tensor.Tensor;

import java.util.List;

/**
 * Internal SPI for CUDA native graph execution.
 *
 * <p>Implementations report availability without throwing. Callers should compile
 * to an unavailable executable when native support is missing and use CPU fallback
 * instead of invoking {@link #execute(CudaBridgeContext, CudaBridgeExecutable, List, List)}.</p>
 */
public interface CudaGraphBridge {
    /**
     * Returns whether the native CUDA bridge can currently be used.
     */
    boolean isAvailable();

    /**
     * Returns the reason the bridge is unavailable, or an empty string when available.
     */
    String unavailableReason();

    /**
     * Creates or returns a native CUDA bridge context.
     */
    CudaBridgeContext createContext();

    /**
     * Releases a context when the implementation owns a releasable native handle.
     */
    default void destroyContext(CudaBridgeContext bridgeContext) {
    }

    /**
     * Compiles a lowered accelerator DAG into a native CUDA executable.
     */
    CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec);

    /**
     * Releases a compiled executable when the implementation owns a releasable native handle.
     */
    default void destroyExecutable(CudaBridgeExecutable executable) {
    }

    /**
     * Returns whether this CUDA bridge can execute through explicit native buffer bindings.
     *
     * <p>The default is {@code false}. Future CUDA native-buffer implementations should override this
     * only after they own a concrete device pointer/graph-buffer lifetime contract.</p>
     */
    default boolean supportsBufferBindings() {
        return false;
    }

    /**
     * Executes a compiled CUDA graph against already resolved runtime tensors.
     */
    void execute(
            CudaBridgeContext bridgeContext,
            CudaBridgeExecutable executable,
            List<Tensor> externalInputs,
            List<Tensor> outputs
    );
}
