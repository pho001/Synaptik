package backend.metal.bridge;

import backend.metal.lowering.MetalPartitionPlan;
import tensor.Tensor;

import java.util.List;

/**
 * Internal SPI for Metal MPSGraph native execution.
 *
 * <p>Implementations report availability without throwing. Prepared executables
 * should use unavailable contexts or executables to trigger CPU fallback rather
 * than calling {@link #execute(MetalMpsBridgeContext, MetalMpsBridgeExecutable, List, List)}.</p>
 */
public interface MetalMpsGraphBridge {
    /**
     * Returns whether the native Metal bridge can currently be used.
     */
    boolean isAvailable();

    /**
     * Returns the reason the bridge is unavailable, or an empty string when available.
     */
    String unavailableReason();

    /**
     * Creates or returns a native Metal bridge context.
     */
    MetalMpsBridgeContext createContext();

    /**
     * Releases a context when the implementation owns a releasable native handle.
     */
    default void destroyContext(MetalMpsBridgeContext bridgeContext) {
    }

    /**
     * Compiles a lowered Metal partition into a native executable.
     */
    MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan);

    /**
     * Releases a compiled executable when the implementation owns a releasable native handle.
     */
    default void destroyExecutable(MetalMpsBridgeExecutable executable) {
    }

    /**
     * Executes a compiled Metal graph against already resolved runtime tensors.
     */
    void execute(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            List<Tensor> externalInputs,
            List<Tensor> outputs
    );
}
