package backend.metal.bridge;

import backend.metal.lowering.MetalPartitionPlan;
import backend.metal.buffer.MetalBufferBinding;
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
     *
     * @return true when native context creation, compilation, and execution can be attempted
     */
    boolean isAvailable();

    /**
     * Returns the reason the bridge is unavailable, or an empty string when available.
     *
     * @return stable diagnostic text, or an empty string when {@link #isAvailable()} is true
     */
    String unavailableReason();

    /**
     * Creates or returns a native Metal bridge context.
     *
     * @return native bridge context; unavailable implementations return an unavailable context
     */
    MetalMpsBridgeContext createContext();

    /**
     * Releases a context when the implementation owns a releasable native handle.
     *
     * @param bridgeContext context previously returned by {@link #createContext()}
     */
    default void destroyContext(MetalMpsBridgeContext bridgeContext) {
    }

    /**
     * Compiles a lowered Metal partition into a native executable.
     *
     * @param bridgeContext native bridge context
     * @param plan lowered Metal partition plan
     * @return compiled native executable or an unavailable executable when compilation cannot proceed
     */
    MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan);

    /**
     * Releases a compiled executable when the implementation owns a releasable native handle.
     *
     * @param executable compiled executable previously returned by {@link #compile(MetalMpsBridgeContext, MetalPartitionPlan)}
     */
    default void destroyExecutable(MetalMpsBridgeExecutable executable) {
    }

    /**
     * Returns whether this bridge can execute using native buffer bindings instead of tensor arrays.
     *
     * @return true when {@link #executeBuffers(MetalMpsBridgeContext, MetalMpsBridgeExecutable, List, List)}
     * is implemented by this bridge
     */
    default boolean supportsBufferBindings() {
        return false;
    }

    /**
     * Executes a compiled Metal graph against already resolved runtime tensors.
     *
     * <p>The current FFM implementation materializes inputs from Java tensor arrays and copies outputs back into
     * Java tensor arrays. Callers should use the returned stats to report transfer cost explicitly instead of
     * treating a Metal step as zero-copy.</p>
     *
     * @param bridgeContext native bridge context
     * @param executable compiled native executable
     * @param externalInputs runtime tensors consumed as external inputs
     * @param outputs runtime tensors receiving output data
     * @return bridge-level timing and byte diagnostics for the execution
     */
    MetalMpsBridgeExecutionStats execute(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            List<Tensor> externalInputs,
            List<Tensor> outputs
    );

    /**
     * Executes a compiled Metal graph against explicit native buffer bindings.
     *
     * <p>The current FFM bridge still uses tensor-array execution and therefore
     * inherits this unsupported default. Future shared-buffer bridges should
     * override this method and keep tensor materialization decisions outside the
     * native bridge.</p>
     *
     * @param bridgeContext native bridge context
     * @param executable compiled native executable
     * @param externalInputs input buffer bindings
     * @param outputs output buffer bindings
     * @return bridge-level timing and byte diagnostics for the execution
     * @throws UnsupportedOperationException when the implementation does not support buffer bindings
     */
    default MetalMpsBridgeExecutionStats executeBuffers(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            List<MetalBufferBinding> externalInputs,
            List<MetalBufferBinding> outputs
    ) {
        throw new UnsupportedOperationException("Metal bridge does not support native buffer bindings yet.");
    }
}
