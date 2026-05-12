package training.optimizer;

/**
 * Optimizer that can update trainable parameters after a compiled forward/backward run.
 *
 * <p>The optimizer consumes compiled gradient bindings directly. Implementations should update only tensors marked
 * as trainable parameters and should avoid publishing every gradient to the public CPU tensor API.</p>
 */
public interface TrainingOptimizer extends AutoCloseable {
    /**
     * Called after per-run execution state is created and before graph steps execute.
     *
     * @param context optimizer step context
     */
    default void beforeExecute(OptimizerStepContext context) {
    }

    /**
     * Updates trainable parameters after forward/backward execution has produced gradients.
     *
     * @param context optimizer step context
     */
    void step(OptimizerStepContext context);

    /**
     * Synchronizes optimizer-owned parameter buffers back into public CPU tensor storage where supported.
     */
    default void syncParametersToCpu() {
    }

    @Override
    default void close() {
    }
}
