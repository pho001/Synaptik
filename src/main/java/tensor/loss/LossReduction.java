package tensor.loss;

/**
 * Reduction mode applied by index-based tensor loss functions.
 */
public enum LossReduction {
    /** Average per-sample losses over valid targets. */
    MEAN,
    /** Sum per-sample losses over valid targets. */
    SUM,
    /** Return the unreduced per-sample loss tensor. */
    NONE
}
