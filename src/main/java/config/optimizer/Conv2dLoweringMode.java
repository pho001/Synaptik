package config.optimizer;

/**
 * Strategy for lowering high-level conv2d operations.
 */
public enum Conv2dLoweringMode {
    /**
     * Keep high-level conv2d operations without lowering.
     */
    OFF,
    /**
     * Always lower eligible conv2d operations to GEMM-style forms.
     */
    ALWAYS,
    /**
     * Let conv2d lowering heuristics decide based on operation shape and options.
     */
    HEURISTIC
}
