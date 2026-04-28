package config.optimizer;

/**
 * Named stage in the graph optimizer pipeline.
 */
public enum OptimizerStage {
    /**
     * Algebraic and semantic rewrite/lowering stage.
     */
    AR,
    /**
     * Common subexpression elimination stage.
     */
    CSE,
    /**
     * Backend partition planning stage.
     */
    PART,
    /**
     * Region fusion stage.
     */
    FUSE,
    /**
     * Memory planning stage.
     */
    MEM
}
