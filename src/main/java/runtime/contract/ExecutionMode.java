package runtime.contract;

/**
 * Runtime execution mode for compiled/prepared graphs.
 */
public enum ExecutionMode {
    /**
     * Execute forward graph and backward/gradient graph.
     */
    FORWARD_BACKWARD,

    /**
     * Execute forward graph only.
     */
    FORWARD
}
