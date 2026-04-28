package utils;

/**
 * Internal codegen classification for where an operator input is stored.
 */
public enum InputType {
    /** Input comes from an outer cluster input array. */
    CLUSTER_INPUT,
    /** Input comes from an intermediate value produced inside the cluster. */
    CLUSTER_INNER,

}
