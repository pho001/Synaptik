package prepare.context;

/**
 * Role of a node inside a prepared backend partition.
 */
public enum PartitionExecutionRole {
    /**
     * Node is not assigned a special partition execution role.
     */
    NONE,
    /**
     * Node anchors a partition executable and triggers partition execution.
     */
    ANCHOR,
    /**
     * Node is covered by the anchor executable and should not execute independently.
     */
    INTERIOR
}
