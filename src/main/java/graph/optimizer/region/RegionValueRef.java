package graph.optimizer.region;

/**
 * Stable reference to a value in region optimization and memory planning.
 *
 * @param valueId textual value id
 */
public record RegionValueRef(
        String valueId
) {
    public RegionValueRef {
        if (valueId == null || valueId.isBlank()) {
            throw new IllegalArgumentException("valueId cannot be blank");
        }
    }

    /**
     * Creates a region value reference for a compiled node.
     *
     * @param nodeId compiled node id
     * @return region value reference
     */
    public static RegionValueRef ofNode(int nodeId) {
        return new RegionValueRef("node-" + nodeId);
    }
}
