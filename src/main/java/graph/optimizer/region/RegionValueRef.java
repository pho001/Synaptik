package graph.optimizer.region;

public record RegionValueRef(
        String valueId
) {
    public RegionValueRef {
        if (valueId == null || valueId.isBlank()) {
            throw new IllegalArgumentException("valueId cannot be blank");
        }
    }

    public static RegionValueRef ofNode(int nodeId) {
        return new RegionValueRef("node-" + nodeId);
    }
}
