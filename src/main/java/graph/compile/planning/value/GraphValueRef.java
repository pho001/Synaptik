package graph.compile.planning.value;

/**
 * Stable typed reference to a graph value.
 *
 * @param kind referenced value kind
 * @param nodeId producer node id for {@link GraphValueKind#NODE} values
 */
public record GraphValueRef(
        GraphValueKind kind,
        int nodeId
) {
    public GraphValueRef {
        if (kind == null) {
            throw new IllegalArgumentException("kind cannot be null");
        }
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId must be >= 0");
        }
    }

    /**
     * Creates a reference to a compiled graph node value.
     *
     * @param nodeId producer node id
     * @return graph value reference
     */
    public static GraphValueRef node(int nodeId) {
        return new GraphValueRef(GraphValueKind.NODE, nodeId);
    }

    /**
     * Returns a stable diagnostic id.
     *
     * @return human-readable value id
     */
    public String valueId() {
        return switch (kind) {
            case NODE -> "node-" + nodeId;
        };
    }
}
