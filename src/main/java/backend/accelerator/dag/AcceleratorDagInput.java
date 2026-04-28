package backend.accelerator.dag;

import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * External runtime tensor consumed by an accelerator DAG.
 *
 * @param nodeId compiled-node id used to resolve the runtime tensor
 * @param shape tensor shape exposed to the native bridge
 * @param dataType tensor element type exposed to the native bridge
 */
public record AcceleratorDagInput(
        int nodeId,
        List<Integer> shape,
        DataType dataType
) {
    public AcceleratorDagInput {
        shape = List.copyOf(shape == null ? List.of() : shape);
        Objects.requireNonNull(dataType, "dataType cannot be null");
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId must be non-negative");
        }
        if (shape.isEmpty()) {
            throw new IllegalArgumentException("shape cannot be empty");
        }
    }
}
