package backend.accelerator.dag;

import tensor.DataType;

import java.util.List;
import java.util.Objects;

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
