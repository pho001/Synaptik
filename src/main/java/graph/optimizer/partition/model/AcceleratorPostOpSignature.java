package graph.optimizer.partition.model;

import java.util.Objects;

public record AcceleratorPostOpSignature(
        AcceleratorPostOpType type,
        boolean inputVector,
        boolean hasScalarValue,
        int scalarValueBits
) {
    public AcceleratorPostOpSignature {
        Objects.requireNonNull(type, "type cannot be null");
        if (!type.binary()) {
            inputVector = false;
        }
        if (!hasScalarValue) {
            scalarValueBits = 0;
        }
    }

    public static AcceleratorPostOpSignature from(AcceleratorPostOp postOp) {
        return new AcceleratorPostOpSignature(
                postOp.type(),
                postOp.inputVector(),
                postOp.hasScalarValue(),
                postOp.scalarValueBits()
        );
    }
}
