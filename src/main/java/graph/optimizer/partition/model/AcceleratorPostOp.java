package graph.optimizer.partition.model;

import java.util.Objects;

public record AcceleratorPostOp(
        AcceleratorPostOpType type,
        int inputNodeId,
        boolean inputVector,
        boolean hasScalarValue,
        int scalarValueBits
) {
    public AcceleratorPostOp {
        Objects.requireNonNull(type, "type cannot be null");
        if (!type.binary()) {
            inputNodeId = -1;
            inputVector = false;
            if (!hasScalarValue) {
                scalarValueBits = 0;
            }
        } else if (inputNodeId < 0) {
            throw new IllegalArgumentException("binary post op requires inputNodeId");
        } else {
            hasScalarValue = false;
            scalarValueBits = 0;
        }
    }

    public static AcceleratorPostOp unary(AcceleratorPostOpType type) {
        return new AcceleratorPostOp(type, -1, false, false, 0);
    }

    public static AcceleratorPostOp scalarUnary(AcceleratorPostOpType type, float scalarValue) {
        return new AcceleratorPostOp(type, -1, false, true, Float.floatToIntBits(scalarValue));
    }

    public static AcceleratorPostOp binary(AcceleratorPostOpType type, int inputNodeId, boolean inputVector) {
        return new AcceleratorPostOp(type, inputNodeId, inputVector, false, 0);
    }

    public float scalarValue() {
        return Float.intBitsToFloat(scalarValueBits);
    }
}
