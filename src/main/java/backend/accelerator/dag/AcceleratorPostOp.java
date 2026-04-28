package backend.accelerator.dag;

import java.util.Objects;

/**
 * Post operation fused onto a legacy accelerator matmul lowering.
 *
 * @param type post operation kind
 * @param inputNodeId compiled-node id for binary post-op input, or {@code -1}
 * @param inputVector whether the binary post-op input is vector-shaped
 * @param hasScalarValue whether {@code scalarValueBits} carries a scalar operand
 * @param scalarValueBits raw {@code float} bits for scalar unary post operations
 */
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

    /**
     * Creates a unary post operation with no scalar operand.
     */
    public static AcceleratorPostOp unary(AcceleratorPostOpType type) {
        return new AcceleratorPostOp(type, -1, false, false, 0);
    }

    /**
     * Creates a unary post operation with a scalar operand encoded as float bits.
     */
    public static AcceleratorPostOp scalarUnary(AcceleratorPostOpType type, float scalarValue) {
        return new AcceleratorPostOp(type, -1, false, true, Float.floatToIntBits(scalarValue));
    }

    /**
     * Creates a binary post operation that reads another compiled-node value.
     */
    public static AcceleratorPostOp binary(AcceleratorPostOpType type, int inputNodeId, boolean inputVector) {
        return new AcceleratorPostOp(type, inputNodeId, inputVector, false, 0);
    }

    /**
     * Returns the scalar operand decoded from {@link #scalarValueBits()}.
     */
    public float scalarValue() {
        return Float.intBitsToFloat(scalarValueBits);
    }
}
