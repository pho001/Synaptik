package backend.accelerator.dag;

import java.util.Objects;

/**
 * Cache-stable post-op signature that excludes graph-specific node ids.
 *
 * @param type post operation kind
 * @param inputVector whether a binary input is vector-shaped
 * @param hasScalarValue whether {@code scalarValueBits} carries a scalar operand
 * @param scalarValueBits raw {@code float} bits for scalar post operations
 */
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

    /**
     * Builds a cache signature from a concrete post operation.
     */
    public static AcceleratorPostOpSignature from(AcceleratorPostOp postOp) {
        return new AcceleratorPostOpSignature(
                postOp.type(),
                postOp.inputVector(),
                postOp.hasScalarValue(),
                postOp.scalarValueBits()
        );
    }
}
