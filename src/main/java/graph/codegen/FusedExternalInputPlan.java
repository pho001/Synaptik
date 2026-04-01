package graph.codegen;

import java.util.Arrays;
import java.util.Objects;

public record FusedExternalInputPlan(
        int index,
        boolean directIndex,
        int[] outShape,
        int[] outStrides,
        int[] effStrides
) {
    public FusedExternalInputPlan {
        if (index < 0) {
            throw new IllegalArgumentException("index must be >= 0");
        }
        Objects.requireNonNull(outShape, "outShape cannot be null");
        Objects.requireNonNull(outStrides, "outStrides cannot be null");
        Objects.requireNonNull(effStrides, "effStrides cannot be null");
        outShape = outShape.clone();
        outStrides = outStrides.clone();
        effStrides = effStrides.clone();
        if (outShape.length != outStrides.length || outShape.length != effStrides.length) {
            throw new IllegalArgumentException(
                    "Broadcast metadata rank mismatch: shape=" + Arrays.toString(outShape)
            );
        }
    }
}
