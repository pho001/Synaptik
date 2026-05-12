package backend.cpu.fused.optimize;

import operations.Operation;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

/**
 * Internal resolver for absorbable access transforms on fused external inputs.
 */
public final class FusedAccessResolver {
    private FusedAccessResolver() {
    }

    /**
     * Resolves the backing tensor for an exposed fused input.
     */
    public static ResolvedInput resolve(Tensor externalInput) {
        Objects.requireNonNull(externalInput, "externalInput cannot be null");
        Tensor current = externalInput;
        int absorbedDepth = 0;

        while (isAbsorbableAccessTransform(current.getOperation())) {
            List<Tensor> prev = current.getPrevTensors();
            if (prev == null || prev.size() != 1) {
                throw new IllegalStateException(
                        "Absorbable fused access transform must have exactly one parent. tensor=" + current.getLabel()
                );
            }
            current = prev.getFirst();
            absorbedDepth++;
            if (current == null) {
                throw new IllegalStateException("Absorbable fused access chain resolved to null backing tensor.");
            }
        }

        return new ResolvedInput(externalInput, current, absorbedDepth);
    }

    /**
     * Returns whether an operation can be absorbed into fused input addressing.
     */
    public static boolean isAbsorbableAccessTransform(Operation operation) {
        if (operation == null || operation.opType() == null) {
            return false;
        }
        return switch (operation.opType()) {
            case SELECT, SLICE, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
    }

    /**
     * Result of resolving a fused external input through absorbable access transforms.
     *
     * @param exposedTensor tensor visible to the fused expression
     * @param backingTensor tensor whose storage should be passed at runtime
     * @param absorbedDepth number of absorbed access-transform nodes
     */
    public record ResolvedInput(
            Tensor exposedTensor,
            Tensor backingTensor,
            int absorbedDepth
    ) {
        public ResolvedInput {
            Objects.requireNonNull(exposedTensor, "exposedTensor cannot be null");
            Objects.requireNonNull(backingTensor, "backingTensor cannot be null");
            if (absorbedDepth < 0) {
                throw new IllegalArgumentException("absorbedDepth cannot be negative");
            }
        }

        /**
         * Returns whether at least one access-transform node was absorbed.
         */
        public boolean absorbedAccessChain() {
            return absorbedDepth > 0;
        }
    }
}
