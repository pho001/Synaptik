package graph.optimizer.fusion;

import operations.Operation;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public final class FusedAccessResolver {
    private FusedAccessResolver() {
    }

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

    public static boolean isAbsorbableAccessTransform(Operation operation) {
        if (operation == null || operation.opType() == null) {
            return false;
        }
        return switch (operation.opType()) {
            case SELECT, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
    }

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

        public boolean absorbedAccessChain() {
            return absorbedDepth > 0;
        }
    }
}
