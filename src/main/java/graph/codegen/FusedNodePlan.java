package graph.codegen;

import operations.Operation;

import java.util.List;
import java.util.Objects;

public record FusedNodePlan(
        int index,
        Operation.OpType opType,
        List<Integer> inputRefs,
        int outputRef,
        Object parameter
) {
    public FusedNodePlan {
        if (index < 0) {
            throw new IllegalArgumentException("index must be >= 0");
        }
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(inputRefs, "inputRefs cannot be null");
        inputRefs = List.copyOf(inputRefs);

        if (outputRef < 0) {
            throw new IllegalArgumentException("outputRef must be >= 0");
        }
    }
}
