package backend.cpu.fused.ir;

import operations.Operation;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Internal expression-plan node consumed by fused CPU code generation.
 */
public record FusedNodePlan(
        int index,
        Operation.OpType opType,
        List<Integer> inputRefs,
        int outputRef,
        DataType outputType,
        FusedNodeAttributes attributes
) {
    public FusedNodePlan {
        if (index < 0) {
            throw new IllegalArgumentException("index must be >= 0");
        }
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(inputRefs, "inputRefs cannot be null");
        Objects.requireNonNull(outputType, "outputType cannot be null");
        Objects.requireNonNull(attributes, "attributes cannot be null");
        inputRefs = List.copyOf(inputRefs);

        if (outputRef < 0) {
            throw new IllegalArgumentException("outputRef must be >= 0");
        }
    }
}
