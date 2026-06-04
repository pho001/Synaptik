package backend.cpu1.fused.ir;

import operations.Operation;
import tensor.DataType;

import java.util.List;

public record Cpu1FusedNodePlan(
        int index,
        int nodeId,
        Operation.OpType opType,
        List<Integer> inputRefs,
        int outputRef,
        DataType outputType,
        Cpu1FusedScalarParameter scalarParameter
) {
    public Cpu1FusedNodePlan {
        if (opType == null) {
            throw new IllegalArgumentException("opType cannot be null");
        }
        if (inputRefs == null) {
            throw new IllegalArgumentException("inputRefs cannot be null");
        }
        inputRefs = List.copyOf(inputRefs);
        if (outputType == null) {
            throw new IllegalArgumentException("outputType cannot be null");
        }
        scalarParameter = scalarParameter == null ? Cpu1FusedScalarParameter.NONE : scalarParameter;
    }
}
