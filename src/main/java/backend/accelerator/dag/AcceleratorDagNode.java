package backend.accelerator.dag;

import java.util.Objects;

public record AcceleratorDagNode(
        int nodeId,
        AcceleratorDagNodeType type,
        AcceleratorDagValueRef input0,
        AcceleratorDagValueRef input1,
        AcceleratorDagValueRef input2,
        AcceleratorDagValueRef input3,
        int scalarValueBits,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3
) {
    public AcceleratorDagNode {
        Objects.requireNonNull(type, "type cannot be null");
        input0 = input0 == null ? AcceleratorDagValueRef.none() : input0;
        input1 = input1 == null ? AcceleratorDagValueRef.none() : input1;
        input2 = input2 == null ? AcceleratorDagValueRef.none() : input2;
        input3 = input3 == null ? AcceleratorDagValueRef.none() : input3;
        outputRank = Math.max(1, outputRank);
        outputDim0 = Math.max(1, outputDim0);
        outputDim1 = Math.max(1, outputDim1);
        outputDim2 = Math.max(1, outputDim2);
        outputDim3 = Math.max(1, outputDim3);
    }
}
