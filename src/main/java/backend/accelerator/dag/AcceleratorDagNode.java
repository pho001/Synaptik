package backend.accelerator.dag;

import tensor.DataType;

import java.util.Objects;

/**
 * ABI-stable description of one lowered accelerator DAG operation.
 *
 * <p>The five inputs reference either external DAG inputs or earlier node outputs.
 * Scalar payloads are stored as raw float bits so Java and native bridges agree on
 * the wire representation.</p>
 *
 * @param nodeId compiled-node id represented by this DAG node
 * @param type lowered operation type
 * @param input0 first operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input1 second operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input2 third operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input3 fourth operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input4 fifth operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param scalarValueBits raw {@code float} bits for scalar-valued operations
 * @param attribute0 first static integer operation attribute
 * @param attribute1 second static integer operation attribute
 * @param attribute2 third static integer operation attribute
 * @param attribute3 fourth static integer operation attribute
 * @param attribute4 fifth static integer operation attribute
 * @param attribute5 sixth static integer operation attribute
 * @param attribute6 seventh static integer operation attribute
 * @param attribute7 eighth static integer operation attribute
 * @param outputRank rank of the produced tensor, clamped to at least one
 * @param outputDim0 first output dimension, clamped to at least one
 * @param outputDim1 second output dimension, clamped to at least one
 * @param outputDim2 third output dimension, clamped to at least one
 * @param outputDim3 fourth output dimension, clamped to at least one
 * @param outputDataType dtype produced by this lowered node
 */
public record AcceleratorDagNode(
        int nodeId,
        AcceleratorDagNodeType type,
        AcceleratorDagValueRef input0,
        AcceleratorDagValueRef input1,
        AcceleratorDagValueRef input2,
        AcceleratorDagValueRef input3,
        AcceleratorDagValueRef input4,
        int scalarValueBits,
        int attribute0,
        int attribute1,
        int attribute2,
        int attribute3,
        int attribute4,
        int attribute5,
        int attribute6,
        int attribute7,
        int outputRank,
        int outputDim0,
        int outputDim1,
        int outputDim2,
        int outputDim3,
        DataType outputDataType
) {
    public AcceleratorDagNode {
        Objects.requireNonNull(type, "type cannot be null");
        input0 = input0 == null ? AcceleratorDagValueRef.none() : input0;
        input1 = input1 == null ? AcceleratorDagValueRef.none() : input1;
        input2 = input2 == null ? AcceleratorDagValueRef.none() : input2;
        input3 = input3 == null ? AcceleratorDagValueRef.none() : input3;
        input4 = input4 == null ? AcceleratorDagValueRef.none() : input4;
        outputRank = Math.max(1, outputRank);
        outputDim0 = Math.max(1, outputDim0);
        outputDim1 = Math.max(1, outputDim1);
        outputDim2 = Math.max(1, outputDim2);
        outputDim3 = Math.max(1, outputDim3);
        outputDataType = outputDataType == null ? DataType.FLOAT32 : outputDataType;
    }

    public boolean hasAttributes() {
        return attribute0 != 0
                || attribute1 != 0
                || attribute2 != 0
                || attribute3 != 0
                || attribute4 != 0
                || attribute5 != 0
                || attribute6 != 0
                || attribute7 != 0;
    }

    public AcceleratorDagNode(
            int nodeId,
            AcceleratorDagNodeType type,
            AcceleratorDagValueRef input0,
            AcceleratorDagValueRef input1,
            AcceleratorDagValueRef input2,
            AcceleratorDagValueRef input3,
            AcceleratorDagValueRef input4,
            int scalarValueBits,
            int outputRank,
            int outputDim0,
            int outputDim1,
            int outputDim2,
            int outputDim3
    ) {
        this(
                nodeId,
                type,
                input0,
                input1,
                input2,
                input3,
                input4,
                scalarValueBits,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                outputRank,
                outputDim0,
                outputDim1,
                outputDim2,
                outputDim3,
                DataType.FLOAT32
        );
    }

    public AcceleratorDagNode(
            int nodeId,
            AcceleratorDagNodeType type,
            AcceleratorDagValueRef input0,
            AcceleratorDagValueRef input1,
            AcceleratorDagValueRef input2,
            AcceleratorDagValueRef input3,
            AcceleratorDagValueRef input4,
            int scalarValueBits,
            int outputRank,
            int outputDim0,
            int outputDim1,
            int outputDim2,
            int outputDim3,
            DataType outputDataType
    ) {
        this(
                nodeId,
                type,
                input0,
                input1,
                input2,
                input3,
                input4,
                scalarValueBits,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                outputRank,
                outputDim0,
                outputDim1,
                outputDim2,
                outputDim3,
                outputDataType
        );
    }

    public AcceleratorDagNode(
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
            int outputDim3,
            DataType outputDataType
    ) {
        this(
                nodeId,
                type,
                input0,
                input1,
                input2,
                input3,
                AcceleratorDagValueRef.none(),
                scalarValueBits,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                outputRank,
                outputDim0,
                outputDim1,
                outputDim2,
                outputDim3,
                outputDataType
        );
    }

    public AcceleratorDagNode(
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
        this(
                nodeId,
                type,
                input0,
                input1,
                input2,
                input3,
                AcceleratorDagValueRef.none(),
                scalarValueBits,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                outputRank,
                outputDim0,
                outputDim1,
                outputDim2,
                outputDim3,
                DataType.FLOAT32
        );
    }
}
