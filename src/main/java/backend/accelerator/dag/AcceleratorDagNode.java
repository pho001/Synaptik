package backend.accelerator.dag;

import java.util.Objects;

/**
 * ABI-stable description of one lowered accelerator DAG operation.
 *
 * <p>The four inputs reference either external DAG inputs or earlier node outputs.
 * Scalar payloads are stored as raw float bits so Java and native bridges agree on
 * the wire representation.</p>
 *
 * @param nodeId compiled-node id represented by this DAG node
 * @param type lowered operation type
 * @param input0 first operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input1 second operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input2 third operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param input3 fourth operand reference, or {@link AcceleratorDagValueRef#none()}
 * @param scalarValueBits raw {@code float} bits for scalar-valued operations
 * @param outputRank rank of the produced tensor, clamped to at least one
 * @param outputDim0 first output dimension, clamped to at least one
 * @param outputDim1 second output dimension, clamped to at least one
 * @param outputDim2 third output dimension, clamped to at least one
 * @param outputDim3 fourth output dimension, clamped to at least one
 */
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
