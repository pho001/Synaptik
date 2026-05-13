package backend.accelerator.lowering;

import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorPostOpSignature;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.metal.lowering.MetalPartitionPlan;
import operations.Operation;

import java.util.List;

/**
 * Cache key for native accelerator executables.
 *
 * <p>The signature captures operation kinds, external-input shape/type contracts,
 * lowered DAG node shapes, and output ids while avoiding bridge handles and runtime
 * tensor objects.</p>
 *
 * @param ops source operation kinds in subgraph order
 * @param externalInputShapes dtype and shape descriptors for external inputs
 * @param dagNodeTypes lowered DAG node kinds
 * @param dagNodeScalarValueBits lowered DAG scalar parameters
 * @param dagNodeAttributes lowered DAG static integer attributes
 * @param dagNodeShapes lowered DAG output shape descriptors
 * @param postOps cache-stable matmul post-op signatures
 * @param outputNodeIds compiled-node ids produced by the partition
 */
public record AcceleratorSubgraphSignature(
        List<Operation.OpType> ops,
        List<String> externalInputShapes,
        List<AcceleratorDagNodeType> dagNodeTypes,
        List<Integer> dagNodeScalarValueBits,
        List<String> dagNodeAttributes,
        List<String> dagNodeShapes,
        List<AcceleratorPostOpSignature> postOps,
        List<Integer> outputNodeIds
) {
    public AcceleratorSubgraphSignature {
        ops = List.copyOf(ops == null ? List.of() : ops);
        externalInputShapes = List.copyOf(externalInputShapes == null ? List.of() : externalInputShapes);
        dagNodeTypes = List.copyOf(dagNodeTypes == null ? List.of() : dagNodeTypes);
        dagNodeScalarValueBits = List.copyOf(dagNodeScalarValueBits == null ? List.of() : dagNodeScalarValueBits);
        dagNodeAttributes = List.copyOf(dagNodeAttributes == null ? List.of() : dagNodeAttributes);
        dagNodeShapes = List.copyOf(dagNodeShapes == null ? List.of() : dagNodeShapes);
        postOps = List.copyOf(postOps == null ? List.of() : postOps);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
    }

    /**
     * Builds the cache signature for a lowered Metal partition plan.
     */
    public static AcceleratorSubgraphSignature from(MetalPartitionPlan plan) {
        List<Operation.OpType> ops = plan.subgraph().ops().stream()
                .map(AcceleratorSubgraphOp::opType)
                .toList();
        AcceleratorDagSpec dagSpec = plan.lowering().dagSpec();
        return new AcceleratorSubgraphSignature(
                ops,
                dagSpec.externalInputs().stream().map(input -> input.dataType().name() + ":" + input.shape()).toList(),
                dagSpec.nodes().stream().map(AcceleratorDagNode::type).toList(),
                dagSpec.nodes().stream().map(AcceleratorDagNode::scalarValueBits).toList(),
                dagSpec.nodes().stream().map(node ->
                        node.attribute0() + ":" + node.attribute1() + ":" + node.attribute2() + ":" + node.attribute3()
                                + ":" + node.attribute4() + ":" + node.attribute5() + ":" + node.attribute6() + ":" + node.attribute7()
                ).toList(),
                dagSpec.nodes().stream().map(node ->
                        node.outputRank() + ":" + node.outputDim0() + ":" + node.outputDim1() + ":" + node.outputDim2() + ":" + node.outputDim3()
                                + ":" + node.outputDataType()
                ).toList(),
                plan.matMulSpec() == null
                        ? List.of()
                        : plan.matMulSpec().postOps().stream().map(AcceleratorPostOpSignature::from).toList(),
                dagSpec.outputNodeIds()
        );
    }
}
