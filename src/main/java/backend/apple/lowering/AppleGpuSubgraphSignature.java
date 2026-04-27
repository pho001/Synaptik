package backend.apple.lowering;

import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorPostOpSignature;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import operations.Operation;

import java.util.List;

public record AppleGpuSubgraphSignature(
        List<Operation.OpType> ops,
        List<String> externalInputShapes,
        List<AcceleratorDagNodeType> dagNodeTypes,
        List<String> dagNodeShapes,
        List<AcceleratorPostOpSignature> postOps,
        List<Integer> outputNodeIds
) {
    public AppleGpuSubgraphSignature {
        ops = List.copyOf(ops == null ? List.of() : ops);
        externalInputShapes = List.copyOf(externalInputShapes == null ? List.of() : externalInputShapes);
        dagNodeTypes = List.copyOf(dagNodeTypes == null ? List.of() : dagNodeTypes);
        dagNodeShapes = List.copyOf(dagNodeShapes == null ? List.of() : dagNodeShapes);
        postOps = List.copyOf(postOps == null ? List.of() : postOps);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
    }

    public static AppleGpuSubgraphSignature from(AppleGpuPartitionPlan plan) {
        List<Operation.OpType> ops = plan.subgraph().ops().stream()
                .map(AcceleratorSubgraphOp::opType)
                .toList();
        AcceleratorDagSpec dagSpec = plan.lowering().dagSpec();
        return new AppleGpuSubgraphSignature(
                ops,
                dagSpec.externalInputs().stream().map(input -> input.dataType().name() + ":" + input.shape()).toList(),
                dagSpec.nodes().stream().map(AcceleratorDagNode::type).toList(),
                dagSpec.nodes().stream().map(node ->
                        node.outputRank() + ":" + node.outputDim0() + ":" + node.outputDim1() + ":" + node.outputDim2() + ":" + node.outputDim3()
                ).toList(),
                plan.matMulSpec() == null
                        ? List.of()
                        : plan.matMulSpec().postOps().stream().map(AcceleratorPostOpSignature::from).toList(),
                dagSpec.outputNodeIds()
        );
    }
}
