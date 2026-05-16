package backend.prepare;

import backend.lowering.LoweringFamily;
import backend.lowering.region.EmptyRegionPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionLegalityStatus;
import backend.lowering.region.RegionNodePlan;
import backend.lowering.region.RegionRole;
import backend.lowering.region.RegionStorageContract;
import config.compile.CompileConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.compile.CompileArtifacts;
import graph.optimizer.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RegionPlanValidatorTest {
    @Test
    void acceptsTerminalBoundaryOutputRegion() {
        Tensor left = Tensor.ones(new int[]{2, 2});
        Tensor right = Tensor.ones(new int[]{2, 2});
        Tensor add = left.add(right);
        Tensor relu = add.relu();
        CompileArtifacts artifacts = compile(relu);
        int reluId = firstNodeIdForOp(artifacts, Operation.OpType.RELU);
        int addId = artifacts.compiledNodes().get(reluId).inputIds().getFirst();

        RegionExecutionPlan plan = plan(List.of(addId, reluId), List.of(reluId));

        assertDoesNotThrow(() -> RegionPlanValidator.requireBoundaryCoverage(plan, context(artifacts)));
    }

    @Test
    void rejectsExternalConsumerForNonBoundaryNode() {
        Tensor left = Tensor.ones(new int[]{2, 2});
        Tensor right = Tensor.ones(new int[]{2, 2});
        Tensor add = left.add(right);
        Tensor relu = add.relu();
        Tensor leakedConsumer = add.neg();
        Tensor root = relu.add(leakedConsumer);
        CompileArtifacts artifacts = compile(root);
        int reluId = firstNodeIdForOp(artifacts, Operation.OpType.RELU);
        int addId = artifacts.compiledNodes().get(reluId).inputIds().getFirst();

        RegionExecutionPlan plan = plan(List.of(addId, reluId), List.of(reluId));

        assertThrows(
                IllegalStateException.class,
                () -> RegionPlanValidator.requireBoundaryCoverage(plan, context(artifacts))
        );
    }

    @Test
    void rejectsTerminalRegionNodeWhenItIsNotBoundaryOutput() {
        Tensor left = Tensor.ones(new int[]{2, 2});
        Tensor right = Tensor.ones(new int[]{2, 2});
        Tensor add = left.add(right);
        Tensor relu = add.relu();
        CompileArtifacts artifacts = compile(relu);
        int reluId = firstNodeIdForOp(artifacts, Operation.OpType.RELU);
        int addId = artifacts.compiledNodes().get(reluId).inputIds().getFirst();

        RegionExecutionPlan plan = plan(List.of(addId, reluId), List.of(addId));

        assertThrows(
                IllegalStateException.class,
                () -> RegionPlanValidator.requireBoundaryCoverage(plan, context(artifacts))
        );
    }

    private static CompileArtifacts compile(Tensor root) {
        return CompiledGraph.compile(root, CompileConfig.noGraphOptimizationBaseline()
                        .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled()))
                .compileArtifacts();
    }

    private static BackendPrepareContext context(CompileArtifacts artifacts) {
        return new BackendPrepareContext(
                RuntimeConfig.inferenceDefaults(),
                artifacts.supportsBackward(),
                artifacts.compiledNodes(),
                artifacts.descriptorIndex(),
                consumers(artifacts.compiledNodes())
        );
    }

    private static Map<Integer, List<CompiledNode>> consumers(List<CompiledNode> nodes) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : nodes) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : nodes) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private static int firstNodeIdForOp(CompileArtifacts artifacts, Operation.OpType opType) {
        return artifacts.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .mapToInt(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static RegionExecutionPlan plan(List<Integer> orderedNodeIds, List<Integer> boundaryOutputNodeIds) {
        int anchorNodeId = boundaryOutputNodeIds.isEmpty()
                ? orderedNodeIds.getLast()
                : boundaryOutputNodeIds.getLast();
        return new RegionExecutionPlan(
                "test-region",
                PartitionTarget.CPU,
                LoweringFamily.CPU_NATIVE_REGION,
                anchorNodeId,
                orderedNodeIds,
                List.of(),
                boundaryOutputNodeIds,
                orderedNodeIds.stream()
                        .map(nodeId -> nodePlan(nodeId, boundaryOutputNodeIds.contains(nodeId)))
                        .toList(),
                List.of(new RegionExecutionGroup(
                        "test-region-group",
                        orderedNodeIds,
                        RegionExecutionKind.DIRECT_KERNEL,
                        "TEST_KERNEL",
                        List.of(),
                        boundaryOutputNodeIds,
                        List.of(),
                        RegionStorageContract.CPU_NATIVE,
                        "test"
                )),
                RegionCost.ofWork(orderedNodeIds.size()),
                RegionDecision.selected("test", "test"),
                EmptyRegionPayload.INSTANCE
        );
    }

    private static RegionNodePlan nodePlan(int nodeId, boolean boundary) {
        return new RegionNodePlan(
                nodeId,
                operations.Operation.OpType.ADD,
                tensor.DataType.FLOAT64,
                boundary ? RegionRole.BOUNDARY_OUTPUT : RegionRole.LOCAL_KERNEL,
                RegionExecutionKind.DIRECT_KERNEL,
                "TEST_KERNEL",
                RegionStorageContract.CPU_NATIVE,
                List.of(),
                boundary ? List.of(nodeId) : List.of(),
                RegionLegalityStatus.SELECTED,
                "test"
        );
    }
}
