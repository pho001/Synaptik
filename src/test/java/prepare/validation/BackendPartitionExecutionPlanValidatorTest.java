package prepare.validation;

import backend.lowering.LoweringFamily;
import backend.lowering.partition.EmptyPartitionPayload;
import backend.lowering.partition.PartitionCost;
import backend.lowering.partition.PartitionDecision;
import backend.lowering.partition.PartitionExecutionGroup;
import backend.lowering.partition.PartitionExecutionKind;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import backend.lowering.partition.PartitionLegalityStatus;
import backend.lowering.partition.PartitionNodePlan;
import backend.lowering.partition.PartitionRole;
import backend.lowering.partition.PartitionStorageContract;
import config.compile.CompileConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.model.CompiledNode;
import graph.compile.CompileArtifacts;
import planning.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import operations.Operation;
import prepare.context.BackendPrepareContext;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

class BackendPartitionExecutionPlanValidatorTest {
    @Test
    void acceptsTerminalBoundaryOutputPartition() {
        Tensor left = Tensor.ones(new int[]{2, 2});
        Tensor right = Tensor.ones(new int[]{2, 2});
        Tensor add = left.add(right);
        Tensor relu = add.relu();
        CompileArtifacts artifacts = compile(relu);
        int reluId = firstNodeIdForOp(artifacts, Operation.OpType.RELU);
        int addId = artifacts.compiledNodes().get(reluId).inputIds().getFirst();

        BackendPartitionExecutionPlan plan = plan(List.of(addId, reluId), List.of(reluId));

        assertDoesNotThrow(() -> BackendPartitionExecutionPlanValidator.requireBoundaryCoverage(plan, context(artifacts)));
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

        BackendPartitionExecutionPlan plan = plan(List.of(addId, reluId), List.of(reluId));

        assertThrows(
                IllegalStateException.class,
                () -> BackendPartitionExecutionPlanValidator.requireBoundaryCoverage(plan, context(artifacts))
        );
    }

    @Test
    void rejectsTerminalPartitionNodeWhenItIsNotBoundaryOutput() {
        Tensor left = Tensor.ones(new int[]{2, 2});
        Tensor right = Tensor.ones(new int[]{2, 2});
        Tensor add = left.add(right);
        Tensor relu = add.relu();
        CompileArtifacts artifacts = compile(relu);
        int reluId = firstNodeIdForOp(artifacts, Operation.OpType.RELU);
        int addId = artifacts.compiledNodes().get(reluId).inputIds().getFirst();

        BackendPartitionExecutionPlan plan = plan(List.of(addId, reluId), List.of(addId));

        assertThrows(
                IllegalStateException.class,
                () -> BackendPartitionExecutionPlanValidator.requireBoundaryCoverage(plan, context(artifacts))
        );
    }

    private static CompileArtifacts compile(Tensor root) {
        CompiledGraph compiled = CompiledGraph.compile(root, CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled()));
        return new CompileArtifacts(compiled.program(), compiled.publication());
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

    private static BackendPartitionExecutionPlan plan(List<Integer> orderedNodeIds, List<Integer> boundaryOutputNodeIds) {
        int anchorNodeId = boundaryOutputNodeIds.isEmpty()
                ? orderedNodeIds.getLast()
                : boundaryOutputNodeIds.getLast();
        return new BackendPartitionExecutionPlan(
                "test-partition",
                LoweringFamily.DIRECT_KERNEL,
                anchorNodeId,
                orderedNodeIds,
                List.of(),
                boundaryOutputNodeIds,
                orderedNodeIds.stream()
                        .map(nodeId -> nodePlan(nodeId, boundaryOutputNodeIds.contains(nodeId)))
                        .toList(),
                List.of(new PartitionExecutionGroup(
                        "test-partition-group",
                        orderedNodeIds,
                        PartitionExecutionKind.DIRECT_KERNEL,
                        "TEST_KERNEL",
                        List.of(),
                        boundaryOutputNodeIds,
                        List.of(),
                        PartitionStorageContract.CPU_NATIVE,
                        "test"
                )),
                PartitionCost.ofWork(orderedNodeIds.size()),
                PartitionDecision.selected("test", "test"),
                EmptyPartitionPayload.INSTANCE
        );
    }

    private static PartitionNodePlan nodePlan(int nodeId, boolean boundary) {
        return new PartitionNodePlan(
                nodeId,
                operations.Operation.OpType.ADD,
                tensor.DataType.FLOAT64,
                boundary ? PartitionRole.BOUNDARY_OUTPUT : PartitionRole.LOCAL_KERNEL,
                PartitionExecutionKind.DIRECT_KERNEL,
                "TEST_KERNEL",
                PartitionStorageContract.CPU_NATIVE,
                List.of(),
                boundary ? List.of(nodeId) : List.of(),
                PartitionLegalityStatus.SELECTED,
                "test"
        );
    }
}
