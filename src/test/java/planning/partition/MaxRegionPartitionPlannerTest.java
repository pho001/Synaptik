package planning.partition;

import backend.contract.ComputeBackend;
import backend.metal.lowering.MetalPartitionSupport;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.model.CompiledNode;
import graph.compile.CompileArtifacts;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.CompileMode;
import tensor.DataType;
import tuning.workload.StandardWorkloads;
import tuning.workload.WorkloadEnvironment;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MaxRegionPartitionPlannerTest {
    @Test
    void trainingGpuRegionAbsorbsForwardAndBackwardConsumers() {
        CompileArtifacts artifacts = transformerTrainingArtifacts();

        List<Partition> gpuPartitions = artifacts.partitions().stream()
                .filter(partition -> partition.target().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuPartitions.size());

        Partition trainingRegion = gpuPartitions.getFirst();
        List<Operation.OpType> regionOps = opTypes(trainingRegion, artifacts.compiledNodes());
        int causalMaskNodeId = nodeIdByLabel(artifacts.compiledNodes(), "causal_mask");
        int causalMaskExpandNodeId = artifacts.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .filter(node -> node.operation().opType() == Operation.OpType.EXPAND)
                .filter(node -> node.dataType() == DataType.BOOL)
                .filter(node -> node.inputIds().contains(causalMaskNodeId))
                .mapToInt(CompiledNode::id)
                .findFirst()
                .orElseThrow();
        assertTrue(trainingRegion.orderedNodeIds().size() >= 90);
        assertTrue(trainingRegion.orderedNodeIds().contains(causalMaskExpandNodeId));
        assertTrue(trainingRegion.externalInputNodeIds().contains(causalMaskNodeId));
        assertFalse(trainingRegion.externalInputNodeIds().contains(causalMaskExpandNodeId));
        assertTrue(trainingRegion.orderedNodeIds().stream()
                .map(artifacts.compiledNodes()::get)
                .anyMatch(CompiledNode::backwardNode));
        assertTrue(trainingRegion.orderedNodeIds().stream()
                .map(artifacts.compiledNodes()::get)
                .anyMatch(node -> !node.backwardNode()));
        assertTrue(regionOps.containsAll(List.of(
                        Operation.OpType.LINEAR,
                        Operation.OpType.MATMUL,
                        Operation.OpType.REDUCE_MAX,
                        Operation.OpType.EXP,
                        Operation.OpType.DIV,
                        Operation.OpType.TANH,
                        Operation.OpType.MEAN,
                        Operation.OpType.NOOP,
                        Operation.OpType.SUM,
                        Operation.OpType.EXPAND,
                        Operation.OpType.WHERE
                )));

        assertFalse(regionOps.contains(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
        assertFalse(regionOps.contains(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void canonicalAttentionBackwardHotPathIsSupportedByCapabilityTruth() {
        CompileArtifacts artifacts = transformerTrainingArtifacts();

        Partition backwardAttentionRegion = artifacts.partitions().stream()
                .filter(partition -> partition.target().backend() == ComputeBackend.GPU_METAL)
                .filter(partition -> partition.orderedNodeIds().stream()
                        .map(artifacts.compiledNodes()::get)
                        .anyMatch(CompiledNode::backwardNode))
                .filter(partition -> opTypes(partition, artifacts.compiledNodes()).containsAll(List.of(
                        Operation.OpType.MATMUL,
                        Operation.OpType.SUM,
                        Operation.OpType.SUB,
                        Operation.OpType.WHERE,
                        Operation.OpType.MUL_SCALAR
                )))
                .findFirst()
                .orElseThrow();

        assertTrue(backwardAttentionRegion.orderedNodeIds().stream()
                .map(artifacts.compiledNodes()::get)
                .filter(node -> node.operation() != null)
                .allMatch(node ->
                        MetalPartitionSupport.isPlannerSupported(node, planningContext(artifacts))));
        assertFalse(opTypes(backwardAttentionRegion, artifacts.compiledNodes()).contains(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    @Test
    void anchorBasedPlanningCoalescesCanonicalTrainingHotPath() {
        CompileArtifacts artifacts = transformerTrainingArtifacts();
        List<Partition> gpuPartitions = artifacts.partitions().stream()
                .filter(partition -> partition.target().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertFalse(gpuPartitions.isEmpty());
        assertTrue(gpuPartitions.stream().allMatch(partition ->
                partition.plannerStrategy() == PartitionPlannerStrategy.ANCHOR_MAX_REGION));
        assertEquals(1, gpuPartitions.size());

        List<CompiledNode> directSdpaNodes = artifacts.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .filter(node -> node.operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION
                        || node.operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)
                .toList();
        assertTrue(directSdpaNodes.isEmpty());
        assertTrue(gpuPartitions.stream()
                .filter(partition -> partition.orderedNodeIds().stream()
                        .map(artifacts.compiledNodes()::get)
                        .anyMatch(CompiledNode::backwardNode))
                .anyMatch(partition -> partition.orderedNodeIds().size() >= 90
                        && opTypes(partition, artifacts.compiledNodes()).containsAll(List.of(
                                Operation.OpType.MATMUL,
                                Operation.OpType.WHERE,
                                Operation.OpType.SUM
                        ))));
    }

    @Test
    void autoMetalPlanningAbsorbsSupportedCpuProducerClosure() {
        ExecutionProfile profile = new ExecutionProfile(
                "metal-reduction-producer-closure-test",
                "metal-reduction-producer-closure-test",
                DataType.BFLOAT16,
                ExecutionMode.FORWARD,
                CompileConfig.inference().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );
        var instance = StandardWorkloads.reductionChain("metal_reduction_producer_closure_bf16", 64, 1024)
                .instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(instance.root(), profile.compile());
        CompileArtifacts artifacts = new CompileArtifacts(compiled.program(), compiled.publication());

        List<Partition> gpuPartitions = artifacts.partitions().stream()
                .filter(partition -> partition.target().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertEquals(1, gpuPartitions.size(), () -> "Expected one Metal region, nodes=" + describeNodes(artifacts.compiledNodes())
                + " trace=" + compiled.compileTrace().partitionPlanning().decisions());
        Partition partition = gpuPartitions.getFirst();
        List<Operation.OpType> regionOps = opTypes(partition, artifacts.compiledNodes());
        assertTrue(regionOps.contains(Operation.OpType.EXPAND));
        assertTrue(regionOps.contains(Operation.OpType.MUL));
        assertTrue(regionOps.containsAll(List.of(
                Operation.OpType.SUM,
                Operation.OpType.MEAN,
                Operation.OpType.REDUCE_MIN,
                Operation.OpType.REDUCE_MAX
        )));
        assertFalse(partition.externalInputNodeIds().stream()
                .map(artifacts.compiledNodes()::get)
                .filter(node -> node.operation() != null)
                .map(node -> node.operation().opType())
                .anyMatch(op -> op == Operation.OpType.EXPAND || op == Operation.OpType.MUL));
    }

    private static CompileArtifacts transformerTrainingArtifacts() {
        ExecutionProfile profile = new ExecutionProfile(
                "metal-greedy-test",
                "metal-greedy-test",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator()),
                RuntimeConfig.trainingDefaults(),
                WorkloadProfile.transformerHotPathMedium()
        );
        var instance = StandardWorkloads.transformerBlockHotPath(
                "transformer_block_hot_path_medium_f32",
                WorkloadProfile.transformerHotPathMedium()
        ).instantiate(new WorkloadEnvironment(profile));
        CompiledGraph compiled = CompiledGraph.compile(instance.root(), profile.compile(), CompileMode.TRAINING);
        return new CompileArtifacts(compiled.program(), compiled.publication());
    }

    private static PartitionPlanningContext planningContext(CompileArtifacts artifacts) {
        java.util.Map<Integer, List<CompiledNode>> consumers = new java.util.LinkedHashMap<>();
        for (CompiledNode node : artifacts.compiledNodes()) {
            consumers.computeIfAbsent(node.id(), ignored -> new java.util.ArrayList<>());
        }
        for (CompiledNode node : artifacts.compiledNodes()) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new java.util.ArrayList<>()).add(node);
            }
        }
        return new PartitionPlanningContext(
                artifacts.supportsBackward(),
                artifacts.compiledNodes(),
                artifacts.descriptorIndex(),
                consumers
        );
    }

    private static List<Operation.OpType> opTypes(Partition partition, List<CompiledNode> nodes) {
        return partition.orderedNodeIds().stream()
                .map(nodes::get)
                .filter(node -> node.operation() != null)
                .map(node -> node.operation().opType())
                .toList();
    }

    private static int nodeIdByLabel(List<CompiledNode> nodes, String label) {
        return nodes.stream()
                .filter(node -> label.equals(node.label()))
                .mapToInt(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static String describeNodes(List<CompiledNode> nodes) {
        return nodes.stream()
                .map(node -> node.id()
                        + ":"
                        + (node.operation() == null ? "LEAF" : node.operation().opType().name())
                        + ":"
                        + node.dataType()
                        + ":"
                        + node.backend()
                        + ":inputs="
                        + node.inputIds())
                .toList()
                .toString();
    }
}
