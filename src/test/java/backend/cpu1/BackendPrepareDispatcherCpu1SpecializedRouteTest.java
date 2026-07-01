package backend.cpu1;

import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import backend.lowering.partition.CpuSpecializedPrimitivePayload;
import backend.lowering.partition.PartitionCost;
import backend.lowering.partition.PartitionDecision;
import backend.lowering.partition.BackendPartitionExecutionPlan;
import prepare.context.BackendPrepareContext;
import prepare.orchestration.BackendPrepareDispatcher;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import planning.partition.PartitionTarget;
import planning.partition.specialization.PartitionSpecializationCandidate;
import planning.partition.specialization.PartitionSpecializationKind;
import planning.value.GraphValueRef;
import runtime.execution.InputResidencyRequirement;
import runtime.execution.OutputResidencyEffect;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;

class BackendPrepareDispatcherCpu1SpecializedRouteTest {
    @Test
    void cpuNativeRuntimeMatmulEpiloguePrepareStaysJavaArrayUntilNativeRouteExists() {
        assertCpuNativeMatmulEpilogueStaysJavaArray(matmulReluFixture());
        assertCpuNativeMatmulEpilogueStaysJavaArray(matmulAddBiasFixture());
    }

    private static void assertCpuNativeMatmulEpilogueStaysJavaArray(Fixture fixture) {
        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        var metadata = BackendPrepareDispatcher.from(runtimeConfig)
                .prepareCpuSpecializedStep(fixture.outputNode(), fixture.loweredUnit(), fixture.context(runtimeConfig));

        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, metadata.executable());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, artifact.preparedMatmulUnit().storageKind());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, artifact.preparedMatmulUnit().route());
        assertEquals(InputResidencyRequirement.Mode.CPU_READABLE_ALL, metadata.inputResidencyRequirement().mode());
        assertEquals(OutputResidencyEffect.Mode.CPU_CURRENT_PRESERVE_NATIVE, metadata.outputResidencyEffect().mode());
    }

    private static Fixture matmulReluFixture() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "reluA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "reluB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        return fixture(
                nodes,
                List.of(matmulNodeId, reluNodeId),
                reluNodeId,
                PartitionSpecializationKind.MATMUL_RELU
        );
    }

    private static Fixture matmulAddBiasFixture() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "biasA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "biasB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.add(bias);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        return fixture(
                nodes,
                List.of(matmulNodeId, addNodeId),
                addNodeId,
                PartitionSpecializationKind.MATMUL_ADD_BIAS
        );
    }

    private static Fixture fixture(
            List<CompiledNode> nodes,
            List<Integer> orderedNodeIds,
            int outputNodeId,
            PartitionSpecializationKind kind
    ) {
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        List<Integer> inputNodeIds = externalInputNodeIds(nodes, orderedNodeIds);
        PartitionSpecializationCandidate candidate = new PartitionSpecializationCandidate(
                kind,
                orderedNodeIds,
                inputNodeIds.stream().map(GraphValueRef::node).toList(),
                GraphValueRef.node(outputNodeId),
                outputNodeId,
                kind.name()
        );
        BackendPartitionExecutionPlan plan = new BackendPartitionExecutionPlan(
                "cpu1-specialized-" + kind.name(),
                LoweringFamily.DIRECT_KERNEL,
                outputNodeId,
                orderedNodeIds,
                inputNodeIds,
                List.of(outputNodeId),
                List.of(),
                List.of(),
                PartitionCost.ofWork(0L),
                PartitionDecision.selected("cpu1", kind.name()),
                new CpuSpecializedPrimitivePayload(candidate)
        );
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "cpu1-specialized-" + kind.name(),
                LoweringFamily.DIRECT_KERNEL,
                orderedNodeIds,
                inputNodeIds,
                plan
        );
        return new Fixture(nodes, descriptorIndex, nodes.getLast(), loweredUnit);
    }

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .mapToInt(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static List<Integer> externalInputNodeIds(List<CompiledNode> nodes, List<Integer> selectedNodeIds) {
        java.util.Set<Integer> selected = java.util.Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = nodes.stream()
                    .filter(candidate -> candidate.id() == nodeId)
                    .findFirst()
                    .orElseThrow();
            node.inputIds().stream()
                    .filter(inputId -> !selected.contains(inputId))
                    .forEach(out::add);
        }
        return List.copyOf(out);
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode outputNode,
            LoweredExecutionUnit loweredUnit
    ) {
        BackendPrepareContext context(RuntimeConfig runtimeConfig) {
            return new BackendPrepareContext(runtimeConfig, false, nodes, descriptorIndex, Map.of());
        }
    }
}
