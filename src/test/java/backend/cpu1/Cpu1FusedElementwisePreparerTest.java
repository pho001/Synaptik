package backend.cpu1;

import backend.ApproxMode;
import backend.cpu1.fused.ir.Cpu1FusedExpressionPlan;
import backend.cpu1.fused.ir.Cpu1FusedIrBuilder;
import backend.cpu1.kernels.fused.codegen.Cpu1FusedCodegenRejectionReason;
import backend.cpu1.prepare.Cpu1FusedElementwisePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.dispatch.Cpu1CostClass;
import backend.cpu1.prepare.dispatch.Cpu1DispatchPolicy;
import backend.cpu1.prepare.dispatch.Cpu1FusedDispatchDecision;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import operations.Operation;
import operations.elementwise.unary.pow;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1FusedElementwisePreparerTest {
    @Test
    void buildsCanonicalizedPlanAndClassifiesCostFromSourceOperationsBeforeCodegen() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor pow2 = new Tensor(new int[]{3}, List.of(input), new pow(2.0f), "pow2", DataType.FLOAT32);
        Fixture fixture = fixture(pow2.relu());
        int powNodeId = nodeId(fixture.nodes(), Operation.OpType.POW);
        int reluNodeId = nodeId(fixture.nodes(), Operation.OpType.RELU);
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "pow-relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(powNodeId, reluNodeId)
        );
        RuntimeConfig runtimeConfig = runtimeConfig(ApproxMode.OFF).withCpuStorageProfile(CpuStorageProfile.CPU_ARRAY);

        Cpu1FusedExpressionPlan plan = Cpu1FusedIrBuilder.build(
                loweredUnit.orderedNodeIds(),
                fixture::compiledNode,
                fixture.descriptorIndex()
        );
        Cpu1FusedDispatchDecision decision = new Cpu1DispatchPolicy().decideFusedElementwise(
                plan,
                operations(fixture.nodes(), loweredUnit.orderedNodeIds()),
                DataType.FLOAT32,
                fixture.outputNode().flatDataSize(),
                Cpu1PrepareConfig.automatic(runtimeConfig, 1, Cpu1StorageKind.JAVA_ARRAY)
        );

        assertEquals(Cpu1CostClass.EXPENSIVE_ELEMENTWISE, decision.costClass());
        assertEquals(Operation.OpType.MUL, plan.nodes().getFirst().opType());
        assertEquals(Operation.OpType.RELU, plan.nodes().getLast().opType());

        UnsupportedOperationException thrown = assertThrows(UnsupportedOperationException.class,
                () -> new Cpu1FusedElementwisePreparer(runtimeConfig)
                        .prepareUnit(fixture.outputNode(), loweredUnit, fixture.context(runtimeConfig)));
        assertTrue(thrown.getMessage().contains(Cpu1FusedCodegenRejectionReason.MISSING_ASM_EMITTER.name()));
    }

    @Test
    void rejectsUnsupportedIntrinsicDuringPrepareBeforeCreatingPreparedUnit() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 1.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Fixture fixture = fixture(input.exp().tanh());
        int expNodeId = nodeId(fixture.nodes(), Operation.OpType.EXP);
        int tanhNodeId = nodeId(fixture.nodes(), Operation.OpType.TANH);
        RuntimeConfig runtimeConfig = runtimeConfig(ApproxMode.ALWAYS)
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "exp-tanh",
                LoweringFamily.FUSED_NATIVE,
                List.of(expNodeId, tanhNodeId)
        );

        UnsupportedOperationException thrown = assertThrows(UnsupportedOperationException.class,
                () -> new Cpu1FusedElementwisePreparer(runtimeConfig)
                        .prepareUnit(fixture.outputNode(), loweredUnit, fixture.context(runtimeConfig)));

        assertTrue(thrown.getMessage().contains(Cpu1FusedCodegenRejectionReason.UNSUPPORTED_INTRINSIC.name()));
    }

    @Test
    void rejectsConcreteNonFusableSourceOperation() {
        Tensor base = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        Fixture fixture = fixture(base.select(0, 1).relu());
        int selectNodeId = nodeId(fixture.nodes(), Operation.OpType.SELECT);
        int reluNodeId = nodeId(fixture.nodes(), Operation.OpType.RELU);
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "select-relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(selectNodeId, reluNodeId)
        );
        Cpu1FusedElementwisePreparer preparer = new Cpu1FusedElementwisePreparer(
                RuntimeConfig.inferenceDefaults(DataType.FLOAT32)
        );

        assertThrows(UnsupportedOperationException.class,
                () -> preparer.prepareUnit(fixture.outputNode(), loweredUnit, fixture.context()));
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(nodes, descriptorIndex, nodes.getLast());
    }

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow()
                .id();
    }

    private static List<Operation> operations(List<CompiledNode> nodes, List<Integer> nodeIds) {
        return nodeIds.stream()
                .map(nodeId -> nodes.stream()
                        .filter(node -> node.id() == nodeId)
                        .findFirst()
                        .orElseThrow()
                        .operation())
                .toList();
    }

    private static RuntimeConfig runtimeConfig(ApproxMode approxMode) {
        return new RuntimeConfig(
                config.backend.CpuKernelConfig.defaultsInference(),
                new ApproximationConfig(approxMode, false),
                BlasConfig.disabled()
        );
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode outputNode
    ) {
        CompiledNode compiledNode(int nodeId) {
            return nodes.stream()
                    .filter(node -> node.id() == nodeId)
                    .findFirst()
                    .orElseThrow();
        }

        BackendPrepareContext context() {
            return context(RuntimeConfig.inferenceDefaults(outputNode.dataType()));
        }

        BackendPrepareContext context(RuntimeConfig runtimeConfig) {
            return new BackendPrepareContext(runtimeConfig, false, nodes, descriptorIndex, Map.of());
        }
    }
}
