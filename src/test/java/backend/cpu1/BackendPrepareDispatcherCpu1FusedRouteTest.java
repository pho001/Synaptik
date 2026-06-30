package backend.cpu1;

import backend.cpu.CpuFusedExecutionArtifact;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.lowering.LoweredExecutionUnit;
import backend.lowering.LoweringFamily;
import backend.prepare.BackendPrepareContext;
import backend.prepare.BackendPrepareDispatcher;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.backend.CpuKernelConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.CompiledGraph;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import graph.execution.PreparedExecution;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.fail;

class BackendPrepareDispatcherCpu1FusedRouteTest {
    @Test
    void defaultFusedPolicyKeepsOldCpuFusedRoute() {
        Fixture fixture = fixture();
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(fixture.reluNodeId())
        );
        RuntimeConfig runtimeConfig = runtimeConfig(FusedExecutionPolicy.defaultsInference());
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        IllegalStateException thrown = assertThrows(
                IllegalStateException.class,
                () -> dispatcher.prepareCpuFusedStep(
                        fixture.outputNode(),
                        loweredUnit,
                        fixture.context(runtimeConfig)
                )
        );

        assertTrue(thrown.getMessage().contains("requires artifact RegionExecutionPlan"));
    }

    @Test
    void explicitCpu1FusedPolicyRoutesToCpu1PreparerWithoutFallback() {
        Fixture fixture = fixture();
        LoweredExecutionUnit loweredUnit = new LoweredExecutionUnit(
                "relu",
                LoweringFamily.FUSED_NATIVE,
                List.of(fixture.reluNodeId())
        );
        RuntimeConfig runtimeConfig = runtimeConfig(new FusedExecutionPolicy(true, true));
        BackendPrepareDispatcher dispatcher = BackendPrepareDispatcher.from(runtimeConfig);

        var metadata = dispatcher.prepareCpuFusedStep(
                fixture.outputNode(),
                loweredUnit,
                fixture.context(runtimeConfig)
        );

        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, metadata.artifact());
        assertTrue(artifact.preparedFusedElementwiseUnit().generatedKernel()
                .generatedClassName()
                .contains("Cpu1GeneratedFusedKernel"));
    }

    @Test
    void compiledGraphCpu1FusedRouteMatchesOldCpuFusedRoute() {
        assertCompiledGraphCpu1FusedRouteMatchesOldCpuFusedRoute(DataType.FLOAT32, 1.0e-5);
    }

    @Test
    void compiledGraphCpu1FusedRouteMatchesOldCpuFusedRouteForF64() {
        assertCompiledGraphCpu1FusedRouteMatchesOldCpuFusedRoute(DataType.FLOAT64, 1.0e-12);
    }

    @Test
    void compiledGraphCpu1FusedRouteMatchesOldCpuFusedRouteForBF16() {
        assertCompiledGraphCpu1FusedRouteMatchesOldCpuFusedRoute(DataType.BFLOAT16, 6.0e-3);
    }

    @Test
    void compiledGraphCpu1FusedRouteRejectsBoolOutputWithoutFallback() {
        GraphRun oldRoute = executeCompiledBoolFusedGraph(false);
        assertTrue(hasOldCpuFusedArtifact(oldRoute.execution()));
        assertArrayEquals(new byte[]{1, 0, 1, 1}, oldRoute.boolOutput());

        UnsupportedOperationException thrown = assertThrows(
                UnsupportedOperationException.class,
                () -> executeCompiledBoolFusedGraph(true)
        );
        assertTrue(thrown.getMessage().contains("UNSUPPORTED_DTYPE"));
    }

    private static void assertCompiledGraphCpu1FusedRouteMatchesOldCpuFusedRoute(DataType dataType, double tolerance) {
        GraphRun oldRoute = executeCompiledFusedGraph(dataType, false);
        GraphRun cpu1Route = executeCompiledFusedGraph(dataType, true);

        assertTrue(hasOldCpuFusedArtifact(oldRoute.execution()));
        Cpu1PreparedArtifact cpu1Artifact = requireCpu1FusedArtifact(cpu1Route.execution());
        assertTrue(cpu1Artifact.preparedFusedElementwiseUnit().generatedKernel()
                .generatedClassName()
                .contains("Cpu1GeneratedFusedKernel"));
        assertArrayEquals(oldRoute.numericOutput(), cpu1Route.numericOutput(), tolerance);
    }

    private static Fixture fixture() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 1.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Tensor output = input.relu();
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(output.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        int reluNodeId = nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.RELU)
                .findFirst()
                .orElseThrow()
                .id();
        return new Fixture(nodes, descriptorIndex, nodes.getLast(), reluNodeId);
    }

    private static RuntimeConfig runtimeConfig(FusedExecutionPolicy fusedExecutionPolicy) {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                fusedExecutionPolicy
        );
    }

    private static GraphRun executeCompiledFusedGraph(DataType dataType, boolean useCpu1Elementwise) {
        Tensor output = fusedExpression(dataType);
        PreparedExecution execution = CompiledGraph.compile(output, CompileConfig.inference())
                .prepare(runtimeConfig(FusedExecutionPolicy.defaultsInference()
                        .withUseCpu1Elementwise(useCpu1Elementwise)));

        execution.execute(ExecutionMode.FORWARD);

        return new GraphRun(execution, output.toDoubleArrayCopy(), null);
    }

    private static GraphRun executeCompiledBoolFusedGraph(boolean useCpu1Elementwise) {
        Tensor output = boolFusedExpression();
        PreparedExecution execution = CompiledGraph.compile(output, CompileConfig.inference())
                .prepare(runtimeConfig(FusedExecutionPolicy.defaultsInference()
                        .withUseCpu1Elementwise(useCpu1Elementwise)));

        execution.execute(ExecutionMode.FORWARD);

        return new GraphRun(execution, null, output.toBoolByteArrayCopy());
    }

    private static Tensor fusedExpression(DataType dataType) {
        double[] a = new double[]{
                -2.0, -1.0, 0.0, 1.0,
                2.0, 3.0, -3.0, 4.0,
                0.5, -0.5, 1.5, -1.5
        };
        double[] b = new double[]{
                0.25, -0.5, 1.0, 1.5,
                -1.25, 0.75, 0.5, -0.25,
                2.0, -2.0, 0.125, -0.125
        };
        double[] c = new double[]{
                0.5, -0.25, 0.75, -1.0,
                1.25, -1.5, 0.0, 2.0,
                -2.5, 0.375, -0.625, 1.125
        };
        return fusedExpression(tensor(a, "a", dataType), tensor(b, "b", dataType), tensor(c, "c", dataType));
    }

    private static Tensor fusedExpression(Tensor a, Tensor b, Tensor c) {
        return a.mul(b).add(c).relu().mul(0.25d).abs();
    }

    private static Tensor boolFusedExpression() {
        int[] shape = new int[]{4};
        Tensor a = new Tensor(toFloatArray(new double[]{1.0, 5.0, -2.0, 7.0}), shape, null, "bool_a", DataType.FLOAT32);
        Tensor b = new Tensor(toFloatArray(new double[]{0.0, 6.0, -3.0, 8.0}), shape, null, "bool_b", DataType.FLOAT32);
        Tensor c = new Tensor(toFloatArray(new double[]{4.0, -1.0, 3.0, 2.0}), shape, null, "bool_c", DataType.FLOAT32);
        Tensor d = new Tensor(toFloatArray(new double[]{4.0, 0.0, 2.0, 1.0}), shape, null, "bool_d", DataType.FLOAT32);
        return a.greaterThan(b).logicalOr(c.greaterThan(d)).logicalNot().logicalNot();
    }

    private static Tensor tensor(double[] values, String label, DataType dataType) {
        int[] shape = new int[]{3, 4};
        return switch (dataType) {
            case FLOAT64 -> new Tensor(values.clone(), shape, null, label, DataType.FLOAT64);
            case FLOAT32 -> new Tensor(toFloatArray(values), shape, null, label, DataType.FLOAT32);
            case BFLOAT16 -> new Tensor(toFloatArray(values), shape, null, label, DataType.BFLOAT16);
            default -> throw new IllegalArgumentException("Unsupported test dtype: " + dataType);
        };
    }

    private static float[] toFloatArray(double[] values) {
        float[] floats = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            floats[i] = (float) values[i];
        }
        return floats;
    }

    private static boolean hasOldCpuFusedArtifact(PreparedExecution execution) {
        return execution.forwardSteps().stream()
                .anyMatch(step -> step.metadata().artifact() instanceof CpuFusedExecutionArtifact);
    }

    private static Cpu1PreparedArtifact requireCpu1FusedArtifact(PreparedExecution execution) {
        for (var step : execution.forwardSteps()) {
            if (step.metadata().artifact() instanceof Cpu1PreparedArtifact artifact) {
                try {
                    artifact.preparedFusedElementwiseUnit();
                    return artifact;
                } catch (IllegalStateException ignored) {
                    // Other cpu1 artifacts are not relevant to this route test.
                }
            }
        }
        return fail("Expected a cpu1 prepared fused elementwise artifact.");
    }

    private record Fixture(
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode outputNode,
            int reluNodeId
    ) {
        BackendPrepareContext context(RuntimeConfig runtimeConfig) {
            return new BackendPrepareContext(runtimeConfig, false, nodes, descriptorIndex, Map.of());
        }
    }

    private record GraphRun(PreparedExecution execution, double[] numericOutput, byte[] boolOutput) {
    }
}
