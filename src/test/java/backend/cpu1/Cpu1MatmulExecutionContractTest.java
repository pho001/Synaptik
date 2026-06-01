package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.exec.Cpu1MatmulExecutableUnit;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.PreparedExecutionStep;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.contrib.StepExecutionTracer;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

class Cpu1MatmulExecutionContractTest {
    @Test
    void preparedF32MatmulRunsDenseJavaScalarRoute() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                new int[]{3, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{58.0f, 64.0f, 139.0f, 154.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64MatmulSupportsBroadcastBatch() {
        Tensor left = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d,
                        7.0d, 8.0d, 9.0d,
                        10.0d, 11.0d, 12.0d
                },
                new int[]{2, 2, 3},
                null,
                "left",
                DataType.FLOAT64
        );
        Tensor right = new Tensor(
                new double[]{
                        1.0d, 2.0d,
                        3.0d, 4.0d,
                        5.0d, 6.0d
                },
                new int[]{1, 3, 2},
                null,
                "right",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 2, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new double[]{
                        22.0d, 28.0d,
                        49.0d, 64.0d,
                        76.0d, 100.0d,
                        103.0d, 136.0d
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedBf16MatmulAccumulatesInF32AndStoresBf16() {
        Tensor left = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "left",
                DataType.BFLOAT16
        );
        Tensor right = new Tensor(
                new float[]{
                        5.0f, 6.0f,
                        7.0f, 8.0f
                },
                new int[]{2, 2},
                null,
                "right",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertMatmulKernel(artifact, Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{19.0f, 22.0f, 43.0f, 50.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-6f
        );
    }

    @Test
    void matmulTraceReportsPreparedRoute() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f},
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.matmul(right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1MatmulKernelId.MATMUL_F32_DENSE_SCALAR.name(), trace.kernel());
        assertEquals("JAVA_SCALAR", trace.metadata().matMul().route());
        assertEquals("JAVA_SCALAR", trace.metadata().attributes().get("cpu1MatmulRoute"));
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertMatmulKernel(Cpu1PreparedArtifact artifact, Cpu1MatmulKernelId expected) {
        Cpu1MatmulExecutableUnit executable = assertInstanceOf(Cpu1MatmulExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedMatmulUnit().kernelId());
        assertSame(artifact.preparedMatmulUnit(), executable.preparedUnit());
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(
            Fixture fixture,
            Map<Integer, CompiledNodeExecutionMetadata> metadataIndex
    ) {
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] source = TensorInternalAccess.bfloat16Data(tensor);
        float[] out = new float[source.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(source[i]);
        }
        return out;
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
