package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelId;
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
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.logSoftmax;
import operations.reduction.softmax;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;

class Cpu1ReductionExecutionContractTest {
    @Test
    void preparedF32SumReducesDenseAxis() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.sum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{6.0f, 15.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64MeanReducesDenseAxisWithKeepDims() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d,
                        3.0d, 4.0d,
                        5.0d, 6.0d,
                        7.0d, 8.0d
                },
                new int[]{2, 2, 2},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.mean(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MEAN_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 1, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new double[]{2.0d, 3.0d, 6.0d, 7.0d},
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedBf16SumAccumulatesInF32AndStoresBf16() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.sum(0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_BF16_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{5.0f, 7.0f, 9.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-6f
        );
    }

    @Test
    void preparedF32MinReducesDenseAxis() {
        Tensor input = new Tensor(
                new float[]{
                        3.0f, -2.0f, 5.0f,
                        7.0f, 1.0f, -4.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.min(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MIN_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{-2.0f, -4.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64MaxReducesDenseAxisWithKeepDims() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 7.0d,
                        3.0d, 4.0d,
                        5.0d, 6.0d,
                        -1.0d, 8.0d
                },
                new int[]{2, 2, 2},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.max(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MAX_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 1, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new double[]{3.0d, 7.0d, 5.0d, 8.0d},
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedBf16ProdAccumulatesInF32AndStoresBf16() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.prod(0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.PROD_BF16_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{4.0f, 10.0f, 18.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-6f
        );
    }

    @Test
    void preparedBoolAllReducesDenseAxis() {
        Tensor input = new Tensor(
                boolBytes(
                        true, true, false,
                        true, true, true
                ),
                new int[]{2, 3},
                null,
                "input",
                DataType.BOOL
        );
        Fixture fixture = fixture(input.all(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ALL_BOOL_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                boolBytes(false, true),
                context.runtimeTensorForNodeId(fixture.node().id()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedBoolAnyReducesDenseAxisWithKeepDims() {
        Tensor input = new Tensor(
                boolBytes(
                        false, false,
                        true, false,
                        false, true,
                        false, false
                ),
                new int[]{2, 2, 2},
                null,
                "input",
                DataType.BOOL
        );
        Fixture fixture = fixture(input.any(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ANY_BOOL_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 1, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                boolBytes(true, false, false, true),
                context.runtimeTensorForNodeId(fixture.node().id()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedF32ArgMaxReducesDenseAxisToInt64() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 5.0f, 5.0f,
                        4.0f, 3.0f, 2.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.argMax(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_F32_TO_I64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new long[]{1L, 0L},
                context.runtimeTensorForNodeId(fixture.node().id()).toInt64ArrayCopy()
        );
    }

    @Test
    void preparedF64ArgMaxSupportsLastIndexTiePolicyAndKeepDims() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 5.0d,
                        3.0d, 5.0d,
                        7.0d, 2.0d,
                        7.0d, 6.0d
                },
                new int[]{2, 2, 2},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.argMax(1, true, ArgMaxTiePolicy.LAST_INDEX));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_F64_TO_I64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(new int[]{2, 1, 2}, context.runtimeTensorForNodeId(fixture.node().id()).getShape());
        assertArrayEquals(
                new long[]{1L, 1L, 1L, 1L},
                context.runtimeTensorForNodeId(fixture.node().id()).toInt64ArrayCopy()
        );
    }

    @Test
    void preparedI32ArgMaxUsesIntInputAndInt64Output() {
        Tensor input = new Tensor(
                new int[]{
                        1, 9, 3,
                        4, 4, 2
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.INT32
        );
        Fixture fixture = fixture(input.argMax(1, false, ArgMaxTiePolicy.LAST_INDEX));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_I32_TO_I64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new long[]{1L, 1L},
                context.runtimeTensorForNodeId(fixture.node().id()).toInt64ArrayCopy()
        );
    }

    @Test
    void preparedF32CumSumScansDenseAxis() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64CumSumSupportsExclusiveReverse() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.cumSum(1, true, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new double[]{5.0d, 3.0d, 0.0d, 11.0d, 6.0d, 0.0d},
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedI64CumSumPreservesIntegralOutput() {
        Tensor input = new Tensor(
                new long[]{
                        10L, 20L, 30L,
                        1L, 2L, 3L
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.INT64
        );
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_I64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new long[]{10L, 30L, 60L, 1L, 3L, 6L},
                context.runtimeTensorForNodeId(fixture.node().id()).toInt64ArrayCopy()
        );
    }

    @Test
    void preparedF32SoftmaxNormalizesDenseAxis() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        1.0f, 1.0f, 1.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Tensor out = new Tensor(input.getShape(), List.of(input), new softmax(1), "softmax", DataType.FLOAT32);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SOFTMAX_F32_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{0.09003057f, 0.24472847f, 0.66524096f, 0.33333334f, 0.33333334f, 0.33333334f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64LogSoftmaxNormalizesDenseAxis() {
        Tensor input = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        1.0d, 1.0d, 1.0d
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Tensor out = new Tensor(input.getShape(), List.of(input), new logSoftmax(1), "logSoftmax", DataType.FLOAT64);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.LOG_SOFTMAX_F64_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new double[]{
                        -2.4076059644443806d, -1.4076059644443804d, -0.4076059644443804d,
                        -1.0986122886681098d, -1.0986122886681098d, -1.0986122886681098d
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedBf16SoftmaxStoresBf16Output() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f
                },
                new int[]{1, 3},
                null,
                "input",
                DataType.BFLOAT16
        );
        Tensor out = new Tensor(input.getShape(), List.of(input), new softmax(1), "softmax", DataType.BFLOAT16);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SOFTMAX_BF16_DENSE_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{0.09003057f, 0.24472847f, 0.66524096f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                3.0e-3f
        );
    }

    @Test
    void reductionTraceReportsPreparedKernelId() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.mean(0));
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
        assertEquals(Cpu1ReductionKernelId.MEAN_F32_DENSE_SCALAR.name(), trace.kernel());
        assertEquals(
                Cpu1ReductionKernelId.MEAN_F32_DENSE_SCALAR.name(),
                trace.metadata().attributes().get("cpu1ReductionKernelId")
        );
        assertEquals("MEAN", trace.metadata().reduction().mode());
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertReductionKernel(Cpu1PreparedArtifact artifact, Cpu1ReductionKernelId expected) {
        Cpu1ReductionExecutableUnit executable = assertInstanceOf(Cpu1ReductionExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedReductionUnit().kernelId());
        assertSame(artifact.preparedReductionUnit(), executable.preparedUnit());
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

    private static byte[] boolBytes(boolean... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] ? (byte) 1 : (byte) 0;
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
