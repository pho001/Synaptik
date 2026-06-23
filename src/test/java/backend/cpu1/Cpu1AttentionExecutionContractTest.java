package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.linalg.attention.Cpu1AttentionKernelId;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.StepTraceContribution;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionWeights;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class Cpu1AttentionExecutionContractTest {
    @Test
    void executesF32JavaArrayVectorSingleThreadExpectedValuesAndKernelId() {
        int depth = FloatVector.SPECIES_PREFERRED.length() + 3;
        int valueDim = FloatVector.SPECIES_PREFERRED.length() + 5;
        int queryLen = 2;
        int keyLen = 3;
        float[] queryValues = f32Pattern(queryLen * depth, 0.125f);
        float[] keyValues = f32Pattern(keyLen * depth, -0.075f);
        float[] valueValues = f32Pattern(keyLen * valueDim, 0.25f);
        Tensor query = new Tensor(queryValues, new int[]{1, queryLen, depth}, null, "f32VectorAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, keyLen, depth}, null, "f32VectorAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, keyLen, valueDim}, null, "f32VectorAttentionV", DataType.FLOAT32);
        Fixture fixture = fixture(directAttention(query, key, value, null, 0.75d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.vectorSingleThread());
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);

        assertEquals(Cpu1AttentionKernelId.ATTENTION_F32_ARRAY_DENSE_VECTOR, artifact.preparedAttentionUnit().kernelId());
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedAttentionUnit().vectorizationKind());
        assertArrayEquals(
                expectedAttentionF32(
                        queryValues,
                        new int[]{1, queryLen, depth},
                        keyValues,
                        new int[]{1, keyLen, depth},
                        valueValues,
                        new int[]{1, keyLen, valueDim},
                        null,
                        0.75d
                ),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-5f
        );
    }

    @Test
    void executesF64JavaArrayVectorParallelExpectedValuesAndTraceDispatch() {
        int depth = DoubleVector.SPECIES_PREFERRED.length() + 3;
        int valueDim = DoubleVector.SPECIES_PREFERRED.length() + 3;
        int batch = 2;
        int queryLen = 3;
        int keyLen = 4;
        double[] queryValues = f64Pattern(batch * queryLen * depth, 0.03125d);
        double[] keyValues = f64Pattern(batch * keyLen * depth, -0.046875d);
        double[] valueValues = f64Pattern(batch * keyLen * valueDim, 0.0625d);
        Tensor query = new Tensor(queryValues, new int[]{batch, queryLen, depth}, null, "f64VectorAttentionQ", DataType.FLOAT64);
        Tensor key = new Tensor(keyValues, new int[]{batch, keyLen, depth}, null, "f64VectorAttentionK", DataType.FLOAT64);
        Tensor value = new Tensor(valueValues, new int[]{batch, keyLen, valueDim}, null, "f64VectorAttentionV", DataType.FLOAT64);
        Fixture fixture = fixture(directAttention(query, key, value, null, 0.5d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.vectorParallel(3));
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);
        StepTraceContribution trace = trace(fixture, artifact);

        assertEquals(Cpu1AttentionKernelId.ATTENTION_F64_ARRAY_DENSE_VECTOR, artifact.preparedAttentionUnit().kernelId());
        assertEquals(Cpu1VectorizationKind.VECTOR.name(), trace.dispatch().mode());
        assertEquals(Cpu1VectorizationKind.VECTOR.name(), trace.attributes().get("cpu1AttentionVectorizationKind"));
        assertEquals(3, trace.dispatch().plannedWorkers());
        assertArrayEquals(
                expectedAttentionF64(
                        queryValues,
                        new int[]{batch, queryLen, depth},
                        keyValues,
                        new int[]{batch, keyLen, depth},
                        valueValues,
                        new int[]{batch, keyLen, valueDim},
                        null,
                        0.5d
                ),
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-10
        );
    }

    @Test
    void executesF32MemorySegmentVectorExpectedValuesWithoutMaterialization() {
        int depth = FloatVector.SPECIES_PREFERRED.length() + 1;
        int valueDim = FloatVector.SPECIES_PREFERRED.length() + 2;
        int queryLen = 2;
        int keyLen = 3;
        float[] queryValues = f32Pattern(queryLen * depth, -0.0625f);
        float[] keyValues = f32Pattern(keyLen * depth, 0.09375f);
        float[] valueValues = f32Pattern(keyLen * valueDim, -0.1875f);
        Tensor query = new Tensor(queryValues, new int[]{1, queryLen, depth}, null, "nativeVectorAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, keyLen, depth}, null, "nativeVectorAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, keyLen, valueDim}, null, "nativeVectorAttentionV", DataType.FLOAT32);
        Fixture fixture = fixture(directAttention(query, key, value, null, 0.625d));
        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.vectorMemorySegmentSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().get(0), queryValues);
        attachNativeF32Input(context, fixture.node().inputIds().get(1), keyValues);
        attachNativeF32Input(context, fixture.node().inputIds().get(2), valueValues);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertEquals(Cpu1AttentionKernelId.ATTENTION_F32_SEGMENT_DENSE_VECTOR, artifact.preparedAttentionUnit().kernelId());
        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedAttentionUnit().vectorizationKind());
        assertArrayEquals(
                expectedAttentionF32(
                        queryValues,
                        new int[]{1, queryLen, depth},
                        keyValues,
                        new int[]{1, keyLen, depth},
                        valueValues,
                        new int[]{1, keyLen, valueDim},
                        null,
                        0.625d
                ),
                nativeF32Values(output),
                1.0e-5f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void bf16VectorConfigSelectsScalarBf16AttentionKernel() {
        float[] queryValues = new float[]{
                1.0f, 0.5f,
                -0.25f, 0.75f
        };
        float[] keyValues = new float[]{
                0.5f, 1.0f,
                1.5f, -0.5f
        };
        float[] valueValues = new float[]{
                3.0f, -1.0f,
                7.0f, 2.0f
        };
        Tensor query = new Tensor(bf16Bits(queryValues), new int[]{1, 2, 2}, null, "bf16VectorConfigQ", DataType.BFLOAT16);
        Tensor key = new Tensor(bf16Bits(keyValues), new int[]{1, 2, 2}, null, "bf16VectorConfigK", DataType.BFLOAT16);
        Tensor value = new Tensor(bf16Bits(valueValues), new int[]{1, 2, 2}, null, "bf16VectorConfigV", DataType.BFLOAT16);
        Fixture fixture = fixture(directAttention(query, key, value, null, 0.5d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.vectorSingleThread());
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);

        assertEquals(Cpu1AttentionKernelId.ATTENTION_BF16_ARRAY_DENSE_SCALAR, artifact.preparedAttentionUnit().kernelId());
        assertEquals(Cpu1VectorizationKind.SCALAR, artifact.preparedAttentionUnit().vectorizationKind());
        assertArrayEquals(
                expectedBf16AttentionAsF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        null,
                        0.5d
                ),
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-2f
        );
    }

    @Test
    void executesF64JavaArrayUnmaskedDirectExpectedValues() {
        double[] queryValues = new double[]{
                1.0d, 0.0d,
                0.0d, 1.0d
        };
        double[] keyValues = new double[]{
                1.0d, 0.0d,
                0.0d, 1.0d
        };
        double[] valueValues = new double[]{
                10.0d, 1.0d,
                1.0d, 10.0d
        };
        Tensor query = new Tensor(queryValues, new int[]{1, 2, 2}, null, "f64AttentionQ", DataType.FLOAT64);
        Tensor key = new Tensor(keyValues, new int[]{1, 2, 2}, null, "f64AttentionK", DataType.FLOAT64);
        Tensor value = new Tensor(valueValues, new int[]{1, 2, 2}, null, "f64AttentionV", DataType.FLOAT64);
        Fixture fixture = fixture(directAttention(query, key, value, null, 1.0d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarSingleThread());
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);

        assertArrayEquals(
                expectedAttentionF64(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        null,
                        1.0d
                ),
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void executesF32JavaArrayMaskedWithAllMaskedRowAsUniformAverage() {
        float[] queryValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] keyValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] valueValues = new float[]{
                2.0f, 4.0f,
                10.0f, 20.0f
        };
        byte[] maskValues = new byte[]{
                1, 0,
                0, 0
        };
        Tensor query = new Tensor(queryValues, new int[]{1, 2, 2}, null, "maskedAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, 2, 2}, null, "maskedAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, 2, 2}, null, "maskedAttentionV", DataType.FLOAT32);
        Tensor mask = new Tensor(maskValues, new int[]{1, 2, 2}, null, "maskedAttentionMask", DataType.BOOL);
        Fixture fixture = fixture(directAttention(query, key, value, mask, 1.0d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarSingleThread());
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);

        float[] actual = context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy();
        assertArrayEquals(
                expectedAttentionF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        maskValues,
                        1.0d
                ),
                actual,
                1.0e-6f
        );
        assertArrayEquals(new float[]{6.0f, 12.0f}, new float[]{actual[2], actual[3]}, 1.0e-6f);
    }

    @Test
    void executesBf16JavaArrayDirectWithinTolerance() {
        float[] queryValues = new float[]{
                1.0f, 0.5f,
                -0.25f, 0.75f
        };
        float[] keyValues = new float[]{
                0.5f, 1.0f,
                1.5f, -0.5f
        };
        float[] valueValues = new float[]{
                3.0f, -1.0f,
                7.0f, 2.0f
        };
        Tensor query = new Tensor(bf16Bits(queryValues), new int[]{1, 2, 2}, null, "bf16AttentionQ", DataType.BFLOAT16);
        Tensor key = new Tensor(bf16Bits(keyValues), new int[]{1, 2, 2}, null, "bf16AttentionK", DataType.BFLOAT16);
        Tensor value = new Tensor(bf16Bits(valueValues), new int[]{1, 2, 2}, null, "bf16AttentionV", DataType.BFLOAT16);
        Fixture fixture = fixture(directAttention(query, key, value, null, 0.5d));

        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarSingleThread());
        ExecutionContext context = executeSingle(fixture, fixture.node(), artifact);

        assertArrayEquals(
                expectedBf16AttentionAsF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        null,
                        0.5d
                ),
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-2f
        );
    }

    @Test
    void executesF32MemorySegmentDirectWithoutMaterialization() {
        float[] queryValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] keyValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] valueValues = new float[]{
                5.0f, 1.0f,
                1.0f, 5.0f
        };
        Tensor query = new Tensor(queryValues, new int[]{1, 2, 2}, null, "nativeAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, 2, 2}, null, "nativeAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, 2, 2}, null, "nativeAttentionV", DataType.FLOAT32);
        Fixture fixture = fixture(directAttention(query, key, value, null, 1.0d));
        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().get(0), queryValues);
        attachNativeF32Input(context, fixture.node().inputIds().get(1), keyValues);
        attachNativeF32Input(context, fixture.node().inputIds().get(2), valueValues);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(
                expectedAttentionF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        null,
                        1.0d
                ),
                nativeF32Values(output),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void memorySegmentAttentionMaterializesArrayInputsThroughRuntimeBoundary() {
        float[] queryValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] keyValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] valueValues = new float[]{
                5.0f, 1.0f,
                1.0f, 5.0f
        };
        Tensor query = new Tensor(queryValues, new int[]{1, 2, 2}, null, "nativeAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, 2, 2}, null, "nativeAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, 2, 2}, null, "nativeAttentionV", DataType.FLOAT32);
        Fixture fixture = fixture(directAttention(query, key, value, null, 1.0d));
        Cpu1PreparedArtifact artifact = prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(
                expectedAttentionF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        valueValues,
                        new int[]{1, 2, 2},
                        null,
                        1.0d
                ),
                nativeF32Values(output),
                1.0e-6f
        );
        assertEquals(3, context.cpuMaterializationTraceCount());
    }

    @Test
    void publishesWeightsAfterAttentionOutputRequiresGrad() {
        float[] queryValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] keyValues = new float[]{
                1.0f, 0.0f,
                0.0f, 1.0f
        };
        float[] valueValues = new float[]{
                10.0f, 1.0f,
                1.0f, 10.0f
        };
        Tensor query = new Tensor(queryValues, new int[]{1, 2, 2}, null, "weightsAttentionQ", DataType.FLOAT32);
        Tensor key = new Tensor(keyValues, new int[]{1, 2, 2}, null, "weightsAttentionK", DataType.FLOAT32);
        Tensor value = new Tensor(valueValues, new int[]{1, 2, 2}, null, "weightsAttentionV", DataType.FLOAT32);
        Tensor attention = directAttention(query, key, value, null, 1.0d);
        attention.setRequiresGrad(true);
        Tensor weights = attentionWeights(attention, new int[]{1, 2, 2});
        Fixture fixture = fixture(weights);
        CompiledNode attentionNode = node(fixture.nodes(), Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        CompiledNode weightsNode = node(fixture.nodes(), Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS);
        Cpu1PreparedArtifact attentionArtifact = prepare(fixture, attentionNode, Cpu1PrepareConfig.scalarSingleThread());
        Cpu1PreparedArtifact weightsArtifact = prepare(fixture, weightsNode, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata attentionMetadata = metadata(attentionNode, attentionArtifact);
        CompiledNodeExecutionMetadata weightsMetadata = metadata(weightsNode, weightsArtifact);
        ExecutionContext context = context(fixture, Map.of(
                attentionNode.id(), attentionMetadata,
                weightsNode.id(), weightsMetadata
        ));

        Cpu1Backend backend = new Cpu1Backend();
        backend.execute(attentionNode, attentionMetadata, context);
        backend.execute(weightsNode, weightsMetadata, context);

        assertArrayEquals(
                expectedAttentionWeightsF32(
                        queryValues,
                        new int[]{1, 2, 2},
                        keyValues,
                        new int[]{1, 2, 2},
                        null,
                        1.0d
                ),
                context.runtimeTensorForNodeId(weightsNode.id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void rejectsWeightsPreparedOverNonAttentionInput() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{1, 2, 2},
                null,
                "plainFloatingInput",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(attentionWeights(input, new int[]{1, 2, 2}));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, fixture.node(), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("input must be a SCALED_DOT_PRODUCT_ATTENTION"));
    }

    @Test
    void rejectsStridedAndDenseOffsetAttentionContracts() {
        Tensor stridedBase = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f
                },
                new int[]{2, 2},
                null,
                "stridedAttentionBase",
                DataType.FLOAT32
        );
        Tensor stridedQuery = stridedBase.permute(1, 0);
        Tensor key = new Tensor(new float[]{1.0f, 0.0f, 0.0f, 1.0f}, new int[]{2, 2}, null, "stridedK", DataType.FLOAT32);
        Tensor value = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "stridedV", DataType.FLOAT32);
        Fixture stridedFixture = fixture(directAttention(stridedQuery, key, value, null, 1.0d));

        UnsupportedOperationException stridedException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(stridedFixture, stridedFixture.node(), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(stridedException.getMessage().contains("query access"));
        assertTrue(stridedException.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));

        Tensor offsetBase = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f
                },
                new int[]{2, 2, 2},
                null,
                "offsetAttentionBase",
                DataType.FLOAT32
        );
        Tensor offsetQuery = offsetBase.select(0, 1);
        Fixture offsetFixture = fixture(directAttention(offsetQuery, key, value, null, 1.0d));

        UnsupportedOperationException offsetException = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(offsetFixture, offsetFixture.node(), Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(offsetException.getMessage().contains("query access"));
        assertTrue(offsetException.getMessage().contains(Cpu1StorageAccessKind.DENSE_WITH_OFFSET.name()));
    }

    private static Cpu1PreparedArtifact prepare(
            Fixture fixture,
            CompiledNode node,
            Cpu1PrepareConfig config
    ) {
        return new Cpu1NodePreparer().prepare(node, fixture.descriptorIndex(), config);
    }

    private static ExecutionContext executeSingle(
            Fixture fixture,
            CompiledNode node,
            Cpu1PreparedArtifact artifact
    ) {
        CompiledNodeExecutionMetadata metadata = metadata(node, artifact);
        ExecutionContext context = context(fixture, Map.of(node.id(), metadata));
        new Cpu1Backend().execute(node, metadata, context);
        return context;
    }

    private static StepTraceContribution trace(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        return artifact.traceContribution(fixture.node(), metadata, context(fixture, Map.of(fixture.node().id(), metadata)));
    }

    private static Tensor directAttention(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            double scale
    ) {
        int[] outShape = query.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = value.getShapeUnsafe()[value.getShapeUnsafe().length - 1];
        List<Tensor> inputs = new ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (mask != null) {
            inputs.add(mask);
        }
        return TensorPrimitiveBuilder.nary(
                outShape,
                inputs,
                new scaledDotProductAttention(scale, mask != null),
                "directScaledDotProductAttention",
                query.getDataType()
        );
    }

    private static Tensor attentionWeights(Tensor attention, int[] scoresShape) {
        return TensorPrimitiveBuilder.unary(
                attention,
                scoresShape,
                new scaledDotProductAttentionWeights(),
                "directScaledDotProductAttentionWeights",
                attention.getDataType()
        );
    }

    private static Fixture fixture(Tensor root) {
        List<CompiledNode> nodes = CompiledNode.snapshot(root.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(root, nodes, descriptorIndex, nodes.getLast());
    }

    private static CompiledNode node(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .findFirst()
                .orElseThrow();
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

    private static double[] expectedAttentionF64(
            double[] query,
            int[] queryShape,
            double[] key,
            int[] keyShape,
            double[] value,
            int[] valueShape,
            byte[] mask,
            double scale
    ) {
        int batchCount = batchCount(queryShape);
        int queryLen = queryShape[queryShape.length - 2];
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        int valueDim = valueShape[valueShape.length - 1];
        double[] out = new double[batchCount * queryLen * valueDim];
        double[] scores = new double[keyLen];
        for (int batch = 0; batch < batchCount; batch++) {
            for (int queryIndex = 0; queryIndex < queryLen; queryIndex++) {
                double[] weights = attentionWeightsForRowF64(
                        query,
                        key,
                        mask,
                        batch,
                        queryIndex,
                        queryLen,
                        keyLen,
                        depth,
                        scale,
                        scores
                );
                int valueBatchBase = batch * keyLen * valueDim;
                int outBase = (batch * queryLen + queryIndex) * valueDim;
                for (int col = 0; col < valueDim; col++) {
                    double sum = 0.0d;
                    for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                        sum += weights[keyIndex] * value[valueBatchBase + keyIndex * valueDim + col];
                    }
                    out[outBase + col] = sum;
                }
            }
        }
        return out;
    }

    private static float[] expectedAttentionF32(
            float[] query,
            int[] queryShape,
            float[] key,
            int[] keyShape,
            float[] value,
            int[] valueShape,
            byte[] mask,
            double scale
    ) {
        int batchCount = batchCount(queryShape);
        int queryLen = queryShape[queryShape.length - 2];
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        int valueDim = valueShape[valueShape.length - 1];
        float[] out = new float[batchCount * queryLen * valueDim];
        float[] scores = new float[keyLen];
        for (int batch = 0; batch < batchCount; batch++) {
            for (int queryIndex = 0; queryIndex < queryLen; queryIndex++) {
                float[] weights = attentionWeightsForRowF32(
                        query,
                        key,
                        mask,
                        batch,
                        queryIndex,
                        queryLen,
                        keyLen,
                        depth,
                        (float) scale,
                        scores
                );
                int valueBatchBase = batch * keyLen * valueDim;
                int outBase = (batch * queryLen + queryIndex) * valueDim;
                for (int col = 0; col < valueDim; col++) {
                    float sum = 0.0f;
                    for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
                        sum += weights[keyIndex] * value[valueBatchBase + keyIndex * valueDim + col];
                    }
                    out[outBase + col] = sum;
                }
            }
        }
        return out;
    }

    private static float[] expectedAttentionWeightsF32(
            float[] query,
            int[] queryShape,
            float[] key,
            int[] keyShape,
            byte[] mask,
            double scale
    ) {
        int batchCount = batchCount(queryShape);
        int queryLen = queryShape[queryShape.length - 2];
        int keyLen = keyShape[keyShape.length - 2];
        int depth = queryShape[queryShape.length - 1];
        float[] out = new float[batchCount * queryLen * keyLen];
        float[] scores = new float[keyLen];
        for (int batch = 0; batch < batchCount; batch++) {
            for (int queryIndex = 0; queryIndex < queryLen; queryIndex++) {
                float[] weights = attentionWeightsForRowF32(
                        query,
                        key,
                        mask,
                        batch,
                        queryIndex,
                        queryLen,
                        keyLen,
                        depth,
                        (float) scale,
                        scores
                );
                System.arraycopy(weights, 0, out, (batch * queryLen + queryIndex) * keyLen, keyLen);
            }
        }
        return out;
    }

    private static float[] f32Pattern(int length, float offset) {
        float[] values = new float[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = offset + ((i % 13) - 6) * 0.03125f + (i % 5) * 0.0078125f;
        }
        return values;
    }

    private static double[] f64Pattern(int length, double offset) {
        double[] values = new double[length];
        for (int i = 0; i < values.length; i++) {
            values[i] = offset + ((i % 17) - 8) * 0.015625d + (i % 7) * 0.00390625d;
        }
        return values;
    }

    private static double[] attentionWeightsForRowF64(
            double[] query,
            double[] key,
            byte[] mask,
            int batch,
            int queryIndex,
            int queryLen,
            int keyLen,
            int depth,
            double scale,
            double[] scores
    ) {
        int queryBase = batch * queryLen * depth + queryIndex * depth;
        int keyBase = batch * keyLen * depth;
        int maskBase = mask == null ? -1 : (batch * queryLen + queryIndex) * keyLen;
        double max = Double.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                scores[keyIndex] = Double.NaN;
                continue;
            }
            double score = 0.0d;
            for (int d = 0; d < depth; d++) {
                score += query[queryBase + d] * key[keyBase + keyIndex * depth + d];
            }
            scores[keyIndex] = score * scale;
            max = Math.max(max, scores[keyIndex]);
            anyValid = true;
        }
        double[] weights = new double[keyLen];
        if (!anyValid) {
            java.util.Arrays.fill(weights, 1.0d / keyLen);
            return weights;
        }
        double sum = 0.0d;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (Double.isNaN(scores[keyIndex])) {
                weights[keyIndex] = 0.0d;
                continue;
            }
            weights[keyIndex] = Math.exp(scores[keyIndex] - max);
            sum += weights[keyIndex];
        }
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            weights[keyIndex] /= sum;
        }
        return weights;
    }

    private static float[] attentionWeightsForRowF32(
            float[] query,
            float[] key,
            byte[] mask,
            int batch,
            int queryIndex,
            int queryLen,
            int keyLen,
            int depth,
            float scale,
            float[] scores
    ) {
        int queryBase = batch * queryLen * depth + queryIndex * depth;
        int keyBase = batch * keyLen * depth;
        int maskBase = mask == null ? -1 : (batch * queryLen + queryIndex) * keyLen;
        float max = Float.NEGATIVE_INFINITY;
        boolean anyValid = false;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (mask != null && mask[maskBase + keyIndex] == 0) {
                scores[keyIndex] = Float.NaN;
                continue;
            }
            float score = 0.0f;
            for (int d = 0; d < depth; d++) {
                score += query[queryBase + d] * key[keyBase + keyIndex * depth + d];
            }
            scores[keyIndex] = score * scale;
            max = Math.max(max, scores[keyIndex]);
            anyValid = true;
        }
        float[] weights = new float[keyLen];
        if (!anyValid) {
            java.util.Arrays.fill(weights, 1.0f / keyLen);
            return weights;
        }
        float sum = 0.0f;
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            if (Float.isNaN(scores[keyIndex])) {
                weights[keyIndex] = 0.0f;
                continue;
            }
            weights[keyIndex] = (float) Math.exp(scores[keyIndex] - max);
            sum += weights[keyIndex];
        }
        for (int keyIndex = 0; keyIndex < keyLen; keyIndex++) {
            weights[keyIndex] /= sum;
        }
        return weights;
    }

    private static float[] expectedBf16AttentionAsF32(
            float[] query,
            int[] queryShape,
            float[] key,
            int[] keyShape,
            float[] value,
            int[] valueShape,
            byte[] mask,
            double scale
    ) {
        float[] rounded = expectedAttentionF32(
                roundBf16(query),
                queryShape,
                roundBf16(key),
                keyShape,
                roundBf16(value),
                valueShape,
                mask,
                scale
        );
        for (int i = 0; i < rounded.length; i++) {
            rounded[i] = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(rounded[i]));
        }
        return rounded;
    }

    private static float[] roundBf16(float[] values) {
        float[] rounded = new float[values.length];
        for (int i = 0; i < values.length; i++) {
            rounded[i] = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        return rounded;
    }

    private static int batchCount(int[] shape) {
        int count = 1;
        for (int i = 0; i < shape.length - 2; i++) {
            count = Math.multiplyExact(count, shape[i]);
        }
        return count;
    }

    private static short[] bf16Bits(float[] values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] bits = tensor.toBFloat16BitsArrayCopy();
        float[] out = new float[bits.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(bits[i]);
        }
        return out;
    }

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1-attention-test-f32-input-" + nodeId)
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 attention test native F32 input");
    }

    private static float[] nativeF32Values(NativeTensorStorage storage) {
        NativeFloat32Storage f32 = assertInstanceOf(NativeFloat32Storage.class, storage);
        float[] out = new float[f32.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f32.getFloat32At(i);
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
