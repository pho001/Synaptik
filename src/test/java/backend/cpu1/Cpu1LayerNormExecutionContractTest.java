package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.nn.normalization.layernorm.Cpu1LayerNormKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.StepTraceContribution;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class Cpu1LayerNormExecutionContractTest {
    @Test
    void executesF32RankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 2.0f, 4.0f,
                -2.0f, 0.5f, 3.0f
        };
        float[] gammaValues = new float[]{1.25f, 0.75f, -0.5f};
        float[] betaValues = new float[]{0.5f, -0.25f, 1.0f};
        Tensor input = new Tensor(inputValues, new int[]{2, 3}, null, "layerNormF32Input", DataType.FLOAT32);
        Tensor gamma = new Tensor(gammaValues, new int[]{3}, null, "layerNormF32Gamma", DataType.FLOAT32);
        Tensor beta = new Tensor(betaValues, new int[]{3}, null, "layerNormF32Beta", DataType.FLOAT32);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-5));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1LayerNormKernelId.LAYER_NORM_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedLayerNormUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, betaValues, 3, 1.0e-5f),
                actual.toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertLayerNormTrace(fixture, artifact, DataType.FLOAT32, 1, 3, 2, 1);
    }

    @Test
    void executesF64HigherRankTrailingAxesInParallel() {
        double[] inputValues = new double[]{
                1.0d, 2.0d, 3.0d,
                4.0d, 5.0d, 6.0d,
                -1.0d, -0.5d, 0.0d,
                0.5d, 1.0d, 1.5d
        };
        double[] gammaValues = new double[]{
                1.0d, 0.5d, 1.5d,
                -0.25d, 2.0d, 0.75d
        };
        double[] betaValues = new double[]{
                0.0d, 0.25d, -0.5d,
                1.0d, -1.0d, 0.5d
        };
        Tensor input = new Tensor(inputValues, new int[]{2, 2, 3}, null, "layerNormF64Input", DataType.FLOAT64);
        Tensor gamma = new Tensor(gammaValues, new int[]{2, 3}, null, "layerNormF64Gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(betaValues, new int[]{2, 3}, null, "layerNormF64Beta", DataType.FLOAT64);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedF64(inputValues, gammaValues, betaValues, 6, 1.0e-12),
                actual.toDoubleArrayCopy(),
                1.0e-12
        );
        assertLayerNormTrace(fixture, artifact, DataType.FLOAT64, 2, 6, 2, 2);
    }

    @Test
    void executesBf16RankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 3.0f,
                -2.0f, 2.0f
        };
        float[] gammaValues = new float[]{2.0f, 3.0f};
        float[] betaValues = new float[]{10.0f, 20.0f};
        Tensor input = new Tensor(bf16Bits(inputValues), new int[]{2, 2}, null, "layerNormBf16Input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(bf16Bits(gammaValues), new int[]{2}, null, "layerNormBf16Gamma", DataType.BFLOAT16);
        Tensor beta = new Tensor(bf16Bits(betaValues), new int[]{2}, null, "layerNormBf16Beta", DataType.BFLOAT16);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedBf16(inputValues, gammaValues, betaValues, 2, 1.0e-12f),
                actual.toBFloat16BitsArrayCopy()
        );
        assertLayerNormTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 1);
    }

    @Test
    void executesF32MemorySegmentRankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 2.0f, 4.0f,
                -2.0f, 0.5f, 3.0f
        };
        float[] gammaValues = new float[]{1.25f, 0.75f, -0.5f};
        float[] betaValues = new float[]{0.5f, -0.25f, 1.0f};
        Tensor input = new Tensor(inputValues, new int[]{2, 3}, null, "nativeLayerNormF32Input", DataType.FLOAT32);
        Tensor gamma = new Tensor(gammaValues, new int[]{3}, null, "nativeLayerNormF32Gamma", DataType.FLOAT32);
        Tensor beta = new Tensor(betaValues, new int[]{3}, null, "nativeLayerNormF32Beta", DataType.FLOAT32);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-5));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1LayerNormKernelId.LAYER_NORM_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedLayerNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF32Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF32Input(ctx, fixture.node().inputIds().get(1), gammaValues);
            attachNativeF32Input(ctx, fixture.node().inputIds().get(2), betaValues);
        });

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, betaValues, 3, 1.0e-5f),
                nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertLayerNormTrace(fixture, artifact, DataType.FLOAT32, 1, 3, 2, 1);
    }

    @Test
    void executesF64MemorySegmentHigherRankTrailingAxesInParallel() {
        double[] inputValues = new double[]{
                1.0d, 2.0d, 3.0d,
                4.0d, 5.0d, 6.0d,
                -1.0d, -0.5d, 0.0d,
                0.5d, 1.0d, 1.5d
        };
        double[] gammaValues = new double[]{
                1.0d, 0.5d, 1.5d,
                -0.25d, 2.0d, 0.75d
        };
        double[] betaValues = new double[]{
                0.0d, 0.25d, -0.5d,
                1.0d, -1.0d, 0.5d
        };
        Tensor input = new Tensor(inputValues, new int[]{2, 2, 3}, null, "nativeLayerNormF64Input", DataType.FLOAT64);
        Tensor gamma = new Tensor(gammaValues, new int[]{2, 3}, null, "nativeLayerNormF64Gamma", DataType.FLOAT64);
        Tensor beta = new Tensor(betaValues, new int[]{2, 3}, null, "nativeLayerNormF64Beta", DataType.FLOAT64);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.parallel(2),
                Cpu1StorageKind.MEMORY_SEGMENT
        ));
        assertEquals(
                Cpu1LayerNormKernelId.LAYER_NORM_F64_SEGMENT_DENSE_SCALAR,
                artifact.preparedLayerNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF64Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF64Input(ctx, fixture.node().inputIds().get(1), gammaValues);
            attachNativeF64Input(ctx, fixture.node().inputIds().get(2), betaValues);
        });

        assertArrayEquals(
                expectedF64(inputValues, gammaValues, betaValues, 6, 1.0e-12),
                nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertLayerNormTrace(fixture, artifact, DataType.FLOAT64, 2, 6, 2, 2);
    }

    @Test
    void executesBf16MemorySegmentRankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 3.0f,
                -2.0f, 2.0f
        };
        float[] gammaValues = new float[]{2.0f, 3.0f};
        float[] betaValues = new float[]{10.0f, 20.0f};
        Tensor input = new Tensor(
                bf16Bits(inputValues),
                new int[]{2, 2},
                null,
                "nativeLayerNormBf16Input",
                DataType.BFLOAT16
        );
        Tensor gamma = new Tensor(
                bf16Bits(gammaValues),
                new int[]{2},
                null,
                "nativeLayerNormBf16Gamma",
                DataType.BFLOAT16
        );
        Tensor beta = new Tensor(
                bf16Bits(betaValues),
                new int[]{2},
                null,
                "nativeLayerNormBf16Beta",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1LayerNormKernelId.LAYER_NORM_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedLayerNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(1), gammaValues);
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(2), betaValues);
        });

        assertArrayEquals(
                expectedBf16(inputValues, gammaValues, betaValues, 2, 1.0e-12f),
                nativeBf16Bits(context.nativeStorageForNodeId(fixture.node().id()))
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertLayerNormTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 1);
    }

    @Test
    void rejectsStridedInputDescriptorForFirstSlice() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "stridedLayerNormBase",
                DataType.FLOAT32
        );
        Tensor view = base.permute(1, 0);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "stridedGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "stridedBeta", DataType.FLOAT32);
        Fixture fixture = fixture(view.layerNorm(gamma, beta, 1.0e-5));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void javaArrayRuntimeMaterializesNativeCurrentInput() {
        float[] inputValues = new float[]{1.0f, 2.0f};
        Tensor input = new Tensor(inputValues, new int[]{1, 2}, null, "runtimeNativeInput", DataType.FLOAT32);
        float[] gammaValues = new float[]{1.0f, 1.0f};
        float[] betaValues = new float[]{0.0f, 0.0f};
        Tensor gamma = new Tensor(gammaValues, new int[]{2}, null, "runtimeNativeGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(betaValues, new int[]{2}, null, "runtimeNativeBeta", DataType.FLOAT32);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-5));
        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeF32Input(context, fixture.node().inputIds().getFirst(), inputValues);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, betaValues, 2, 1.0e-5f),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertEquals(1, context.cpuMaterializationTraceCount());
    }

    @Test
    void memorySegmentRuntimeMaterializesCpuArrayInputs() {
        float[] inputValues = new float[]{1.0f, 2.0f};
        float[] gammaValues = new float[]{1.0f, 1.0f};
        float[] betaValues = new float[]{0.0f, 0.0f};
        Tensor input = new Tensor(inputValues, new int[]{1, 2}, null, "segmentRuntimeArrayInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(gammaValues, new int[]{2}, null, "segmentRuntimeArrayGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(betaValues, new int[]{2}, null, "segmentRuntimeArrayBeta", DataType.FLOAT32);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-5));
        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, betaValues, 2, 1.0e-5f),
                nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-6f
        );
        assertEquals(3, context.cpuMaterializationTraceCount());
    }

    @Test
    void rejectsMixedInputParameterOutputDType() {
        Tensor input = new Tensor(new double[]{1.0d, 2.0d}, new int[]{1, 2}, null, "mixedInput", DataType.FLOAT64);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "mixedGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new double[]{0.0d, 0.0d}, new int[]{2}, null, "mixedBeta", DataType.FLOAT64);
        Fixture fixture = fixture(input.layerNorm(gamma, beta, 1.0e-5));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("requires matching input/gamma/beta/output dtype"));
    }

    private static Cpu1PreparedArtifact prepare(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static Tensor execute(Fixture fixture, Cpu1PreparedArtifact artifact) {
        ExecutionContext context = executeContext(fixture, artifact, ignored -> {
        });
        return context.runtimeTensorForNodeId(fixture.node().id());
    }

    private static ExecutionContext executeContext(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            Consumer<ExecutionContext> beforeExecute
    ) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        beforeExecute.accept(context);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context;
    }

    private static void assertLayerNormTrace(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            DataType expectedDType,
            int expectedNormalizedRank,
            int expectedNormalizedSize,
            int expectedGroupCount,
            int expectedWorkers
    ) {
        StepTraceContribution trace = trace(fixture, artifact);
        Map<String, Object> attrs = trace.attributes();
        assertEquals(artifact.preparedLayerNormUnit().kernelId().name(), attrs.get("cpu1LayerNormKernelId"));
        assertEquals("LAYER_NORM", attrs.get("cpu1NormalizationOpType"));
        assertEquals(expectedDType.name(), attrs.get("cpu1NormalizationDType"));
        assertEquals(artifact.preparedLayerNormUnit().storageKind().name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedNormalizedRank, attrs.get("cpu1LayerNormNormalizedRank"));
        assertEquals(expectedNormalizedSize, attrs.get("cpu1LayerNormNormalizedSize"));
        assertEquals(expectedGroupCount, attrs.get("cpu1LayerNormGroupCount"));
        assertEquals(expectedWorkers, attrs.get("cpu1LayerNormLaunchWorkers"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1LayerNormInputAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1LayerNormOutputAccessKind"));
    }

    private static StepTraceContribution trace(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        return artifact.traceContribution(fixture.node(), metadata, context(fixture, metadata));
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(Fixture fixture, CompiledNodeExecutionMetadata metadata) {
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = Map.of(fixture.node().id(), metadata);
        ExecutionState state = ExecutionState.create(
                fixture.nodes(),
                fixture.descriptorIndex(),
                metadataIndex,
                fixture.node().id(),
                testsupport.PublicationPlans.forRoot(fixture.root(), fixture.nodes(), fixture.node().id())
        );
        return ExecutionContext.fromRuntimeConfig(
                RuntimeConfigHolder.RUNTIME_CONFIG,
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static CompiledNodeExecutionMetadata cpu1Metadata(
            CompiledNode node,
            Cpu1PreparedArtifact artifact
    ) {
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact
        );
    }

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 layernorm native f32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 layernorm test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1 layernorm native f64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 layernorm test native F64 input");
    }

    private static void attachNativeBf16Input(ExecutionContext context, int nodeId, float[] values) {
        NativeBFloat16Storage storage = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.allocateNativeStorage(DataType.BFLOAT16, values.length, "cpu1 layernorm native bf16 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 layernorm test native BF16 input");
    }

    private static float[] nativeF32Values(NativeTensorStorage storage) {
        NativeFloat32Storage f32 = assertInstanceOf(NativeFloat32Storage.class, storage);
        float[] out = new float[f32.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f32.getFloat32At(i);
        }
        return out;
    }

    private static double[] nativeF64Values(NativeTensorStorage storage) {
        NativeFloat64Storage f64 = assertInstanceOf(NativeFloat64Storage.class, storage);
        double[] out = new double[f64.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f64.getFloat64At(i);
        }
        return out;
    }

    private static short[] nativeBf16Bits(NativeTensorStorage storage) {
        NativeBFloat16Storage bf16 = assertInstanceOf(NativeBFloat16Storage.class, storage);
        short[] out = new short[bf16.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = bf16.getBFloat16BitsAt(i);
        }
        return out;
    }

    private static float[] expectedF32(
            float[] input,
            float[] gamma,
            float[] beta,
            int normalizedSize,
            float epsilon
    ) {
        float[] out = new float[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double total = 0.0d;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                float value = input[base + i];
                total += value;
                totalSquares += value * value;
            }
            float mean = (float) (total / normalizedSize);
            float variance = (float) Math.max(totalSquares / normalizedSize - mean * mean, 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                out[base + i] = ((input[base + i] - mean) * invStd) * gamma[i] + beta[i];
            }
        }
        return out;
    }

    private static double[] expectedF64(
            double[] input,
            double[] gamma,
            double[] beta,
            int normalizedSize,
            double epsilon
    ) {
        double[] out = new double[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double total = 0.0d;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                double value = input[base + i];
                total += value;
                totalSquares += value * value;
            }
            double mean = total / normalizedSize;
            double variance = Math.max(totalSquares / normalizedSize - mean * mean, 0.0d);
            double invStd = 1.0d / Math.sqrt(variance + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                out[base + i] = ((input[base + i] - mean) * invStd) * gamma[i] + beta[i];
            }
        }
        return out;
    }

    private static short[] expectedBf16(
            float[] input,
            float[] gamma,
            float[] beta,
            int normalizedSize,
            float epsilon
    ) {
        short[] out = new short[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double total = 0.0d;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(input[base + i]));
                total += value;
                totalSquares += value * value;
            }
            float mean = (float) (total / normalizedSize);
            float variance = (float) Math.max(totalSquares / normalizedSize - mean * mean, 0.0d);
            float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(input[base + i]));
                float scale = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(gamma[i]));
                float shift = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(beta[i]));
                out[base + i] = TensorDTypeOps.toBFloat16Bits(((value - mean) * invStd) * scale + shift);
            }
        }
        return out;
    }

    private static short[] bf16Bits(float[] values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static final class RuntimeConfigHolder {
        private static final config.runtime.RuntimeConfig RUNTIME_CONFIG = config.runtime.RuntimeConfig.inferenceDefaults();
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
