package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.kernels.nn.normalization.rmsnorm.Cpu1RmsNormKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import graph.CompiledNode;
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

class Cpu1RmsNormExecutionContractTest {
    @Test
    void executesF32RankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 2.0f, 4.0f,
                -2.0f, 0.5f, 3.0f
        };
        float[] gammaValues = new float[]{1.25f, 0.75f, -0.5f};
        Tensor input = new Tensor(inputValues, new int[]{2, 3}, null, "rmsNormF32Input", DataType.FLOAT32);
        Tensor gamma = new Tensor(gammaValues, new int[]{3}, null, "rmsNormF32Gamma", DataType.FLOAT32);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-5));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1RmsNormKernelId.RMS_NORM_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedRmsNormUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, 3, 1.0e-5f),
                actual.toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertRmsNormTrace(fixture, artifact, DataType.FLOAT32, 1, 3, 2, 1);
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
        Tensor input = new Tensor(inputValues, new int[]{2, 2, 3}, null, "rmsNormF64Input", DataType.FLOAT64);
        Tensor gamma = new Tensor(gammaValues, new int[]{2, 3}, null, "rmsNormF64Gamma", DataType.FLOAT64);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedF64(inputValues, gammaValues, 6, 1.0e-12),
                actual.toDoubleArrayCopy(),
                1.0e-12
        );
        assertRmsNormTrace(fixture, artifact, DataType.FLOAT64, 2, 6, 2, 2);
    }

    @Test
    void executesBf16RankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 3.0f,
                -2.0f, 2.0f
        };
        float[] gammaValues = new float[]{2.0f, 3.0f};
        Tensor input = new Tensor(bf16Bits(inputValues), new int[]{2, 2}, null, "rmsNormBf16Input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(bf16Bits(gammaValues), new int[]{2}, null, "rmsNormBf16Gamma", DataType.BFLOAT16);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(
                expectedBf16(inputValues, gammaValues, 2, 1.0e-12f),
                actual.toBFloat16BitsArrayCopy()
        );
        assertRmsNormTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 1);
    }

    @Test
    void executesF32MemorySegmentRankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 2.0f, 4.0f,
                -2.0f, 0.5f, 3.0f
        };
        float[] gammaValues = new float[]{1.25f, 0.75f, -0.5f};
        Tensor input = new Tensor(inputValues, new int[]{2, 3}, null, "nativeRmsNormF32Input", DataType.FLOAT32);
        Tensor gamma = new Tensor(gammaValues, new int[]{3}, null, "nativeRmsNormF32Gamma", DataType.FLOAT32);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-5));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1RmsNormKernelId.RMS_NORM_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedRmsNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF32Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF32Input(ctx, fixture.node().inputIds().get(1), gammaValues);
        });

        assertArrayEquals(
                expectedF32(inputValues, gammaValues, 3, 1.0e-5f),
                nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertRmsNormTrace(fixture, artifact, DataType.FLOAT32, 1, 3, 2, 1);
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
        Tensor input = new Tensor(inputValues, new int[]{2, 2, 3}, null, "nativeRmsNormF64Input", DataType.FLOAT64);
        Tensor gamma = new Tensor(gammaValues, new int[]{2, 3}, null, "nativeRmsNormF64Gamma", DataType.FLOAT64);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.parallel(2),
                Cpu1StorageKind.MEMORY_SEGMENT
        ));
        assertEquals(
                Cpu1RmsNormKernelId.RMS_NORM_F64_SEGMENT_DENSE_SCALAR,
                artifact.preparedRmsNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF64Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF64Input(ctx, fixture.node().inputIds().get(1), gammaValues);
        });

        assertArrayEquals(
                expectedF64(inputValues, gammaValues, 6, 1.0e-12),
                nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertRmsNormTrace(fixture, artifact, DataType.FLOAT64, 2, 6, 2, 2);
    }

    @Test
    void executesBf16MemorySegmentRankTwoTrailingAxis() {
        float[] inputValues = new float[]{
                1.0f, 3.0f,
                -2.0f, 2.0f
        };
        float[] gammaValues = new float[]{2.0f, 3.0f};
        Tensor input = new Tensor(
                bf16Bits(inputValues),
                new int[]{2, 2},
                null,
                "nativeRmsNormBf16Input",
                DataType.BFLOAT16
        );
        Tensor gamma = new Tensor(
                bf16Bits(gammaValues),
                new int[]{2},
                null,
                "nativeRmsNormBf16Gamma",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-12));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1RmsNormKernelId.RMS_NORM_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedRmsNormUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(1), gammaValues);
        });

        assertArrayEquals(
                expectedBf16(inputValues, gammaValues, 2, 1.0e-12f),
                nativeBf16Bits(context.nativeStorageForNodeId(fixture.node().id()))
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertRmsNormTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 1);
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
                "stridedRmsNormBase",
                DataType.FLOAT32
        );
        Tensor view = base.permute(1, 0);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "stridedRmsGamma", DataType.FLOAT32);
        Fixture fixture = fixture(view.rmsNorm(gamma, 1.0e-5));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void rejectsRuntimeNativeCurrentInputWithoutMaterialization() {
        float[] inputValues = new float[]{1.0f, 2.0f};
        Tensor input = new Tensor(inputValues, new int[]{1, 2}, null, "runtimeNativeRmsInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "runtimeNativeRmsGamma", DataType.FLOAT32);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-5));
        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeF32Input(context, fixture.node().inputIds().getFirst(), inputValues);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> new Cpu1Backend().execute(fixture.node(), metadata, context)
        );
        assertTrue(exception.getMessage().contains("requires current CPU array input storage"));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void rejectsMemorySegmentRuntimeCpuArrayInputWithoutMaterialization() {
        float[] inputValues = new float[]{1.0f, 2.0f};
        Tensor input = new Tensor(inputValues, new int[]{1, 2}, null, "segmentRuntimeRmsArrayInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "segmentRuntimeRmsArrayGamma", DataType.FLOAT32);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-5));
        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> new Cpu1Backend().execute(fixture.node(), metadata, context)
        );
        assertTrue(exception.getMessage().contains("MEMORY_SEGMENT requires current native CPU segment input"));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void rejectsMixedInputParameterOutputDType() {
        Tensor input = new Tensor(new double[]{1.0d, 2.0d}, new int[]{1, 2}, null, "mixedRmsInput", DataType.FLOAT64);
        Tensor gamma = new Tensor(new float[]{1.0f, 1.0f}, new int[]{2}, null, "mixedRmsGamma", DataType.FLOAT32);
        Fixture fixture = fixture(input.rmsNorm(gamma, 1.0e-5));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("requires matching input/gamma/output dtype"));
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

    private static void assertRmsNormTrace(
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
        assertEquals(artifact.preparedRmsNormUnit().kernelId().name(), attrs.get("cpu1RmsNormKernelId"));
        assertEquals("RMS_NORM", attrs.get("cpu1NormalizationOpType"));
        assertEquals(expectedDType.name(), attrs.get("cpu1NormalizationDType"));
        assertEquals(artifact.preparedRmsNormUnit().storageKind().name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedNormalizedRank, attrs.get("cpu1RmsNormNormalizedRank"));
        assertEquals(expectedNormalizedSize, attrs.get("cpu1RmsNormNormalizedSize"));
        assertEquals(expectedGroupCount, attrs.get("cpu1RmsNormGroupCount"));
        assertEquals(expectedWorkers, attrs.get("cpu1RmsNormLaunchWorkers"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1RmsNormInputAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1RmsNormOutputAccessKind"));
    }

    private static StepTraceContribution trace(Fixture fixture, Cpu1PreparedArtifact artifact) {
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        return artifact.traceContribution(fixture.node(), metadata, context(fixture, metadata));
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 rmsnorm native f32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 rmsnorm test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1 rmsnorm native f64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 rmsnorm test native F64 input");
    }

    private static void attachNativeBf16Input(ExecutionContext context, int nodeId, float[] values) {
        NativeBFloat16Storage storage = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.allocateNativeStorage(DataType.BFLOAT16, values.length, "cpu1 rmsnorm native bf16 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 rmsnorm test native BF16 input");
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

    private static float[] expectedF32(float[] input, float[] gamma, int normalizedSize, float epsilon) {
        float[] out = new float[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                float value = input[base + i];
                totalSquares += value * value;
            }
            float invRms = (float) (1.0d / Math.sqrt(totalSquares / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                out[base + i] = input[base + i] * invRms * gamma[i];
            }
        }
        return out;
    }

    private static double[] expectedF64(double[] input, double[] gamma, int normalizedSize, double epsilon) {
        double[] out = new double[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                double value = input[base + i];
                totalSquares += value * value;
            }
            double invRms = 1.0d / Math.sqrt(totalSquares / normalizedSize + epsilon);
            for (int i = 0; i < normalizedSize; i++) {
                out[base + i] = input[base + i] * invRms * gamma[i];
            }
        }
        return out;
    }

    private static short[] expectedBf16(float[] input, float[] gamma, int normalizedSize, float epsilon) {
        short[] out = new short[input.length];
        int groupCount = input.length / normalizedSize;
        for (int group = 0; group < groupCount; group++) {
            int base = group * normalizedSize;
            double totalSquares = 0.0d;
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(input[base + i]));
                totalSquares += value * value;
            }
            float invRms = (float) (1.0d / Math.sqrt(totalSquares / normalizedSize + epsilon));
            for (int i = 0; i < normalizedSize; i++) {
                float value = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(input[base + i]));
                float scale = TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(gamma[i]));
                out[base + i] = TensorDTypeOps.toBFloat16Bits(value * invRms * scale);
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
