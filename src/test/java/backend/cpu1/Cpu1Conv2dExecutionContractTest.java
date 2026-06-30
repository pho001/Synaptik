package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.nn.conv.conv2d.Cpu1Conv2dKernelId;
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
import tensor.options.Conv2dOptions;
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

class Cpu1Conv2dExecutionContractTest {
    @Test
    void executesF64DenseArrayWithoutBias() {
        Tensor input = f64Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, "convF64Input");
        Tensor weight = f64Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, "convF64Weight");
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults()));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(Cpu1Conv2dKernelId.CONV2D_F64_ARRAY_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{-4, -4, -4, -4}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertConv2dTrace(fixture, artifact, DataType.FLOAT64, 1, false, 1, 1, 1, 0);
    }

    @Test
    void executesF64DenseArrayWithBiasStrideAndPadding() {
        Tensor input = f64Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, "convBiasInput");
        Tensor weight = f64Tensor(new double[]{
                1, 1,
                1, 1
        }, new int[]{1, 1, 2, 2}, "convBiasWeight");
        Tensor bias = f64Tensor(new double[]{0.5}, new int[]{1}, "convBias");
        Fixture fixture = fixture(input.conv2d(
                weight,
                bias,
                Conv2dOptions.defaults().withStride(2, 2).withPadding(1, 1)
        ));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{1.5, 5.5, 11.5, 28.5}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertConv2dTrace(fixture, artifact, DataType.FLOAT64, 1, true, 1, 2, 1, 1);
    }

    @Test
    void executesF32DenseArrayInParallel() {
        Tensor input = new Tensor(new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "convF32Input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "convF32Weight", DataType.FLOAT32);
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults()));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertEquals(Cpu1Conv2dKernelId.CONV2D_F32_ARRAY_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new float[]{-4, -4, -4, -4}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertConv2dTrace(fixture, artifact, DataType.FLOAT32, 2, false, 1, 1, 1, 0);
    }

    @Test
    void executesBf16DenseArray() {
        Tensor input = new Tensor(bf16Bits(new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }), new int[]{1, 1, 3, 3}, null, "convBf16Input", DataType.BFLOAT16);
        Tensor weight = new Tensor(bf16Bits(new float[]{
                1, 0,
                0, -1
        }), new int[]{1, 1, 2, 2}, null, "convBf16Weight", DataType.BFLOAT16);
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults()));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(Cpu1Conv2dKernelId.CONV2D_BF16_ARRAY_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(bf16Bits(new float[]{-4, -4, -4, -4}), actual.toBFloat16BitsArrayCopy());
        assertConv2dTrace(fixture, artifact, DataType.BFLOAT16, 1, false, 1, 1, 1, 0);
    }

    @Test
    void executesGroupedDenseArray() {
        Tensor input = f64Tensor(new double[]{
                1, 2, 3, 4,
                10, 20, 30, 40
        }, new int[]{1, 2, 1, 4}, "groupedConvInput");
        Tensor weight = f64Tensor(new double[]{
                1, 1,
                2, 0
        }, new int[]{2, 1, 1, 2}, "groupedConvWeight");
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults().withGroups(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 2, 1, 3}, actual.getShape());
        assertArrayEquals(new double[]{3, 5, 7, 20, 40, 60}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertConv2dTrace(fixture, artifact, DataType.FLOAT64, 1, false, 2, 1, 1, 0);
    }

    @Test
    void executesF32MemorySegmentWithoutBias() {
        float[] inputValues = new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        };
        float[] weightValues = new float[]{
                1, 0,
                0, -1
        };
        Tensor input = new Tensor(inputValues, new int[]{1, 1, 3, 3}, null, "nativeConvF32Input", DataType.FLOAT32);
        Tensor weight = new Tensor(weightValues, new int[]{1, 1, 2, 2}, null, "nativeConvF32Weight", DataType.FLOAT32);
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults()));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(Cpu1Conv2dKernelId.CONV2D_F32_SEGMENT_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF32Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF32Input(ctx, fixture.node().inputIds().get(1), weightValues);
        });

        assertArrayEquals(
                new float[]{-4, -4, -4, -4},
                nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertConv2dTrace(fixture, artifact, DataType.FLOAT32, 1, false, 1, 1, 1, 0);
    }

    @Test
    void executesF64MemorySegmentWithBias() {
        double[] inputValues = new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        };
        double[] weightValues = new double[]{
                1, 1,
                1, 1
        };
        double[] biasValues = new double[]{0.5};
        Tensor input = f64Tensor(inputValues, new int[]{1, 1, 3, 3}, "nativeConvF64Input");
        Tensor weight = f64Tensor(weightValues, new int[]{1, 1, 2, 2}, "nativeConvF64Weight");
        Tensor bias = f64Tensor(biasValues, new int[]{1}, "nativeConvF64Bias");
        Fixture fixture = fixture(input.conv2d(weight, bias, Conv2dOptions.defaults().withStride(2, 2).withPadding(1, 1)));

        Cpu1PreparedArtifact artifact = prepare(fixture, new Cpu1PrepareConfig(
                backend.cpu1.kernels.Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.parallel(2),
                Cpu1StorageKind.MEMORY_SEGMENT
        ));
        assertEquals(Cpu1Conv2dKernelId.CONV2D_F64_SEGMENT_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeF64Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeF64Input(ctx, fixture.node().inputIds().get(1), weightValues);
            attachNativeF64Input(ctx, fixture.node().inputIds().get(2), biasValues);
        });

        assertArrayEquals(
                new double[]{1.5, 5.5, 11.5, 28.5},
                nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertConv2dTrace(fixture, artifact, DataType.FLOAT64, 2, true, 1, 2, 1, 1);
    }

    @Test
    void executesBf16MemorySegmentWithoutBias() {
        float[] inputValues = new float[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        };
        float[] weightValues = new float[]{
                1, 0,
                0, -1
        };
        Tensor input = new Tensor(
                bf16Bits(inputValues),
                new int[]{1, 1, 3, 3},
                null,
                "nativeConvBf16Input",
                DataType.BFLOAT16
        );
        Tensor weight = new Tensor(
                bf16Bits(weightValues),
                new int[]{1, 1, 2, 2},
                null,
                "nativeConvBf16Weight",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.conv2d(weight, Conv2dOptions.defaults()));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(Cpu1Conv2dKernelId.CONV2D_BF16_SEGMENT_DENSE_SCALAR, artifact.preparedConv2dUnit().kernelId());
        ExecutionContext context = executeContext(fixture, artifact, ctx -> {
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(0), inputValues);
            attachNativeBf16Input(ctx, fixture.node().inputIds().get(1), weightValues);
        });

        assertArrayEquals(
                bf16Bits(new float[]{-4, -4, -4, -4}),
                nativeBf16Bits(context.nativeStorageForNodeId(fixture.node().id()))
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertConv2dTrace(fixture, artifact, DataType.BFLOAT16, 1, false, 1, 1, 1, 0);
    }

    @Test
    void rejectsStridedInputDescriptorForDenseDirectRoute() {
        Tensor base = f64Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, "stridedConvBase");
        Tensor view = base.permute(0, 1, 3, 2);
        Tensor weight = f64Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, "stridedConvWeight");
        Fixture fixture = fixture(view.conv2d(weight, Conv2dOptions.defaults()));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
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

    private static void assertConv2dTrace(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            DataType expectedDType,
            int expectedWorkers,
            boolean expectedHasBias,
            int expectedGroups,
            int expectedStride,
            int expectedDilation,
            int expectedPad
    ) {
        StepTraceContribution trace = trace(fixture, artifact);
        Map<String, Object> attrs = trace.attributes();
        assertEquals(artifact.preparedConv2dUnit().kernelId().name(), attrs.get("cpu1Conv2dKernelId"));
        assertEquals("CONV2D", attrs.get("cpu1Conv2dOpType"));
        assertEquals(expectedDType.name(), attrs.get("cpu1Conv2dDType"));
        assertEquals(artifact.preparedConv2dUnit().storageKind().name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedHasBias, attrs.get("cpu1Conv2dHasBias"));
        assertEquals(expectedGroups, attrs.get("cpu1Conv2dGroups"));
        assertEquals(expectedStride, attrs.get("cpu1Conv2dStrideH"));
        assertEquals(expectedStride, attrs.get("cpu1Conv2dStrideW"));
        assertEquals(expectedDilation, attrs.get("cpu1Conv2dDilationH"));
        assertEquals(expectedDilation, attrs.get("cpu1Conv2dDilationW"));
        assertEquals(expectedPad, attrs.get("cpu1Conv2dPadH"));
        assertEquals(expectedPad, attrs.get("cpu1Conv2dPadW"));
        assertEquals(expectedWorkers, attrs.get("cpu1Conv2dLaunchWorkers"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Conv2dInputAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Conv2dWeightAccessKind"));
        assertEquals(expectedHasBias ? Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name() : "NONE",
                attrs.get("cpu1Conv2dBiasAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Conv2dOutputAccessKind"));
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

    private static Tensor f64Tensor(double[] values, int[] shape, String name) {
        return new Tensor(values, shape, null, name, DataType.FLOAT64);
    }

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 conv2d native f32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 conv2d test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1 conv2d native f64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 conv2d test native F64 input");
    }

    private static void attachNativeBf16Input(ExecutionContext context, int nodeId, float[] values) {
        NativeBFloat16Storage storage = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.allocateNativeStorage(DataType.BFLOAT16, values.length, "cpu1 conv2d native bf16 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 conv2d test native BF16 input");
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
