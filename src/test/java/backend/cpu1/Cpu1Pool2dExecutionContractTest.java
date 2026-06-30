package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.kernels.nn.pool.avgpool.Cpu1AvgPool2dKernelId;
import backend.cpu1.kernels.nn.pool.maxpool.Cpu1MaxPool2dKernelId;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
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
import tensor.options.Pool2dOptions;
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

class Cpu1Pool2dExecutionContractTest {
    @Test
    void executesF64DenseArrayMaxPool2d() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "poolF64Input", DataType.FLOAT64);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_F64_ARRAY_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{6, 8, 14, 16}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertMaxPoolTrace(fixture, artifact, DataType.FLOAT64, 1, 2, 2, 2);
    }

    @Test
    void executesF32DenseArrayMaxPool2dInParallel() {
        Tensor input = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "poolF32Input", DataType.FLOAT32);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new float[]{6, 8, 14, 16}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertMaxPoolTrace(fixture, artifact, DataType.FLOAT32, 2, 2, 2, 2);
    }

    @Test
    void executesBf16DenseArrayMaxPool2d() {
        Tensor input = new Tensor(bf16Bits(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }), new int[]{1, 1, 4, 4}, null, "poolBf16Input", DataType.BFLOAT16);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_BF16_ARRAY_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(bf16Bits(new float[]{6, 8, 14, 16}), actual.toBFloat16BitsArrayCopy());
        assertMaxPoolTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 2);
    }

    @Test
    void executesF64DenseArrayAvgPool2d() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "avgPoolF64Input", DataType.FLOAT64);
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_F64_ARRAY_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{3.5, 5.5, 11.5, 13.5}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertAvgPoolTrace(fixture, artifact, DataType.FLOAT64, 1, 2, 2, 2, false);
    }

    @Test
    void executesF32DenseArrayAvgPool2dInParallel() {
        Tensor input = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "avgPoolF32Input", DataType.FLOAT32);
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.vectorParallel(2));
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_F32_ARRAY_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new float[]{3.5f, 5.5f, 11.5f, 13.5f}, actual.toFloat32ArrayCopy(), 1.0e-6f);
        assertAvgPoolTrace(fixture, artifact, DataType.FLOAT32, 2, 2, 2, 2, false);
    }

    @Test
    void executesBf16DenseArrayAvgPool2d() {
        Tensor input = new Tensor(bf16Bits(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }), new int[]{1, 1, 4, 4}, null, "avgPoolBf16Input", DataType.BFLOAT16);
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_BF16_ARRAY_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(bf16Bits(new float[]{3.5f, 5.5f, 11.5f, 13.5f}), actual.toBFloat16BitsArrayCopy());
        assertAvgPoolTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 2, false);
    }

    @Test
    void maxPool2dPaddingDoesNotBeatNegativeInputs() {
        Tensor input = new Tensor(new double[]{
                -4, -3,
                -2, -1
        }, new int[]{1, 1, 2, 2}, null, "negativePoolInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.maxPool2d(
                Pool2dOptions.square(2)
                        .withStride(1, 1)
                        .withPadding(1, 1)
        ));

        Tensor actual = execute(fixture, prepare(fixture, Cpu1PrepareConfig.scalarSingleThread()));

        assertArrayEquals(new int[]{1, 1, 3, 3}, actual.getShape());
        assertArrayEquals(new double[]{
                -4, -3, -3,
                -2, -1, -1,
                -2, -1, -1
        }, actual.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void maxPool2dCeilModeMatchesTensorShapeContract() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "ceilModePoolInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.maxPool2d(
                Pool2dOptions.square(2)
                        .withStride(2, 2)
                        .withCeilMode(true)
        ));

        Tensor actual = execute(fixture, prepare(fixture, Cpu1PrepareConfig.scalarSingleThread()));

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{5, 6, 8, 9}, actual.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void avgPool2dCountIncludePadFalseUsesValidBorderCount() {
        Tensor input = new Tensor(new double[]{4}, new int[]{1, 1, 1, 1}, null, "avgPoolExcludePadInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.avgPool2d(
                Pool2dOptions.square(2)
                        .withStride(1, 1)
                        .withPadding(1, 1)
        ));

        Tensor actual = execute(fixture, prepare(fixture, Cpu1PrepareConfig.scalarSingleThread()));

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{4, 4, 4, 4}, actual.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void avgPool2dCountIncludePadTrueUsesFullKernelCount() {
        Tensor input = new Tensor(new double[]{4}, new int[]{1, 1, 1, 1}, null, "avgPoolIncludePadInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.avgPool2d(
                Pool2dOptions.square(2)
                        .withStride(1, 1)
                        .withPadding(1, 1)
                        .withCountIncludePad(true)
        ));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarSingleThread());
        Tensor actual = execute(fixture, artifact);

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{1, 1, 1, 1}, actual.toDoubleArrayCopy(), 1.0e-12);
        assertAvgPoolTrace(fixture, artifact, DataType.FLOAT64, 1, 2, 2, 2, true);
    }

    @Test
    void avgPool2dCeilModeMatchesTensorShapeContract() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "ceilModeAvgPoolInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.avgPool2d(
                Pool2dOptions.square(2)
                        .withStride(2, 2)
                        .withCeilMode(true)
        ));

        Tensor actual = execute(fixture, prepare(fixture, Cpu1PrepareConfig.scalarSingleThread()));

        assertArrayEquals(new int[]{1, 1, 2, 2}, actual.getShape());
        assertArrayEquals(new double[]{3, 4.5, 7.5, 9}, actual.toDoubleArrayCopy(), 1.0e-12);
    }

    @Test
    void automaticLaunchUsesTunedThresholds() {
        Tensor input = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "autoPoolInput", DataType.FLOAT32);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 1, 1);

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.automatic(tuned, 4));

        assertEquals(4, artifact.preparedMaxPool2dUnit().launchConfig().workerCount());
        assertTrue(artifact.preparedMaxPool2dUnit().launchConfig().chunkSize() >= tuned.minReductionChunkSize());
    }

    @Test
    void executesF32MemorySegmentMaxPool2d() {
        float[] inputValues = new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(inputValues, new int[]{1, 1, 4, 4}, null, "nativePoolF32Input", DataType.FLOAT32);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeF32Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(new float[]{6, 8, 14, 16}, nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertMaxPoolTrace(fixture, artifact, DataType.FLOAT32, 1, 2, 2, 2);
    }

    @Test
    void executesF64MemorySegmentMaxPool2d() {
        double[] inputValues = new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(inputValues, new int[]{1, 1, 4, 4}, null, "nativePoolF64Input", DataType.FLOAT64);
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, new Cpu1PrepareConfig(
                backend.cpu1.kernels.Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.parallel(2),
                Cpu1StorageKind.MEMORY_SEGMENT
        ));
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_F64_SEGMENT_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeF64Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(new double[]{6, 8, 14, 16}, nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())), 1.0e-12);
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertMaxPoolTrace(fixture, artifact, DataType.FLOAT64, 2, 2, 2, 2);
    }

    @Test
    void executesBf16MemorySegmentMaxPool2d() {
        float[] inputValues = new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(
                bf16Bits(inputValues),
                new int[]{1, 1, 4, 4},
                null,
                "nativePoolBf16Input",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.maxPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1MaxPool2dKernelId.MAX_POOL2D_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedMaxPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeBf16Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(bf16Bits(new float[]{6, 8, 14, 16}), nativeBf16Bits(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertMaxPoolTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 2);
    }

    @Test
    void executesF32MemorySegmentAvgPool2d() {
        float[] inputValues = new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(inputValues, new int[]{1, 1, 4, 4}, null, "nativeAvgPoolF32Input", DataType.FLOAT32);
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_F32_SEGMENT_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeF32Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(
                new float[]{3.5f, 5.5f, 11.5f, 13.5f},
                nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertAvgPoolTrace(fixture, artifact, DataType.FLOAT32, 1, 2, 2, 2, false);
    }

    @Test
    void executesF64MemorySegmentAvgPool2d() {
        double[] inputValues = new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(inputValues, new int[]{1, 1, 4, 4}, null, "nativeAvgPoolF64Input", DataType.FLOAT64);
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, new Cpu1PrepareConfig(
                backend.cpu1.kernels.Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.parallel(2),
                Cpu1StorageKind.MEMORY_SEGMENT
        ));
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_F64_SEGMENT_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeF64Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(
                new double[]{3.5, 5.5, 11.5, 13.5},
                nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertAvgPoolTrace(fixture, artifact, DataType.FLOAT64, 2, 2, 2, 2, false);
    }

    @Test
    void executesBf16MemorySegmentAvgPool2d() {
        float[] inputValues = new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        };
        Tensor input = new Tensor(
                bf16Bits(inputValues),
                new int[]{1, 1, 4, 4},
                null,
                "nativeAvgPoolBf16Input",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(input.avgPool2d(Pool2dOptions.square(2)));

        Cpu1PreparedArtifact artifact = prepare(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertEquals(
                Cpu1AvgPool2dKernelId.AVG_POOL2D_BF16_SEGMENT_DENSE_SCALAR,
                artifact.preparedAvgPool2dUnit().kernelId()
        );
        ExecutionContext context = executeContext(fixture, artifact, ctx ->
                attachNativeBf16Input(ctx, fixture.node().inputIds().getFirst(), inputValues)
        );

        assertArrayEquals(
                bf16Bits(new float[]{3.5f, 5.5f, 11.5f, 13.5f}),
                nativeBf16Bits(context.nativeStorageForNodeId(fixture.node().id()))
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
        assertAvgPoolTrace(fixture, artifact, DataType.BFLOAT16, 1, 2, 2, 2, false);
    }

    @Test
    void rejectsStridedInputDescriptorForDenseDirectRoute() {
        Tensor base = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "stridedPoolBase", DataType.FLOAT32);
        Tensor view = base.permute(0, 1, 3, 2);
        Fixture fixture = fixture(view.maxPool2d(Pool2dOptions.square(2)));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepare(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );
        assertTrue(exception.getMessage().contains("input access"));
        assertTrue(exception.getMessage().contains(Cpu1StorageAccessKind.STRIDED.name()));
    }

    @Test
    void rejectsStridedAvgPool2dInputDescriptorForDenseDirectRoute() {
        Tensor base = new Tensor(new float[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "stridedAvgPoolBase", DataType.FLOAT32);
        Tensor view = base.permute(0, 1, 3, 2);
        Fixture fixture = fixture(view.avgPool2d(Pool2dOptions.square(2)));

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

    private static void assertMaxPoolTrace(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            DataType expectedDType,
            int expectedWorkers,
            int expectedOutputH,
            int expectedOutputW,
            int expectedKernel
    ) {
        StepTraceContribution trace = trace(fixture, artifact);
        Map<String, Object> attrs = trace.attributes();
        assertEquals(artifact.preparedMaxPool2dUnit().kernelId().name(), attrs.get("cpu1MaxPool2dKernelId"));
        assertEquals("MAX_POOL2D", attrs.get("cpu1Pool2dOpType"));
        assertEquals(expectedDType.name(), attrs.get("cpu1Pool2dDType"));
        assertEquals(artifact.preparedMaxPool2dUnit().storageKind().name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedOutputH, attrs.get("cpu1Pool2dOutputH"));
        assertEquals(expectedOutputW, attrs.get("cpu1Pool2dOutputW"));
        assertEquals(expectedKernel, attrs.get("cpu1Pool2dKernelH"));
        assertEquals(expectedKernel, attrs.get("cpu1Pool2dKernelW"));
        assertEquals(expectedWorkers, attrs.get("cpu1Pool2dLaunchWorkers"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Pool2dInputAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Pool2dOutputAccessKind"));
    }

    private static void assertAvgPoolTrace(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            DataType expectedDType,
            int expectedWorkers,
            int expectedOutputH,
            int expectedOutputW,
            int expectedKernel,
            boolean expectedCountIncludePad
    ) {
        StepTraceContribution trace = trace(fixture, artifact);
        Map<String, Object> attrs = trace.attributes();
        assertEquals(artifact.preparedAvgPool2dUnit().kernelId().name(), attrs.get("cpu1AvgPool2dKernelId"));
        assertEquals("AVG_POOL2D", attrs.get("cpu1Pool2dOpType"));
        assertEquals(expectedDType.name(), attrs.get("cpu1Pool2dDType"));
        assertEquals(artifact.preparedAvgPool2dUnit().storageKind().name(), attrs.get("cpu1StorageKind"));
        assertEquals(expectedOutputH, attrs.get("cpu1Pool2dOutputH"));
        assertEquals(expectedOutputW, attrs.get("cpu1Pool2dOutputW"));
        assertEquals(expectedKernel, attrs.get("cpu1Pool2dKernelH"));
        assertEquals(expectedKernel, attrs.get("cpu1Pool2dKernelW"));
        assertEquals(expectedCountIncludePad, attrs.get("cpu1Pool2dCountIncludePad"));
        assertEquals(expectedWorkers, attrs.get("cpu1Pool2dLaunchWorkers"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Pool2dInputAccessKind"));
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS.name(), attrs.get("cpu1Pool2dOutputAccessKind"));
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
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 maxpool native f32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 maxpool test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1 maxpool native f64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 maxpool test native F64 input");
    }

    private static void attachNativeBf16Input(ExecutionContext context, int nodeId, float[] values) {
        NativeBFloat16Storage storage = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.allocateNativeStorage(DataType.BFLOAT16, values.length, "cpu1 maxpool native bf16 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 maxpool test native BF16 input");
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
