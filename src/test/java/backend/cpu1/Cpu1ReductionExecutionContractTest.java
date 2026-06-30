package backend.cpu1;

import backend.contract.ComputeBackend;
import backend.cpu1.exec.Cpu1ReductionExecutableUnit;
import backend.cpu1.kernels.reduction.Cpu1ReductionKernelId;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
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
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeBoolStorage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeFloat64Storage;
import tensor.storage.NativeInt32Storage;
import tensor.storage.NativeInt64Storage;
import tensor.storage.NativeTensorStorage;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
    void preparedF32SumReducesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Fixture fixture = fixture(input.sum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_F32_DENSE_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        NativeTensorStorage secondOutput = context.nativeStorageForNodeId(fixture.node().id());
        assertSame(output, secondOutput);
        assertArrayEquals(new float[]{6.0f, 15.0f}, nativeF32Values(secondOutput), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF64MeanReducesNativeSegmentWithoutCpuMaterialization() {
        double[] values = new double[]{
                1.0d, 2.0d,
                3.0d, 4.0d,
                5.0d, 6.0d,
                7.0d, 8.0d
        };
        Tensor input = new Tensor(values, new int[]{2, 2, 2}, null, "input", DataType.FLOAT64);
        Fixture fixture = fixture(input.mean(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MEAN_F64_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF64Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat64Storage output = assertInstanceOf(
                NativeFloat64Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new double[]{2.0d, 3.0d, 6.0d, 7.0d}, nativeF64Values(output), 1.0e-12);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedBf16MeanReducesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "input", DataType.BFLOAT16);
        Fixture fixture = fixture(input.mean(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MEAN_BF16_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeBf16Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeBFloat16Storage output = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new float[]{2.0f, 5.0f}, nativeBf16Values(output), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF32MinReducesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                3.0f, -2.0f, 5.0f,
                7.0f, 1.0f, -4.0f
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Fixture fixture = fixture(input.min(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MIN_F32_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new float[]{-2.0f, -4.0f}, nativeF32Values(output), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF64ProdReducesNativeSegmentWithoutCpuMaterialization() {
        double[] values = new double[]{
                1.0d, 2.0d,
                3.0d, 4.0d,
                5.0d, 6.0d,
                7.0d, 8.0d
        };
        Tensor input = new Tensor(values, new int[]{2, 2, 2}, null, "input", DataType.FLOAT64);
        Fixture fixture = fixture(input.prod(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.PROD_F64_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF64Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat64Storage output = assertInstanceOf(
                NativeFloat64Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new double[]{3.0d, 8.0d, 35.0d, 48.0d}, nativeF64Values(output), 1.0e-12);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedBf16ProdReducesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "input", DataType.BFLOAT16);
        Fixture fixture = fixture(input.prod(0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.PROD_BF16_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeBf16Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeBFloat16Storage output = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new float[]{4.0f, 10.0f, 18.0f}, nativeBf16Values(output), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedBoolAnyReducesNativeSegmentWithoutCpuMaterialization() {
        byte[] values = boolBytes(
                false, false,
                true, false,
                false, true,
                false, false
        );
        Tensor input = new Tensor(values, new int[]{2, 2, 2}, null, "input", DataType.BOOL);
        Fixture fixture = fixture(input.any(1, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ANY_BOOL_DENSE_SCALAR);
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeBoolInput(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeBoolStorage output = assertInstanceOf(
                NativeBoolStorage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(boolBytes(true, false, false, true), nativeBoolValues(output));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedArgMaxReducesNativeSegmentsToInt64WithoutCpuMaterialization() {
        assertNativeArgMaxF32(
                new float[]{
                        1.0f, 5.0f, 5.0f,
                        4.0f, 3.0f, 2.0f
                },
                new int[]{2, 3},
                new long[]{1L, 0L},
                ArgMaxTiePolicy.FIRST_INDEX
        );
        assertNativeArgMaxF64(
                new double[]{
                        1.0d, 5.0d,
                        3.0d, 5.0d,
                        7.0d, 2.0d,
                        7.0d, 6.0d
                },
                new int[]{2, 2, 2},
                new long[]{1L, 1L, 1L, 1L},
                ArgMaxTiePolicy.LAST_INDEX
        );
        assertNativeArgMaxBf16(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 9.0f, 6.0f
                },
                new int[]{2, 3},
                new long[]{2L, 1L},
                ArgMaxTiePolicy.FIRST_INDEX
        );
        assertNativeArgMaxI32(
                new int[]{
                        1, 9, 3,
                        4, 4, 2
                },
                new int[]{2, 3},
                new long[]{1L, 1L},
                ArgMaxTiePolicy.LAST_INDEX
        );
        assertNativeArgMaxI64(
                new long[]{
                        1L, 9L, 3L,
                        4L, 4L, 12L
                },
                new int[]{2, 3},
                new long[]{1L, 2L},
                ArgMaxTiePolicy.FIRST_INDEX
        );
    }

    @Test
    void preparedCumSumScansNativeSegmentsWithoutCpuMaterialization() {
        assertNativeCumSumF32(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                new float[]{1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}
        );
        assertNativeCumSumF64(
                new double[]{1.0d, 2.0d, 3.0d, 4.0d, 5.0d, 6.0d},
                new int[]{2, 3},
                new double[]{5.0d, 3.0d, 0.0d, 11.0d, 6.0d, 0.0d}
        );
        assertNativeCumSumBf16(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                new float[]{1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}
        );
        assertNativeCumSumI32(
                new int[]{1, 2, 3, 4, 5, 6},
                new int[]{2, 3},
                new int[]{1, 3, 6, 4, 9, 15}
        );
        assertNativeCumSumI64(
                new long[]{10L, 20L, 30L, 1L, 2L, 3L},
                new int[]{2, 3},
                new long[]{10L, 30L, 60L, 1L, 3L, 6L}
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
    void preparedF32SumAcceptsDenseWithOffsetInputAccess() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "offsetBase",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.select(0, 1).sum(0, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_F32_DENSE_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1StorageAccessKind.DENSE_WITH_OFFSET,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{15.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF32SumReducesStridedInputInLogicalOrder() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "stridedF32Base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0).sum(0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_F32_STRIDED_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1StorageAccessKind.STRIDED,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{6.0f, 15.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedF64MeanReducesStridedInputInLogicalOrder() {
        Tensor base = new Tensor(
                new double[]{
                        1.0d, 2.0d, 3.0d,
                        4.0d, 5.0d, 6.0d
                },
                new int[]{2, 3},
                null,
                "stridedF64Base",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(base.permute(1, 0).mean(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.MEAN_F64_STRIDED_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1StorageAccessKind.STRIDED,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new double[]{2.5d, 3.5d, 4.5d},
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedF32SumReducesNativeStridedSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                1.0f, 2.0f, 3.0f,
                4.0f, 5.0f, 6.0f
        };
        Tensor base = new Tensor(values, new int[]{2, 3}, null, "nativeStridedBase", DataType.FLOAT32);
        Fixture fixture = fixture(base.permute(1, 0).sum(0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SUM_F32_STRIDED_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1StorageAccessKind.STRIDED,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        attachNativeF32Input(context, fixture.node().inputIds().getFirst(), values);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(new float[]{6.0f, 15.0f}, nativeF32Values(output), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
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
    void preparedF32SoftmaxNormalizesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{
                1.0f, 2.0f, 3.0f,
                0.0f, 0.0f, 0.0f
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "nativeSoftmaxF32", DataType.FLOAT32);
        Tensor out = new Tensor(input.getShape(), List.of(input), new softmax(1), "softmax", DataType.FLOAT32);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SOFTMAX_F32_DENSE_SCALAR);
        assertReductionPolicy(
                artifact,
                Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS
        );
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF32Input(ctx, fixture.node().inputIds().getFirst(), values));

        NativeFloat32Storage output = assertInstanceOf(
                NativeFloat32Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(
                new float[]{0.09003057f, 0.24472847f, 0.66524096f, 0.33333334f, 0.33333334f, 0.33333334f},
                nativeF32Values(output),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF64LogSoftmaxNormalizesNativeSegmentWithoutCpuMaterialization() {
        double[] values = new double[]{
                1000.0d, 1001.0d, 1002.0d,
                0.0d, 0.0d, 0.0d
        };
        Tensor input = new Tensor(values, new int[]{2, 3}, null, "nativeLogSoftmaxF64", DataType.FLOAT64);
        Tensor out = new Tensor(input.getShape(), List.of(input), new logSoftmax(1), "logSoftmax", DataType.FLOAT64);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.LOG_SOFTMAX_F64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF64Input(ctx, fixture.node().inputIds().getFirst(), values));

        NativeFloat64Storage output = assertInstanceOf(
                NativeFloat64Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(
                new double[]{
                        -2.4076059644443806d, -1.4076059644443804d, -0.4076059644443804d,
                        -1.0986122886681098d, -1.0986122886681098d, -1.0986122886681098d
                },
                nativeF64Values(output),
                1.0e-12
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedBf16LogSoftmaxNormalizesNativeSegmentWithoutCpuMaterialization() {
        float[] values = new float[]{1.0f, 2.0f, 3.0f};
        Tensor input = new Tensor(values, new int[]{1, 3}, null, "nativeLogSoftmaxBf16", DataType.BFLOAT16);
        Tensor out = new Tensor(input.getShape(), List.of(input), new logSoftmax(1), "logSoftmax", DataType.BFLOAT16);
        Fixture fixture = fixture(out);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.LOG_SOFTMAX_BF16_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeBf16Input(ctx, fixture.node().inputIds().getFirst(), values));

        NativeBFloat16Storage output = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.nativeStorageForNodeId(fixture.node().id())
        );
        assertArrayEquals(
                new float[]{-2.4076059f, -1.4076060f, -0.4076060f},
                nativeBf16Values(output),
                8.0e-3f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedF32SoftmaxAutomaticLaunchUsesReductionParallelThreshold() {
        int rows = 100_000;
        int cols = 2;
        float[] values = new float[rows * cols];
        float[] expected = new float[values.length];
        for (int row = 0; row < rows; row++) {
            values[row * cols] = 0.0f;
            values[row * cols + 1] = 1.0f;
            expected[row * cols] = 0.26894143f;
            expected[row * cols + 1] = 0.7310586f;
        }
        Tensor input = new Tensor(values, new int[]{rows, cols}, null, "parallelSoftmaxF32", DataType.FLOAT32);
        Tensor out = new Tensor(input.getShape(), List.of(input), new softmax(1), "softmax", DataType.FLOAT32);
        Fixture fixture = fixture(out);
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, rows);
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.automatic(tuned, 4));
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
        assertReductionKernel(artifact, Cpu1ReductionKernelId.SOFTMAX_F32_DENSE_SCALAR);
        assertEquals(4, unit.launchConfig().workerCount());
        assertEquals(25_000, unit.launchConfig().chunkSize());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expected,
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
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

    @Test
    void reductionPrepareCarriesParallelLaunchConfigIntoTrace() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.sum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();

        assertEquals(4, unit.launchConfig().workerCount());
        assertInstanceOf(Cpu1ParallelLaunch.class, unit.launchPolicy());

        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(4, trace.metadata().attributes().get("cpu1ReductionLaunchWorkers"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionLaunchChunkSize"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionScratchF32"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionScratchF64"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionScratchI32"));
        assertEquals(4, trace.metadata().reduction().plannedWorkers());
        assertEquals(0, trace.metadata().reduction().chunkSize());
    }

    @Test
    void preparedF32MeanUsesParallelOutputWorkItems() {
        int rows = 64;
        int cols = 8;
        float[] values = new float[rows * cols];
        float[] expected = new float[rows];
        for (int row = 0; row < rows; row++) {
            float sum = 0.0f;
            for (int col = 0; col < cols; col++) {
                float value = row * 10.0f + col;
                values[row * cols + col] = value;
                sum += value;
            }
            expected[row] = sum / cols;
        }
        Tensor input = new Tensor(values, new int[]{rows, cols}, null, "parallelF32MeanInput", DataType.FLOAT32);
        Fixture fixture = fixture(input.mean(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
        assertEquals(0, unit.scratchBufferSpec().f64ArrayElements());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(expected, context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(), 1.0e-6f);
        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(4, trace.metadata().attributes().get("cpu1ReductionLaunchWorkers"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionScratchF64"));
    }

    @Test
    void preparedBf16SumUsesParallelOutputWorkItemsWithDoubleAccumulator() {
        int rows = 16;
        int cols = 4;
        float[] values = new float[rows * cols];
        float[] expected = new float[rows];
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                values[row * cols + col] = col + 1.0f;
            }
            expected[row] = 10.0f;
        }
        Tensor input = new Tensor(values, new int[]{rows, cols}, null, "parallelBf16SumInput", DataType.BFLOAT16);
        Fixture fixture = fixture(input.sum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
        assertEquals(0, unit.scratchBufferSpec().f64ArrayElements());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(expected, bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())), 1.0e-6f);
        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(4, trace.metadata().attributes().get("cpu1ReductionLaunchWorkers"));
        assertEquals(0, trace.metadata().attributes().get("cpu1ReductionScratchF64"));
    }

    @Test
    void preparedF64ScalarMeanUsesF64ScratchPartials() {
        int elements = 64;
        double[] values = new double[elements];
        for (int i = 0; i < elements; i++) {
            values[i] = i + 1.0d;
        }
        Tensor input = new Tensor(values, new int[]{elements}, null, "partialF64MeanInput", DataType.FLOAT64);
        Fixture fixture = fixture(input.mean(0, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorParallel(4));
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
        int expectedSlots = Cpu1RangeLauncher.slotCount(unit.axisSize(), unit.launchConfig());
        assertEquals(expectedSlots, unit.scratchBufferSpec().f64ArrayElements());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new double[]{32.5d},
                context.runtimeTensorForNodeId(fixture.node().id()).toDoubleArrayCopy(),
                1.0e-12
        );
        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(4, trace.metadata().attributes().get("cpu1ReductionLaunchWorkers"));
        assertEquals(expectedSlots, trace.metadata().attributes().get("cpu1ReductionScratchF64"));
    }

    @Test
    void reductionPrepareRejectsUnsupportedStridedInputPolicies() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "stridedReductionBase",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0).min(0));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("DENSE_CONTIGUOUS input access"));
        assertTrue(exception.getMessage().contains("actual=STRIDED"));
        assertTrue(exception.getMessage().contains("SUM/MEAN FLOAT32/FLOAT64"));
    }

    @Test
    void reductionPrepareRejectsBf16StridedInputPolicy() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f
                },
                new int[]{2, 3},
                null,
                "stridedBf16Base",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(base.permute(1, 0).sum(0));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("actual=STRIDED"));
        assertTrue(exception.getMessage().contains("input=BFLOAT16"));
    }

    @Test
    void reductionPrepareRejectsBroadcastInputPolicy() {
        Tensor row = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f},
                new int[]{1, 3},
                null,
                "broadcastRow",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(row.expand(2, 3).sum(1));

        UnsupportedOperationException exception = assertThrows(
                UnsupportedOperationException.class,
                () -> prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread())
        );

        assertTrue(exception.getMessage().contains("actual=BROADCAST"));
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertReductionKernel(Cpu1PreparedArtifact artifact, Cpu1ReductionKernelId expected) {
        Cpu1ReductionExecutableUnit executable = assertInstanceOf(Cpu1ReductionExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedReductionUnit().kernelId());
        assertSame(artifact.preparedReductionUnit(), executable.preparedUnit());
    }

    private static void assertReductionPolicy(
            Cpu1PreparedArtifact artifact,
            Cpu1StorageKind storageKind,
            Cpu1StorageAccessKind inputAccessKind,
            Cpu1StorageAccessKind outputAccessKind
    ) {
        Cpu1PreparedReductionUnit unit = artifact.preparedReductionUnit();
        assertEquals(storageKind, unit.storageKind());
        assertEquals(inputAccessKind, unit.inputAccessPlan().kind());
        assertEquals(outputAccessKind, unit.outputAccessPlan().kind());
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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

    private static void attachNativeF32Input(ExecutionContext context, int nodeId, float[] values) {
        NativeFloat32Storage storage = assertInstanceOf(
                NativeFloat32Storage.class,
                context.allocateNativeStorage(DataType.FLOAT32, values.length, "cpu1 reduction native f32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native F32 input");
    }

    private static void attachNativeF64Input(ExecutionContext context, int nodeId, double[] values) {
        NativeFloat64Storage storage = assertInstanceOf(
                NativeFloat64Storage.class,
                context.allocateNativeStorage(DataType.FLOAT64, values.length, "cpu1 reduction native f64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setFloat64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native F64 input");
    }

    private static void attachNativeBf16Input(ExecutionContext context, int nodeId, float[] values) {
        NativeBFloat16Storage storage = assertInstanceOf(
                NativeBFloat16Storage.class,
                context.allocateNativeStorage(DataType.BFLOAT16, values.length, "cpu1 reduction native bf16 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBFloat16BitsAt(i, TensorDTypeOps.toBFloat16Bits(values[i]));
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native BF16 input");
    }

    private static void attachNativeBoolInput(ExecutionContext context, int nodeId, byte[] values) {
        NativeBoolStorage storage = assertInstanceOf(
                NativeBoolStorage.class,
                context.allocateNativeStorage(DataType.BOOL, values.length, "cpu1 reduction native bool input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setBoolAt(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native BOOL input");
    }

    private static void attachNativeI32Input(ExecutionContext context, int nodeId, int[] values) {
        NativeInt32Storage storage = assertInstanceOf(
                NativeInt32Storage.class,
                context.allocateNativeStorage(DataType.INT32, values.length, "cpu1 reduction native i32 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setInt32At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native I32 input");
    }

    private static void attachNativeI64Input(ExecutionContext context, int nodeId, long[] values) {
        NativeInt64Storage storage = assertInstanceOf(
                NativeInt64Storage.class,
                context.allocateNativeStorage(DataType.INT64, values.length, "cpu1 reduction native i64 input")
        );
        for (int i = 0; i < values.length; i++) {
            storage.setInt64At(i, values[i]);
        }
        context.attachNativeStorage(nodeId, storage, "cpu1 reduction test native I64 input");
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

    private static float[] nativeBf16Values(NativeTensorStorage storage) {
        NativeBFloat16Storage bf16 = assertInstanceOf(NativeBFloat16Storage.class, storage);
        float[] out = new float[bf16.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(bf16.getBFloat16BitsAt(i));
        }
        return out;
    }

    private static byte[] nativeBoolValues(NativeTensorStorage storage) {
        NativeBoolStorage bool = assertInstanceOf(NativeBoolStorage.class, storage);
        byte[] out = new byte[bool.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = bool.getBoolAt(i) == 0 ? (byte) 0 : (byte) 1;
        }
        return out;
    }

    private static int[] nativeI32Values(NativeTensorStorage storage) {
        NativeInt32Storage i32 = assertInstanceOf(NativeInt32Storage.class, storage);
        int[] out = new int[i32.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = i32.getInt32At(i);
        }
        return out;
    }

    private static long[] nativeI64Values(NativeTensorStorage storage) {
        NativeInt64Storage i64 = assertInstanceOf(NativeInt64Storage.class, storage);
        long[] out = new long[i64.getSize()];
        for (int i = 0; i < out.length; i++) {
            out[i] = i64.getInt64At(i);
        }
        return out;
    }

    private static void assertNativeArgMaxF32(
            float[] values,
            int[] shape,
            long[] expected,
            ArgMaxTiePolicy tiePolicy
    ) {
        Tensor input = new Tensor(values, shape, null, "nativeArgMaxF32", DataType.FLOAT32);
        Fixture fixture = fixture(input.argMax(1, false, tiePolicy));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_F32_TO_I64_DENSE_SCALAR);
        assertReductionPolicy(artifact, Cpu1StorageKind.MEMORY_SEGMENT,
                Cpu1StorageAccessKind.DENSE_CONTIGUOUS, Cpu1StorageAccessKind.DENSE_CONTIGUOUS);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF32Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeArgMaxF64(
            double[] values,
            int[] shape,
            long[] expected,
            ArgMaxTiePolicy tiePolicy
    ) {
        Tensor input = new Tensor(values, shape, null, "nativeArgMaxF64", DataType.FLOAT64);
        Fixture fixture = fixture(input.argMax(1, true, tiePolicy));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_F64_TO_I64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF64Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeArgMaxBf16(
            float[] values,
            int[] shape,
            long[] expected,
            ArgMaxTiePolicy tiePolicy
    ) {
        Tensor input = new Tensor(values, shape, null, "nativeArgMaxBf16", DataType.BFLOAT16);
        Fixture fixture = fixture(input.argMax(1, false, tiePolicy));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_BF16_TO_I64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeBf16Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeArgMaxI32(
            int[] values,
            int[] shape,
            long[] expected,
            ArgMaxTiePolicy tiePolicy
    ) {
        Tensor input = new Tensor(values, shape, null, "nativeArgMaxI32", DataType.INT32);
        Fixture fixture = fixture(input.argMax(1, false, tiePolicy));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_I32_TO_I64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeI32Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeArgMaxI64(
            long[] values,
            int[] shape,
            long[] expected,
            ArgMaxTiePolicy tiePolicy
    ) {
        Tensor input = new Tensor(values, shape, null, "nativeArgMaxI64", DataType.INT64);
        Fixture fixture = fixture(input.argMax(1, false, tiePolicy));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.ARGMAX_I64_TO_I64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeI64Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeCumSumF32(float[] values, int[] shape, float[] expected) {
        Tensor input = new Tensor(values, shape, null, "nativeCumSumF32", DataType.FLOAT32);
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_F32_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF32Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeF32Values(context.nativeStorageForNodeId(fixture.node().id())), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeCumSumF64(double[] values, int[] shape, double[] expected) {
        Tensor input = new Tensor(values, shape, null, "nativeCumSumF64", DataType.FLOAT64);
        Fixture fixture = fixture(input.cumSum(1, true, true));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_F64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeF64Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeF64Values(context.nativeStorageForNodeId(fixture.node().id())), 1.0e-12);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeCumSumBf16(float[] values, int[] shape, float[] expected) {
        Tensor input = new Tensor(values, shape, null, "nativeCumSumBf16", DataType.BFLOAT16);
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_BF16_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeBf16Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeBf16Values(context.nativeStorageForNodeId(fixture.node().id())), 1.0e-6f);
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeCumSumI32(int[] values, int[] shape, int[] expected) {
        Tensor input = new Tensor(values, shape, null, "nativeCumSumI32", DataType.INT32);
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_I32_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeI32Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI32Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static void assertNativeCumSumI64(long[] values, int[] shape, long[] expected) {
        Tensor input = new Tensor(values, shape, null, "nativeCumSumI64", DataType.INT64);
        Fixture fixture = fixture(input.cumSum(1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertReductionKernel(artifact, Cpu1ReductionKernelId.CUMSUM_I64_DENSE_SCALAR);
        ExecutionContext context = executeWithNativeInput(fixture, artifact, ctx ->
                attachNativeI64Input(ctx, fixture.node().inputIds().getFirst(), values));
        assertArrayEquals(expected, nativeI64Values(context.nativeStorageForNodeId(fixture.node().id())));
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    private static ExecutionContext executeWithNativeInput(
            Fixture fixture,
            Cpu1PreparedArtifact artifact,
            NativeInputBinder nativeInputBinder
    ) {
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));
        nativeInputBinder.attach(context);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context;
    }

    @FunctionalInterface
    private interface NativeInputBinder {
        void attach(ExecutionContext context);
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
