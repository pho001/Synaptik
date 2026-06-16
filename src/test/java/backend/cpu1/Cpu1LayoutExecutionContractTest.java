package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.cpu1.exec.Cpu1LayoutExecutableUnit;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.kernels.layout.Cpu1LayoutKernelId;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
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
import operations.layout.sliceBackward;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.Window2dOptions;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;

class Cpu1LayoutExecutionContractTest {
    @Test
    void preparedPermuteAliasesArrayStorageWithoutCopy() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.PERMUTE_ALIAS);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedReshapeOfContiguousInputAliasesArrayStorage() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.reshape(3, 2));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.RESHAPE_ALIAS);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedReshapeOfStridedInputCopiesInLogicalOrder() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0).reshape(2, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.RESHAPE_COPY_LINEARIZED_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertNotSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedContiguousMaterializesStridedInputInLogicalOrder() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0).contiguous());
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONTIGUOUS_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertNotSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new int[]{2, 1}, output.getStrides());
        assertArrayEquals(new float[]{1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedContiguousOfDenseOffsetInputUsesBlockCopy() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f,
                        10.0f, 11.0f, 12.0f
                },
                new int[]{4, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.slice(new int[]{1}, new int[]{3}, new int[]{0}, new int[]{1}).contiguous());
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertEquals(3, input.getStorageOffsetUnsafe());
        assertArrayEquals(new float[]{4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedVectorContiguousOfDenseOffsetInputUsesVectorBlockCopy() {
        Tensor base = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f, 4.0f,
                        5.0f, 6.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f, 12.0f
                },
                new int[]{3, 4},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.slice(new int[]{1}, new int[]{3}, new int[]{0}, new int[]{1}).contiguous());
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONTIGUOUS_OFFSET_DENSE_BLOCK_COPY_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedCommonViewOpsAliasArrayStorage() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Tensor out = base.select(1, 1).expandDims(0).squeeze(0);
        ExecutionResult result = executeAll(fixture(out), Cpu1PrepareConfig.scalarSingleThread(), false);

        Tensor source = result.context().runtimeTensorForNodeId(0);
        Tensor output = result.context().runtimeTensorForNodeId(result.fixture().node().id());
        assertSame(TensorInternalAccess.storage(source), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{2.0f, 5.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedExpandAliasesArrayStorage() {
        Tensor base = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{1, 3}, null, "base", DataType.FLOAT32);
        Fixture fixture = fixture(base.expand(2, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.EXPAND_ALIAS);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedSliceAliasesArrayStorage() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.slice(new int[]{0, 1}, new int[]{2, 3}, new int[]{0, 1}, new int[]{1, 1}));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.SLICE_ALIAS);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        Tensor input = context.runtimeTensorForNodeId(fixture.node().inputIds().getFirst());
        Tensor output = context.runtimeTensorForNodeId(fixture.node().id());
        assertSame(TensorInternalAccess.storage(input), TensorInternalAccess.storage(output));
        assertArrayEquals(new float[]{2.0f, 3.0f, 5.0f, 6.0f}, output.toFloat32ArrayCopy(), 1.0e-6f);
    }

    @Test
    void preparedSliceBackwardScattersF32ArrayUpdates() {
        Tensor updates = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f},
                new int[]{2, 2},
                null,
                "updates",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(sliceBackward(updates, new int[]{2, 4}));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.SLICE_BACKWARD_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{0.0f, 10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 40.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedSliceBackwardScattersBfloat16ArrayUpdates() {
        Tensor updates = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0},
                new int[]{2, 2},
                null,
                "updates",
                DataType.BFLOAT16
        );
        Fixture fixture = fixture(sliceBackward(updates, new int[]{2, 4}));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.SLICE_BACKWARD_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{0.0f, 1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f},
                bf16ToF32(context.runtimeTensorForNodeId(fixture.node().id())),
                1.0e-3f
        );
    }

    @Test
    void preparedMemorySegmentSliceBackwardScattersF32Updates() {
        Tensor updates = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f},
                new int[]{2, 2},
                null,
                "updates",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(sliceBackward(updates, new int[]{2, 4}));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.SLICE_BACKWARD_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{0.0f, 10.0f, 0.0f, 20.0f, 0.0f, 30.0f, 0.0f, 40.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 8),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentSliceBackwardScattersF64Updates() {
        Tensor updates = new Tensor(
                new double[]{10.0, 20.0, 30.0, 40.0},
                new int[]{2, 2},
                null,
                "updates",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(sliceBackward(updates, new int[]{2, 4}));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.SLICE_BACKWARD_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new double[]{0.0, 10.0, 0.0, 20.0, 0.0, 30.0, 0.0, 40.0},
                readNativeF64(context.nativeStorageForNodeId(fixture.node().id()), 8),
                1.0e-12
        );
    }

    @Test
    void preparedElementwiseAfterPermuteRunsThroughLayoutView() {
        Tensor base = new Tensor(
                new float[]{-1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        ExecutionResult result = executeAll(fixture(base.permute(1, 0).relu()), Cpu1PrepareConfig.scalarSingleThread(), false);

        assertArrayEquals(
                new float[]{0.0f, 4.0f, 2.0f, 0.0f, 3.0f, 6.0f},
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentPermuteAliasesNativeStorageWithoutCopy() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        int inputNodeId = fixture.node().inputIds().getFirst();
        assertSame(
                context.nativeStorageForNodeId(inputNodeId),
                context.nativeStorageForNodeId(fixture.node().id())
        );
    }

    @Test
    void preparedMemorySegmentLayoutViewFeedsElementwiseKernel() {
        Tensor base = new Tensor(
                new float[]{-1.0f, 2.0f, 3.0f, 4.0f, -5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        ExecutionResult result = executeAll(
                fixture(base.permute(1, 0).relu()),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread(),
                true
        );

        NativeTensorStorage output = result.context().nativeStorageForNodeId(result.fixture().node().id());
        assertArrayEquals(new float[]{0.0f, 4.0f, 2.0f, 0.0f, 3.0f, 6.0f}, readNativeF32(output, 6), 1.0e-6f);
    }

    @Test
    void preparedConcatMaterializesInputsAlongAxis() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f},
                new int[]{2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(Tensor.concat(1, left, right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONCAT_INNER_AXIS_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedConcatAxis0UsesBlockCopy() {
        Tensor top = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "top",
                DataType.FLOAT32
        );
        Tensor bottom = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f},
                new int[]{2, 2},
                null,
                "bottom",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(Tensor.concat(0, top, bottom));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONCAT_AXIS0_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 10.0f, 20.0f, 30.0f, 40.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedConcatMiddleAxisUsesBlockCopy() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 1, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        70.0f, 80.0f, 90.0f,
                        100.0f, 110.0f, 120.0f
                },
                new int[]{2, 2, 3},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(Tensor.concat(1, left, right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONCAT_MIDDLE_AXIS_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        10.0f, 20.0f, 30.0f,
                        40.0f, 50.0f, 60.0f,
                        4.0f, 5.0f, 6.0f,
                        70.0f, 80.0f, 90.0f,
                        100.0f, 110.0f, 120.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedVectorConcatMiddleAxisUsesVectorBlockCopy() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 1, 2},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{
                        10.0f, 20.0f,
                        30.0f, 40.0f,
                        50.0f, 60.0f,
                        70.0f, 80.0f
                },
                new int[]{2, 2, 2},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(Tensor.concat(1, left, right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.CONCAT_MIDDLE_AXIS_BLOCK_COPY_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        10.0f, 20.0f,
                        30.0f, 40.0f,
                        3.0f, 4.0f,
                        50.0f, 60.0f,
                        70.0f, 80.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedPadMaterializesConstantBorder() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.pad(new int[]{1, 1}, new int[]{0, 1}, -1.0));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.PAD_DENSE_INNER_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        -1.0f, -1.0f, -1.0f, -1.0f,
                        -1.0f, 1.0f, 2.0f, -1.0f,
                        -1.0f, 3.0f, 4.0f, -1.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedTileLastAxisUsesBlockCopy() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(1, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedTileAxis0UsesBlockCopy() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(3, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_AXIS0_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        1.0f, 2.0f,
                        3.0f, 4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedVectorTileAxis0UsesVectorBlockCopy() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(2, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_AXIS0_BLOCK_COPY_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        1.0f, 2.0f,
                        3.0f, 4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedTileSingleMiddleAxisRepeatUsesDenseBlockCopy() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                new int[]{2, 3, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(1, 2, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_DENSE_BLOCK_REPEAT_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void layoutTraceReportsPreparedKernelId() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(1, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        CompiledNodeExecutionMetadata metadata = metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata));

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        var trace = StepExecutionTracer.toStepTrace(
                0,
                new PreparedExecutionStep(fixture.node(), metadata),
                1L,
                context
        );
        assertEquals(Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_VECTOR.name(), trace.kernel());
        assertEquals(
                Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_VECTOR.name(),
                trace.metadata().attributes().get("cpu1LayoutKernelId")
        );
        assertEquals(
                Cpu1LayoutKernelId.TILE_LAST_AXIS_BLOCK_COPY_VECTOR.name(),
                trace.metadata().layout().targetType()
        );
    }

    @Test
    void preparedVectorTileSingleMiddleAxisRepeatUsesDenseBlockCopy() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                new int[]{2, 3, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(1, 2, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_DENSE_BLOCK_REPEAT_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f,
                        7.0f, 8.0f,
                        9.0f, 10.0f,
                        11.0f, 12.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedTileRepeatsInputAlongAxes() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(2, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_DENSE_MULTI_AXIS_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedVectorTileRepeatsInputAlongAxesWithDenseBlockCopy() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.tile(2, 3));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.TILE_DENSE_MULTI_AXIS_BLOCK_COPY_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f,
                        1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                        3.0f, 4.0f, 3.0f, 4.0f, 3.0f, 4.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedUnfoldAxisMaterializesSlidingWindows() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{4},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.unfold(0, 3, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 2.0f, 3.0f, 2.0f, 3.0f, 4.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedVectorUnfoldLastAxisUsesBlockCopy() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f, 4.0f,
                        10.0f, 20.0f, 30.0f, 40.0f
                },
                new int[]{2, 4},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.unfold(1, 2, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.UNFOLD_AXIS_LAST_AXIS_BLOCK_COPY_VECTOR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f,
                        2.0f, 3.0f,
                        3.0f, 4.0f,
                        10.0f, 20.0f,
                        20.0f, 30.0f,
                        30.0f, 40.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedUnfoldNonLastAxisUsesGenericScalarCopy() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        5.0f, 6.0f
                },
                new int[]{3, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.unfold(0, 2, 1));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.UNFOLD_AXIS_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 3.0f,
                        2.0f, 4.0f,
                        3.0f, 5.0f,
                        4.0f, 6.0f
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMaterializedLayoutOutputFeedsElementwiseKernel() {
        Tensor input = new Tensor(
                new float[]{-1.0f, 2.0f, -3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.FLOAT32
        );
        ExecutionResult result = executeAll(
                fixture(input.tile(1, 2).relu()),
                Cpu1PrepareConfig.scalarSingleThread(),
                false
        );

        assertArrayEquals(
                new float[]{0.0f, 2.0f, 0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 4.0f},
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentConcatMaterializesNativeOutput() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{1, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{1, 2}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(Tensor.concat(0, left, right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 4),
                1.0e-6f
        );
    }

    @Test
    void preparedUnfold2dMaterializesIm2colColumns() {
        Tensor input = new Tensor(
                new double[]{
                        1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0
                },
                new int[]{1, 1, 3, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.unfold2d(Window2dOptions.of(2, 2)));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.UNFOLD2D_COPY_SCALAR);
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new double[]{
                        1.0, 2.0, 4.0, 5.0,
                        2.0, 3.0, 5.0, 6.0,
                        4.0, 5.0, 7.0, 8.0,
                        5.0, 6.0, 8.0, 9.0
                },
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedFold2dAccumulatesOverlappingColumns() {
        Tensor input = new Tensor(
                new double[]{
                        1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0
                },
                new int[]{1, 1, 3, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.unfold2d(Window2dOptions.of(2, 2))
                .fold2d(new int[]{1, 1, 3, 3}, Window2dOptions.of(2, 2)));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertEquals(9, artifact.scratchBufferSpec().f64ArrayElements());
        ExecutionResult result = executeAll(fixture, Cpu1PrepareConfig.scalarSingleThread(), false);

        assertLayoutKernel(artifact, Cpu1LayoutKernelId.FOLD2D_COPY_SCALAR);
        Cpu1ScratchBuffer scratchBuffer = assertInstanceOf(
                Cpu1ScratchBuffer.class,
                result.context().workspaceForNodeId(result.fixture().node().id())
        );
        assertEquals(9, scratchBuffer.requireF64Array(9).length);
        assertArrayEquals(
                new double[]{
                        1.0, 4.0, 3.0,
                        8.0, 20.0, 12.0,
                        7.0, 16.0, 9.0
                },
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedFold2dNonOverlapUsesDirectPathWithoutScratchBuffer() {
        Tensor input = new Tensor(
                new double[]{
                        1.0, 2.0, 3.0, 4.0,
                        5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0,
                        13.0, 14.0, 15.0, 16.0
                },
                new int[]{1, 1, 4, 4},
                null,
                "input",
                DataType.FLOAT64
        );
        Window2dOptions nonOverlap = Window2dOptions.of(2, 2).withStride(2, 2);
        Fixture fixture = fixture(input.unfold2d(nonOverlap).fold2d(new int[]{1, 1, 4, 4}, nonOverlap));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarSingleThread());
        assertLayoutKernel(artifact, Cpu1LayoutKernelId.FOLD2D_NON_OVERLAP_DIRECT_SCALAR);
        assertEquals(0, artifact.scratchBufferSpec().f64ArrayElements());

        ExecutionResult result = executeAll(fixture, Cpu1PrepareConfig.scalarSingleThread(), false);

        assertArrayEquals(
                input.toFloat64ArrayCopy(),
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedParallelFold2dUsesScratchBufferSlotPerRangeTask() {
        Tensor input = new Tensor(
                new double[]{
                        1.0, 2.0, 3.0,
                        4.0, 5.0, 6.0,
                        7.0, 8.0, 9.0
                },
                new int[]{1, 1, 3, 3},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.unfold2d(Window2dOptions.of(2, 2))
                .fold2d(new int[]{1, 1, 3, 3}, Window2dOptions.of(2, 2)));
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.parallel(2, 5),
                Cpu1StorageKind.JAVA_ARRAY
        );
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, config);
        assertEquals(36, artifact.scratchBufferSpec().f64ArrayElements());

        ExecutionResult result = executeAll(fixture, config, false);

        Cpu1ScratchBuffer scratchBuffer = assertInstanceOf(
                Cpu1ScratchBuffer.class,
                result.context().workspaceForNodeId(result.fixture().node().id())
        );
        assertEquals(36, scratchBuffer.requireF64Array(36).length);
        assertArrayEquals(
                new double[]{
                        1.0, 4.0, 3.0,
                        8.0, 20.0, 12.0,
                        7.0, 16.0, 9.0
                },
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedParallelVectorLayoutOpsUseConfiguredLaunchAndBulkPaths() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, new int[]{2, 2}, null, "right", DataType.FLOAT32);
        Fixture concatFixture = fixture(Tensor.concat(1, left, right));
        Cpu1PreparedArtifact concatArtifact = prepareRoot(concatFixture, Cpu1PrepareConfig.vectorParallel(3));
        Cpu1LayoutExecutableUnit concatExecutable = assertInstanceOf(
                Cpu1LayoutExecutableUnit.class,
                concatArtifact.executableUnit()
        );
        assertEquals(3, concatExecutable.preparedUnit().launchConfig().workerCount());
        assertEquals(
                Cpu1LayoutKernelId.CONCAT_INNER_AXIS_BLOCK_COPY_VECTOR,
                concatExecutable.preparedUnit().kernelId()
        );

        ExecutionResult concatResult = executeAll(concatFixture, Cpu1PrepareConfig.vectorParallel(3), false);
        assertArrayEquals(
                new float[]{1.0f, 2.0f, 10.0f, 20.0f, 3.0f, 4.0f, 30.0f, 40.0f},
                concatResult.context().runtimeTensorForNodeId(concatResult.fixture().node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );

        Tensor contiguousInput = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        ExecutionResult contiguousResult = executeAll(fixture(contiguousInput.contiguous()), Cpu1PrepareConfig.vectorParallel(2), false);
        assertArrayEquals(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                contiguousResult.context().runtimeTensorForNodeId(contiguousResult.fixture().node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void automaticLayoutDispatchUsesTunedVectorChunkConfig() {
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);
        int length = tuned.minVectorChunkSize() * 4;
        float[] data = new float[length];
        for (int i = 0; i < data.length; i++) {
            data[i] = i + 1.0f;
        }
        Tensor input = new Tensor(data, new int[]{length}, null, "input", DataType.FLOAT32);
        Fixture fixture = fixture(input.contiguous());
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.automatic(tuned, 4));
        Cpu1LayoutExecutableUnit executable = assertInstanceOf(
                Cpu1LayoutExecutableUnit.class,
                artifact.executableUnit()
        );
        assertEquals(Cpu1VectorizationKind.VECTOR, executable.preparedUnit().vectorizationKind());
        assertEquals(Cpu1LayoutKernelId.CONTIGUOUS_COPY_VECTOR, executable.preparedUnit().kernelId());
        assertEquals(4, executable.preparedUnit().launchConfig().workerCount());
        assertEquals(tuned.minVectorChunkSize(), executable.preparedUnit().launchConfig().chunkSize());

        ExecutionResult result = executeAll(fixture, Cpu1PrepareConfig.automatic(tuned, 4), false);

        assertArrayEquals(
                data,
                result.context().runtimeTensorForNodeId(result.fixture().node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentVectorConcatMaterializesNativeOutput() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{1, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 4.0f}, new int[]{1, 2}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(Tensor.concat(1, left, right));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.vectorMemorySegmentSingleThread());
        assertEquals(Cpu1LayoutKernelId.CONCAT_INNER_AXIS_BLOCK_COPY_VECTOR, artifact.preparedLayoutUnit().kernelId());
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 4),
                1.0e-6f
        );
    }

    @Test
    void preparedBfloat16VectorPadMaterializesConstantBorder() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f},
                new int[]{2, 2},
                null,
                "input",
                DataType.BFLOAT16
        );
        Fixture padFixture = fixture(input.pad(new int[]{1, 1}, new int[]{0, 1}, -1.0));
        Cpu1PreparedArtifact artifact = prepareRoot(padFixture, Cpu1PrepareConfig.vectorSingleThread());
        assertEquals(Cpu1LayoutKernelId.PAD_DENSE_INNER_BLOCK_COPY_VECTOR, artifact.preparedLayoutUnit().kernelId());
        ExecutionResult result = executeAll(
                padFixture,
                Cpu1PrepareConfig.vectorSingleThread(),
                false
        );

        assertArrayEquals(
                new float[]{
                        -1.0f, -1.0f, -1.0f, -1.0f,
                        -1.0f, 1.0f, 2.0f, -1.0f,
                        -1.0f, 3.0f, 4.0f, -1.0f
                },
                bf16ToF32(result.context().runtimeTensorForNodeId(result.fixture().node().id())),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentUnfold2dMaterializesNativeOutput() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, 2.0f, 3.0f,
                        4.0f, 5.0f, 6.0f,
                        7.0f, 8.0f, 9.0f
                },
                new int[]{1, 1, 3, 3},
                null,
                "input",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(input.unfold2d(Window2dOptions.of(2, 2)));
        Cpu1PreparedArtifact artifact = prepareRoot(fixture, Cpu1PrepareConfig.scalarMemorySegmentSingleThread());
        ExecutionContext context = context(fixture, Map.of(fixture.node().id(), metadata(fixture.node(), artifact)));
        attachNativeLeaves(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata(fixture.node(), artifact), context);

        assertArrayEquals(
                new float[]{
                        1.0f, 2.0f, 4.0f, 5.0f,
                        2.0f, 3.0f, 5.0f, 6.0f,
                        4.0f, 5.0f, 7.0f, 8.0f,
                        5.0f, 6.0f, 8.0f, 9.0f
                },
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 16),
                1.0e-6f
        );
    }

    private static Cpu1PreparedArtifact prepareRoot(Fixture fixture, Cpu1PrepareConfig config) {
        return new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
    }

    private static void assertLayoutKernel(Cpu1PreparedArtifact artifact, Cpu1LayoutKernelId expected) {
        Cpu1LayoutExecutableUnit executable = assertInstanceOf(Cpu1LayoutExecutableUnit.class, artifact.executableUnit());
        assertEquals(expected, artifact.preparedLayoutUnit().kernelId());
        assertSame(artifact.preparedLayoutUnit(), executable.preparedUnit());
    }

    private static ExecutionResult executeAll(Fixture fixture, Cpu1PrepareConfig config, boolean attachNativeLeaves) {
        Cpu1NodePreparer preparer = new Cpu1NodePreparer();
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = new LinkedHashMap<>();
        for (CompiledNode node : fixture.nodes()) {
            if (node.operation() == null) {
                continue;
            }
            Cpu1PreparedArtifact artifact = preparer.prepare(node, fixture.descriptorIndex(), config);
            metadataIndex.put(node.id(), metadata(node, artifact));
        }
        ExecutionContext context = context(fixture, metadataIndex);
        if (attachNativeLeaves) {
            attachNativeLeaves(context, fixture);
        }
        Cpu1Backend backend = new Cpu1Backend();
        for (CompiledNode node : fixture.nodes()) {
            CompiledNodeExecutionMetadata metadata = metadataIndex.get(node.id());
            if (metadata != null) {
                backend.execute(node, metadata, context);
            }
        }
        return new ExecutionResult(fixture, context);
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

    private static void attachNativeLeaves(ExecutionContext context, Fixture fixture) {
        NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
        for (CompiledNode node : fixture.nodes()) {
            if (!node.leaf()) {
                continue;
            }
            Tensor tensor = context.runtimeTensorForNodeId(node.id());
            NativeTensorStorage storage = storageFactory.allocate(
                    tensor.getDataType(),
                    tensor.getFlatDataSize(),
                    "cpu1-layout-test-input-" + node.id()
            );
            copyToNative(tensor, storage);
            context.attachNativeStorage(node.id(), storage, "cpu1 layout test native leaf");
        }
    }

    private static void copyToNative(Tensor tensor, NativeTensorStorage storage) {
        MemorySegment segment = storage.segment();
        switch (tensor.getDataType()) {
            case FLOAT32 -> {
                float[] source = TensorInternalAccess.float32Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    segment.set(JAVA_FLOAT, (long) i * Float.BYTES, source[i]);
                }
            }
            case FLOAT64 -> {
                double[] source = TensorInternalAccess.float64Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    segment.set(JAVA_DOUBLE, (long) i * Double.BYTES, source[i]);
                }
            }
            case BFLOAT16 -> {
                short[] source = TensorInternalAccess.bfloat16Data(tensor);
                for (int i = 0; i < source.length; i++) {
                    segment.set(JAVA_SHORT, (long) i * Short.BYTES, source[i]);
                }
            }
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout test native dtype "
                    + tensor.getDataType());
        }
        storage.markModified();
    }

    private static float[] readNativeF32(NativeTensorStorage storage, int length) {
        float[] out = new float[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
        return out;
    }

    private static double[] readNativeF64(NativeTensorStorage storage, int length) {
        double[] out = new double[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_DOUBLE, (long) i * Double.BYTES);
        }
        return out;
    }

    private static float[] bf16ToF32(Tensor tensor) {
        short[] source = TensorInternalAccess.bfloat16Data(tensor);
        float[] out = new float[source.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = TensorDTypeOps.fromBFloat16Bits(source[i]);
        }
        return out;
    }

    private static Tensor sliceBackward(Tensor updates, int[] inputShape) {
        return TensorPrimitiveBuilder.unaryNoGrad(
                updates,
                inputShape,
                new sliceBackward(new int[]{0, 1}, new int[]{0, 1}, new int[]{1, 2}, inputShape),
                "slice_backward",
                updates.getDataType()
        );
    }

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }

    private record ExecutionResult(Fixture fixture, ExecutionContext context) {
    }
}
