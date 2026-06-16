package backend.cpu1;

import backend.ComputeBackend;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.elementwise.Cpu1ElementwiseKernelId;
import backend.cpu1.kernels.Cpu1KernelRegistry;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.Cpu1VectorizationKind;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1ParallelLaunch;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.plan.Cpu1IterationPlan;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedElementwiseUnit;
import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPlan;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;
import utils.FastTranscendentals;
import utils.SpecialFunctions;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class Cpu1ExecutionContractTest {
    @Test
    void preparedAddRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "right", DataType.FLOAT32);
        Tensor out = left.add(right);
        Fixture fixture = fixture(out);

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedAddReadsRank3StridedInputViews() {
        float[] leftData = new float[]{
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f
        };
        float[] rightData = new float[]{
                10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f,
                70.0f, 80.0f, 90.0f, 100.0f, 110.0f, 120.0f
        };
        Tensor left = new Tensor(leftData, new int[]{2, 2, 3}, null, "left", DataType.FLOAT32).permute(0, 2, 1);
        Tensor right = new Tensor(rightData, new int[]{2, 2, 3}, null, "right", DataType.FLOAT32).permute(0, 2, 1);
        Fixture fixture = fixture(left.add(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.ADD_F32_ARRAY_STRIDED_RANK3_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedRank3PermutedAdd(leftData, rightData),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMulReadsRank3StridedInputViews() {
        double[] leftData = new double[]{
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0
        };
        double[] rightData = new double[]{
                0.5, 1.5, 2.5, 3.5, 4.5, 5.5,
                6.5, 7.5, 8.5, 9.5, 10.5, 11.5
        };
        Tensor left = new Tensor(leftData, new int[]{2, 2, 3}, null, "left", DataType.FLOAT64).permute(0, 2, 1);
        Tensor right = new Tensor(rightData, new int[]{2, 2, 3}, null, "right", DataType.FLOAT64).permute(0, 2, 1);
        Fixture fixture = fixture(left.mul(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.MUL_F64_ARRAY_STRIDED_RANK3_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedRank3PermutedMul(leftData, rightData),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedReluReadsRank3StridedInputView() {
        float[] data = new float[]{
                1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f,
                7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f
        };
        Tensor base = new Tensor(data, new int[]{2, 2, 3}, null, "base", DataType.FLOAT32);
        Tensor transposed = base.permute(0, 2, 1);
        Tensor out = transposed.relu();
        Fixture fixture = fixture(out);

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.RELU_F32_ARRAY_STRIDED_RANK3_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedRank3PermutedRelu(data),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedReluReadsRank4StridedInputView() {
        float[] data = new float[24];
        for (int i = 0; i < data.length; i++) {
            data[i] = i % 2 == 0 ? i + 1.0f : -(i + 1.0f);
        }
        Tensor base = new Tensor(data, new int[]{2, 3, 2, 2}, null, "base", DataType.FLOAT32);
        Tensor transposed = base.permute(0, 2, 3, 1);
        Tensor out = transposed.relu();
        Fixture fixture = fixture(out);

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.RELU_F32_ARRAY_STRIDED_RANK4_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expectedRank4PermutedRelu(data),
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedReluReadsGenericStridedInputView() {
        double[] data = new double[]{1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
        Tensor base = new Tensor(data, new int[]{1, 1, 2, 1, 3}, null, "base", DataType.FLOAT64);
        Tensor transposed = base.permute(4, 2, 0, 1, 3);
        Tensor out = transposed.relu();
        Fixture fixture = fixture(out);

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.RELU_F64_ARRAY_STRIDED_GENERIC_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new double[]{1.0, 0.0, 0.0, 5.0, 3.0, 0.0},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedUnitCanOverrideSemanticNodeOperationWhenRunDirectly() {
        Tensor left = new Tensor(new float[]{1.0f, -2.0f, 3.0f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "right", DataType.FLOAT32);
        Tensor add = left.add(right);
        Fixture fixture = fixture(add);
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();
        Cpu1PreparedElementwiseUnit reluLeftInsteadOfAdd = new Cpu1PreparedElementwiseUnit(
                fixture.node().id(),
                List.of(fixture.node().inputIds().getFirst()),
                fixture.node().id(),
                Operation.OpType.RELU,
                DataType.FLOAT32,
                Cpu1IterationPlan.contiguous(fixture.node().flatDataSize(), fixture.node().shape()),
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                registry.resolve(Operation.OpType.RELU, DataType.FLOAT32),
                new Cpu1SingleThreadLaunch()
        );
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(
                fixture.node(),
                new Cpu1PreparedArtifact(reluLeftInsteadOfAdd)
        );
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{1.0f, 0.0f, 3.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparerRejectsMismatchedInputDtypeBeforeRuntime() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "right", DataType.FLOAT64);
        Tensor out = left.add(right);
        Fixture fixture = fixture(out);

        assertThrows(
                UnsupportedOperationException.class,
                () -> new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex())
        );
    }

    @Test
    void preparedAddSupportsRank2BroadcastInput() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(bias));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.ADD_F32_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f},
                executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedElementwiseUnitStoresInputAndOutputAccessPlans() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{1, 3}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(bias));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        Cpu1PreparedElementwiseUnit unit = artifact.preparedUnit();

        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.outputAccessPlan().kind());
        assertArrayEquals(new int[]{2, 3}, unit.outputAccessPlan().shape());
        assertArrayEquals(new int[]{3, 1}, unit.outputAccessPlan().strides());
        assertEquals(2, unit.inputAccessPlans().size());
        assertEquals(Cpu1StorageAccessKind.DENSE_CONTIGUOUS, unit.inputAccessPlan(0).kind());
        assertArrayEquals(new int[]{3, 1}, unit.inputAccessPlan(0).strides());
        assertEquals(Cpu1StorageAccessKind.BROADCAST, unit.inputAccessPlan(1).kind());
        assertArrayEquals(new int[]{0, 1}, unit.inputAccessPlan(1).strides());
    }

    @Test
    void preparedBinaryOpsSupportRank2BroadcastInputs() {
        Tensor left = new Tensor(
                new float[]{8.0f, 9.0f, 4.0f, 16.0f, 25.0f, 36.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(new float[]{2.0f, 3.0f, 4.0f}, new int[]{3}, null, "right", DataType.FLOAT32);

        Fixture mulFixture = fixture(left.mul(right));
        Cpu1PreparedArtifact mulArtifact = new Cpu1NodePreparer().prepare(
                mulFixture.node(),
                mulFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.MUL_F32_ARRAY_BROADCAST_INNER_VECTOR, mulArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                new float[]{16.0f, 27.0f, 16.0f, 32.0f, 75.0f, 144.0f},
                executeCpu1(mulFixture, Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{6.0f, 6.0f, 0.0f, 14.0f, 22.0f, 32.0f},
                executeCpu1(fixture(left.sub(right)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{4.0f, 3.0f, 1.0f, 8.0f, 25.0f / 3.0f, 9.0f},
                executeCpu1(fixture(left.div(right)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{2.0f, 3.0f, 4.0f, 2.0f, 3.0f, 4.0f},
                executeCpu1(fixture(left.min(right)), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{8.0f, 9.0f, 4.0f, 16.0f, 25.0f, 36.0f},
                executeCpu1(fixture(left.max(right)), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{64.0f, 729.0f, 256.0f, 256.0f, 15625.0f, 1679616.0f},
                executeCpu1(fixture(left.pow(right)), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-1f
        );
    }

    @Test
    void preparedUnaryOpsSupportBroadcastViews() {
        Tensor input = new Tensor(new float[]{-2.0f, 0.5f, 3.0f}, new int[]{1, 3}, null, "input", DataType.FLOAT32)
                .expand(2, 3);

        Fixture reluFixture = fixture(input.relu());
        Cpu1PreparedArtifact reluArtifact = new Cpu1NodePreparer().prepare(
                reluFixture.node(),
                reluFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.RELU_F32_ARRAY_BROADCAST_INNER_VECTOR, reluArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                new float[]{0.0f, 0.5f, 3.0f, 0.0f, 0.5f, 3.0f},
                executeCpu1(reluFixture, Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-2.0f, 0.5f, 2.0f, -2.0f, 0.5f, 2.0f},
                executeCpu1(fixture(input.clampMax(2.0)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{
                        SpecialFunctions.erf(-2.0f),
                        SpecialFunctions.erf(0.5f),
                        SpecialFunctions.erf(3.0f),
                        SpecialFunctions.erf(-2.0f),
                        SpecialFunctions.erf(0.5f),
                        SpecialFunctions.erf(3.0f)
                },
                executeCpu1(fixture(input.erf()), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedVectorBinaryOpsSupportScalarBroadcastInput() {
        Tensor left = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0, 5.0}, new int[]{5}, null, "left", DataType.FLOAT64);
        Tensor scalar = new Tensor(new double[]{10.0}, new int[]{1}, null, "scalar", DataType.FLOAT64);
        Fixture fixture = fixture(left.add(scalar));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.ADD_F64_ARRAY_BROADCAST_INNER_VECTOR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new double[]{11.0, 12.0, 13.0, 14.0, 15.0},
                executeCpu1(fixture, Cpu1PrepareConfig.vectorSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedCompareSupportsBroadcastInput() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(new float[]{2.0f, 2.0f, 5.0f}, new int[]{3}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.greaterThan(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.GT_F32_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                boolBytes(false, false, false, true, true, true),
                executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedLogicalOpsSupportBroadcastInputs() {
        Tensor left = new Tensor(
                boolBytes(true, false, true, false, true, false),
                new int[]{2, 3},
                null,
                "left",
                DataType.BOOL
        );
        Tensor right = new Tensor(boolBytes(true, false, true), new int[]{3}, null, "right", DataType.BOOL);

        Fixture andFixture = fixture(left.logicalAnd(right));
        Cpu1PreparedArtifact andArtifact = new Cpu1NodePreparer().prepare(
                andFixture.node(),
                andFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.LOGICAL_AND_BOOL_ARRAY_STRIDED_RANK2_SCALAR, andArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                boolBytes(true, false, true, false, false, false),
                executeCpu1(andFixture, Cpu1PrepareConfig.vectorSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(true, false, true, true, true, true),
                executeCpu1(fixture(left.logicalOr(right)), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedVectorAddRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f},
                new int[]{9},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor right = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f},
                new int[]{9},
                null,
                "right",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(left.add(right));
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                Cpu1VectorizationKind.VECTOR,
                Cpu1LaunchConfig.singleThread()
        );

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f, 44.0f, 55.0f, 66.0f, 77.0f, 88.0f, 99.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedParallelVectorAddRunsAgainstRuntimeTensors() {
        float[] leftData = new float[33];
        float[] rightData = new float[33];
        float[] expected = new float[33];
        for (int i = 0; i < expected.length; i++) {
            leftData[i] = i + 1.0f;
            rightData[i] = (i + 1.0f) * 10.0f;
            expected[i] = leftData[i] + rightData[i];
        }
        Tensor left = new Tensor(leftData, new int[]{33}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(rightData, new int[]{33}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorParallel(4)
        );
        assertInstanceOf(Cpu1ParallelLaunch.class, artifact.preparedUnit().launchPolicy());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                expected,
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void automaticDispatchDecisionIsStoredOnPreparedUnit() {
        CpuKernelConfig tuned = new CpuKernelConfig(4, 32, 32, 32, 16, 64);
        int length = tuned.cheapParallelMinSize();
        Tensor left = new Tensor(new float[length], new int[]{length}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[length], new int[]{length}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.automatic(tuned, 4)
        );

        assertEquals(Cpu1VectorizationKind.VECTOR, artifact.preparedUnit().dispatchDecision().requestedVectorizationKind());
        assertEquals(4, artifact.preparedUnit().dispatchDecision().plannedWorkers());
        assertEquals(tuned.minVectorChunkSize(), artifact.preparedUnit().dispatchDecision().vectorChunkSize());
        assertEquals(
                artifact.preparedUnit().dispatchDecision().vectorChunkSize(),
                artifact.preparedUnit().dispatchDecision().launchConfig().chunkSize()
        );
        assertInstanceOf(Cpu1ParallelLaunch.class, artifact.preparedUnit().launchPolicy());
    }

    @Test
    void preparedParallelReluRunsAgainstRuntimeTensors() {
        Tensor input = new Tensor(
                new double[]{1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0},
                new int[]{9},
                null,
                "input",
                DataType.FLOAT64
        );
        Fixture fixture = fixture(input.relu());
        Cpu1PrepareConfig config = new Cpu1PrepareConfig(
                Cpu1VectorizationKind.SCALAR,
                Cpu1LaunchConfig.parallel(3)
        );

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new double[]{1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedBfloat16AddRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{1.5f, -2.0f, 3.25f}, new int[]{3}, null, "left", DataType.BFLOAT16);
        Tensor right = new Tensor(new float[]{2.25f, 10.0f, -1.25f}, new int[]{3}, null, "right", DataType.BFLOAT16);

        Tensor actual = executeCpu1(fixture(left.add(right)), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(
                bf16Bits(3.75f, 8.0f, 2.0f),
                actual.toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedBfloat16MulRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{1.5f, -2.0f, 3.0f}, new int[]{3}, null, "left", DataType.BFLOAT16);
        Tensor right = new Tensor(new float[]{2.0f, 4.0f, -0.5f}, new int[]{3}, null, "right", DataType.BFLOAT16);

        Tensor actual = executeCpu1(fixture(left.mul(right)), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(
                bf16Bits(3.0f, -8.0f, -1.5f),
                actual.toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedBfloat16ReluRunsAgainstRuntimeTensors() {
        Tensor input = new Tensor(new float[]{1.5f, -2.0f, 0.0f, 3.25f}, new int[]{4}, null, "input", DataType.BFLOAT16);

        Tensor actual = executeCpu1(fixture(input.relu()), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(
                bf16Bits(1.5f, 0.0f, 0.0f, 3.25f),
                actual.toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedBfloat16StridedAddRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.BFLOAT16
        ).permute(1, 0);
        Tensor right = new Tensor(
                new float[]{10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f},
                new int[]{2, 3},
                null,
                "right",
                DataType.BFLOAT16
        ).permute(1, 0);

        Tensor actual = executeCpu1(fixture(left.add(right)), Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(
                bf16Bits(11.0f, 44.0f, 22.0f, 55.0f, 33.0f, 66.0f),
                actual.toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedSubAndDivRunAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{8.0f, 12.0f, -6.0f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2.0f, 3.0f, -2.0f}, new int[]{3}, null, "right", DataType.FLOAT32);

        assertArrayEquals(
                new float[]{6.0f, 9.0f, -4.0f},
                executeCpu1(fixture(left.sub(right)), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{4.0f, 4.0f, 3.0f},
                executeCpu1(fixture(left.div(right)), Cpu1PrepareConfig.scalarSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedMinMaxAndPowTensorRunAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{2.0f, 3.0f, 4.0f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{3.0f, 2.0f, 0.5f}, new int[]{3}, null, "right", DataType.FLOAT32);

        assertArrayEquals(
                new float[]{2.0f, 2.0f, 0.5f},
                executeCpu1(fixture(left.min(right)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{3.0f, 3.0f, 4.0f},
                executeCpu1(fixture(left.max(right)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{8.0f, 9.0f, 2.0f},
                executeCpu1(fixture(left.pow(right)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedNewUnaryOpsRunAgainstRuntimeTensors() {
        Tensor input = new Tensor(new double[]{0.25, 1.0, 4.0}, new int[]{3}, null, "input", DataType.FLOAT64);

        assertArrayEquals(
                new double[]{-0.25, -1.0, -4.0},
                executeCpu1(fixture(input.neg()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{Math.exp(0.25), Math.exp(1.0), Math.exp(4.0)},
                executeCpu1(fixture(input.exp()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{SpecialFunctions.erf(0.25), SpecialFunctions.erf(1.0), SpecialFunctions.erf(4.0)},
                executeCpu1(fixture(input.erf()), Cpu1PrepareConfig.vectorSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{Math.log(0.25), Math.log(1.0), Math.log(4.0)},
                executeCpu1(fixture(input.log()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{Math.tanh(0.25), Math.tanh(1.0), Math.tanh(4.0)},
                executeCpu1(fixture(input.tanh()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{
                        1.0d / (1.0d + Math.exp(-0.25)),
                        1.0d / (1.0d + Math.exp(-1.0)),
                        1.0d / (1.0d + Math.exp(-4.0))
                },
                executeCpu1(fixture(input.sigmoid()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
        assertArrayEquals(
                new double[]{0.5, 1.0, 2.0},
                executeCpu1(fixture(input.sqrt()), Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedAdditionalUnaryOpsRunAgainstRuntimeTensors() {
        Tensor input = new Tensor(new float[]{-2.75f, -0.0f, 3.25f}, new int[]{3}, null, "input", DataType.FLOAT32);

        assertArrayEquals(
                new float[]{2.75f, 0.0f, 3.25f},
                executeCpu1(fixture(input.abs()), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-1.0f / 2.75f, Float.NEGATIVE_INFINITY, 1.0f / 3.25f},
                executeCpu1(fixture(input.inv()), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-3.0f, -0.0f, 3.0f},
                executeCpu1(fixture(input.floor()), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-2.0f, -0.0f, 4.0f},
                executeCpu1(fixture(input.ceil()), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );

        Fixture signFixture = fixture(input.sign());
        Cpu1PreparedArtifact signArtifact = new Cpu1NodePreparer().prepare(
                signFixture.node(),
                signFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.SIGN_F32_ARRAY_CONTIGUOUS_SCALAR, signArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                new float[]{-1.0f, 0.0f, 1.0f},
                executeCpu1(signFixture, Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedScalarParameterUnaryOpsRunAgainstRuntimeTensors() {
        Tensor input = new Tensor(new float[]{-2.0f, 0.5f, 3.0f}, new int[]{3}, null, "input", DataType.FLOAT32);

        assertArrayEquals(
                new float[]{-5.0f, 1.25f, 7.5f},
                executeCpu1(fixture(input.mul(2.5)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-8.0f, 0.125f, 27.0f},
                executeCpu1(fixture(input.pow(3.0)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-1.0f, 0.5f, 3.0f},
                executeCpu1(fixture(input.clampMin(-1.0)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
        assertArrayEquals(
                new float[]{-2.0f, 0.5f, 1.0f},
                executeCpu1(fixture(input.clampMax(1.0)), Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );

        Fixture powFixture = fixture(input.pow(3.0));
        Cpu1PreparedArtifact powArtifact = new Cpu1NodePreparer().prepare(
                powFixture.node(),
                powFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.POW_F32_ARRAY_CONTIGUOUS_SCALAR, powArtifact.preparedUnit().kernelId());
        assertEquals(3.0f, powArtifact.preparedUnit().scalarParameterF32(), 0.0f);
    }

    @Test
    void preparedBfloat16ScalarParameterUnaryOpsRunAgainstRuntimeTensors() {
        Tensor input = new Tensor(new float[]{-2.0f, 0.5f, 3.0f}, new int[]{3}, null, "input", DataType.BFLOAT16);

        assertArrayEquals(
                bf16Bits(-5.0f, 1.25f, 7.5f),
                executeCpu1(fixture(input.mul(2.5)), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
        assertArrayEquals(
                bf16Bits(-1.0f, 0.5f, 3.0f),
                executeCpu1(fixture(input.clampMin(-1.0)), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
        assertArrayEquals(
                bf16Bits(-2.0f, 0.5f, 1.0f),
                executeCpu1(fixture(input.clampMax(1.0)), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedFastApproxConfigSelectsFastKernelsAtPrepareTime() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 2.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Cpu1PrepareConfig config = Cpu1PrepareConfig.vectorSingleThread().withApproximation(true, true);

        Fixture expFixture = fixture(input.exp());
        Cpu1PreparedArtifact expArtifact = new Cpu1NodePreparer().prepare(
                expFixture.node(),
                expFixture.descriptorIndex(),
                config
        );
        assertEquals(Cpu1ElementwiseKernelId.FAST_EXP_F32_ARRAY_CONTIGUOUS_SCALAR, expArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                new float[]{
                        FastTranscendentals.fastExpF32(-1.0f),
                        FastTranscendentals.fastExpF32(0.0f),
                        FastTranscendentals.fastExpF32(2.0f)
                },
                executeCpu1(expFixture, config).toFloat32ArrayCopy(),
                1.0e-6f
        );

        Fixture tanhFixture = fixture(input.tanh());
        Cpu1PreparedArtifact tanhArtifact = new Cpu1NodePreparer().prepare(
                tanhFixture.node(),
                tanhFixture.descriptorIndex(),
                config
        );
        assertEquals(Cpu1ElementwiseKernelId.FAST_TANH_F32_ARRAY_CONTIGUOUS_SCALAR, tanhArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                new float[]{
                        FastTranscendentals.fastTanhF32(-1.0f),
                        FastTranscendentals.fastTanhF32(0.0f),
                        FastTranscendentals.fastTanhF32(2.0f)
                },
                executeCpu1(tanhFixture, config).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedBfloat16SubDivAndNegRunAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{8.0f, 12.0f, -6.0f}, new int[]{3}, null, "left", DataType.BFLOAT16);
        Tensor right = new Tensor(new float[]{2.0f, 3.0f, -2.0f}, new int[]{3}, null, "right", DataType.BFLOAT16);

        assertArrayEquals(
                bf16Bits(6.0f, 9.0f, -4.0f),
                executeCpu1(fixture(left.sub(right)), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
        assertArrayEquals(
                bf16Bits(4.0f, 4.0f, 3.0f),
                executeCpu1(fixture(left.div(right)), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
        assertArrayEquals(
                bf16Bits(-8.0f, -12.0f, 6.0f),
                executeCpu1(fixture(left.neg()), Cpu1PrepareConfig.scalarSingleThread()).toBFloat16BitsArrayCopy()
        );
    }

    @Test
    void preparedCompareOpsRunAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{4}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2.0f, 2.0f, 1.0f, 4.0f}, new int[]{4}, null, "right", DataType.FLOAT32);

        Fixture gtFixture = fixture(left.greaterThan(right));
        Cpu1PreparedArtifact gtArtifact = new Cpu1NodePreparer().prepare(
                gtFixture.node(),
                gtFixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.GT_F32_ARRAY_CONTIGUOUS_SCALAR, gtArtifact.preparedUnit().kernelId());
        assertArrayEquals(
                boolBytes(false, false, true, false),
                executeCpu1(gtFixture, Cpu1PrepareConfig.vectorSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(true, true, false, true),
                executeCpu1(fixture(left.lessOrEqual(right)), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(false, true, false, true),
                executeCpu1(fixture(left.equalTo(right)), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(true, false, true, false),
                executeCpu1(fixture(left.notEqualTo(right)), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedLogicalOpsRunAgainstRuntimeTensors() {
        Tensor left = new Tensor(boolBytes(true, true, false, false), new int[]{4}, null, "left", DataType.BOOL);
        Tensor right = new Tensor(boolBytes(true, false, true, false), new int[]{4}, null, "right", DataType.BOOL);

        assertArrayEquals(
                boolBytes(true, false, false, false),
                executeCpu1(fixture(left.logicalAnd(right)), Cpu1PrepareConfig.vectorSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(true, true, true, false),
                executeCpu1(fixture(left.logicalOr(right)), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
        assertArrayEquals(
                boolBytes(false, false, true, true),
                executeCpu1(fixture(left.logicalNot()), Cpu1PrepareConfig.scalarSingleThread()).toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedWhereRunsAgainstRuntimeTensors() {
        Tensor condition = new Tensor(boolBytes(true, false, true, false), new int[]{4}, null, "condition", DataType.BOOL);
        Tensor ifTrue = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new int[]{4}, null, "ifTrue", DataType.FLOAT32);
        Tensor ifFalse = new Tensor(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, new int[]{4}, null, "ifFalse", DataType.FLOAT32);
        Fixture fixture = fixture(Tensor.where(condition, ifTrue, ifFalse));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.WHERE_F32_FROM_F32_F32_ARRAY_CONTIGUOUS_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new float[]{1.0f, 20.0f, 3.0f, 40.0f},
                executeCpu1(fixture, Cpu1PrepareConfig.vectorSingleThread()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedWhereReadsRank2StridedInputViews() {
        Tensor condition = new Tensor(
                boolBytes(true, false, false, true, true, false),
                new int[]{2, 3},
                null,
                "condition",
                DataType.BOOL
        ).permute(1, 0);
        Tensor ifTrue = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                new int[]{2, 3},
                null,
                "ifTrue",
                DataType.FLOAT64
        ).permute(1, 0);
        Tensor ifFalse = new Tensor(
                new double[]{10.0, 20.0, 30.0, 40.0, 50.0, 60.0},
                new int[]{2, 3},
                null,
                "ifFalse",
                DataType.FLOAT64
        ).permute(1, 0);
        Fixture fixture = fixture(Tensor.where(condition, ifTrue, ifFalse));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.WHERE_F64_FROM_F64_F64_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new double[]{1.0, 4.0, 20.0, 5.0, 30.0, 60.0},
                executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedWhereSupportsBroadcastInputs() {
        Tensor condition = new Tensor(boolBytes(true, false), new int[]{2, 1}, null, "condition", DataType.BOOL);
        Tensor ifTrue = new Tensor(
                new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                new int[]{2, 3},
                null,
                "ifTrue",
                DataType.FLOAT64
        );
        Tensor ifFalse = new Tensor(new double[]{10.0, 20.0, 30.0}, new int[]{1, 3}, null, "ifFalse", DataType.FLOAT64);
        Fixture fixture = fixture(Tensor.where(condition, ifTrue, ifFalse));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.WHERE_F64_FROM_F64_F64_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new double[]{1.0, 2.0, 3.0, 10.0, 20.0, 30.0},
                executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedWherePromotesBranchDtypesAtPrepareTime() {
        Tensor condition = new Tensor(boolBytes(true, false), new int[]{2}, null, "condition", DataType.BOOL);
        Tensor ifTrue = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "ifTrue", DataType.FLOAT32);
        Tensor ifFalse = new Tensor(new double[]{10.0, 20.0}, new int[]{2}, null, "ifFalse", DataType.FLOAT64);
        Fixture fixture = fixture(Tensor.where(condition, ifTrue, ifFalse));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.WHERE_F64_FROM_F32_F64_ARRAY_CONTIGUOUS_SCALAR, artifact.preparedUnit().kernelId());

        assertArrayEquals(
                new double[]{1.0, 20.0},
                executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread()).toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedMemorySegmentWhereRunsAgainstRuntimeTensors() {
        Tensor condition = new Tensor(boolBytes(true, false, true), new int[]{3}, null, "condition", DataType.BOOL);
        Tensor ifTrue = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3}, null, "ifTrue", DataType.FLOAT32);
        Tensor ifFalse = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "ifFalse", DataType.FLOAT32);
        Fixture fixture = fixture(Tensor.where(condition, ifTrue, ifFalse));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.WHERE_F32_FROM_F32_F32_SEGMENT_CONTIGUOUS_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{1.0f, 20.0f, 3.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 3),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentElementwiseReusesNativeOutputStorage() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{3}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(right));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);
        NativeTensorStorage first = context.nativeStorageForNodeId(fixture.node().id());
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        NativeTensorStorage second = context.nativeStorageForNodeId(fixture.node().id());

        assertSame(first, second);
        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f},
                readNativeF32(second, 3),
                1.0e-6f
        );
        assertEquals(0, context.cpuMaterializationTraceCount());
    }

    @Test
    void preparedCompareReadsRank2StridedInputViews() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        ).permute(1, 0);
        Tensor right = new Tensor(
                new float[]{0.0f, 3.0f, 2.0f, 5.0f, 4.0f, 7.0f},
                new int[]{2, 3},
                null,
                "right",
                DataType.FLOAT32
        ).permute(1, 0);
        Fixture fixture = fixture(left.greaterOrEqual(right));

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.GE_F32_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());

        Tensor actual = executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(
                boolBytes(true, false, false, true, true, false),
                actual.toBoolByteArrayCopy()
        );
    }

    @Test
    void preparedMemorySegmentVectorAddRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, new int[]{5}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{10.0f, 20.0f, 30.0f, 40.0f, 50.0f}, new int[]{5}, null, "right", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(right));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorMemorySegmentSingleThread()
        );
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f, 44.0f, 55.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 5),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentVectorDivRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(new double[]{8.0, 12.0, -6.0, 10.0, 2.5}, new int[]{5}, null, "left", DataType.FLOAT64);
        Tensor right = new Tensor(new double[]{2.0, 3.0, -2.0, 4.0, 0.5}, new int[]{5}, null, "right", DataType.FLOAT64);
        Fixture fixture = fixture(left.div(right));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorMemorySegmentSingleThread()
        );
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new double[]{4.0, 4.0, 3.0, 2.5, 5.0},
                readNativeF64(context.nativeStorageForNodeId(fixture.node().id()), 5),
                1.0e-12
        );
    }

    @Test
    void preparedMemorySegmentAddSupportsBroadcastInput() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(bias));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.ADD_F32_SEGMENT_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 6),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentVectorAddSupportsBroadcastInput() {
        Tensor left = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "left",
                DataType.FLOAT32
        );
        Tensor bias = new Tensor(new float[]{10.0f, 20.0f, 30.0f}, new int[]{3}, null, "bias", DataType.FLOAT32);
        Fixture fixture = fixture(left.add(bias));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.vectorMemorySegmentSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.ADD_F32_SEGMENT_BROADCAST_INNER_VECTOR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{11.0f, 22.0f, 33.0f, 14.0f, 25.0f, 36.0f},
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 6),
                1.0e-6f
        );
    }

    @Test
    void preparedMemorySegmentErfRunsAgainstRuntimeTensors() {
        Tensor input = new Tensor(new float[]{-1.0f, 0.0f, 2.0f}, new int[]{3}, null, "input", DataType.FLOAT32);
        Fixture fixture = fixture(input.erf());
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        assertEquals(Cpu1ElementwiseKernelId.ERF_F32_SEGMENT_CONTIGUOUS_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{
                        SpecialFunctions.erf(-1.0f),
                        SpecialFunctions.erf(0.0f),
                        SpecialFunctions.erf(2.0f)
                },
                readNativeF32(context.nativeStorageForNodeId(fixture.node().id()), 3),
                1.0e-6f
        );
    }

    @Test
    void preparedBfloat16MemorySegmentStridedReluRunsAgainstRuntimeTensors() {
        Tensor input = new Tensor(
                new float[]{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.BFLOAT16
        ).permute(1, 0);
        Fixture fixture = fixture(input.relu());
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarMemorySegmentSingleThread()
        );
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                bf16Bits(1.0f, 0.0f, 0.0f, 5.0f, 3.0f, 0.0f),
                readNativeBF16(context.nativeStorageForNodeId(fixture.node().id()), 6)
        );
    }

    @Test
    void preparedParallelVectorMulRunsAgainstRuntimeTensors() {
        Tensor left = new Tensor(
                new double[]{
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                        9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0
                },
                new int[]{17},
                null,
                "left",
                DataType.FLOAT64
        );
        Tensor right = new Tensor(
                new double[]{
                        0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5,
                        8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5,
                        16.5
                },
                new int[]{17},
                null,
                "right",
                DataType.FLOAT64
        );

        Tensor actual = executeCpu1(fixture(left.mul(right)), Cpu1PrepareConfig.vectorParallel(4));

        assertArrayEquals(
                new double[]{
                        0.5, 3.0, 7.5, 14.0, 22.5, 33.0, 45.5, 60.0,
                        76.5, 95.0, 115.5, 138.0, 162.5, 189.0, 217.5, 248.0,
                        280.5
                },
                actual.toFloat64ArrayCopy(),
                1.0e-12
        );
    }

    @Test
    void preparedParallelVectorReluRunsAgainstRuntimeTensors() {
        Tensor input = new Tensor(
                new float[]{
                        1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
                        9.0f, -10.0f, 11.0f, -12.0f, 13.0f, -14.0f, 15.0f, -16.0f,
                        17.0f
                },
                new int[]{17},
                null,
                "input",
                DataType.FLOAT32
        );

        Tensor actual = executeCpu1(fixture(input.relu()), Cpu1PrepareConfig.vectorParallel(4));

        assertArrayEquals(
                new float[]{
                        1.0f, 0.0f, 3.0f, 0.0f, 5.0f, 0.0f, 7.0f, 0.0f,
                        9.0f, 0.0f, 11.0f, 0.0f, 13.0f, 0.0f, 15.0f, 0.0f,
                        17.0f
                },
                actual.toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedReluReadsStridedInputView() {
        Tensor base = new Tensor(new float[]{1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f},
                new int[]{3, 2}, null, "base", DataType.FLOAT32);
        Tensor transposed = base.permute(1, 0);
        Tensor out = transposed.relu();
        Fixture fixture = fixture(out);

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.RELU_F32_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{1.0f, 3.0f, 5.0f, 0.0f, 0.0f, 0.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void preparedSqrtReadsRank2StridedInputView() {
        Tensor base = new Tensor(new float[]{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f},
                new int[]{3, 2}, null, "base", DataType.FLOAT32);
        Tensor transposed = base.permute(1, 0);
        Fixture fixture = fixture(transposed.sqrt());

        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex());
        assertEquals(Cpu1ElementwiseKernelId.SQRT_F32_ARRAY_STRIDED_RANK2_SCALAR, artifact.preparedUnit().kernelId());
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);

        new Cpu1Backend().execute(fixture.node(), metadata, context);

        assertArrayEquals(
                new float[]{1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f},
                context.runtimeTensorForNodeId(fixture.node().id()).toFloat32ArrayCopy(),
                1.0e-6f
        );
    }

    @Test
    void kernelArgsExposeGenericOffsetPlansOnlyForGenericStridedUnits() {
        Tensor input = new Tensor(new float[]{1.0f, -2.0f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor output = new Tensor(new float[]{0.0f, 0.0f}, new int[]{2}, null, "output", DataType.FLOAT32);
        Cpu1KernelRegistry registry = new Cpu1KernelRegistry();
        Cpu1PreparedElementwiseUnit unit = new Cpu1PreparedElementwiseUnit(
                10,
                List.of(1),
                10,
                Operation.OpType.RELU,
                DataType.FLOAT32,
                Cpu1IterationPlan.contiguous(2, new int[]{2}),
                Cpu1LayoutKind.CONTIGUOUS,
                Cpu1StorageKind.JAVA_ARRAY,
                registry.resolve(Operation.OpType.RELU, DataType.FLOAT32),
                new Cpu1SingleThreadLaunch()
        );
        Cpu1KernelArgs args = new Cpu1KernelArgs(
                unit,
                List.of(Cpu1TensorView.fromTensor(input)),
                Cpu1TensorView.fromTensor(output)
        );

        assertThrows(IllegalStateException.class, () -> args.inputGenericOffsetPlan(0));
        assertThrows(IllegalStateException.class, args::outputGenericOffsetPlan);
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static float[] expectedRank4PermutedRelu(float[] data) {
        float[] expected = new float[data.length];
        int index = 0;
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 2; c1++) {
                for (int c2 = 0; c2 < 2; c2++) {
                    for (int c3 = 0; c3 < 3; c3++) {
                        int offset = c0 * 12 + c3 * 4 + c1 * 2 + c2;
                        expected[index++] = Math.max(0.0f, data[offset]);
                    }
                }
            }
        }
        return expected;
    }

    private static float[] expectedRank3PermutedRelu(float[] data) {
        float[] expected = new float[data.length];
        int index = 0;
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 3; c1++) {
                for (int c2 = 0; c2 < 2; c2++) {
                    int offset = c0 * 6 + c2 * 3 + c1;
                    expected[index++] = Math.max(0.0f, data[offset]);
                }
            }
        }
        return expected;
    }

    private static float[] expectedRank3PermutedAdd(float[] left, float[] right) {
        float[] expected = new float[left.length];
        int index = 0;
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 3; c1++) {
                for (int c2 = 0; c2 < 2; c2++) {
                    int offset = c0 * 6 + c2 * 3 + c1;
                    expected[index++] = left[offset] + right[offset];
                }
            }
        }
        return expected;
    }

    private static double[] expectedRank3PermutedMul(double[] left, double[] right) {
        double[] expected = new double[left.length];
        int index = 0;
        for (int c0 = 0; c0 < 2; c0++) {
            for (int c1 = 0; c1 < 3; c1++) {
                for (int c2 = 0; c2 < 2; c2++) {
                    int offset = c0 * 6 + c2 * 3 + c1;
                    expected[index++] = left[offset] * right[offset];
                }
            }
        }
        return expected;
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
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );
    }

    private static short[] bf16Bits(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static byte[] boolBytes(boolean... values) {
        byte[] out = new byte[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] ? (byte) 1 : (byte) 0;
        }
        return out;
    }

    private static void attachNativeInputs(ExecutionContext context, Fixture fixture) {
        NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
        for (int inputNodeId : fixture.node().inputIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            NativeTensorStorage storage = storageFactory.allocate(
                    tensor.getDataType(),
                    physicalElementCount(tensor),
                    "cpu1-test-input-" + inputNodeId
            );
            copyToNative(tensor, storage);
            context.attachNativeStorage(inputNodeId, storage, "cpu1 test native input");
        }
    }

    private static int physicalElementCount(Tensor tensor) {
        int maxOffset = tensor.getStorageOffsetUnsafe();
        int[] shape = tensor.getShapeUnsafe();
        int[] strides = tensor.getStridesUnsafe();
        for (int i = 0; i < shape.length; i++) {
            maxOffset += Math.max(0, shape[i] - 1) * strides[i];
        }
        return maxOffset + 1;
    }

    private static void copyToNative(Tensor tensor, NativeTensorStorage storage) {
        MemorySegment segment = storage.segment();
        switch (tensor.getDataType()) {
            case FLOAT32 -> {
                float[] source = TensorInternalAccess.float32Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_FLOAT, (long) i * Float.BYTES, source[i]);
                }
            }
            case FLOAT64 -> {
                double[] source = TensorInternalAccess.float64Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_DOUBLE, (long) i * Double.BYTES, source[i]);
                }
            }
            case BFLOAT16 -> {
                short[] source = TensorInternalAccess.bfloat16Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_SHORT, (long) i * Short.BYTES, source[i]);
                }
            }
            case BOOL -> {
                byte[] source = TensorInternalAccess.boolData(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_BYTE, (long) i * Byte.BYTES, source[i]);
                }
            }
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 test native copy dtype=" + tensor.getDataType());
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

    private static short[] readNativeBF16(NativeTensorStorage storage, int length) {
        short[] out = new short[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_SHORT, (long) i * Short.BYTES);
        }
        return out;
    }

    private static Tensor executeCpu1(Fixture fixture, Cpu1PrepareConfig config) {
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        CompiledNodeExecutionMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context.runtimeTensorForNodeId(fixture.node().id());
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

    private record Fixture(
            Tensor root,
            List<CompiledNode> nodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            CompiledNode node
    ) {
    }
}
