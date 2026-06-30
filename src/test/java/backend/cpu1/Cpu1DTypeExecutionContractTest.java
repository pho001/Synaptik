package backend.cpu1;

import backend.contract.ComputeBackend;
import runtime.memory.nativecpu.NativeCpuStorageFactory;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.kernels.Cpu1LayoutKind;
import backend.cpu1.kernels.dtype.Cpu1DTypeKernelId;
import backend.cpu1.kernels.dtype.cast.Cpu1CastLoops;
import backend.cpu1.launch.Cpu1LaunchConfig;
import backend.cpu1.launch.Cpu1SingleThreadLaunch;
import backend.cpu1.prepare.Cpu1NodePreparer;
import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import runtime.execution.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;
import runtime.execution.PreparedStepMetadata;
import runtime.execution.ExecutionState;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1DTypeExecutionContractTest {
    @Test
    void preparedCastConvertsF32AndBF16Arrays() {
        Tensor f32 = new Tensor(new float[]{1.25f, -2.5f, 0.0f}, new int[]{3}, null, "f32", DataType.FLOAT32);
        Tensor f32ToBf16 = executeCpu1(fixture(f32.cast(DataType.BFLOAT16)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(
                bf16Bits(1.25f, -2.5f, 0.0f),
                f32ToBf16.toBFloat16BitsArrayCopy()
        );

        Tensor bf16 = new Tensor(bf16Bits(3.5f, -4.25f, 0.5f), new int[]{3}, null, "bf16", DataType.BFLOAT16);
        Tensor bf16ToF32 = executeCpu1(fixture(bf16.cast(DataType.FLOAT32)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(
                new float[]{bf16(3.5f), bf16(-4.25f), bf16(0.5f)},
                bf16ToF32.toFloat32ArrayCopy(),
                0.0f
        );
    }

    @Test
    void preparedCastConvertsF64AndF32Arrays() {
        Tensor f64 = new Tensor(new double[]{1.25, -2.5, 1.0 / 3.0}, new int[]{3}, null, "f64", DataType.FLOAT64);
        Tensor f64ToF32 = executeCpu1(fixture(f64.cast(DataType.FLOAT32)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(new float[]{1.25f, -2.5f, (float) (1.0 / 3.0)}, f64ToF32.toFloat32ArrayCopy(), 0.0f);

        Tensor f32 = new Tensor(new float[]{2.0f, -3.5f}, new int[]{2}, null, "f32", DataType.FLOAT32);
        Tensor f32ToF64 = executeCpu1(fixture(f32.cast(DataType.FLOAT64)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(new double[]{2.0d, -3.5d}, f32ToF64.toFloat64ArrayCopy(), 0.0d);
    }

    @Test
    void preparedCastConvertsInt32AndF32Arrays() {
        Tensor i32 = new Tensor(new int[]{3, -4, 0}, new int[]{3}, null, "i32", DataType.INT32);
        Tensor i32ToF32 = executeCpu1(fixture(i32.cast(DataType.FLOAT32)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(new float[]{3.0f, -4.0f, 0.0f}, i32ToF32.toFloat32ArrayCopy(), 0.0f);

        Tensor f32 = new Tensor(new float[]{3.75f, -4.25f, 0.0f}, new int[]{3}, null, "f32", DataType.FLOAT32);
        Tensor f32ToI32 = executeCpu1(fixture(f32.cast(DataType.INT32)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(new int[]{3, -4, 0}, f32ToI32.toInt32ArrayCopy());
    }

    @Test
    void preparedCastConvertsBoolAndF32Arrays() {
        Tensor bool = new Tensor(boolBytes(true, false, true), new int[]{3}, null, "bool", DataType.BOOL);
        Tensor boolToF32 = executeCpu1(fixture(bool.cast(DataType.FLOAT32)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(new float[]{1.0f, 0.0f, 1.0f}, boolToF32.toFloat32ArrayCopy(), 0.0f);

        Tensor f32 = new Tensor(new float[]{0.0f, -2.0f, 3.0f}, new int[]{3}, null, "f32", DataType.FLOAT32);
        Tensor f32ToBool = executeCpu1(fixture(f32.cast(DataType.BOOL)), Cpu1PrepareConfig.scalarSingleThread());
        assertArrayEquals(boolBytes(false, true, true), f32ToBool.toBoolByteArrayCopy());
    }

    @Test
    void preparedCastReadsStridedInputInLogicalOrder() {
        Tensor base = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "base",
                DataType.FLOAT32
        );
        Fixture fixture = fixture(base.permute(1, 0).cast(DataType.FLOAT64));
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(
                fixture.node(),
                fixture.descriptorIndex(),
                Cpu1PrepareConfig.scalarSingleThread()
        );
        assertEquals(Cpu1DTypeKernelId.CAST_ARRAY_SCALAR, artifact.preparedDTypeUnit().kernelId());
        assertEquals(Cpu1LayoutKind.STRIDED_RANK2, artifact.preparedDTypeUnit().layoutKind());

        Tensor actual = executeCpu1(fixture, Cpu1PrepareConfig.scalarSingleThread());

        assertArrayEquals(new double[]{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}, actual.toFloat64ArrayCopy(), 0.0d);
    }

    @Test
    void castLoopWritesStridedOutputInLogicalOrder() {
        Tensor input = new Tensor(
                new float[]{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
                new int[]{2, 3},
                null,
                "input",
                DataType.FLOAT32
        ).permute(1, 0);
        Tensor output = new Tensor(new double[6], new int[]{2, 3}, null, "output", DataType.FLOAT64)
                .permute(1, 0);
        Cpu1PreparedDTypeUnit unit = new Cpu1PreparedDTypeUnit(
                1,
                0,
                Operation.OpType.CAST,
                DataType.FLOAT32,
                DataType.FLOAT64,
                6,
                Cpu1LayoutKind.STRIDED_RANK2,
                Cpu1StorageKind.JAVA_ARRAY,
                Cpu1DTypeKernelId.CAST_ARRAY_SCALAR,
                Cpu1LaunchConfig.singleThread(),
                new Cpu1SingleThreadLaunch()
        );

        Cpu1CastLoops.cast(unit, Cpu1TensorView.fromTensor(input), Cpu1TensorView.fromTensor(output));

        assertArrayEquals(new double[]{1.0, 4.0, 2.0, 5.0, 3.0, 6.0}, output.toFloat64ArrayCopy(), 0.0d);
    }

    @Test
    void preparedMemorySegmentCastConvertsF32AndBF16() {
        Tensor f32 = new Tensor(new float[]{1.25f, -2.5f, 0.0f}, new int[]{3}, null, "f32", DataType.FLOAT32);
        Fixture f32Fixture = fixture(f32.cast(DataType.BFLOAT16));
        ExecutionContext f32Context = executeCpu1WithNativeInputs(f32Fixture);
        assertArrayEquals(
                bf16Bits(1.25f, -2.5f, 0.0f),
                readNativeBF16(f32Context.nativeStorageForNodeId(f32Fixture.node().id()), 3)
        );

        Tensor bf16 = new Tensor(bf16Bits(3.5f, -4.25f, 0.5f), new int[]{3}, null, "bf16", DataType.BFLOAT16);
        Fixture bf16Fixture = fixture(bf16.cast(DataType.FLOAT32));
        ExecutionContext bf16Context = executeCpu1WithNativeInputs(bf16Fixture);
        assertArrayEquals(
                new float[]{bf16(3.5f), bf16(-4.25f), bf16(0.5f)},
                readNativeF32(bf16Context.nativeStorageForNodeId(bf16Fixture.node().id()), 3),
                0.0f
        );
    }

    private static Tensor executeCpu1(Fixture fixture, Cpu1PrepareConfig config) {
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        PreparedStepMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context.runtimeTensorForNodeId(fixture.node().id());
    }

    private static ExecutionContext executeCpu1WithNativeInputs(Fixture fixture) {
        Cpu1PrepareConfig config = Cpu1PrepareConfig.scalarMemorySegmentSingleThread();
        Cpu1PreparedArtifact artifact = new Cpu1NodePreparer().prepare(fixture.node(), fixture.descriptorIndex(), config);
        assertEquals(Cpu1DTypeKernelId.CAST_SEGMENT_SCALAR, artifact.preparedDTypeUnit().kernelId());
        PreparedStepMetadata metadata = cpu1Metadata(fixture.node(), artifact);
        ExecutionContext context = context(fixture, metadata);
        attachNativeInputs(context, fixture);
        new Cpu1Backend().execute(fixture.node(), metadata, context);
        return context;
    }

    private static Fixture fixture(Tensor out) {
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(nodes);
        return new Fixture(out, nodes, descriptorIndex, nodes.getLast());
    }

    private static ExecutionContext context(Fixture fixture, PreparedStepMetadata metadata) {
        Map<Integer, PreparedStepMetadata> metadataIndex = Map.of(fixture.node().id(), metadata);
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

    private static PreparedStepMetadata cpu1Metadata(CompiledNode node, Cpu1PreparedArtifact artifact) {
        return new PreparedStepMetadata(
                ComputeBackend.CPU,
                null,
                node.inputIds(),
                artifact,
                runtime.execution.InputResidencyRequirement.cpuReadableAll(),
                runtime.execution.OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }

    private static void attachNativeInputs(ExecutionContext context, Fixture fixture) {
        NativeCpuStorageFactory storageFactory = new NativeCpuStorageFactory();
        for (int inputNodeId : fixture.node().inputIds()) {
            Tensor tensor = context.runtimeTensorForNodeId(inputNodeId);
            NativeTensorStorage storage = storageFactory.allocate(
                    tensor.getDataType(),
                    physicalElementCount(tensor),
                    "cpu1-dtype-test-input-" + inputNodeId
            );
            copyToNative(tensor, storage);
            context.attachNativeStorage(inputNodeId, storage, "cpu1 dtype test native input");
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
            case INT32 -> {
                int[] source = TensorInternalAccess.int32Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_INT, (long) i * Integer.BYTES, source[i]);
                }
            }
            case INT64 -> {
                long[] source = TensorInternalAccess.int64Data(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_LONG, (long) i * Long.BYTES, source[i]);
                }
            }
            case BOOL -> {
                byte[] source = TensorInternalAccess.boolData(tensor);
                for (int i = 0; i < storage.getSize(); i++) {
                    segment.set(JAVA_BYTE, (long) i * Byte.BYTES, source[i]);
                }
            }
        }
        storage.markModified();
    }

    private static short[] readNativeBF16(NativeTensorStorage storage, int length) {
        short[] out = new short[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_SHORT, (long) i * Short.BYTES);
        }
        return out;
    }

    private static float[] readNativeF32(NativeTensorStorage storage, int length) {
        float[] out = new float[length];
        MemorySegment segment = storage.segment();
        for (int i = 0; i < out.length; i++) {
            out[i] = segment.get(JAVA_FLOAT, (long) i * Float.BYTES);
        }
        return out;
    }

    private static short[] bf16Bits(float... values) {
        short[] bits = new short[values.length];
        for (int i = 0; i < values.length; i++) {
            bits[i] = TensorDTypeOps.toBFloat16Bits(values[i]);
        }
        return bits;
    }

    private static float bf16(float value) {
        return TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits(value));
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
