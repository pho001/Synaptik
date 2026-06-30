package backend.cpu.kernels.layout;

import backend.contract.ComputeBackend;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.storage.CpuStorageBindings;
import backend.cpu.storage.CpuStorageResolver;
import backend.cpu.storage.CpuStorageView;
import runtime.execution.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.runtime.RuntimeConfig;
import runtime.execution.PreparedStepMetadata;
import operations.Operation;
import operations.dtype.cast;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuCastKernelStorageViewTest {
    private static final int[] SHAPE = {3};
    private static final int[] STRIDES = {2};
    private static final int STORAGE_OFFSET = 1;
    private static final int LOGICAL_SIZE = 3;
    private static final int PHYSICAL_SIZE = 7;
    private static final double[] SOURCE_VALUES = {-2.0d, 0.0d, 3.0d};

    @Test
    void castsAllDtypePairsFromStridedMemorySegmentInputToStridedMemorySegmentOutput() {
        for (DataType inputType : DataType.values()) {
            for (DataType outputType : DataType.values()) {
                Object inputStorage = inputStorage(inputType);
                Object outputStorage = outputStorage(outputType);
                Tensor input = tensor(inputType, "input");
                Tensor out = new Tensor(SHAPE, STRIDES, STORAGE_OFFSET,
                        List.of(input), new cast(outputType), "out", outputType);

                new CpuCastKernel().execute(call(
                        new cast(outputType),
                        List.of(input),
                        out,
                        List.of(segment(inputType, inputStorage)),
                        segment(outputType, outputStorage)));

                assertOutput(outputType, outputStorage, expectedLogical(inputType), inputType + "->" + outputType);
            }
        }
    }

    @Test
    void arrayStorageFallbackMarksOutputTensorStorageModified() {
        Tensor input = new Tensor(new float[]{1.0f, 2.0f}, new int[]{2}, null, "input", DataType.FLOAT32);
        Tensor out = new Tensor(new int[]{2}, List.of(input), new cast(DataType.FLOAT64), "out", DataType.FLOAT64);
        long before = TensorInternalAccess.storage(out).version();
        CpuStorageBindings storage = new CpuStorageResolver().bindArrayOnly(List.of(input), out);

        new CpuCastKernel().execute(call(
                new cast(DataType.FLOAT64),
                List.of(input),
                out,
                storage.inputs(),
                storage.output()));

        assertArrayEquals(new double[]{1.0d, 2.0d}, TensorInternalAccess.float64Data(out));
        assertTrue(TensorInternalAccess.storage(out).version() > before);
    }

    private static Tensor tensor(DataType dtype, String label) {
        return new Tensor(SHAPE, STRIDES, STORAGE_OFFSET, null, null, label, dtype);
    }

    private static CpuStorageView segment(DataType dtype, Object storage) {
        return CpuStorageView.segment(
                dtype,
                memorySegment(storage),
                SHAPE,
                STRIDES,
                STORAGE_OFFSET,
                LOGICAL_SIZE
        );
    }

    private static MemorySegment memorySegment(Object storage) {
        if (storage instanceof double[] values) {
            return MemorySegment.ofArray(values);
        }
        if (storage instanceof float[] values) {
            return MemorySegment.ofArray(values);
        }
        if (storage instanceof short[] values) {
            return MemorySegment.ofArray(values);
        }
        if (storage instanceof int[] values) {
            return MemorySegment.ofArray(values);
        }
        if (storage instanceof long[] values) {
            return MemorySegment.ofArray(values);
        }
        if (storage instanceof byte[] values) {
            return MemorySegment.ofArray(values);
        }
        throw new IllegalArgumentException("Unsupported storage array: " + storage.getClass());
    }

    private static Object inputStorage(DataType dtype) {
        Object storage = outputStorage(dtype);
        for (int logical = 0; logical < LOGICAL_SIZE; logical++) {
            writeStorageValue(dtype, storage, STORAGE_OFFSET + logical * STRIDES[0], SOURCE_VALUES[logical]);
        }
        return storage;
    }

    private static Object outputStorage(DataType dtype) {
        return switch (dtype) {
            case FLOAT64 -> filledF64();
            case FLOAT32 -> filledF32();
            case BFLOAT16 -> filledBF16();
            case INT32 -> filledI32();
            case INT64 -> filledI64();
            case BOOL -> filledBool();
        };
    }

    private static double[] filledF64() {
        double[] values = new double[PHYSICAL_SIZE];
        Arrays.fill(values, -99.0d);
        return values;
    }

    private static float[] filledF32() {
        float[] values = new float[PHYSICAL_SIZE];
        Arrays.fill(values, -99.0f);
        return values;
    }

    private static short[] filledBF16() {
        short[] values = new short[PHYSICAL_SIZE];
        Arrays.fill(values, TensorDTypeOps.toBFloat16Bits(-99.0f));
        return values;
    }

    private static int[] filledI32() {
        int[] values = new int[PHYSICAL_SIZE];
        Arrays.fill(values, -99);
        return values;
    }

    private static long[] filledI64() {
        long[] values = new long[PHYSICAL_SIZE];
        Arrays.fill(values, -99L);
        return values;
    }

    private static byte[] filledBool() {
        byte[] values = new byte[PHYSICAL_SIZE];
        Arrays.fill(values, (byte) 9);
        return values;
    }

    private static void writeStorageValue(DataType dtype, Object storage, int offset, double value) {
        switch (dtype) {
            case FLOAT64 -> ((double[]) storage)[offset] = value;
            case FLOAT32 -> ((float[]) storage)[offset] = (float) value;
            case BFLOAT16 -> ((short[]) storage)[offset] = TensorDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> ((int[]) storage)[offset] = (int) value;
            case INT64 -> ((long[]) storage)[offset] = (long) value;
            case BOOL -> ((byte[]) storage)[offset] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }

    private static double[] expectedLogical(DataType inputType) {
        double[] expected = new double[LOGICAL_SIZE];
        for (int i = 0; i < LOGICAL_SIZE; i++) {
            double value = SOURCE_VALUES[i];
            expected[i] = switch (inputType) {
                case FLOAT64 -> value;
                case FLOAT32 -> (float) value;
                case BFLOAT16 -> TensorDTypeOps.fromBFloat16Bits(TensorDTypeOps.toBFloat16Bits((float) value));
                case INT32 -> (int) value;
                case INT64 -> (double) ((long) value);
                case BOOL -> value == 0.0d ? 0.0d : 1.0d;
            };
        }
        return expected;
    }

    private static void assertOutput(DataType dtype, Object storage, double[] expectedLogical, String castLabel) {
        switch (dtype) {
            case FLOAT64 -> assertF64((double[]) storage, expectedLogical, castLabel);
            case FLOAT32 -> assertF32((float[]) storage, expectedLogical, castLabel);
            case BFLOAT16 -> assertBF16((short[]) storage, expectedLogical, castLabel);
            case INT32 -> assertI32((int[]) storage, expectedLogical, castLabel);
            case INT64 -> assertI64((long[]) storage, expectedLogical, castLabel);
            case BOOL -> assertBool((byte[]) storage, expectedLogical, castLabel);
        }
    }

    private static void assertF64(double[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            double expected = logical < 0 ? -99.0d : expectedLogical[logical];
            assertEquals(expected, actual[i], 0.0d, castLabel + " physical offset " + i);
        }
    }

    private static void assertF32(float[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            float expected = logical < 0 ? -99.0f : (float) expectedLogical[logical];
            assertEquals(expected, actual[i], 0.0f, castLabel + " physical offset " + i);
        }
    }

    private static void assertBF16(short[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            short expected = logical < 0
                    ? TensorDTypeOps.toBFloat16Bits(-99.0f)
                    : TensorDTypeOps.toBFloat16Bits((float) expectedLogical[logical]);
            assertEquals(expected, actual[i], castLabel + " physical offset " + i);
        }
    }

    private static void assertI32(int[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            int expected = logical < 0 ? -99 : (int) expectedLogical[logical];
            assertEquals(expected, actual[i], castLabel + " physical offset " + i);
        }
    }

    private static void assertI64(long[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            long expected = logical < 0 ? -99L : (long) expectedLogical[logical];
            assertEquals(expected, actual[i], castLabel + " physical offset " + i);
        }
    }

    private static void assertBool(byte[] actual, double[] expectedLogical, String castLabel) {
        for (int i = 0; i < actual.length; i++) {
            int logical = logicalIndexForPhysicalOffset(i);
            byte expected = logical < 0
                    ? (byte) 9
                    : expectedLogical[logical] == 0.0d ? (byte) 0 : (byte) 1;
            assertEquals(expected, actual[i], castLabel + " physical offset " + i);
        }
    }

    private static int logicalIndexForPhysicalOffset(int physicalOffset) {
        int shifted = physicalOffset - STORAGE_OFFSET;
        if (shifted < 0 || shifted % STRIDES[0] != 0) {
            return -1;
        }
        int logical = shifted / STRIDES[0];
        return logical >= 0 && logical < LOGICAL_SIZE ? logical : -1;
    }

    private static CpuKernelCall call(
            Operation operation,
            List<Tensor> inputTensors,
            Tensor outputTensor,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        CpuNodeExecutionPlan plan = plan(output.dtype());
        PreparedStepMetadata metadata = new PreparedStepMetadata(
                ComputeBackend.CPU,
                operation,
                List.of(),
                testsupport.MetadataArtifacts.noopExecutable(),
                runtime.execution.InputResidencyRequirement.cpuReadableAll(),
                runtime.execution.OutputResidencyEffect.cpuCurrentPreserveNative()
                );
        CpuKernelContext context = new CpuKernelContext(
                1,
                inputTensors.stream().map(tensor -> 0).toList(),
                plan,
                ExecutionContext.fromRuntimeConfig(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD),
                metadata,
                List.of(),
                operation);
        return new CpuKernelCall(operation, inputTensors, outputTensor, inputs, output, plan, context, null);
    }

    private static CpuNodeExecutionPlan plan(DataType dtype) {
        return new CpuNodeExecutionPlan(
                new CpuLayoutPlan(StridedLayoutDecision.NONE, dtype, 0, null, null, List.of()),
                null,
                false,
                1,
                0,
                null,
                null,
                null,
                null,
                null);
    }
}
