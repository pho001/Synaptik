package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class CpuSoftmaxInputValidatorTest {
    @Test void temporalSlicesSkipInactiveHighRankRowsAndReadCurrentOffsets() {
        float[] input = {Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                1, 2, Float.NaN, Float.NaN, 3, 4, Float.NaN, Float.NaN};
        byte[] active = {1, 0};
        MemorySegment storage = MemorySegment.ofArray(input);
        MemorySegment activity = MemorySegment.ofArray(active);
        Runnable scan = () -> CpuSoftmaxInputValidator.validateTemporal(storage, DataType.FLOAT32,
                0L, new long[] {2, 2, 2}, new long[] {4, 2, 1}, 2, 2L, 0L, 4L, 4L,
                true, activity, 0L, 1L, 2L);
        assertDoesNotThrow(scan::run);
        input[9] = Float.POSITIVE_INFINITY;
        assertThrows(IllegalArgumentException.class, scan::run);
        input[9] = 4;
        assertDoesNotThrow(scan::run);
        active[1] = 1;
        assertThrows(IllegalArgumentException.class, scan::run);
        assertDoesNotThrow(() -> CpuSoftmaxInputValidator.validateTemporal(null, DataType.FLOAT32,
                0L, null, null, 0, 1L, 0L, 1L, 0L, false, null, 0L, 0L, 0L));
    }

    @Test void lowScansDecodeDistinctTypesAndCaptureNoValuesDuringBinding() {
        var layout = new CpuSoftmaxLowering.Layout(new long[] {2}, 1, new long[] {2});
        for (DataType type : new DataType[] {DataType.FLOAT16, DataType.BFLOAT16}) {
            var geometry = new CpuSoftmaxLowering.Geometry(SoftmaxKind.LOG_SOFTMAX, type, 0,
                    layout, layout, 1, 2, 2);
            short[] values = new short[7];
            java.util.Arrays.fill(values, (short) 0x7fff);
            var scan = assertDoesNotThrow(() -> CpuSoftmaxInputValidator.bind(
                    new CpuBufferArgument.Shorts(values, 2, 10, true), geometry));
            // Both logical values are initially NaN, but only invocation may observe them.
            assertThrows(IllegalArgumentException.class, scan::run);
            values[2] = type == DataType.FLOAT16 ? (short) 0x7bff : (short) 0x7f7f;
            values[4] = type == DataType.FLOAT16 ? (short) 0xfbff : (short) 0xff7f;
            if (type == DataType.FLOAT16) assertDoesNotThrow(scan::run);
            else assertThrows(IllegalArgumentException.class, scan::run,
                    "BFLOAT16 maximum shift overflows FLOAT32, unlike FLOAT16");
            values[2] = values[4] = 0;
            assertDoesNotThrow(scan::run);
        }
    }

    @Test void admitsFiniteDoubleShiftOverflowAndRejectsNonFiniteInputs() {
        var layout = new CpuSoftmaxLowering.Layout(new long[] {2}, 0, new long[] {1});
        var geometry = new CpuSoftmaxLowering.Geometry(SoftmaxKind.SOFTMAX, DataType.FLOAT64, 0,
                layout, layout, 1, 2, 2);
        assertAll(() -> assertDoesNotThrow(() -> CpuSoftmaxInputValidator.validate(
                        new CpuBufferArgument.Doubles(new double[] {-1, 1}, 0, 16, true), geometry)),
                () -> assertDoesNotThrow(() ->
                        CpuSoftmaxInputValidator.validate(new CpuBufferArgument.Doubles(
                                new double[] {Double.MAX_VALUE, -Double.MAX_VALUE}, 0, 16, true), geometry)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        CpuSoftmaxInputValidator.validate(new CpuBufferArgument.Doubles(
                                new double[] {0, Double.NaN}, 0, 16, true), geometry)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        CpuSoftmaxInputValidator.validate(new CpuBufferArgument.Segment(
                                DataType.FLOAT64,
                                MemorySegment.ofArray(new double[] {Double.NEGATIVE_INFINITY, 0}),
                                16, true), geometry)),
                () -> assertThrows(IllegalArgumentException.class, () ->
                        CpuSoftmaxInputValidator.validate(new CpuBufferArgument.Segment(
                                DataType.FLOAT64,
                                MemorySegment.ofArray(new double[] {0, Double.POSITIVE_INFINITY}),
                                16, true), geometry)));
    }

    @Test void preservesFloat32FiniteExtremeAdmissionDespiteWorkingShiftOverflow() {
        var layout = new CpuSoftmaxLowering.Layout(new long[] {2}, 0, new long[] {1});
        var geometry = new CpuSoftmaxLowering.Geometry(SoftmaxKind.LOG_SOFTMAX,
                DataType.FLOAT32, 0, layout, layout, 1, 2, 2);
        assertDoesNotThrow(() -> CpuSoftmaxInputValidator.validate(
                new CpuBufferArgument.Floats(new float[] {-Float.MAX_VALUE, Float.MAX_VALUE},
                        0, 8, true), geometry));
        assertThrows(IllegalArgumentException.class, () -> CpuSoftmaxInputValidator.validate(
                new CpuBufferArgument.Floats(new float[] {Float.NaN, 0}, 0, 8, true), geometry));
    }

    @Test void boundScansReadCurrentValuesAcrossStridedSlicesAndCarrierOffsets() {
        var layout = new CpuSoftmaxLowering.Layout(new long[] {3, 2}, 1, new long[] {2, 8});
        for (DataType type : new DataType[] {DataType.FLOAT32, DataType.FLOAT64})
            for (SoftmaxKind kind : SoftmaxKind.values()) for (boolean segment : new boolean[] {false, true}) {
                var geometry = new CpuSoftmaxLowering.Geometry(kind, type, 0,
                        layout, layout, 2, 3, 6);
                double maximum = type == DataType.FLOAT32 ? Float.MAX_VALUE : Double.MAX_VALUE;
                double[] doubles = new double[16];
                float[] floats = new float[16];
                MemorySegment storage = type == DataType.FLOAT32
                        ? MemorySegment.ofArray(floats) : MemorySegment.ofArray(doubles);
                CpuBufferArgument input = segment
                        ? new CpuBufferArgument.Segment(type, storage.asSlice(type.byteWidth()),
                                15L * type.byteWidth(), true)
                        : type == DataType.FLOAT32
                                ? new CpuBufferArgument.Floats(floats, 4, 60, true)
                                : new CpuBufferArgument.Doubles(doubles, 8, 120, true);
                Runnable scan = CpuSoftmaxInputValidator.bind(input, geometry);
                for (int index : new int[] {2, 4, 6, 10, 12, 14}) {
                    doubles[index] = index == 2 || index == 10 ? -maximum : maximum;
                    floats[index] = (float) doubles[index];
                }
                assertDoesNotThrow(scan::run);
                for (double invalid : new double[] {Double.NaN, Double.POSITIVE_INFINITY,
                        Double.NEGATIVE_INFINITY}) {
                    doubles[14] = invalid; floats[14] = (float) invalid;
                    assertThrows(IllegalArgumentException.class, scan::run);
                    doubles[14] = maximum; floats[14] = (float) maximum;
                    assertDoesNotThrow(scan::run);
                }
            }
    }
}
