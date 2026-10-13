package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class CpuSoftmaxInputValidatorTest {
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

    @Test void rejectsNonFiniteInputsAndFiniteShiftOverflow() {
        var layout = new CpuSoftmaxLowering.Layout(new long[] {2}, 0, new long[] {1});
        var geometry = new CpuSoftmaxLowering.Geometry(SoftmaxKind.SOFTMAX, DataType.FLOAT64, 0,
                layout, layout, 1, 2, 2);
        assertAll(() -> assertDoesNotThrow(() -> CpuSoftmaxInputValidator.validate(
                        new CpuBufferArgument.Doubles(new double[] {-1, 1}, 0, 16, true), geometry)),
                () -> assertThrows(IllegalArgumentException.class, () ->
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
}
