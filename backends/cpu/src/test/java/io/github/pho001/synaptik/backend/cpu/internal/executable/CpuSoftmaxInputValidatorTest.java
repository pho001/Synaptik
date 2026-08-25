package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class CpuSoftmaxInputValidatorTest {
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
