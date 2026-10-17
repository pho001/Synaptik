package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.*;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuPool2dReferenceTest {
    @Test
    void preservesMaxAndAverageSpecialValues() {
        var max =
                new CpuPool2dLowering()
                        .lower(
                                CpuPool2dLoweringTest.context(
                                        Pool2dKind.MAX_POOL2D,
                                        new MaxPool2dAttrs(2, 2, 1, 1, 2, 2, 1, 1, true),
                                        DataType.FLOAT64,
                                        Shape.of(1, 1, 1, 1),
                                        Shape.of(1, 1, 4, 4)))
                        .pool2dGeometry()
                        .orElseThrow();
        double[] mo = new double[16];
        CpuPool2dReferenceKernel.evaluate(max, new double[] {-0.0}, mo, 0, 16);
        assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(mo[5]));
        assertEquals(Double.NEGATIVE_INFINITY, mo[0]);
        var avg =
                new CpuPool2dLowering()
                        .lower(
                                CpuPool2dLoweringTest.context(
                                        Pool2dKind.AVERAGE_POOL2D,
                                        new AveragePool2dAttrs(2, 2, 1, 1, 1, 1, 1, 1, true),
                                        DataType.FLOAT32,
                                        Shape.of(1, 1, 1, 1),
                                        Shape.of(1, 1, 2, 2)))
                        .pool2dGeometry()
                        .orElseThrow();
        float[] ao = new float[4];
        CpuPool2dReferenceKernel.evaluate(avg, new float[] {-0.0f}, ao, 0, 4);
        for (float v : ao) assertEquals(0, Float.floatToRawIntBits(v));
    }

    @Test
    void maxNanWinsAndPositiveZeroWins() {
        var g =
                new CpuPool2dLowering()
                        .lower(
                                CpuPool2dLoweringTest.context(
                                        Pool2dKind.MAX_POOL2D,
                                        new MaxPool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false),
                                        DataType.FLOAT32,
                                        Shape.of(1, 1, 1, 2),
                                        Shape.of(1, 1, 1, 1)))
                        .pool2dGeometry()
                        .orElseThrow();
        float[] out = new float[1];
        CpuPool2dReferenceKernel.evaluate(g, new float[] {-0.0f, +0.0f}, out, 0, 1);
        assertEquals(0, Float.floatToRawIntBits(out[0]));
        CpuPool2dReferenceKernel.evaluate(g, new float[] {Float.NaN, 1}, out, 0, 1);
        assertTrue(Float.isNaN(out[0]));
    }

    @Test
    void averagePreservesAnAllNegativeZeroWindow() {
        var attrs = new AveragePool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false);
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            var geometry =
                    new CpuPool2dLowering()
                            .lower(
                                    CpuPool2dLoweringTest.context(
                                            Pool2dKind.AVERAGE_POOL2D,
                                            attrs,
                                            type,
                                            Shape.of(1, 1, 1, 2),
                                            Shape.of(1, 1, 1, 1)))
                            .pool2dGeometry()
                            .orElseThrow();
            switch (type) {
                case BFLOAT16 -> {
                    short[] output = new short[1];
                    CpuPool2dReferenceKernel.evaluate(
                            geometry,
                            new short[] {BFloat16Bits.fromFloat(-0.0f), BFloat16Bits.fromFloat(-0.0f)},
                            output,
                            0,
                            1);
                    assertEquals(BFloat16Bits.fromFloat(-0.0f), output[0]);
                }
                case FLOAT32 -> {
                    float[] output = new float[1];
                    CpuPool2dReferenceKernel.evaluate(geometry, new float[] {-0.0f, -0.0f}, output, 0, 1);
                    assertEquals(Integer.MIN_VALUE, Float.floatToRawIntBits(output[0]));
                }
                case FLOAT64 -> {
                    double[] output = new double[1];
                    CpuPool2dReferenceKernel.evaluate(geometry, new double[] {-0.0d, -0.0d}, output, 0, 1);
                    assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(output[0]));
                }
                default -> throw new AssertionError(type);
            }
        }
    }

    @Test
    void rejectsCarrierTypeCapacityAndInvalidRangeBeforeEvaluation() {
        var geometry =
                new CpuPool2dLowering()
                        .lower(
                                CpuPool2dLoweringTest.context(
                                        Pool2dKind.AVERAGE_POOL2D,
                                        new AveragePool2dAttrs(1, 2, 1, 1, 0, 0, 1, 1, false),
                                        DataType.FLOAT32,
                                        Shape.of(1, 1, 1, 2),
                                        Shape.of(1, 1, 1, 1)))
                        .pool2dGeometry()
                        .orElseThrow();
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuPool2dReferenceKernel.evaluate(
                                geometry, new float[1], new float[1], 0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuPool2dReferenceKernel.evaluate(
                                geometry, new double[2], new float[1], 0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> CpuPool2dReferenceKernel.evaluate(
                                geometry, new float[2], new float[1], 1, 0)));
    }
}
