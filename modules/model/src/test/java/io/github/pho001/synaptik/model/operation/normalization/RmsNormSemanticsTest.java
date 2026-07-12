package io.github.pho001.synaptik.model.operation.normalization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class RmsNormSemanticsTest {
    @Test
    void declaresOneKindAndOneSafeContiguousInputRangeSignature() {
        assertEquals(List.of(RmsNormKind.RMS_NORM), List.of(RmsNormKind.values()));
        assertEquals(
                List.of(OperationSignature.inputRange(RmsNormAttrs.class, 1, 2, 1)),
                RmsNormKind.RMS_NORM.signatures());
        OperationSignature signature = RmsNormKind.RMS_NORM.signatures().getFirst();
        assertAll(
                () -> signature.validateOccurrence(1, 1),
                () -> signature.validateOccurrence(2, 1),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(3, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(1, 2)));
    }

    @Test
    void retainsExactAttributeReferencesForBothSafeInputCounts() {
        Shape shape = Shape.of(3, 4);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        RmsNormAttrs attrs = new RmsNormAttrs(shape, epsilon);
        Operation operation = new Operation(RmsNormKind.RMS_NORM, attrs);

        assertAll(
                () -> assertSame(shape, attrs.normalizedShape()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertSame(attrs, operation.attrs()));
    }

    @Test
    void validatesIntrinsicNullAndRankFailuresInOrder() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        assertEquals("normalizedShape", assertThrows(NullPointerException.class,
                () -> new RmsNormAttrs(null, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> new RmsNormAttrs(Shape.of(1), null)).getMessage());
        assertEquals("normalizedShape rank must be positive", assertThrows(
                IllegalArgumentException.class,
                () -> new RmsNormAttrs(Shape.scalar(), epsilon)).getMessage());
    }

    @Test
    void acceptsEveryFloatingPositiveRepresentation() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16(Float.MIN_NORMAL),
                ScalarValue.float32(Float.MIN_VALUE),
                ScalarValue.float64(Double.MIN_VALUE))) {
            assertSame(epsilon, new RmsNormAttrs(Shape.of(1), epsilon).epsilon());
        }
    }

    @Test
    void rejectsNonFloatingEpsilonWithExactTypeMessage() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.int32(1), ScalarValue.int64(1), ScalarValue.bool(true))) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new RmsNormAttrs(Shape.of(1), epsilon));
            assertEquals(
                    "epsilon must have a floating data type, but was " + epsilon.dataType(),
                    failure.getMessage());
        }
    }

    @Test
    void rejectsZeroNegativeInfinityAndEveryRepresentativeNaNEncoding() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16Bits((short) 0x0000),
                ScalarValue.bfloat16Bits((short) 0x8000),
                ScalarValue.bfloat16Bits((short) 0xBF80),
                ScalarValue.bfloat16Bits((short) 0x7F80),
                ScalarValue.bfloat16Bits((short) 0xFF80),
                ScalarValue.bfloat16Bits((short) 0x7FC1),
                ScalarValue.bfloat16Bits((short) 0xFFC1),
                ScalarValue.float32(0.0f),
                ScalarValue.float32(-0.0f),
                ScalarValue.float32(-1.0f),
                ScalarValue.float32(Float.POSITIVE_INFINITY),
                ScalarValue.float32(Float.NEGATIVE_INFINITY),
                ScalarValue.float32(Float.NaN),
                ScalarValue.float32(Float.intBitsToFloat(0xFFC00001)),
                ScalarValue.float64(0.0d),
                ScalarValue.float64(-0.0d),
                ScalarValue.float64(-1.0d),
                ScalarValue.float64(Double.POSITIVE_INFINITY),
                ScalarValue.float64(Double.NEGATIVE_INFINITY),
                ScalarValue.float64(Double.NaN),
                ScalarValue.float64(Double.longBitsToDouble(0xFFF8000000000001L)))) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new RmsNormAttrs(Shape.of(1), epsilon));
            assertEquals("epsilon must be finite and positive: " + epsilon, failure.getMessage());
        }
    }

    @Test
    void locksUncenteredFormulaPopulationDivisorEpsilonPlacementAndScaleExample() {
        double[] input = {1.0, 2.0, 3.0};
        double sumSquares = 0.0;
        for (double value : input) {
            sumSquares += value * value;
        }
        double meanSquare = sumSquares / input.length;
        double rms = Math.sqrt(meanSquare + 1.0e-5);

        assertAll(
                () -> assertEquals(14.0 / 3.0, meanSquare, 0.0),
                () -> assertEquals(2.1602492140182963, rms, 1.0e-15),
                () -> assertEquals(0.4629095539120194, input[0] / rms, 1.0e-15),
                () -> assertEquals(0.9258191078240388, input[1] / rms, 1.0e-15),
                () -> assertEquals(1.3887286617360581, input[2] / rms, 1.0e-15),
                () -> assertEquals(2.7774573234721163, input[2] / rms * 2.0, 1.0e-15));
    }

    @Test
    void locksSpecialValueSignedZeroOverflowAndScaleClassesWithoutExecution() {
        double positiveRms = Math.sqrt(1.0e-5);
        assertAll(
                () -> assertEquals(0L, Double.doubleToRawLongBits(0.0d / positiveRms)),
                () -> assertEquals(Long.MIN_VALUE,
                        Double.doubleToRawLongBits(-0.0d / positiveRms)),
                () -> assertEquals(0L, Double.doubleToRawLongBits(1.0d
                        / Math.sqrt(Double.POSITIVE_INFINITY))),
                () -> assertEquals(Long.MIN_VALUE, Double.doubleToRawLongBits(-1.0d
                        / Math.sqrt(Double.POSITIVE_INFINITY))),
                () -> assertEquals(Double.POSITIVE_INFINITY, 1.0d * Double.POSITIVE_INFINITY),
                () -> assertEquals(Double.NEGATIVE_INFINITY, -1.0d * Double.POSITIVE_INFINITY),
                () -> assertEquals(true, Double.isNaN(Double.POSITIVE_INFINITY
                        / Double.POSITIVE_INFINITY)),
                () -> assertEquals(true, Double.isNaN(0.0d * Double.POSITIVE_INFINITY)));
    }
}
