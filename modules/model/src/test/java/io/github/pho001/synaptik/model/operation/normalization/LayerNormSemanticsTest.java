package io.github.pho001.synaptik.model.operation.normalization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import org.junit.jupiter.api.Test;

class LayerNormSemanticsTest {
    @Test
    void declaresOneKindAndTheTwoExactOrderedSignatureVariants() {
        assertEquals(List.of(LayerNormKind.LAYER_NORM), List.of(LayerNormKind.values()));
        assertEquals(
                List.of(
                        OperationSignature.fixed(LayerNormAttrs.class, 1, 1),
                        OperationSignature.fixed(AffineLayerNormAttrs.class, 3, 1)),
                LayerNormKind.LAYER_NORM.signatures());
    }

    @Test
    void retainsExactAttributeReferencesAndUsesDistinctAttributeClasses() {
        Shape shape = Shape.of(3, 4);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        LayerNormAttrs plain = new LayerNormAttrs(shape, epsilon);
        AffineLayerNormAttrs affine = new AffineLayerNormAttrs(shape, epsilon);
        Operation plainOperation = new Operation(LayerNormKind.LAYER_NORM, plain);
        Operation affineOperation = new Operation(LayerNormKind.LAYER_NORM, affine);

        assertAll(
                () -> assertSame(shape, plain.normalizedShape()),
                () -> assertSame(epsilon, plain.epsilon()),
                () -> assertSame(shape, affine.normalizedShape()),
                () -> assertSame(epsilon, affine.epsilon()),
                () -> assertSame(plain, plainOperation.attrs()),
                () -> assertSame(affine, affineOperation.attrs()),
                () -> assertNotEquals(plainOperation, affineOperation));
    }

    @Test
    void validatesIntrinsicNullAndRankFailuresInOrder() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        assertEquals("normalizedShape", assertThrows(NullPointerException.class,
                () -> new LayerNormAttrs(null, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> new AffineLayerNormAttrs(Shape.of(1), null)).getMessage());
        assertEquals("normalizedShape rank must be positive", assertThrows(
                IllegalArgumentException.class,
                () -> new LayerNormAttrs(Shape.scalar(), epsilon)).getMessage());
    }

    @Test
    void acceptsEveryFloatingPositiveRepresentation() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16(Float.MIN_NORMAL),
                ScalarValue.float32(Float.MIN_VALUE),
                ScalarValue.float64(Double.MIN_VALUE))) {
            assertSame(epsilon, new LayerNormAttrs(Shape.of(1), epsilon).epsilon());
            assertSame(epsilon, new AffineLayerNormAttrs(Shape.of(1), epsilon).epsilon());
        }
    }

    @Test
    void rejectsNonFloatingEpsilonWithExactTypeMessage() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.int32(1), ScalarValue.int64(1), ScalarValue.bool(true))) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new LayerNormAttrs(Shape.of(1), epsilon));
            assertEquals(
                    "epsilon must have a floating data type, but was " + epsilon.dataType(),
                    failure.getMessage());
        }
    }

    @Test
    void rejectsZeroNegativeInfinityAndNaNByExactTypedValue() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16Bits((short) 0x0000),
                ScalarValue.bfloat16Bits((short) 0x8000),
                ScalarValue.bfloat16Bits((short) 0xBF80),
                ScalarValue.bfloat16Bits((short) 0x7F80),
                ScalarValue.bfloat16Bits((short) 0x7FC1),
                ScalarValue.float32(0.0f),
                ScalarValue.float32(-0.0f),
                ScalarValue.float32(-1.0f),
                ScalarValue.float32(Float.NEGATIVE_INFINITY),
                ScalarValue.float32(Float.NaN),
                ScalarValue.float64(0.0d),
                ScalarValue.float64(-0.0d),
                ScalarValue.float64(-1.0d),
                ScalarValue.float64(Double.POSITIVE_INFINITY),
                ScalarValue.float64(Double.NaN))) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new AffineLayerNormAttrs(Shape.of(1), epsilon));
            assertEquals("epsilon must be finite and positive: " + epsilon, failure.getMessage());
        }
    }

    @Test
    void locksPopulationFormulaEpsilonPlacementAndAffineExample() {
        double[] input = {1.0, 2.0, 3.0};
        double mean = 2.0;
        double variance = ((1.0 - mean) * (1.0 - mean)
                + (2.0 - mean) * (2.0 - mean)
                + (3.0 - mean) * (3.0 - mean)) / input.length;
        double epsilon = 1.0e-5;
        double scale = 2.0;
        double bias = 0.5;
        double denominator = Math.sqrt(variance + epsilon);

        assertAll(
                () -> assertEquals(2.0 / 3.0, variance, 0.0),
                () -> assertEquals((1.0 - mean) / denominator,
                        -1.2247356859083902, 1.0e-12),
                () -> assertEquals(((3.0 - mean) / denominator) * scale + bias,
                        2.9494713718167804, 1.0e-12));
    }
}
