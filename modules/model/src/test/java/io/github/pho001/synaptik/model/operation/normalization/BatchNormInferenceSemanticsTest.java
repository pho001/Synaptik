package io.github.pho001.synaptik.model.operation.normalization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;
import org.junit.jupiter.api.Test;

class BatchNormInferenceSemanticsTest {
    @Test
    void declaresExactlyOneInferenceKindAndFixedFiveInputOneOutputSignature() {
        assertEquals(
                List.of(BatchNormKind.BATCH_NORM_INFERENCE,
                        BatchNormKind.BATCH_NORM_TRAINING),
                List.of(BatchNormKind.values()));
        assertEquals(
                List.of(OperationSignature.fixed(BatchNormInferenceAttrs.class, 5, 1)),
                BatchNormKind.BATCH_NORM_INFERENCE.signatures());
        OperationSignature signature = BatchNormKind.BATCH_NORM_INFERENCE.signatures().getFirst();
        assertAll(
                () -> signature.validateOccurrence(5, 1),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(4, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(6, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(5, 2)));
    }

    @Test
    void retainsNormalizedAxisAndExactEpsilonReference() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        BatchNormInferenceAttrs attrs = new BatchNormInferenceAttrs(3, epsilon);
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_INFERENCE, attrs);

        assertAll(
                () -> assertEquals(3, attrs.channelAxis()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertSame(attrs, operation.attrs()));
    }

    @Test
    void validatesAxisThenEpsilonWithExactMessages() {
        assertEquals("channelAxis must be non-negative: -1", assertThrows(
                IllegalArgumentException.class,
                () -> new BatchNormInferenceAttrs(-1, null)).getMessage());
        assertEquals("epsilon", assertThrows(NullPointerException.class,
                () -> new BatchNormInferenceAttrs(0, null)).getMessage());
        assertEquals("epsilon must have a floating data type, but was INT32", assertThrows(
                IllegalArgumentException.class,
                () -> new BatchNormInferenceAttrs(0, ScalarValue.int32(1))).getMessage());
    }

    @Test
    void acceptsEveryFloatingPositiveRepresentationAndRejectsInvalidClasses() {
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16(Float.MIN_NORMAL),
                ScalarValue.float32(Float.MIN_VALUE),
                ScalarValue.float64(Double.MIN_VALUE))) {
            assertSame(epsilon, new BatchNormInferenceAttrs(0, epsilon).epsilon());
        }
        for (ScalarValue epsilon : List.of(
                ScalarValue.bfloat16Bits((short) 0x0000),
                ScalarValue.bfloat16Bits((short) 0x8000),
                ScalarValue.bfloat16Bits((short) 0xBF80),
                ScalarValue.bfloat16Bits((short) 0x7F80),
                ScalarValue.bfloat16Bits((short) 0xFF80),
                ScalarValue.bfloat16Bits((short) 0x7FC1),
                ScalarValue.float32(0.0f),
                ScalarValue.float32(-0.0f),
                ScalarValue.float32(-1.0f),
                ScalarValue.float32(Float.POSITIVE_INFINITY),
                ScalarValue.float32(Float.NEGATIVE_INFINITY),
                ScalarValue.float32(Float.NaN),
                ScalarValue.float64(0.0d),
                ScalarValue.float64(-0.0d),
                ScalarValue.float64(-1.0d),
                ScalarValue.float64(Double.POSITIVE_INFINITY),
                ScalarValue.float64(Double.NEGATIVE_INFINITY),
                ScalarValue.float64(Double.NaN))) {
            assertEquals("epsilon must be finite and positive: " + epsilon, assertThrows(
                    IllegalArgumentException.class,
                    () -> new BatchNormInferenceAttrs(0, epsilon)).getMessage());
        }
    }

    @Test
    void locksInferenceFormulaEpsilonPlacementAndRepresentativeSpecialValues() {
        double epsilon = 1.0e-5;
        double first = ((1.0 - 1.0) / Math.sqrt(4.0 + epsilon)) * 2.0 + 0.5;
        double second = ((2.0 - 1.0) / Math.sqrt(4.0 + epsilon)) * 2.0 + 0.5;
        assertAll(
                () -> assertEquals(0.5, first, 0.0),
                () -> assertEquals(1.4999987500023437, second, 1.0e-15),
                () -> assertEquals(true, Double.isNaN(Math.sqrt(-1.0))),
                () -> assertEquals(true, Double.isNaN(0.0 / 0.0)),
                () -> assertEquals(Double.POSITIVE_INFINITY, 1.0 / 0.0),
                () -> assertEquals(true, Double.isNaN(
                        Double.POSITIVE_INFINITY / Double.POSITIVE_INFINITY)),
                () -> assertEquals(true, Double.isNaN(0.0 * Double.POSITIVE_INFINITY)));
    }
}
