package io.github.pho001.synaptik.model.operation.loss;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;
import org.junit.jupiter.api.Test;

class LossSemanticsTest {
    @Test
    void declaresExactlyTheSelectedKindReductionVocabularyAndFixedSignature() {
        assertAll(
                () -> assertEquals(
                        List.of(LossKind.MEAN_SQUARED_ERROR), List.of(LossKind.values())),
                () -> assertEquals(
                        List.of(LossReduction.NONE, LossReduction.SUM, LossReduction.MEAN),
                        List.of(LossReduction.values())),
                () -> assertFalse(OperationAttrs.class.isAssignableFrom(LossReduction.class)),
                () -> assertEquals(
                        List.of(OperationSignature.fixed(MeanSquaredErrorAttrs.class, 2, 1)),
                        LossKind.MEAN_SQUARED_ERROR.signatures()));
    }

    @Test
    void retainsExactReductionAndAcceptsOnlyTwoInputOneOutputOccurrences() {
        MeanSquaredErrorAttrs attrs = new MeanSquaredErrorAttrs(LossReduction.MEAN);
        Operation operation = new Operation(LossKind.MEAN_SQUARED_ERROR, attrs);
        OperationSignature signature = operation.signature();

        assertAll(
                () -> assertSame(LossReduction.MEAN, attrs.reduction()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertSame(LossKind.MEAN_SQUARED_ERROR, operation.kind()),
                () -> signature.validateOccurrence(2, 1),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(1, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(3, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> signature.validateOccurrence(2, 2)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(
                                LossKind.MEAN_SQUARED_ERROR, NoOperationAttrs.INSTANCE)));
    }

    @Test
    void rejectsNullReductionWithExactMessage() {
        NullPointerException failure = assertThrows(
                NullPointerException.class, () -> new MeanSquaredErrorAttrs(null));
        assertEquals("reduction", failure.getMessage());
    }

    @Test
    void locksFormulaReductionDenominatorAndEmptyDomainMeaningWithoutTensorExecution() {
        double[] prediction = {1.0, 2.0, 4.0};
        double[] target = {1.0, 4.0, 1.0};
        double[] losses = new double[prediction.length];
        double sum = 0.0;
        for (int index = 0; index < prediction.length; index++) {
            double difference = prediction[index] - target[index];
            losses[index] = difference * difference;
            sum += losses[index];
        }
        double reducedSum = sum;

        assertAll(
                () -> assertEquals(List.of(0.0, 4.0, 9.0),
                        List.of(losses[0], losses[1], losses[2])),
                () -> assertEquals(13.0, reducedSum),
                () -> assertEquals(13.0 / 3.0, reducedSum / prediction.length),
                () -> assertEquals(0L, Double.doubleToRawLongBits(0.0d)),
                () -> assertTrue(Double.isNaN(0.0d / 0.0d)));
    }

    @Test
    void locksRequiredSpecialValueClassesWithoutSelectingAnAlgorithm() {
        double equalPositiveInfinity = Double.POSITIVE_INFINITY - Double.POSITIVE_INFINITY;
        double finiteFromInfinity = Double.POSITIVE_INFINITY - 1.0d;
        double oppositeInfinities = Double.POSITIVE_INFINITY - Double.NEGATIVE_INFINITY;
        double signedZeroDifference = -0.0d - 0.0d;

        assertAll(
                () -> assertTrue(Double.isNaN(Double.NaN * Double.NaN)),
                () -> assertTrue(Double.isNaN(equalPositiveInfinity * equalPositiveInfinity)),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        finiteFromInfinity * finiteFromInfinity),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        oppositeInfinities * oppositeInfinities),
                () -> assertEquals(0L, Double.doubleToRawLongBits(
                        signedZeroDifference * signedZeroDifference)),
                () -> assertTrue(Double.isNaN(Double.NaN + Double.POSITIVE_INFINITY)),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        Double.MAX_VALUE * Double.MAX_VALUE),
                () -> assertEquals(0L, Double.doubleToRawLongBits(
                        Double.MIN_VALUE * Double.MIN_VALUE)));
    }
}
