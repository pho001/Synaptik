package io.github.pho001.synaptik.model.operation.normalization;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import java.util.List;
import org.junit.jupiter.api.Test;

final class BatchNormTrainingSemanticsTest {
    @Test
    void appendsTrainingKindWithExactAttrsAndFixedFiveByFiveSignature() {
        ScalarValue momentum = ScalarValue.float32(0.25f);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        BatchNormTrainingAttrs attrs = new BatchNormTrainingAttrs(2, momentum, epsilon);
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_TRAINING, attrs);

        assertAll(
                () -> assertEquals(
                        List.of(BatchNormKind.BATCH_NORM_INFERENCE,
                                BatchNormKind.BATCH_NORM_TRAINING),
                        List.of(BatchNormKind.values())),
                () -> assertEquals(2, attrs.channelAxis()),
                () -> assertSame(momentum, attrs.momentum()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertEquals(BatchNormTrainingAttrs.class,
                        operation.signature().attributesType()),
                () -> assertEquals(5, operation.signature().minimumInputs()),
                () -> assertEquals(5, operation.signature().maximumInputs()),
                () -> assertEquals(5, operation.signature().minimumOutputs()),
                () -> assertEquals(5, operation.signature().maximumOutputs()),
                () -> assertEquals(1,
                        BatchNormKind.BATCH_NORM_TRAINING.signatures().size()));
    }

    @Test
    void preservesCompletedInferenceSignatureAndRejectsCrossKindAttrs() {
        BatchNormInferenceAttrs inference = new BatchNormInferenceAttrs(
                0, ScalarValue.float64(1.0e-5));
        BatchNormTrainingAttrs training = new BatchNormTrainingAttrs(
                0, ScalarValue.float64(0.1), ScalarValue.float64(1.0e-5));
        Operation operation = new Operation(BatchNormKind.BATCH_NORM_INFERENCE, inference);

        assertAll(
                () -> assertEquals(5, operation.signature().minimumInputs()),
                () -> assertEquals(1, operation.signature().minimumOutputs()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(BatchNormKind.BATCH_NORM_INFERENCE, training)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new Operation(BatchNormKind.BATCH_NORM_TRAINING, inference)));
    }

    @Test
    void validatesIntrinsicComponentsInExactOrderAndAcceptsBothMomentumZeros() {
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);

        IllegalArgumentException axis = assertThrows(IllegalArgumentException.class,
                () -> new BatchNormTrainingAttrs(-1, null, null));
        NullPointerException nullMomentum = assertThrows(NullPointerException.class,
                () -> new BatchNormTrainingAttrs(0, null, null));
        IllegalArgumentException integralMomentum = assertThrows(IllegalArgumentException.class,
                () -> new BatchNormTrainingAttrs(0, ScalarValue.int32(0), null));
        IllegalArgumentException invalidMomentum = assertThrows(IllegalArgumentException.class,
                () -> new BatchNormTrainingAttrs(
                        0, ScalarValue.float32(Float.NaN), null));
        NullPointerException nullEpsilon = assertThrows(NullPointerException.class,
                () -> new BatchNormTrainingAttrs(0, ScalarValue.float32(0.5f), null));
        IllegalArgumentException integralEpsilon = assertThrows(IllegalArgumentException.class,
                () -> new BatchNormTrainingAttrs(
                        0, ScalarValue.float32(0.5f), ScalarValue.int64(1)));
        IllegalArgumentException invalidEpsilon = assertThrows(IllegalArgumentException.class,
                () -> new BatchNormTrainingAttrs(
                        0, ScalarValue.float32(0.5f), ScalarValue.float32(-0.0f)));

        assertAll(
                () -> assertEquals("channelAxis must be non-negative: -1", axis.getMessage()),
                () -> assertEquals("momentum", nullMomentum.getMessage()),
                () -> assertEquals(
                        "momentum must have a floating data type, but was INT32",
                        integralMomentum.getMessage()),
                () -> assertEquals(
                        "momentum must be finite and in [0, 1]: "
                                + ScalarValue.float32(Float.NaN),
                        invalidMomentum.getMessage()),
                () -> assertEquals("epsilon", nullEpsilon.getMessage()),
                () -> assertEquals(
                        "epsilon must have a floating data type, but was INT64",
                        integralEpsilon.getMessage()),
                () -> assertEquals(
                        "epsilon must be finite and positive: " + ScalarValue.float32(-0.0f),
                        invalidEpsilon.getMessage()),
                () -> assertSame(ScalarValue.float32(0.0f).dataType(),
                        new BatchNormTrainingAttrs(
                                0, ScalarValue.float32(0.0f), epsilon).momentum().dataType()),
                () -> assertEquals(ScalarValue.float32(-0.0f),
                        new BatchNormTrainingAttrs(
                                0, ScalarValue.float32(-0.0f), epsilon).momentum()));
    }

    @Test
    void rejectsEveryNonFiniteOrOutOfRangeMomentumAndNonPositiveEpsilon() {
        for (ScalarValue momentum : List.of(
                ScalarValue.float64(-0.01),
                ScalarValue.float64(1.01),
                ScalarValue.float64(Double.NEGATIVE_INFINITY),
                ScalarValue.float64(Double.POSITIVE_INFINITY),
                ScalarValue.float64(Double.NaN))) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> new BatchNormTrainingAttrs(
                            0, momentum, ScalarValue.float64(1.0e-5))).getMessage()
                    .startsWith("momentum must be finite and in [0, 1]: "));
        }
        for (ScalarValue epsilon : List.of(
                ScalarValue.float64(0.0),
                ScalarValue.float64(-0.0),
                ScalarValue.float64(-1.0),
                ScalarValue.float64(Double.POSITIVE_INFINITY),
                ScalarValue.float64(Double.NaN))) {
            assertTrue(assertThrows(IllegalArgumentException.class,
                    () -> new BatchNormTrainingAttrs(
                            0, ScalarValue.float64(0.5), epsilon)).getMessage()
                    .startsWith("epsilon must be finite and positive: "));
        }
        assertEquals(DataType.BFLOAT16,
                new BatchNormTrainingAttrs(
                        0, ScalarValue.bfloat16(1.0f), ScalarValue.bfloat16(0.125f))
                        .momentum().dataType());
    }
}
