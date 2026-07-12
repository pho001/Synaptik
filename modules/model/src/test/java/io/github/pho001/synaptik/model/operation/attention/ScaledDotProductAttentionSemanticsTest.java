package io.github.pho001.synaptik.model.operation.attention;

import static io.github.pho001.synaptik.model.operation.OperationSignatureTest.assertSignatureEnumShape;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

public final class ScaledDotProductAttentionSemanticsTest {
    @Test
    void exposesTheExactImmutableAttributeAndKindSurface() {
        var components = ScaledDotProductAttentionAttrs.class.getRecordComponents();
        var attrs = new ScaledDotProductAttentionAttrs(
                Optional.of(ScalarValue.float32(0.5f)), true);
        OperationSignature expected = OperationSignature.inputRange(
                ScaledDotProductAttentionAttrs.class, 3, 4, 1);
        Operation operation = new Operation(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION, attrs);

        assertAll(
                () -> assertTrue(ScaledDotProductAttentionAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(ScaledDotProductAttentionAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("scale", "causal"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Optional.class, boolean.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, ScaledDotProductAttentionAttrs.class.getDeclaredClasses().length),
                () -> assertSame(attrs.scale(), attrs.scale()),
                () -> assertTrue(attrs.causal()),
                () -> assertEquals(List.of(expected),
                        ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION
                                .signatures().clear()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(ScaledDotProductAttentionKind.class.getInterfaces())));
        assertSignatureEnumShape(ScaledDotProductAttentionKind.class);
    }

    @Test
    void acceptsEmptyAndEveryPositiveFiniteFloatingScaleWithExactBits() {
        var empty = new ScaledDotProductAttentionAttrs(Optional.empty(), false);
        List<ScalarValue> values = List.of(
                ScalarValue.float64(Double.MIN_VALUE),
                ScalarValue.float64(Double.MAX_VALUE),
                ScalarValue.float32(Float.MIN_VALUE),
                ScalarValue.float32(Float.MAX_VALUE),
                ScalarValue.bfloat16Bits((short) 0x0001),
                ScalarValue.bfloat16Bits((short) 0x7F7F));

        assertAll(
                () -> assertTrue(empty.scale().isEmpty()),
                () -> assertFalse(empty.causal()),
                () -> assertEquals(empty,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)));
        for (ScalarValue value : values) {
            var attrs = new ScaledDotProductAttentionAttrs(Optional.of(value), false);
            assertSame(value, attrs.scale().orElseThrow());
        }
        assertNotEquals(
                new ScaledDotProductAttentionAttrs(Optional.empty(), false),
                new ScaledDotProductAttentionAttrs(Optional.empty(), true));
    }

    @Test
    void rejectsNullNonFloatingNonFiniteAndNonPositiveScalesWithExactMessages() {
        assertEquals("scale", assertThrows(NullPointerException.class,
                () -> new ScaledDotProductAttentionAttrs(null, false)).getMessage());
        for (ScalarValue value : List.of(
                ScalarValue.int32(1), ScalarValue.int64(1), ScalarValue.bool(true))) {
            IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                    () -> new ScaledDotProductAttentionAttrs(Optional.of(value), false));
            assertEquals("scale must have a floating data type, but was " + value.dataType(),
                    failure.getMessage());
        }

        assertScaleFailure(ScalarValue.float64(0.0d), "0.0");
        assertScaleFailure(ScalarValue.float64(-0.0d), "-0.0");
        assertScaleFailure(ScalarValue.float64(-1.0d), "-1.0");
        assertScaleFailure(ScalarValue.float64(Double.NaN), "NaN");
        assertScaleFailure(ScalarValue.float64(Double.POSITIVE_INFINITY), "Infinity");
        assertScaleFailure(ScalarValue.float32(Float.NEGATIVE_INFINITY), "-Infinity");
        assertScaleFailure(ScalarValue.bfloat16Bits((short) 0x7FC0), "NaN");
    }

    private static void assertScaleFailure(ScalarValue value, String decoded) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> new ScaledDotProductAttentionAttrs(Optional.of(value), false));
        assertEquals("scale must be finite and positive: " + decoded, failure.getMessage());
    }
}
