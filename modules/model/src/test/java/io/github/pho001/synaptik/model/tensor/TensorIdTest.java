package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class TensorIdTest {
    @Test
    void acceptsBoundaryValuesAndReturnsStoredValue() {
        assertAll(
                () -> assertEquals(0, new TensorId(0).value()),
                () -> assertEquals(Long.MAX_VALUE, new TensorId(Long.MAX_VALUE).value()));
    }

    @Test
    void rejectsNegativeValuesWithoutSentinels() {
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> new TensorId(-1)),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> new TensorId(Long.MIN_VALUE)));
    }

    @Test
    void usesStructuralRecordEqualityAndHashing() {
        TensorId first = new TensorId(42);
        TensorId equal = new TensorId(42);
        TensorId different = new TensorId(43);

        assertAll(
                () -> assertTrue(TensorId.class.isRecord()),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different));
    }

    @Test
    void diagnosticTextIdentifiesTypeAndValue() {
        String text = new TensorId(42).toString();

        assertAll(
                () -> assertTrue(text.contains("TensorId")),
                () -> assertTrue(text.contains("42")));
    }
}
