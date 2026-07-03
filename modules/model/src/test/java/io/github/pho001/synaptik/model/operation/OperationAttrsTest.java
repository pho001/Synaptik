package io.github.pho001.synaptik.model.operation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class OperationAttrsTest {
    @Test
    void remainsAZeroMethodMarkerInterface() {
        assertAll(
                () -> assertTrue(OperationAttrs.class.isInterface()),
                () -> assertEquals(0, OperationAttrs.class.getDeclaredMethods().length));
    }

    @Test
    void acceptsTypedStructuralRecordAttributes() {
        SampleAttrs first = new SampleAttrs(2, true);
        SampleAttrs equal = new SampleAttrs(2, true);
        SampleAttrs different = new SampleAttrs(3, false);

        assertAll(
                () -> assertTrue(SampleAttrs.class.isRecord()),
                () -> assertInstanceOf(OperationAttrs.class, first),
                () -> assertEquals(2, first.axis()),
                () -> assertTrue(first.keepDimensions()),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, different));
    }

    @Test
    void providesOneCanonicalEmptyAttributeValue() {
        NoOperationAttrs[] values = NoOperationAttrs.values();

        assertAll(
                () -> assertEquals(1, values.length),
                () -> assertSame(NoOperationAttrs.INSTANCE, values[0]),
                () -> assertSame(
                        NoOperationAttrs.INSTANCE, NoOperationAttrs.valueOf("INSTANCE")),
                () -> assertInstanceOf(OperationAttrs.class, NoOperationAttrs.INSTANCE));
    }

    @Test
    void suppliesDeterministicDiagnosticText() {
        String emptyText = NoOperationAttrs.INSTANCE.toString();
        String typedText = new SampleAttrs(2, true).toString();

        assertAll(
                () -> assertEquals("INSTANCE", emptyText),
                () -> assertTrue(typedText.contains("SampleAttrs")),
                () -> assertTrue(typedText.contains("axis=2")),
                () -> assertTrue(typedText.contains("keepDimensions=true")));
    }

    private record SampleAttrs(int axis, boolean keepDimensions) implements OperationAttrs {}
}
