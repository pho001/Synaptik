package io.github.pho001.synaptik.model.operation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import org.junit.jupiter.api.Test;

class OperationKindTest {
    @Test
    void exposesOnlyTheSemanticNameContract() {
        var methods = OperationKind.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(OperationKind.class.isInterface()),
                () -> assertEquals(1, methods.length),
                () -> assertEquals("name", methods[0].getName()),
                () -> assertEquals(String.class, methods[0].getReturnType()),
                () -> assertEquals(0, methods[0].getParameterCount()),
                () -> assertTrue(Modifier.isPublic(methods[0].getModifiers())),
                () -> assertTrue(Modifier.isAbstract(methods[0].getModifiers())));
    }

    @Test
    void enumKindsSatisfyTheContractThroughTheirStableEnumName() {
        OperationKind kind = SampleKind.SAMPLE;

        assertAll(
                () -> assertEquals("SAMPLE", kind.name()),
                () -> assertEquals(kind.name(), kind.name()),
                () -> assertEquals("SAMPLE", kind.toString()));
    }

    @Test
    void equalNamesDoNotCollapseDifferentConcreteKindTypes() {
        OperationKind first = SampleKind.SAMPLE;
        OperationKind second = OtherSampleKind.SAMPLE;

        assertAll(
                () -> assertEquals(first.name(), second.name()),
                () -> assertNotEquals(first, second),
                () -> assertNotEquals(first.getClass(), second.getClass()));
    }

    private enum SampleKind implements OperationKind {
        SAMPLE
    }

    private enum OtherSampleKind implements OperationKind {
        SAMPLE
    }
}
