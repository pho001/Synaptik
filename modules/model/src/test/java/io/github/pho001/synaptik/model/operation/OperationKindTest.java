package io.github.pho001.synaptik.model.operation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class OperationKindTest {
    @Test
    void exposesNameAndFamilyOwnedSignatureContracts() {
        var publicMethods = Arrays.stream(OperationKind.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName() + ":" + method.getReturnType().getSimpleName())
                .sorted()
                .toList();

        assertAll(
                () -> assertTrue(OperationKind.class.isInterface()),
                () -> assertEquals(
                        List.of(
                                "name:String",
                                "signatureFor:OperationSignature",
                                "signatures:List"),
                        publicMethods));
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
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private enum OtherSampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
