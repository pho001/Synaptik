package io.github.pho001.synaptik.model.operation;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class OperationTest {
    @Test
    void hasExactlyTheKindAndAttributesRecordState() {
        var components = Operation.class.getRecordComponents();
        var instanceFields =
                Arrays.stream(Operation.class.getDeclaredFields())
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .toList();

        assertAll(
                () -> assertTrue(Operation.class.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("kind", components[0].getName()),
                () -> assertEquals(OperationKind.class, components[0].getType()),
                () -> assertEquals("attrs", components[1].getName()),
                () -> assertEquals(OperationAttrs.class, components[1].getType()),
                () -> assertEquals(List.of("kind", "attrs"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))));
    }

    @Test
    void preservesTheExactKindAndAttributesReferences() {
        OperationKind kind = SampleKind.SAMPLE;
        OperationAttrs attrs = new SampleAttrs(2, true);

        Operation operation = new Operation(kind, attrs);

        assertAll(
                () -> assertSame(kind, operation.kind()),
                () -> assertSame(attrs, operation.attrs()),
                () -> assertEquals(SampleKind.SAMPLE, operation.kind()),
                () -> assertEquals(new SampleAttrs(2, true), operation.attrs()));
    }

    @Test
    void rejectsEachNullComponentWithItsComponentName() {
        NullPointerException nullKind =
                assertThrows(
                        NullPointerException.class,
                        () -> new Operation(null, NoOperationAttrs.INSTANCE));
        NullPointerException nullAttrs =
                assertThrows(
                        NullPointerException.class,
                        () -> new Operation(SampleKind.SAMPLE, null));

        assertAll(
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals("attrs", nullAttrs.getMessage()));
    }

    @Test
    void usesStructuralRecordEqualityAndHashing() {
        Operation first = new Operation(SampleKind.SAMPLE, new SampleAttrs(2, true));
        Operation equal = new Operation(SampleKind.SAMPLE, new SampleAttrs(2, true));
        Operation differentKind = new Operation(SampleKind.OTHER, new SampleAttrs(2, true));
        Operation differentAttrs = new Operation(SampleKind.SAMPLE, new SampleAttrs(3, false));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentKind),
                () -> assertNotEquals(first, differentAttrs));
    }

    @Test
    void keepsKindsFromDifferentConcreteTypesDistinct() {
        Operation first = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);
        Operation second = new Operation(OtherSampleKind.SAMPLE, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertEquals(first.kind().name(), second.kind().name()),
                () -> assertNotEquals(first.kind(), second.kind()),
                () -> assertNotEquals(first, second));
    }

    @Test
    void retainsTheCanonicalNoAttributesSingleton() {
        Operation operation = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);

        assertSame(NoOperationAttrs.INSTANCE, operation.attrs());
    }

    @Test
    void includesComponentsAndRepresentativeValuesInDiagnosticText() {
        String text = new Operation(SampleKind.SAMPLE, new SampleAttrs(2, true)).toString();

        assertAll(
                () -> assertTrue(text.contains("Operation")),
                () -> assertTrue(text.contains("kind=")),
                () -> assertTrue(text.contains("SAMPLE")),
                () -> assertTrue(text.contains("attrs=")),
                () -> assertTrue(text.contains("SampleAttrs")),
                () -> assertTrue(text.contains("axis=2")),
                () -> assertTrue(text.contains("keepDimensions=true")));
    }

    @Test
    void resolvesStableSignatureAndRejectsAnIncompatibleAttributesClass() {
        Operation operation =
                new Operation(SampleKind.SAMPLE, new SampleAttrs(2, true));

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> new Operation(OtherSampleKind.SAMPLE, new SampleAttrs(2, true)));

        assertAll(
                () -> assertSame(operation.signature(), operation.signature()),
                () -> assertSame(SampleAttrs.class, operation.signature().attributesType()),
                () -> assertTrue(failure.getMessage().contains("OtherSampleKind.SAMPLE")),
                () -> assertTrue(failure.getMessage().contains(SampleAttrs.class.getName())),
                () -> assertTrue(
                        failure.getMessage().contains(NoOperationAttrs.class.getName())));
    }

    @Test
    void failsClosedForMissingOrDuplicateKindSignatures() {
        assertAll(
                () -> assertThrows(
                        IllegalStateException.class,
                        () -> new Operation(EmptyKind.EMPTY, NoOperationAttrs.INSTANCE)),
                () -> assertThrows(
                        IllegalStateException.class,
                        () -> new Operation(DuplicateKind.DUPLICATE, NoOperationAttrs.INSTANCE)));
    }

    private enum SampleKind implements OperationKind {
        SAMPLE,
        OTHER;

        private static final List<OperationSignature> SIGNATURES = List.of(
                OperationSignature.fixed(SampleAttrs.class, 1, 1),
                OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

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

    private enum EmptyKind implements OperationKind {
        EMPTY;

        private static final List<OperationSignature> SIGNATURES = List.of();

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private enum DuplicateKind implements OperationKind {
        DUPLICATE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                OperationSignature.fixed(NoOperationAttrs.class, 1, 1),
                OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private record SampleAttrs(int axis, boolean keepDimensions) implements OperationAttrs {}
}
