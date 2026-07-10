package io.github.pho001.synaptik.model.operation.elementwise.classification;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class FloatingClassificationKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        FloatingClassificationKind[] values = FloatingClassificationKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new FloatingClassificationKind[] {
                            FloatingClassificationKind.IS_FINITE,
                            FloatingClassificationKind.IS_NAN,
                            FloatingClassificationKind.IS_INF
                        },
                        values),
                () -> assertEquals(
                        List.of("IS_FINITE", "IS_NAN", "IS_INF"),
                        Arrays.stream(values).map(FloatingClassificationKind::name).toList()));
    }

    @Test
    void hasTheExactFamilyShapeAndCanonicalParameterlessSignature() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(FloatingClassificationKind.class);

        for (FloatingClassificationKind kind : FloatingClassificationKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);
            assertAll(
                    () -> assertInstanceOf(OperationKind.class, kind),
                    () -> assertEquals(
                            List.of(io.github.pho001.synaptik.model.operation.OperationSignature
                                    .fixed(NoOperationAttrs.class, 1, 1)),
                            kind.signatures()),
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                    () -> assertThrows(
                            IllegalArgumentException.class,
                            () -> new Operation(kind, new OtherAttrs())));
        }
    }

    @Test
    void preservesTypedIdentityAndDiagnosticNames() {
        OperationKind classification = FloatingClassificationKind.IS_NAN;
        OperationKind other = OtherKind.IS_NAN;

        assertAll(
                () -> assertEquals("IS_NAN", classification.name()),
                () -> assertEquals("IS_NAN", classification.toString()),
                () -> assertSame(
                        FloatingClassificationKind.IS_NAN,
                        FloatingClassificationKind.valueOf("IS_NAN")),
                () -> assertNotEquals(classification, other),
                () -> assertNotEquals(
                        new Operation(classification, NoOperationAttrs.INSTANCE),
                        new Operation(other, NoOperationAttrs.INSTANCE)));
    }

    private record OtherAttrs() implements OperationAttrs {}

    private enum OtherKind implements OperationKind {
        IS_NAN;

        @Override
        public List<io.github.pho001.synaptik.model.operation.OperationSignature> signatures() {
            return List.of(io.github.pho001.synaptik.model.operation.OperationSignature.fixed(
                    NoOperationAttrs.class, 1, 1));
        }
    }
}
