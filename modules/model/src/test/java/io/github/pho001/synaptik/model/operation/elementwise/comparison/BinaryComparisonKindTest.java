package io.github.pho001.synaptik.model.operation.elementwise.comparison;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class BinaryComparisonKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        BinaryComparisonKind[] values = BinaryComparisonKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new BinaryComparisonKind[] {
                            BinaryComparisonKind.GREATER_THAN,
                            BinaryComparisonKind.GREATER_OR_EQUAL,
                            BinaryComparisonKind.LESS_THAN,
                            BinaryComparisonKind.LESS_OR_EQUAL,
                            BinaryComparisonKind.EQUAL,
                            BinaryComparisonKind.NOT_EQUAL
                        },
                        values),
                () -> assertEquals(
                        List.of(
                                "GREATER_THAN",
                                "GREATER_OR_EQUAL",
                                "LESS_THAN",
                                "LESS_OR_EQUAL",
                                "EQUAL",
                                "NOT_EQUAL"),
                        Arrays.stream(values).map(BinaryComparisonKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = BinaryComparisonKind.GREATER_THAN;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("GREATER_THAN", kind.name()),
                () -> assertEquals("GREATER_THAN", kind.toString()),
                () -> assertSame(
                        BinaryComparisonKind.GREATER_THAN,
                        BinaryComparisonKind.valueOf("GREATER_THAN")),
                () -> assertEquals(
                        BinaryComparisonKind.GREATER_THAN,
                        BinaryComparisonKind.GREATER_THAN),
                () -> assertEquals(
                        BinaryComparisonKind.GREATER_THAN.hashCode(),
                        BinaryComparisonKind.GREATER_THAN.hashCode()),
                () -> assertNotEquals(
                        BinaryComparisonKind.GREATER_THAN,
                        BinaryComparisonKind.GREATER_OR_EQUAL));
    }

    @Test
    void exposesOnlyTheExactEnumShape() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(BinaryComparisonKind.class);
    }

    @Test
    void composesEveryKindWithTheCanonicalNoAttributesValue() {
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind comparison = BinaryComparisonKind.EQUAL;
        OperationKind other = OtherKind.EQUAL;

        assertAll(
                () -> assertEquals(comparison.name(), other.name()),
                () -> assertNotEquals(comparison, other),
                () -> assertNotEquals(
                        new Operation(comparison, NoOperationAttrs.INSTANCE),
                        new Operation(other, NoOperationAttrs.INSTANCE)));
    }

    private static String methodSignature(java.lang.reflect.Method method) {
        String parameters = Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(java.util.stream.Collectors.joining(","));
        return method.getName()
                + "("
                + parameters
                + "):"
                + method.getReturnType().getName();
    }

    private enum OtherKind implements OperationKind {
        EQUAL;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
