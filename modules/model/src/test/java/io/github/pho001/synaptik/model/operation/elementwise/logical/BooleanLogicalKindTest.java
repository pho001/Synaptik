package io.github.pho001.synaptik.model.operation.elementwise.logical;

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

class BooleanLogicalKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        BooleanLogicalKind[] values = BooleanLogicalKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new BooleanLogicalKind[] {
                            BooleanLogicalKind.AND,
                            BooleanLogicalKind.OR,
                            BooleanLogicalKind.NOT
                        },
                        values),
                () -> assertEquals(
                        List.of("AND", "OR", "NOT"),
                        Arrays.stream(values).map(BooleanLogicalKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = BooleanLogicalKind.AND;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("AND", kind.name()),
                () -> assertEquals("AND", kind.toString()),
                () -> assertSame(BooleanLogicalKind.AND, BooleanLogicalKind.valueOf("AND")),
                () -> assertEquals(BooleanLogicalKind.AND, BooleanLogicalKind.AND),
                () -> assertEquals(
                        BooleanLogicalKind.AND.hashCode(), BooleanLogicalKind.AND.hashCode()),
                () -> assertNotEquals(BooleanLogicalKind.AND, BooleanLogicalKind.OR));
    }

    @Test
    void exposesOnlyTheExactEnumShapeWithoutArityState() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(BooleanLogicalKind.class);
    }

    @Test
    void composesEveryKindWithTheCanonicalNoAttributesValue() {
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind logical = BooleanLogicalKind.AND;
        OperationKind other = OtherKind.AND;

        assertAll(
                () -> assertEquals(logical.name(), other.name()),
                () -> assertNotEquals(logical, other),
                () -> assertNotEquals(
                        new Operation(logical, NoOperationAttrs.INSTANCE),
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
        AND;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
