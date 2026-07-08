package io.github.pho001.synaptik.model.operation.elementwise.selection;

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

class WhereSelectionKindTest {
    @Test
    void declaresExactlyTheRequiredVocabulary() {
        WhereSelectionKind[] values = WhereSelectionKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new WhereSelectionKind[] {WhereSelectionKind.WHERE}, values),
                () -> assertEquals(
                        List.of("WHERE"),
                        Arrays.stream(values).map(WhereSelectionKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = WhereSelectionKind.WHERE;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("WHERE", kind.name()),
                () -> assertEquals("WHERE", kind.toString()),
                () -> assertSame(WhereSelectionKind.WHERE, WhereSelectionKind.valueOf("WHERE")),
                () -> assertEquals(WhereSelectionKind.WHERE, WhereSelectionKind.WHERE),
                () -> assertEquals(
                        WhereSelectionKind.WHERE.hashCode(),
                        WhereSelectionKind.WHERE.hashCode()));
    }

    @Test
    void exposesOnlyTheExactEnumShapeWithoutArityState() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(WhereSelectionKind.class);
    }

    @Test
    void composesWithTheCanonicalNoAttributesValue() {
        Operation operation =
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertSame(WhereSelectionKind.WHERE, operation.kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind selection = WhereSelectionKind.WHERE;
        OperationKind other = OtherKind.WHERE;

        assertAll(
                () -> assertEquals(selection.name(), other.name()),
                () -> assertNotEquals(selection, other),
                () -> assertNotEquals(
                        new Operation(selection, NoOperationAttrs.INSTANCE),
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
        WHERE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 3, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
