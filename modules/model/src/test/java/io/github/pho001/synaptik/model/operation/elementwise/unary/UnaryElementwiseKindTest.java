package io.github.pho001.synaptik.model.operation.elementwise.unary;

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
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class UnaryElementwiseKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        UnaryElementwiseKind[] values = UnaryElementwiseKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new UnaryElementwiseKind[] {
                            UnaryElementwiseKind.ABS,
                            UnaryElementwiseKind.NEG,
                            UnaryElementwiseKind.INV,
                            UnaryElementwiseKind.LOG,
                            UnaryElementwiseKind.EXP,
                            UnaryElementwiseKind.ERF,
                            UnaryElementwiseKind.SQRT,
                            UnaryElementwiseKind.FLOOR,
                            UnaryElementwiseKind.CEIL,
                            UnaryElementwiseKind.SIGN,
                            UnaryElementwiseKind.RELU,
                            UnaryElementwiseKind.SIGMOID,
                            UnaryElementwiseKind.TANH,
                            UnaryElementwiseKind.FAST_EXP,
                            UnaryElementwiseKind.FAST_TANH
                        },
                        values),
                () -> assertEquals(
                        List.of(
                                "ABS",
                                "NEG",
                                "INV",
                                "LOG",
                                "EXP",
                                "ERF",
                                "SQRT",
                                "FLOOR",
                                "CEIL",
                                "SIGN",
                                "RELU",
                                "SIGMOID",
                                "TANH",
                                "FAST_EXP",
                                "FAST_TANH"),
                        Arrays.stream(values).map(UnaryElementwiseKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = UnaryElementwiseKind.EXP;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("EXP", kind.name()),
                () -> assertEquals("EXP", kind.toString()),
                () -> assertSame(UnaryElementwiseKind.EXP, UnaryElementwiseKind.valueOf("EXP")),
                () -> assertEquals(UnaryElementwiseKind.EXP, UnaryElementwiseKind.EXP),
                () -> assertEquals(
                        UnaryElementwiseKind.EXP.hashCode(),
                        UnaryElementwiseKind.EXP.hashCode()),
                () -> assertNotEquals(UnaryElementwiseKind.EXP, UnaryElementwiseKind.LOG));
    }

    @Test
    void keepsStrictAndFastRequestsDistinct() {
        assertAll(
                () -> assertNotEquals(UnaryElementwiseKind.EXP, UnaryElementwiseKind.FAST_EXP),
                () -> assertNotEquals(UnaryElementwiseKind.TANH, UnaryElementwiseKind.FAST_TANH),
                () -> assertNotEquals(
                        new Operation(
                                UnaryElementwiseKind.EXP, NoOperationAttrs.INSTANCE),
                        new Operation(
                                UnaryElementwiseKind.FAST_EXP, NoOperationAttrs.INSTANCE)),
                () -> assertNotEquals(
                        new Operation(
                                UnaryElementwiseKind.TANH, NoOperationAttrs.INSTANCE),
                        new Operation(
                                UnaryElementwiseKind.FAST_TANH, NoOperationAttrs.INSTANCE)));
    }

    @Test
    void declaresNoProjectStateBehaviorOrNestedTypes() {
        var instanceFields =
                Arrays.stream(UnaryElementwiseKind.class.getDeclaredFields())
                        .filter(field -> !Modifier.isStatic(field.getModifiers()))
                        .toList();
        var instanceMethods =
                Arrays.stream(UnaryElementwiseKind.class.getDeclaredMethods())
                        .filter(method -> !Modifier.isStatic(method.getModifiers()))
                        .toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(UnaryElementwiseKind.class.getModifiers())),
                () -> assertTrue(UnaryElementwiseKind.class.isEnum()),
                () -> assertTrue(instanceFields.isEmpty()),
                () -> assertTrue(instanceMethods.isEmpty()),
                () -> assertEquals(0, UnaryElementwiseKind.class.getDeclaredClasses().length),
                () -> assertTrue(Arrays.stream(UnaryElementwiseKind.values())
                        .allMatch(value -> value.getClass() == UnaryElementwiseKind.class)));
    }

    @Test
    void composesEveryKindWithTheCanonicalNoAttributesValue() {
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind unary = UnaryElementwiseKind.EXP;
        OperationKind other = OtherKind.EXP;

        assertAll(
                () -> assertEquals(unary.name(), other.name()),
                () -> assertNotEquals(unary, other),
                () -> assertNotEquals(
                        new Operation(unary, NoOperationAttrs.INSTANCE),
                        new Operation(other, NoOperationAttrs.INSTANCE)));
    }

    private enum OtherKind implements OperationKind {
        EXP
    }
}
