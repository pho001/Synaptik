package io.github.pho001.synaptik.model.operation.elementwise.unary;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
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
                            UnaryElementwiseKind.RECIPROCAL,
                            UnaryElementwiseKind.LOG,
                            UnaryElementwiseKind.LOG1P,
                            UnaryElementwiseKind.EXP,
                            UnaryElementwiseKind.EXPM1,
                            UnaryElementwiseKind.ERF,
                            UnaryElementwiseKind.SQRT,
                            UnaryElementwiseKind.RSQRT,
                            UnaryElementwiseKind.FLOOR,
                            UnaryElementwiseKind.CEIL,
                            UnaryElementwiseKind.SIGN,
                            UnaryElementwiseKind.RELU,
                            UnaryElementwiseKind.SIGMOID,
                            UnaryElementwiseKind.TANH,
                            UnaryElementwiseKind.GELU,
                            UnaryElementwiseKind.GELU_TANH_APPROXIMATION,
                            UnaryElementwiseKind.SILU
                        },
                        values),
                () -> assertEquals(
                        List.of(
                                "ABS",
                                "NEG",
                                "RECIPROCAL",
                                "LOG",
                                "LOG1P",
                                "EXP",
                                "EXPM1",
                                "ERF",
                                "SQRT",
                                "RSQRT",
                                "FLOOR",
                                "CEIL",
                                "SIGN",
                                "RELU",
                                "SIGMOID",
                                "TANH",
                                "GELU",
                                "GELU_TANH_APPROXIMATION",
                                "SILU"),
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
    void retainsReciprocalIdentityAndRejectsRemovedNames() {
        assertAll(
                () -> assertSame(
                        UnaryElementwiseKind.RECIPROCAL,
                        UnaryElementwiseKind.valueOf("RECIPROCAL")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> UnaryElementwiseKind.valueOf("IN" + "V")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> UnaryElementwiseKind.valueOf("FAST" + "_EXP")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> UnaryElementwiseKind.valueOf("FAST" + "_TANH")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> UnaryElementwiseKind.valueOf("SW" + "ISH")));
    }

    @Test
    void declaresNoProjectStateBehaviorOrNestedTypes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(UnaryElementwiseKind.class);
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
        EXP;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
