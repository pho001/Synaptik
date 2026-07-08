package io.github.pho001.synaptik.model.operation.elementwise.binary;

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

class BinaryArithmeticKindTest {
    @Test
    void declaresExactlyTheRequiredVocabularyInOrder() {
        BinaryArithmeticKind[] values = BinaryArithmeticKind.values();

        assertAll(
                () -> assertArrayEquals(
                        new BinaryArithmeticKind[] {
                            BinaryArithmeticKind.ADD,
                            BinaryArithmeticKind.SUB,
                            BinaryArithmeticKind.MUL,
                            BinaryArithmeticKind.DIV,
                            BinaryArithmeticKind.MIN,
                            BinaryArithmeticKind.MAX,
                            BinaryArithmeticKind.POW
                        },
                        values),
                () -> assertEquals(
                        List.of("ADD", "SUB", "MUL", "DIV", "MIN", "MAX", "POW"),
                        Arrays.stream(values).map(BinaryArithmeticKind::name).toList()));
    }

    @Test
    void implementsOperationKindThroughInheritedEnumBehavior() {
        OperationKind kind = BinaryArithmeticKind.ADD;

        assertAll(
                () -> assertInstanceOf(OperationKind.class, kind),
                () -> assertEquals("ADD", kind.name()),
                () -> assertEquals("ADD", kind.toString()),
                () -> assertSame(BinaryArithmeticKind.ADD, BinaryArithmeticKind.valueOf("ADD")),
                () -> assertEquals(BinaryArithmeticKind.ADD, BinaryArithmeticKind.ADD),
                () -> assertEquals(
                        BinaryArithmeticKind.ADD.hashCode(), BinaryArithmeticKind.ADD.hashCode()),
                () -> assertNotEquals(BinaryArithmeticKind.ADD, BinaryArithmeticKind.SUB));
    }

    @Test
    void declaresNoProjectStateBehaviorOrNestedTypes() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(BinaryArithmeticKind.class);
    }

    @Test
    void composesEveryKindWithTheCanonicalNoAttributesValue() {
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            Operation operation = new Operation(kind, NoOperationAttrs.INSTANCE);

            assertAll(
                    () -> assertSame(kind, operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()));
        }
    }

    @Test
    void keepsEqualDiagnosticNamesTypedByTheirKindFamily() {
        OperationKind binary = BinaryArithmeticKind.ADD;
        OperationKind other = OtherKind.ADD;

        assertAll(
                () -> assertEquals(binary.name(), other.name()),
                () -> assertNotEquals(binary, other),
                () -> assertNotEquals(
                        new Operation(binary, NoOperationAttrs.INSTANCE),
                        new Operation(other, NoOperationAttrs.INSTANCE)));
    }

    private enum OtherKind implements OperationKind {
        ADD;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
