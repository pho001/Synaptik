package io.github.pho001.synaptik.model.operation.linalg;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MatmulSemanticsTest {
    @Test
    void exposesExactlyOneTypedParameterlessSemanticKind() {
        assertAll(
                () -> assertTrue(MatmulKind.class.isEnum()),
                () -> assertTrue(Modifier.isPublic(MatmulKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(MatmulKind.class.getModifiers())),
                () -> assertEquals(Set.of(OperationKind.class), Set.of(MatmulKind.class.getInterfaces())),
                () -> assertEquals(List.of(MatmulKind.MATMUL), List.of(MatmulKind.values())),
                () -> assertEquals("MATMUL", MatmulKind.MATMUL.name()),
                () -> assertSame(MatmulKind.MATMUL, MatmulKind.valueOf("MATMUL")));
    }

    @Test
    void declaresOneStableImmutableTwoInputOneOutputSignature() {
        List<OperationSignature> first = MatmulKind.MATMUL.signatures();
        List<OperationSignature> second = MatmulKind.MATMUL.signatures();
        OperationSignature signature = first.getFirst();

        assertAll(
                () -> assertSame(first, second),
                () -> assertEquals(1, first.size()),
                () -> assertSame(NoOperationAttrs.class, signature.attributesType()),
                () -> assertEquals(2, signature.minimumInputs()),
                () -> assertEquals(2, signature.maximumInputs()),
                () -> assertEquals(1, signature.minimumOutputs()),
                () -> assertEquals(1, signature.maximumOutputs()),
                () -> assertSame(signature, MatmulKind.MATMUL.signatureFor(NoOperationAttrs.INSTANCE)),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.add(signature)));
    }

    @Test
    void operationRetainsTheExactKindAndOnlyNoAttributes() {
        Operation operation = new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);

        assertAll(
                () -> assertSame(MatmulKind.MATMUL, operation.kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                () -> assertEquals(2, operation.signature().minimumInputs()),
                () -> assertEquals(1, operation.signature().minimumOutputs()),
                () -> assertFalse(operation.toString().isBlank()));
    }
}
