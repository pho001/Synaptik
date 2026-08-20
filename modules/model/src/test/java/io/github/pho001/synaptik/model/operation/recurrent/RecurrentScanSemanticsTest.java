package io.github.pho001.synaptik.model.operation.recurrent;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public final class RecurrentScanSemanticsTest {
    @Test
    void exposesOnlyTheClosedDirectionAndKindEnums() {
        assertAll(
                () -> assertEquals(List.of("FORWARD", "REVERSE"),
                        Arrays.stream(RecurrentDirection.values()).map(Enum::name).toList()),
                () -> assertEquals(
                        List.of("RNN_TANH", "GRU_RESET_AFTER", "LSTM"),
                        Arrays.stream(RecurrentScanKind.values()).map(Enum::name).toList()),
                () -> assertEquals(
                        List.of(OperationAttrs.class),
                        Arrays.asList(RecurrentDirection.class.getInterfaces())),
                () -> assertEquals(
                        List.of(OperationKind.class),
                        Arrays.asList(RecurrentScanKind.class.getInterfaces())),
                () -> assertTrue(Modifier.isPublic(RecurrentDirection.class.getModifiers())),
                () -> assertTrue(Modifier.isPublic(RecurrentScanKind.class.getModifiers())));
    }

    @Test
    void declaresExactDirectionTypedStructuralSignatures() {
        assertSignature(RecurrentScanKind.RNN_TANH, 5, 6, 2);
        assertSignature(RecurrentScanKind.GRU_RESET_AFTER, 5, 6, 2);
        assertSignature(RecurrentScanKind.LSTM, 6, 7, 3);
    }

    @Test
    void operationRetainsExactDirectionAndRejectsOtherAttributes() {
        for (RecurrentScanKind kind : RecurrentScanKind.values()) {
            for (RecurrentDirection direction : RecurrentDirection.values()) {
                Operation operation = new Operation(kind, direction);
                assertAll(
                        () -> assertSame(kind, operation.kind()),
                        () -> assertSame(direction, operation.attrs()),
                        () -> assertSame(kind.signatures().getFirst(), operation.signature()));
            }
            assertThrows(IllegalArgumentException.class,
                    () -> new Operation(kind, NoOperationAttrs.INSTANCE));
        }
    }

    @Test
    void signatureListsAreStableAndImmutable() {
        for (RecurrentScanKind kind : RecurrentScanKind.values()) {
            List<OperationSignature> signatures = kind.signatures();
            assertAll(
                    () -> assertSame(signatures, kind.signatures()),
                    () -> assertEquals(1, signatures.size()),
                    () -> assertThrows(UnsupportedOperationException.class,
                            () -> signatures.add(signatures.getFirst())));
        }
    }

    private static void assertSignature(
            RecurrentScanKind kind, int minimumInputs, int maximumInputs, int outputs) {
        OperationSignature signature = kind.signatures().getFirst();
        assertAll(
                () -> assertSame(RecurrentDirection.class, signature.attributesType()),
                () -> assertEquals(minimumInputs, signature.minimumInputs()),
                () -> assertEquals(maximumInputs, signature.maximumInputs()),
                () -> assertEquals(outputs, signature.minimumOutputs()),
                () -> assertEquals(outputs, signature.maximumOutputs()),
                () -> assertTrue(signature.acceptsInputCount(minimumInputs)),
                () -> assertTrue(signature.acceptsInputCount(maximumInputs)),
                () -> assertFalse(signature.acceptsInputCount(minimumInputs - 1)),
                () -> assertFalse(signature.acceptsInputCount(maximumInputs + 1)));
    }
}
