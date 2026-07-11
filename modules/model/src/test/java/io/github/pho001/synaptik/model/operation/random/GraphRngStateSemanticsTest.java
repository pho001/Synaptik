package io.github.pho001.synaptik.model.operation.random;

import static io.github.pho001.synaptik.model.operation.OperationSignatureTest.assertSignatureEnumShape;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

public final class GraphRngStateSemanticsTest {
    @Test
    void preservesEveryRawKeyAndCounterBitPatternWithRecordValueSemantics() {
        for (long key : new long[] {0L, 1L, -1L, Long.MIN_VALUE, Long.MAX_VALUE}) {
            for (long counter : new long[] {0L, 1L, -1L, Long.MIN_VALUE, Long.MAX_VALUE}) {
                GraphRngStateAttrs attrs = new GraphRngStateAttrs(key, counter);
                GraphRngStateAttrs equal = new GraphRngStateAttrs(key, counter);

                assertAll(
                        () -> assertEquals(key, attrs.key()),
                        () -> assertEquals(counter, attrs.counter()),
                        () -> assertEquals(attrs, equal),
                        () -> assertEquals(attrs.hashCode(), equal.hashCode()));
            }
        }

        assertNotEquals(new GraphRngStateAttrs(-1L, 0L), new GraphRngStateAttrs(0L, -1L));
    }

    @Test
    void exposesOnlyTheExactImmutableAttributeRecordSurface() {
        var components = GraphRngStateAttrs.class.getRecordComponents();

        assertAll(
                () -> assertTrue(GraphRngStateAttrs.class.isRecord()),
                () -> assertEquals(List.of(OperationAttrs.class),
                        List.of(GraphRngStateAttrs.class.getInterfaces())),
                () -> assertEquals(List.of("key", "counter"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(long.class, long.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, GraphRngStateAttrs.class.getDeclaredClasses().length));
    }

    @Test
    void declaresTheExactZeroInputOneOutputInitializerSignature() {
        OperationSignature expected =
                OperationSignature.fixed(GraphRngStateAttrs.class, 0, 1);
        Operation operation = new Operation(
                GraphRngKind.INITIAL_STATE, new GraphRngStateAttrs(-1L, Long.MIN_VALUE));

        assertAll(
                () -> assertEquals(List.of(expected), GraphRngKind.INITIAL_STATE.signatures()),
                () -> assertEquals(expected, operation.signature()),
                () -> assertSame(GraphRngKind.INITIAL_STATE, operation.kind()),
                () -> assertTrue(operation.attrs() instanceof GraphRngStateAttrs),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> GraphRngKind.INITIAL_STATE.signatures().clear()),
                () -> assertEquals(List.of(OperationKind.class),
                        List.of(GraphRngKind.class.getInterfaces())));
        assertSignatureEnumShape(GraphRngKind.class);
    }
}
