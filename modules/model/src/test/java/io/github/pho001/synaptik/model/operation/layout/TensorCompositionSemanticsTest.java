package io.github.pho001.synaptik.model.operation.layout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;
import org.junit.jupiter.api.Test;

class TensorCompositionSemanticsTest {
    @Test
    void declaresExactlyConcatAndStack() {
        assertArrayEquals(
                new TensorCompositionKind[] {
                    TensorCompositionKind.CONCAT, TensorCompositionKind.STACK
                },
                TensorCompositionKind.values());
        assertSame(TensorCompositionKind.CONCAT, TensorCompositionKind.valueOf("CONCAT"));
        assertSame(TensorCompositionKind.STACK, TensorCompositionKind.valueOf("STACK"));
        assertThrows(
                IllegalArgumentException.class,
                () -> TensorCompositionKind.valueOf("UN" + "STACK"));
    }

    @Test
    void bothKindsUseTheExactVariadicSignature() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(TensorCompositionKind.class);
        List<OperationSignature> expected = List.of(OperationSignature.inputRange(
                CompositionAxisAttrs.class, 1, Integer.MAX_VALUE, 1));
        for (TensorCompositionKind kind : TensorCompositionKind.values()) {
            assertEquals(expected, kind.signatures());
            CompositionAxisAttrs attrs = new CompositionAxisAttrs(1);
            Operation operation = new Operation(kind, attrs);
            assertSame(kind, operation.kind());
            assertSame(attrs, operation.attrs());
        }
    }

    @Test
    void compositionAxisRetainsAndValidatesNormalizedValues() {
        assertEquals(0, new CompositionAxisAttrs(0).axis());
        assertEquals(Integer.MAX_VALUE, new CompositionAxisAttrs(Integer.MAX_VALUE).axis());
        assertEquals(
                "axis must be non-negative: -1",
                assertThrows(IllegalArgumentException.class, () -> new CompositionAxisAttrs(-1))
                        .getMessage());
    }
}
