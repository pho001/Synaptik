package io.github.pho001.synaptik.model.operation.index;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;
import org.junit.jupiter.api.Test;

class AxisGatherSemanticsTest {
    @Test
    void declaresExactlyTheFinalOrderedKinds() {
        assertArrayEquals(
                new AxisGatherKind[] {
                    AxisGatherKind.GATHER, AxisGatherKind.GATHER_ELEMENTS
                },
                AxisGatherKind.values());
        assertSame(AxisGatherKind.GATHER, AxisGatherKind.valueOf("GATHER"));
        assertSame(
                AxisGatherKind.GATHER_ELEMENTS,
                AxisGatherKind.valueOf("GATHER_ELEMENTS"));
        assertThrows(
                IllegalArgumentException.class,
                () -> AxisGatherKind.valueOf("GATHER" + "_AXIS"));
        assertThrows(
                IllegalArgumentException.class,
                () -> AxisGatherKind.valueOf("TAKE" + "_ALONG_AXIS"));
    }

    @Test
    void exposesOnlyTheExactEnumShapeAndSignature() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(AxisGatherKind.class);
        OperationSignature expected = OperationSignature.fixed(IndexAxisAttrs.class, 2, 1);
        for (AxisGatherKind kind : AxisGatherKind.values()) {
            assertEquals(List.of(expected), kind.signatures());
            IndexAxisAttrs attrs = new IndexAxisAttrs(1);
            Operation operation = new Operation(kind, attrs);
            assertSame(kind, operation.kind());
            assertSame(attrs, operation.attrs());
        }
    }

    @Test
    void indexAxisAttributesRetainAndValidateTheNormalizedAxis() {
        assertEquals(0, new IndexAxisAttrs(0).axis());
        assertEquals(Integer.MAX_VALUE, new IndexAxisAttrs(Integer.MAX_VALUE).axis());
        assertEquals(
                "axis must be non-negative: -1",
                assertThrows(IllegalArgumentException.class, () -> new IndexAxisAttrs(-1))
                        .getMessage());
    }
}
