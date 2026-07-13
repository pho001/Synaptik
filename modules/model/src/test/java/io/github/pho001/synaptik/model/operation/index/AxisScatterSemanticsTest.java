package io.github.pho001.synaptik.model.operation.index;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;
import org.junit.jupiter.api.Test;

class AxisScatterSemanticsTest {
    @Test
    void declaresExactlyFinalScatterKindsInStableOrder() {
        assertArrayEquals(
                new AxisScatterKind[] {
                    AxisScatterKind.SCATTER_ELEMENTS,
                    AxisScatterKind.SCATTER_ADD
                },
                AxisScatterKind.values());
        assertSame(
                AxisScatterKind.SCATTER_ELEMENTS,
                AxisScatterKind.valueOf("SCATTER_ELEMENTS"));
        assertSame(AxisScatterKind.SCATTER_ADD, AxisScatterKind.valueOf("SCATTER_ADD"));
        assertThrows(
                IllegalArgumentException.class,
                () -> AxisScatterKind.valueOf("SCATTER" + "_AXIS_ADD"));
    }

    @Test
    void exposesOnlyTheExactEnumShapeAndFamilyOwnedSignatures() {
        io.github.pho001.synaptik.model.operation.OperationSignatureTest
                .assertSignatureEnumShape(AxisScatterKind.class);
        assertEquals(
                List.of(OperationSignature.fixed(ScatterElementsAttrs.class, 3, 1)),
                AxisScatterKind.SCATTER_ELEMENTS.signatures());
        assertEquals(
                List.of(OperationSignature.fixed(IndexAxisAttrs.class, 3, 1)),
                AxisScatterKind.SCATTER_ADD.signatures());
        ScatterElementsAttrs attrs =
                new ScatterElementsAttrs(1, ScatterReduction.ADD);
        Operation operation = new Operation(AxisScatterKind.SCATTER_ELEMENTS, attrs);
        IndexAxisAttrs addAttrs = new IndexAxisAttrs(1);
        Operation add = new Operation(AxisScatterKind.SCATTER_ADD, addAttrs);
        assertSame(AxisScatterKind.SCATTER_ELEMENTS, operation.kind());
        assertSame(attrs, operation.attrs());
        assertSame(AxisScatterKind.SCATTER_ADD, add.kind());
        assertSame(addAttrs, add.attrs());
        assertThrows(
                IllegalArgumentException.class,
                () -> new Operation(AxisScatterKind.SCATTER_ELEMENTS, addAttrs));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Operation(AxisScatterKind.SCATTER_ADD, attrs));
    }
}
