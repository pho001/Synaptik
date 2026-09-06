package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuPartialReductionIrTest {
    @Test void partitionsQuotientAndRemainderWithoutGapsOrOverlap() {
        var ir = new CpuPartialReductionIr(CpuPartialReductionIr.Kind.SUM, DataType.INT32,
                CpuAggregateIr.Form.FULL, 3, 11, 4);
        assertEquals(0, ir.begin(0, 0));
        assertEquals(3, ir.end(0, 0));
        assertEquals(3, ir.begin(0, 1));
        assertEquals(6, ir.end(0, 1));
        assertEquals(6, ir.begin(0, 2));
        assertEquals(9, ir.end(0, 2));
        assertEquals(9, ir.begin(0, 3));
        assertEquals(11, ir.end(0, 3));
        assertEquals(96, ir.workspaceBytes());
        assertEquals(80, ir.stateOffset(2, 2));
    }

    @Test void rejectsExcludedFactsAndInvalidOrdinals() {
        assertThrows(IllegalArgumentException.class, () -> new CpuPartialReductionIr(
                CpuPartialReductionIr.Kind.PROD, DataType.FLOAT32, CpuAggregateIr.Form.FULL,
                1, 8, 2));
        assertThrows(IllegalArgumentException.class, () -> new CpuPartialReductionIr(
                CpuPartialReductionIr.Kind.PROD, DataType.INT64, CpuAggregateIr.Form.SUM_TO_SHAPE,
                1, 8, 2));
        var ir = new CpuPartialReductionIr(CpuPartialReductionIr.Kind.PROD, DataType.INT64,
                CpuAggregateIr.Form.MULTI_AXIS, 1, 4, 2);
        assertThrows(IllegalArgumentException.class, () -> ir.begin(1, 0));
        assertThrows(IllegalArgumentException.class, () -> ir.end(0, 2));
    }
}
