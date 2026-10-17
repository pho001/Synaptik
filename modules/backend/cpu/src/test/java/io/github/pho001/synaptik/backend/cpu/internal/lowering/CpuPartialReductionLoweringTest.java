package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;

class CpuPartialReductionLoweringTest {
    private final CpuPartialReductionLowering lowering = new CpuPartialReductionLowering();

    @Test void admitsOnlyACompleteIntegralDensePreparedProof() {
        var admitted = lowering.admit(inputs(CpuAggregateIr.Kind.SUM, DataType.INT32,
                CpuAggregateIr.Form.SINGLE_AXIS, 64, 8192, 4, true));
        assertTrue(admitted.isPresent());
        assertEquals(4, admitted.orElseThrow().partialCount());
    }

    @Test void failsClosedForEveryUnprovedOrExcludedFact() {
        assertTrue(lowering.admit(inputs(CpuAggregateIr.Kind.SUM, DataType.INT64,
                CpuAggregateIr.Form.FULL, 1, 524288, 2, false)).isEmpty());
        assertTrue(lowering.admit(inputs(CpuAggregateIr.Kind.MEAN, DataType.INT32,
                CpuAggregateIr.Form.FULL, 1, 524288, 2, true)).isEmpty());
        assertTrue(lowering.admit(inputs(CpuAggregateIr.Kind.PROD, DataType.FLOAT32,
                CpuAggregateIr.Form.FULL, 1, 524288, 2, true)).isEmpty());
        assertTrue(lowering.admit(inputs(CpuAggregateIr.Kind.SUM, DataType.INT32,
                CpuAggregateIr.Form.SUM_TO_SHAPE, 1, 524288, 2, true)).isEmpty());
    }

    private static CpuPartialReductionLowering.AdmissionInputs inputs(CpuAggregateIr.Kind kind,
            DataType type, CpuAggregateIr.Form form, long cells, long domain, int partials,
            boolean evidence) {
        return new CpuPartialReductionLowering.AdmissionInputs(kind, type, form, cells, domain,
                partials, 1024, 4, true, true, true, evidence);
    }
}
