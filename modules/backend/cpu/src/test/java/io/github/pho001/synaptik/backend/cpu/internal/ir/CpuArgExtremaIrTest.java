package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuArgExtremaIrTest {
    @Test void structuralIdentityIncludesKindTypeAxisFormTieAndAccess() {
        var read = access(CpuAccessPlan.AccessKind.READ, 2);
        var write = access(CpuAccessPlan.AccessKind.WRITE, 1);
        var base = new CpuArgExtremaIr(CpuArgExtremaIr.Kind.ARG_MIN, DataType.FLOAT32, 1,
                false, ArgExtremaTiePolicy.FIRST_INDEX, true, true, read, write);
        assertAll(
                () -> assertNotEquals(base.structuralKey(), new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MAX, DataType.FLOAT32, 1, false,
                        ArgExtremaTiePolicy.FIRST_INDEX, true, true, read, write).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MIN, DataType.INT32, 1, false,
                        ArgExtremaTiePolicy.FIRST_INDEX, true, true, read, write).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MIN, DataType.FLOAT32, 1, false,
                        ArgExtremaTiePolicy.LAST_INDEX, true, true, read, write).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MIN, DataType.FLOAT32, 1, false,
                        ArgExtremaTiePolicy.FIRST_INDEX, false, true, read, write).structuralKey()),
                () -> assertNotEquals(base.structuralKey(), new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MIN, DataType.FLOAT32, 1, false,
                        ArgExtremaTiePolicy.FIRST_INDEX, true, false, read, write).structuralKey()),
                () -> assertEquals(DataType.INT64,
                        base.encodedKernelIr().values().getLast().dataType()),
                () -> assertThrows(IllegalArgumentException.class, () -> new CpuArgExtremaIr(
                        CpuArgExtremaIr.Kind.ARG_MIN, DataType.BOOL, 0, false,
                        ArgExtremaTiePolicy.FIRST_INDEX, true, true, read, write)));
    }

    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind, int rank) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
    }
}
