package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuBatchNormTrainingIrTest {
    private static CpuAccessPlan access(CpuAccessPlan.AccessKind kind,int rank){return new CpuAccessPlan(kind,CpuAccessPlan.Regime.DENSE_LINEAR,rank,java.util.Collections.nCopies(rank,CpuAccessPlan.AxisRole.CONTIGUOUS),rank);}
    @Test void locksFiveInputsFiveOutputsScalarsPassesAndScratch(){var ir=new CpuBatchNormTrainingIr(
            java.util.Collections.nCopies(5,DataType.FLOAT32),DataType.FLOAT32,
            Float.floatToRawIntBits(.25f)&0xffffffffL,Float.floatToRawIntBits(1e-5f)&0xffffffffL,
            3,1,1,3,8,5,48,List.of(0,1,2,3,4),java.util.Collections.nCopies(5,access(CpuAccessPlan.AccessKind.READ,1)),
            List.of(access(CpuAccessPlan.AccessKind.WRITE,3),access(CpuAccessPlan.AccessKind.WRITE,1),access(CpuAccessPlan.AccessKind.WRITE,1),access(CpuAccessPlan.AccessKind.WRITE,1),access(CpuAccessPlan.AccessKind.WRITE,1)));
        assertAll(()->assertEquals(10,ir.encodedKernelIr().values().size()),()->assertEquals(5,ir.encodedKernelIr().stores().size()),()->assertTrue(ir.encodedKernelIr().familyIdentity().contains("batch-normalization-training")));}
}
