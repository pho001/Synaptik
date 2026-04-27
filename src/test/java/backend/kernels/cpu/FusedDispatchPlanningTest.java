package backend.kernels.cpu;

import backend.kernels.cpu.fused.plan.PreparedFusedDispatch;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import backend.cpu.fused.codegen.FusedAccessKind;
import backend.cpu.fused.codegen.FusedDTypeOps;
import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import backend.cpu.fused.codegen.NoAttributes;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import jdk.incubator.vector.FloatVector;
import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FusedDispatchPlanningTest {

    @Test
    void resolvesCheapContiguousDispatchDuringPrepare() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedOperation fused = fusedUnary(
                Operation.OpType.NEG,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                FusedAccessKind.DIRECT_CONTIGUOUS,
                DataType.FLOAT32
        );
        Tensor out = new Tensor(new int[]{64}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        int expectedWidth = Math.min(4, FloatVector.SPECIES_PREFERRED.length());
        int expectedVectorMinSize = planner.fusedDirectVectorMinSize(fused);
        assertEquals(expectedVectorMinSize, prepared.cpuVectorMinSize());
        assertEquals(expectedWidth, prepared.asmVectorWidth());
        assertEquals(expectedWidth, prepared.dispatchHints().vectorWidth());
        assertEquals(
                expectedWidth > 1 && out.getFlatDataSize() >= expectedVectorMinSize ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR,
                prepared.dispatchHints().mode()
        );
    }

    @Test
    void resolvesNonCheapStridedDispatchWithFamilySpecificAsmWidth() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedOperation fused = fusedUnary(
                Operation.OpType.LOG,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                FusedAccessKind.DIRECT_STRIDED,
                DataType.FLOAT32
        );
        Tensor out = new Tensor(new int[]{64}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        int expectedWidth = Math.min(2, FloatVector.SPECIES_PREFERRED.length());
        int expectedVectorMinSize = planner.fusedDirectVectorMinSize(fused);
        assertEquals(expectedVectorMinSize, prepared.cpuVectorMinSize());
        assertEquals(expectedWidth, prepared.asmVectorWidth());
        assertEquals(expectedWidth, prepared.dispatchHints().vectorWidth());
        assertEquals(
                expectedWidth > 1 && out.getFlatDataSize() >= expectedVectorMinSize ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR,
                prepared.dispatchHints().mode()
        );
    }

    @Test
    void clampsBf16AffineRationalNonCheapStridedDispatchToScalarAsmWidth() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(
                        new FusedNodePlan(0, Operation.OpType.NEG, List.of(0), 5, DataType.BFLOAT16, NoAttributes.INSTANCE),
                        new FusedNodePlan(1, Operation.OpType.MUL, List.of(5, 1), 6, DataType.BFLOAT16, NoAttributes.INSTANCE),
                        new FusedNodePlan(2, Operation.OpType.ADD, List.of(6, 2), 7, DataType.BFLOAT16, NoAttributes.INSTANCE),
                        new FusedNodePlan(3, Operation.OpType.MUL, List.of(7, 3), 8, DataType.BFLOAT16, NoAttributes.INSTANCE),
                        new FusedNodePlan(4, Operation.OpType.DIV, List.of(8, 4), 9, DataType.BFLOAT16, NoAttributes.INSTANCE),
                        new FusedNodePlan(5, Operation.OpType.ADD, List.of(9, 1), 10, DataType.BFLOAT16, NoAttributes.INSTANCE)
                ),
                List.of(
                        new FusedExternalInputPlan(0, DataType.BFLOAT16, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{1, 256}, FusedAccessKind.DIRECT_STRIDED),
                        new FusedExternalInputPlan(1, DataType.BFLOAT16, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{256, 1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                        new FusedExternalInputPlan(2, DataType.BFLOAT16, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{256, 1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                        new FusedExternalInputPlan(3, DataType.BFLOAT16, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{256, 1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                        new FusedExternalInputPlan(4, DataType.BFLOAT16, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{256, 1}, FusedAccessKind.DIRECT_CONTIGUOUS)
                ),
                10
        );
        FusedOperation fused = new FusedOperation(
                "bf16-affine-rational-strided",
                FusedDTypeOps.MODE_BF16,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "bf16-affine-rational-strided",
                1,
                plan
        );
        Tensor out = new Tensor(new int[]{8, 256}, null, "fused_out", DataType.BFLOAT16);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.BFLOAT16,
                CpuComputeDType.BF16_NATIVE,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        assertEquals(1, prepared.asmVectorWidth());
        assertEquals(1, prepared.dispatchHints().vectorWidth());
    }

    private static CpuKernelConfig testKernelConfig() {
        return new CpuKernelConfig(
                1,
                16,
                16,
                16,
                1_024,
                1_024,
                8,
                8,
                1_024,
                1_024,
                1_024,
                1_024,
                1_024,
                1_000_000_000,
                1,
                1,
                1,
                1,
                8,
                8,
                16,
                1_024,
                4,
                4,
                4,
                2,
                SumAccuracyMode.FAST,
                2_000_000,
                AttentionMatMulPolicy.AUTO
        );
    }

    private static FusedOperation fusedUnary(
            Operation.OpType opType,
            boolean lowCostHint,
            FusedDispatchFamily family,
            FusedAccessKind accessKind,
            DataType dataType
    ) {
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, opType, List.of(0), 1, dataType, NoAttributes.INSTANCE)),
                List.of(new FusedExternalInputPlan(0, dataType, new int[]{64}, new int[]{1}, 0, new int[]{1}, accessKind)),
                1
        );
        return new FusedOperation(
                "fused-test",
                0,
                lowCostHint,
                family,
                "fused-test-sig",
                1,
                plan
        );
    }
}
