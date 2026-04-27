package backend.cpu.kernels;

import backend.cpu.kernels.plan.CpuExecutionPlanner;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import operations.elementwise.unary.exp;
import operations.elementwise.unary.neg;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ElementwiseDispatchPlanningTest {

    @Test
    void cheapElementwiseUsesPrepareTimeVectorDispatch() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        Tensor out = new Tensor(new int[]{64}, null, "out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_ELEMENTWISE,
                CpuAccumulateDType.NONE
        );

        ResolvedDispatchHints hints = planner.resolveDispatchHints(new neg(), out, contract);

        assertEquals(4, hints.vectorWidth());
        assertEquals(CpuExecutionMode.VECTOR, hints.mode());
    }

    @Test
    void transcendentalElementwiseRespectsStricterPrepareThresholds() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        Tensor out = new Tensor(new int[]{64}, null, "out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_ELEMENTWISE,
                CpuAccumulateDType.NONE
        );

        ResolvedDispatchHints hints = planner.resolveDispatchHints(new exp(), out, contract);

        assertEquals(4, hints.vectorWidth());
        assertEquals(CpuExecutionMode.SCALAR, hints.mode());
    }

    private static CpuKernelConfig testKernelConfig() {
        return new CpuKernelConfig(
                1,
                16,
                16,
                16,
                8,
                128,
                128,
                128,
                1_024,
                128,
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
                1,
                1,
                1,
                1,
                SumAccuracyMode.FAST,
                2_000_000,
                AttentionMatMulPolicy.AUTO
        );
    }
}
