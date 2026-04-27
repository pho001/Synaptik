import backend.kernels.cpu.CpuAccumulateDType;
import backend.kernels.cpu.CpuComputeDType;
import backend.kernels.cpu.CpuExecutionBackend;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import config.backend.CpuKernelConfig;
import backend.cpu.fused.codegen.FusedAccessKind;
import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CpuExecutionPlannerDispatchHeuristicsTest {
    @Test
    void smallCheapContiguousFusedPlanStaysScalarEvenWhenAsmWidthIsVectorized() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(vectorizedFusedConfig());
        FusedOperation fused = new FusedOperation(
                "fused(2)",
                1,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                "test",
                1,
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT32, backend.cpu.fused.codegen.NoAttributes.INSTANCE),
                                new FusedNodePlan(1, Operation.OpType.MUL, List.of(2, 1), 3, DataType.FLOAT32, backend.cpu.fused.codegen.NoAttributes.INSTANCE)
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS)
                        ),
                        3
                )
        );

        var hints = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        ).dispatchHints();

        assertEquals(CpuExecutionMode.PARALLEL, hints.mode());
        assertEquals(4, hints.vectorWidth());
    }

    @Test
    void smallNonCheapMaskedScaleWherePlanStaysScalarEvenWhenAsmWidthIsVectorized() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(vectorizedFusedConfig());
        FusedOperation fused = new FusedOperation(
                "fused(where)",
                1,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "test",
                2,
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.MUL_SCALAR, List.of(2), 3, DataType.FLOAT32, new backend.cpu.fused.codegen.ScalarDoubleAttribute(0.5)),
                                new FusedNodePlan(1, Operation.OpType.WHERE, List.of(0, 1, 3), 4, DataType.FLOAT32, new backend.cpu.fused.codegen.WhereAttributes())
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.BOOL, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{0}, FusedAccessKind.BROADCAST_STRIDED),
                                new FusedExternalInputPlan(2, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS)
                        ),
                        4
                )
        );

        var hints = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        ).dispatchHints();

        assertEquals(CpuExecutionMode.SCALAR, hints.mode());
        assertEquals(1, hints.vectorWidth());
    }

    @Test
    void nonCheapStridedPlanWithoutWherePatternStaysScalarWhenAsmWidthIsVectorized() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(vectorizedFusedConfig());
        FusedOperation fused = new FusedOperation(
                "fused(noncheap-strided)",
                1,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "test",
                2,
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.DIV, List.of(0, 1), 2, DataType.FLOAT32, backend.cpu.fused.codegen.NoAttributes.INSTANCE),
                                new FusedNodePlan(1, Operation.OpType.ADD, List.of(2, 1), 3, DataType.FLOAT32, backend.cpu.fused.codegen.NoAttributes.INSTANCE)
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{65_536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65_536}, new int[]{1}, 0, new int[]{0}, FusedAccessKind.BROADCAST_STRIDED)
                        ),
                        3
                )
        );

        var hints = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        ).dispatchHints();

        assertEquals(CpuExecutionMode.PARALLEL, hints.mode());
        assertEquals(4, hints.vectorWidth());
    }

    private static CpuKernelConfig vectorizedFusedConfig() {
        return new CpuKernelConfig(
                1,
                32,
                32,
                32,
                1024,
                1024,
                8192,
                8192,
                1024,
                4096,
                4096,
                4096,
                4096,
                4096,
                Integer.MAX_VALUE,
                4,
                2,
                1,
                4096,
                8192,
                16384,
                16384,
                4,
                4,
                4,
                4,
                config.backend.SumAccuracyMode.FAST,
                2_000_000,
                config.backend.AttentionMatMulPolicy.AUTO
        );
    }
}
