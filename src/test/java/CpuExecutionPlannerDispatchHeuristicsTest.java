import backend.cpu.plan.CpuAccumulateDType;
import backend.cpu.plan.CpuComputeDType;
import backend.cpu.plan.CpuExecutionBackend;
import backend.cpu.plan.CpuExecutionMode;
import backend.cpu.plan.ResolvedCpuComputeContract;
import backend.cpu.prepare.CpuExecutionPlanner;
import config.backend.CpuKernelConfig;
import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.plan.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedVectorFallbackReason;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.numeric.FusedValueLane;
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
                numeric(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                "test",
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT32, backend.cpu.fused.ir.NoAttributes.INSTANCE),
                                new FusedNodePlan(1, Operation.OpType.MUL, List.of(2, 1), 3, DataType.FLOAT32, backend.cpu.fused.ir.NoAttributes.INSTANCE)
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS)
                        ),
                        3
                )
        );

        var prepared = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        );
        var hints = prepared.dispatchHints();

        assertEquals(CpuExecutionMode.PARALLEL, hints.mode());
        assertEquals(4, hints.vectorWidth());
        assertEquals(FusedVectorFallbackReason.BELOW_VECTOR_THRESHOLD, prepared.vectorFallbackReason());
    }

    @Test
    void smallNonCheapMaskedScaleWherePlanStaysScalarEvenWhenAsmWidthIsVectorized() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(vectorizedFusedConfig());
        FusedOperation fused = new FusedOperation(
                "fused(where)",
                numeric(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "test",
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.MUL_SCALAR, List.of(2), 3, DataType.FLOAT32, new backend.cpu.fused.ir.ScalarDoubleAttribute(0.5)),
                                new FusedNodePlan(1, Operation.OpType.WHERE, List.of(0, 1, 3), 4, DataType.FLOAT32, new backend.cpu.fused.ir.WhereAttributes())
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.BOOL, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{0}, FusedAccessKind.BROADCAST_STRIDED),
                                new FusedExternalInputPlan(2, DataType.FLOAT32, new int[]{65536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS)
                        ),
                        4
                )
        );

        var prepared = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        );
        var hints = prepared.dispatchHints();

        assertEquals(CpuExecutionMode.SCALAR, hints.mode());
        assertEquals(1, hints.vectorWidth());
        assertEquals(FusedVectorFallbackReason.MASKED_SCALE_WHERE_VECTOR_DISABLED, prepared.vectorFallbackReason());
    }

    @Test
    void nonCheapStridedPlanWithoutWherePatternStaysScalarWhenAsmWidthIsVectorized() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(vectorizedFusedConfig());
        FusedOperation fused = new FusedOperation(
                "fused(noncheap-strided)",
                numeric(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "test",
                new FusedExpressionPlan(
                        List.of(
                                new FusedNodePlan(0, Operation.OpType.DIV, List.of(0, 1), 2, DataType.FLOAT32, backend.cpu.fused.ir.NoAttributes.INSTANCE),
                                new FusedNodePlan(1, Operation.OpType.ADD, List.of(2, 1), 3, DataType.FLOAT32, backend.cpu.fused.ir.NoAttributes.INSTANCE)
                        ),
                        List.of(
                                new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{65_536}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                                new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{65_536}, new int[]{1}, 0, new int[]{0}, FusedAccessKind.BROADCAST_STRIDED)
                        ),
                        3
                )
        );

        var prepared = planner.resolveFusedDispatch(
                fused,
                new Tensor(new float[65_536], new int[]{65_536}, null, "out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        );
        var hints = prepared.dispatchHints();

        assertEquals(CpuExecutionMode.PARALLEL, hints.mode());
        assertEquals(4, hints.vectorWidth());
        assertEquals(FusedVectorFallbackReason.BELOW_VECTOR_THRESHOLD, prepared.vectorFallbackReason());
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
                config.backend.SumAccuracyMode.FAST,
                2_000_000,
                config.backend.AttentionMatMulPolicy.AUTO
        );
    }

    private static FusedNumericContract numeric(FusedValueLane lane) {
        return new FusedNumericContract(
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedStorageKind.CPU_JAVA_ARRAY,
                lane,
                lane == FusedValueLane.F64 ? FusedComputeKind.F64 : FusedComputeKind.F32,
                lane
        );
    }
}
