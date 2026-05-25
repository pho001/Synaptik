package backend.cpu.kernels;

import backend.cpu.kernels.fused.plan.PreparedFusedDispatch;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.NoAttributes;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.numeric.FusedValueLane;
import backend.cpu.fused.plan.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedVectorFallbackReason;
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
        assertEquals(FusedVectorFallbackReason.BELOW_VECTOR_THRESHOLD, prepared.vectorFallbackReason());
        assertEquals(
                expectedWidth > 1 && out.getFlatDataSize() >= expectedVectorMinSize ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR,
                prepared.dispatchHints().mode()
        );
    }

    @Test
    void blocksNonAllocationFreeStridedTranscendentalVectorPath() {
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

        int expectedVectorMinSize = planner.fusedDirectVectorMinSize(fused);
        assertEquals(expectedVectorMinSize, prepared.cpuVectorMinSize());
        assertEquals(1, prepared.asmVectorWidth());
        assertEquals(1, prepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, prepared.vectorFallbackReason());
        assertEquals(CpuExecutionMode.SCALAR, prepared.dispatchHints().mode());
    }

    @Test
    void genericTranscendentalFusedAsmWidthStaysScalar() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(CpuKernelConfig.defaultsInference());
        FusedOperation fused = fusedUnary(
                Operation.OpType.LOG,
                false,
                FusedDispatchFamily.NON_CHEAP_CONTIGUOUS,
                FusedAccessKind.DIRECT_CONTIGUOUS,
                DataType.FLOAT32
        );
        Tensor out = new Tensor(new int[]{2_048}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        assertEquals(1, prepared.asmVectorWidth());
        assertEquals(1, prepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, prepared.vectorFallbackReason());
    }

    @Test
    void memorySegmentF32ContiguousDispatchKeepsVectorAsmWidth() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedOperation fused = fusedUnary(
                Operation.OpType.RELU,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                FusedAccessKind.DIRECT_CONTIGUOUS,
                DataType.FLOAT32
        ).withNumericContract(numericSegment(FusedValueLane.F32));
        Tensor out = new Tensor(new int[]{2_048}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        int expectedWidth = Math.min(4, FloatVector.SPECIES_PREFERRED.length());
        assertEquals(expectedWidth, prepared.asmVectorWidth());
        assertEquals(expectedWidth, prepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.NONE, prepared.vectorFallbackReason());
    }

    @Test
    void memorySegmentF32ScalarBroadcastDispatchKeepsVectorAsmWidth() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, Operation.OpType.ADD, List.of(0, 1), 2, DataType.FLOAT32, NoAttributes.INSTANCE)),
                List.of(
                        new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{2_048}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                        new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{2_048}, new int[]{1}, 0, new int[]{0}, FusedAccessKind.BROADCAST_STRIDED)
                ),
                2
        );
        FusedOperation fused = new FusedOperation(
                "fused-broadcast-test",
                numericSegment(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                true,
                FusedDispatchFamily.CHEAP_STRIDED,
                "fused-broadcast-test-sig",
                plan
        );
        Tensor out = new Tensor(new int[]{2_048}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        int expectedWidth = Math.min(4, FloatVector.SPECIES_PREFERRED.length());
        assertEquals(expectedWidth, prepared.asmVectorWidth());
        assertEquals(expectedWidth, prepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.NONE, prepared.vectorFallbackReason());
    }

    @Test
    void memorySegmentF32GeneralStridedInputKeepsVectorAsmWidth() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, Operation.OpType.RELU, List.of(0), 2, DataType.FLOAT32, NoAttributes.INSTANCE)),
                List.of(
                        new FusedExternalInputPlan(0, DataType.FLOAT32, new int[]{2_048}, new int[]{1}, 0, new int[]{1}, FusedAccessKind.DIRECT_CONTIGUOUS),
                        new FusedExternalInputPlan(1, DataType.FLOAT32, new int[]{8, 256}, new int[]{256, 1}, 0, new int[]{1, 8}, FusedAccessKind.DIRECT_STRIDED)
                ),
                2
        );
        FusedOperation fused = new FusedOperation(
                "fused-general-strided-segment-input",
                numericSegment(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                true,
                FusedDispatchFamily.CHEAP_STRIDED,
                "fused-general-strided-segment-input",
                plan
        );
        Tensor out = new Tensor(new int[]{2_048}, null, "fused_out", DataType.FLOAT32);
        ResolvedCpuComputeContract contract = new ResolvedCpuComputeContract(
                DataType.FLOAT32,
                CpuComputeDType.F32,
                CpuExecutionBackend.CPU_FUSED,
                CpuAccumulateDType.NONE
        );

        PreparedFusedDispatch prepared = planner.resolveFusedDispatch(fused, out, contract);

        int expectedWidth = Math.min(4, FloatVector.SPECIES_PREFERRED.length());
        assertEquals(expectedWidth, prepared.asmVectorWidth());
        assertEquals(expectedWidth, prepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.NONE, prepared.vectorFallbackReason());
    }

    @Test
    void memorySegmentBf16DispatchStaysScalarOnlyWhileF32StridedDispatchVectorizes() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedOperation bf16 = fusedUnary(
                Operation.OpType.RELU,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                FusedAccessKind.DIRECT_CONTIGUOUS,
                DataType.BFLOAT16
        ).withNumericContract(numericSegment(FusedValueLane.BF16));
        FusedOperation strided = fusedUnary(
                Operation.OpType.RELU,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                FusedAccessKind.DIRECT_STRIDED,
                DataType.FLOAT32
        ).withNumericContract(numericSegment(FusedValueLane.F32));

        PreparedFusedDispatch bf16Prepared = planner.resolveFusedDispatch(
                bf16,
                new Tensor(new int[]{2_048}, null, "bf16_out", DataType.BFLOAT16),
                new ResolvedCpuComputeContract(DataType.BFLOAT16, CpuComputeDType.BF16_NATIVE, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        );
        PreparedFusedDispatch stridedPrepared = planner.resolveFusedDispatch(
                strided,
                new Tensor(new int[]{2_048}, null, "strided_out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(DataType.FLOAT32, CpuComputeDType.F32, CpuExecutionBackend.CPU_FUSED, CpuAccumulateDType.NONE)
        );

        assertEquals(1, bf16Prepared.asmVectorWidth());
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, bf16Prepared.vectorFallbackReason());
        int expectedWidth = Math.min(4, FloatVector.SPECIES_PREFERRED.length());
        assertEquals(expectedWidth, stridedPrepared.asmVectorWidth());
        assertEquals(FusedVectorFallbackReason.NONE, stridedPrepared.vectorFallbackReason());
    }

    @Test
    void memorySegmentBoolCompareAndWhereDispatchStayScalarOnly() {
        CpuExecutionPlanner planner = CpuExecutionPlanner.from(testKernelConfig());
        FusedOperation compare = new FusedOperation(
                "fused-segment-compare",
                new FusedNumericContract(
                        FusedStorageKind.CPU_MEMORY_SEGMENT,
                        FusedStorageKind.CPU_MEMORY_SEGMENT,
                        FusedValueLane.F32,
                        FusedComputeKind.F32,
                        FusedValueLane.BOOL
                ),
                FusedApproximationContract.STRICT,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                "fused-segment-compare",
                new FusedExpressionPlan(
                        List.of(new FusedNodePlan(
                                0,
                                Operation.OpType.GT,
                                List.of(0, 1),
                                2,
                                DataType.BOOL,
                                NoAttributes.INSTANCE
                        )),
                        List.of(
                                contiguousSegmentInput(0, DataType.FLOAT32),
                                contiguousSegmentInput(1, DataType.FLOAT32)
                        ),
                        2
                )
        );
        FusedOperation where = new FusedOperation(
                "fused-segment-where",
                numericSegment(FusedValueLane.F32),
                FusedApproximationContract.STRICT,
                true,
                FusedDispatchFamily.CHEAP_CONTIGUOUS,
                "fused-segment-where",
                new FusedExpressionPlan(
                        List.of(new FusedNodePlan(
                                0,
                                Operation.OpType.WHERE,
                                List.of(0, 1, 2),
                                3,
                                DataType.FLOAT32,
                                NoAttributes.INSTANCE
                        )),
                        List.of(
                                contiguousSegmentInput(0, DataType.BOOL),
                                contiguousSegmentInput(1, DataType.FLOAT32),
                                contiguousSegmentInput(2, DataType.FLOAT32)
                        ),
                        3
                )
        );

        PreparedFusedDispatch comparePrepared = planner.resolveFusedDispatch(
                compare,
                new Tensor(new int[]{2_048}, null, "compare_out", DataType.BOOL),
                new ResolvedCpuComputeContract(
                        DataType.BOOL,
                        CpuComputeDType.BOOL,
                        CpuExecutionBackend.CPU_FUSED,
                        CpuAccumulateDType.NONE
                )
        );
        PreparedFusedDispatch wherePrepared = planner.resolveFusedDispatch(
                where,
                new Tensor(new int[]{2_048}, null, "where_out", DataType.FLOAT32),
                new ResolvedCpuComputeContract(
                        DataType.FLOAT32,
                        CpuComputeDType.F32,
                        CpuExecutionBackend.CPU_FUSED,
                        CpuAccumulateDType.NONE
                )
        );

        assertEquals(1, comparePrepared.asmVectorWidth());
        assertEquals(1, comparePrepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, comparePrepared.vectorFallbackReason());
        assertEquals(1, wherePrepared.asmVectorWidth());
        assertEquals(1, wherePrepared.dispatchHints().vectorWidth());
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, wherePrepared.vectorFallbackReason());
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
                numeric(FusedValueLane.BF16),
                FusedApproximationContract.STRICT,
                false,
                FusedDispatchFamily.NON_CHEAP_STRIDED,
                "bf16-affine-rational-strided",
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
        assertEquals(FusedVectorFallbackReason.VECTOR_PATH_UNSUPPORTED, prepared.vectorFallbackReason());
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
                numeric(FusedValueLane.fromDataType(dataType)),
                FusedApproximationContract.STRICT,
                lowCostHint,
                family,
                "fused-test-sig",
                plan
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

    private static FusedNumericContract numericSegment(FusedValueLane lane) {
        return new FusedNumericContract(
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                lane,
                lane == FusedValueLane.F64 ? FusedComputeKind.F64 : FusedComputeKind.F32,
                lane
        );
    }

    private static FusedExternalInputPlan contiguousSegmentInput(int index, DataType dataType) {
        return new FusedExternalInputPlan(
                index,
                dataType,
                new int[]{2_048},
                new int[]{1},
                0,
                new int[]{1},
                FusedAccessKind.DIRECT_CONTIGUOUS
        );
    }
}
