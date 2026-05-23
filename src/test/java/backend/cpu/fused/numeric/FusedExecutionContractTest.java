package backend.cpu.fused.numeric;

import backend.ApproxMode;
import backend.cpu.fused.asm.FusedAsmSpecializationKind;
import backend.cpu.fused.asm.FusedKernelCacheKey;
import backend.cpu.fused.exec.FusedExecutablePreparer;
import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.NoAttributes;
import backend.cpu.fused.plan.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedExecutionPlan;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.kernels.CpuAccumulateDType;
import backend.cpu.kernels.CpuComputeDType;
import backend.cpu.kernels.CpuExecutionBackend;
import backend.cpu.kernels.ResolvedCpuComputeContract;
import config.runtime.ApproximationConfig;
import config.runtime.FusedExecutionPolicy;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FusedExecutionContractTest {
    @Test
    void approximationContractIsPreparedFromRuntimePolicyAndCompileMode() {
        assertEquals(
                FusedApproximationContract.STRICT,
                FusedApproximationContract.from(new ApproximationConfig(ApproxMode.OFF, false), true)
        );
        assertEquals(
                FusedApproximationContract.FAST_EXP_AND_TANH,
                FusedApproximationContract.from(new ApproximationConfig(ApproxMode.ALWAYS, false), false)
        );
        assertEquals(
                FusedApproximationContract.FAST_EXP_AND_TANH,
                FusedApproximationContract.from(new ApproximationConfig(ApproxMode.TRAINING_ONLY, false), true)
        );
        assertEquals(
                FusedApproximationContract.STRICT,
                FusedApproximationContract.from(new ApproximationConfig(ApproxMode.ALWAYS, true), true)
        );
    }

    @Test
    void approximationContractChangesSchedulerAndCacheIdentity() {
        FusedOperation strict = operation(
                javaArrayContract(FusedValueLane.F32),
                FusedApproximationContract.STRICT
        );
        FusedOperation fast = strict.withApproximationContract(FusedApproximationContract.FAST_EXP_AND_TANH);

        assertTrue(strict.getSchedulerSignature().contains("approx=STRICT"));
        assertTrue(fast.getSchedulerSignature().contains("approx=FAST_EXP_AND_TANH"));

        FusedKernelCacheKey strictKey = new FusedKernelCacheKey(
                strict.getSchedulerSignature(),
                strict.getNumericContract().signatureToken(),
                strict.getApproximationContract().signatureToken(),
                4,
                FusedAsmSpecializationKind.NONE
        );
        FusedKernelCacheKey fastKey = new FusedKernelCacheKey(
                fast.getSchedulerSignature(),
                fast.getNumericContract().signatureToken(),
                fast.getApproximationContract().signatureToken(),
                4,
                FusedAsmSpecializationKind.NONE
        );
        assertTrue(!strictKey.equals(fastKey));
    }

    @Test
    void memorySegmentBf16ContractIsPreparedWithoutInterpreterFallback() {
        FusedNumericContract segmentBf16 = new FusedNumericContract(
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedValueLane.BF16,
                FusedComputeKind.F32,
                FusedValueLane.BF16
        );
        assertTrue(segmentBf16.usesMemorySegmentStorage());
        assertEquals(
                "CPU_MEMORY_SEGMENT:BF16->F32->CPU_MEMORY_SEGMENT:BF16",
                segmentBf16.signatureToken()
        );

        FusedExecutionPlan plan = new FusedExecutionPlan(
                operation(segmentBf16, FusedApproximationContract.STRICT),
                new ResolvedCpuComputeContract(
                        DataType.BFLOAT16,
                        CpuComputeDType.BF16_NATIVE,
                        CpuExecutionBackend.CPU_FUSED,
                        CpuAccumulateDType.NONE
                ),
                8,
                1,
                1
        );

        assertTrue(new FusedExecutablePreparer().prepare(plan, FusedExecutionPolicy.defaultsInference())
                .getClass()
                .getName()
                .contains("GeneratedFusedExecutable"));
    }

    @Test
    void numericContractChangesSchedulerAndCacheIdentity() {
        FusedOperation array = operation(
                javaArrayContract(FusedValueLane.F32),
                FusedApproximationContract.STRICT
        );
        FusedNumericContract segmentContract = new FusedNumericContract(
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedValueLane.F32,
                FusedComputeKind.F32,
                FusedValueLane.F32
        );
        FusedOperation segment = array.withNumericContract(segmentContract);

        assertTrue(array.getSchedulerSignature().contains("numeric=CPU_JAVA_ARRAY:F32"));
        assertTrue(segment.getSchedulerSignature().contains("numeric=CPU_MEMORY_SEGMENT:F32"));

        FusedKernelCacheKey arrayKey = new FusedKernelCacheKey(
                array.getSchedulerSignature(),
                array.getNumericContract().signatureToken(),
                array.getApproximationContract().signatureToken(),
                1,
                FusedAsmSpecializationKind.NONE
        );
        FusedKernelCacheKey segmentKey = new FusedKernelCacheKey(
                segment.getSchedulerSignature(),
                segment.getNumericContract().signatureToken(),
                segment.getApproximationContract().signatureToken(),
                1,
                FusedAsmSpecializationKind.NONE
        );
        assertTrue(!arrayKey.equals(segmentKey));
    }

    private static FusedOperation operation(
            FusedNumericContract numericContract,
            FusedApproximationContract approximationContract
    ) {
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, Operation.OpType.EXP, List.of(0), 1, DataType.BFLOAT16, NoAttributes.INSTANCE)),
                List.of(new FusedExternalInputPlan(
                        0,
                        DataType.BFLOAT16,
                        new int[]{8},
                        new int[]{1},
                        0,
                        new int[]{1},
                        FusedAccessKind.DIRECT_CONTIGUOUS
                )),
                1
        );
        return new FusedOperation(
                "contract-test",
                numericContract,
                approximationContract,
                false,
                FusedDispatchFamily.NON_CHEAP_CONTIGUOUS,
                backend.cpu.fused.plan.FusedSignatureBuilder.buildFromPlan(plan, numericContract, approximationContract),
                plan
        );
    }

    private static FusedNumericContract javaArrayContract(FusedValueLane lane) {
        return new FusedNumericContract(
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedStorageKind.CPU_JAVA_ARRAY,
                lane,
                lane == FusedValueLane.F64 ? FusedComputeKind.F64 : FusedComputeKind.F32,
                lane
        );
    }
}
