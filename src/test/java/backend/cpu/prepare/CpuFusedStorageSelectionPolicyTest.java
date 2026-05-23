package backend.cpu.prepare;

import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.NoAttributes;
import backend.cpu.fused.numeric.FusedApproximationContract;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.numeric.FusedValueLane;
import backend.cpu.fused.plan.FusedDispatchFamily;
import backend.cpu.fused.plan.FusedOperation;
import backend.cpu.fused.plan.FusedSignatureBuilder;
import config.runtime.CpuStorageProfile;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CpuFusedStorageSelectionPolicyTest {
    @Test
    void cpuNativeSelectsMemorySegmentForSupportedScalarPlan() {
        FusedOperation selected = CpuFusedStorageSelectionPolicy.specialize(
                operation(DataType.FLOAT32, Operation.OpType.ADD, DataType.FLOAT32),
                CpuStorageProfile.CPU_NATIVE
        );

        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, selected.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, selected.getNumericContract().outputStorageKind());
    }

    @Test
    void arrayAndAutoProfilesKeepSupportedFusedPlanOnJavaArrays() {
        for (CpuStorageProfile profile : List.of(CpuStorageProfile.CPU_ARRAY, CpuStorageProfile.AUTO)) {
            FusedOperation selected = CpuFusedStorageSelectionPolicy.specialize(
                    operation(DataType.FLOAT32, Operation.OpType.ADD, DataType.FLOAT32),
                    profile
            );

            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().inputStorageKind());
            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().outputStorageKind());
        }
    }

    @Test
    void cpuNativeKeepsUnsupportedDtypeAndUnsupportedPlanOnJavaArrays() {
        FusedOperation intPlan = CpuFusedStorageSelectionPolicy.specialize(
                operation(DataType.INT32, Operation.OpType.ADD, DataType.INT32),
                CpuStorageProfile.CPU_NATIVE
        );
        FusedOperation unsupportedOp = CpuFusedStorageSelectionPolicy.specialize(
                operation(DataType.FLOAT32, Operation.OpType.ERF, DataType.FLOAT32),
                CpuStorageProfile.CPU_NATIVE
        );

        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, intPlan.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, intPlan.getNumericContract().outputStorageKind());
        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, unsupportedOp.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, unsupportedOp.getNumericContract().outputStorageKind());
    }

    private static FusedOperation operation(
            DataType inputType,
            Operation.OpType opType,
            DataType outputType
    ) {
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, opType, List.of(0, 0), 1, outputType, NoAttributes.INSTANCE)),
                List.of(new FusedExternalInputPlan(
                        0,
                        inputType,
                        new int[]{8},
                        new int[]{1},
                        0,
                        new int[]{1},
                        FusedAccessKind.DIRECT_CONTIGUOUS
                )),
                1
        );
        FusedNumericContract numericContract = new FusedNumericContract(
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedStorageKind.CPU_JAVA_ARRAY,
                FusedValueLane.F32,
                FusedComputeKind.F32,
                FusedValueLane.F32
        );
        return new FusedOperation(
                "policy-test",
                numericContract,
                FusedApproximationContract.STRICT,
                false,
                FusedDispatchFamily.NON_CHEAP_CONTIGUOUS,
                FusedSignatureBuilder.buildFromPlan(plan, numericContract, FusedApproximationContract.STRICT),
                plan
        );
    }
}
