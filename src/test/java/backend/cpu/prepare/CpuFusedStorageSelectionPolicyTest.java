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

    @Test
    void cpuNativeKeepsNonBindableSegmentInputLayoutsOnJavaArrays() {
        for (FusedExternalInputPlan input : List.of(
                input(DataType.FLOAT32, FusedAccessKind.DIRECT_STRIDED, 0, new int[]{1, 8}),
                input(DataType.FLOAT32, FusedAccessKind.OFFSET_CONTIGUOUS, 4, new int[]{1}),
                input(DataType.FLOAT32, FusedAccessKind.OFFSET_STRIDED, 4, new int[]{1, 8}),
                input(DataType.FLOAT32, FusedAccessKind.BROADCAST_STRIDED, 4, new int[]{0})
        )) {
            FusedOperation selected = CpuFusedStorageSelectionPolicy.specialize(
                    operation(input, Operation.OpType.ADD, DataType.FLOAT32),
                    CpuStorageProfile.CPU_NATIVE
            );

            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().inputStorageKind());
            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().outputStorageKind());
        }
    }

    @Test
    void cpuNativeSelectsMemorySegmentForDenseScalarBroadcastInput() {
        FusedExternalInputPlan scalarBroadcast = input(
                DataType.FLOAT32,
                FusedAccessKind.BROADCAST_STRIDED,
                new int[]{1},
                new int[]{1},
                new int[]{8},
                new int[]{1},
                0,
                new int[]{0}
        );

        FusedOperation selected = CpuFusedStorageSelectionPolicy.specialize(
                operation(scalarBroadcast, Operation.OpType.ADD, DataType.FLOAT32),
                CpuStorageProfile.CPU_NATIVE
        );

        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, selected.getNumericContract().inputStorageKind());
        assertEquals(FusedStorageKind.CPU_MEMORY_SEGMENT, selected.getNumericContract().outputStorageKind());
    }

    @Test
    void cpuNativeKeepsExplicitAndNonDenseBroadcastViewsOnJavaArrays() {
        for (FusedExternalInputPlan input : List.of(
                input(
                        DataType.FLOAT32,
                        FusedAccessKind.BROADCAST_STRIDED,
                        new int[]{8},
                        new int[]{0},
                        new int[]{8},
                        new int[]{1},
                        0,
                        new int[]{0}
                ),
                input(
                        DataType.FLOAT32,
                        FusedAccessKind.BROADCAST_STRIDED,
                        new int[]{3, 4},
                        new int[]{0, 1},
                        new int[]{3, 4},
                        new int[]{4, 1},
                        0,
                        new int[]{0, 1}
                ),
                input(
                        DataType.FLOAT32,
                        FusedAccessKind.BROADCAST_STRIDED,
                        new int[]{3, 4},
                        new int[]{0, 2},
                        new int[]{3, 4},
                        new int[]{4, 1},
                        0,
                        new int[]{0, 2}
                )
        )) {
            FusedOperation selected = CpuFusedStorageSelectionPolicy.specialize(
                    operation(input, Operation.OpType.ADD, DataType.FLOAT32),
                    CpuStorageProfile.CPU_NATIVE
            );

            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().inputStorageKind());
            assertEquals(FusedStorageKind.CPU_JAVA_ARRAY, selected.getNumericContract().outputStorageKind());
        }
    }

    private static FusedOperation operation(
            DataType inputType,
            Operation.OpType opType,
            DataType outputType
    ) {
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, opType, List.of(0, 0), 1, outputType, NoAttributes.INSTANCE)),
                List.of(input(inputType, FusedAccessKind.DIRECT_CONTIGUOUS, 0, new int[]{1})),
                1
        );
        return operation(plan, outputType);
    }

    private static FusedOperation operation(
            FusedExternalInputPlan input,
            Operation.OpType opType,
            DataType outputType
    ) {
        FusedExpressionPlan plan = new FusedExpressionPlan(
                List.of(new FusedNodePlan(0, opType, List.of(0, 0), 1, outputType, NoAttributes.INSTANCE)),
                List.of(input),
                1
        );
        return operation(plan, outputType);
    }

    private static FusedOperation operation(FusedExpressionPlan plan, DataType outputType) {
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

    private static FusedExternalInputPlan input(
            DataType inputType,
            FusedAccessKind accessKind,
            int storageOffset,
            int[] effectiveStrides
    ) {
        int[] shape = effectiveStrides.length == 1 ? new int[]{8} : new int[]{2, 4};
        int[] denseStrides = effectiveStrides.length == 1 ? new int[]{1} : new int[]{4, 1};
        return new FusedExternalInputPlan(
                0,
                inputType,
                shape,
                effectiveStrides,
                shape,
                denseStrides,
                storageOffset,
                effectiveStrides,
                accessKind
        );
    }

    private static FusedExternalInputPlan input(
            DataType inputType,
            FusedAccessKind accessKind,
            int[] inputShape,
            int[] inputStrides,
            int[] logicalOutputShape,
            int[] logicalOutputDenseStrides,
            int storageOffset,
            int[] effectiveStrides
    ) {
        return new FusedExternalInputPlan(
                0,
                inputType,
                inputShape,
                inputStrides,
                logicalOutputShape,
                logicalOutputDenseStrides,
                storageOffset,
                effectiveStrides,
                accessKind
        );
    }
}
