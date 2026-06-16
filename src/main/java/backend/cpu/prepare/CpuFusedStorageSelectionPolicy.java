package backend.cpu.prepare;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.numeric.FusedNumericContractResolver;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.plan.FusedOperation;
import config.runtime.CpuStorageProfile;
import operations.Operation;
import tensor.DataType;
import tensor.TensorMetadata;

import java.util.Arrays;

/**
 * Prepare-time storage policy for CPU fused execution.
 */
final class CpuFusedStorageSelectionPolicy {
    private CpuFusedStorageSelectionPolicy() {
    }

    static FusedOperation specialize(FusedOperation fused, CpuStorageProfile requestedStorage) {
        if (!usesMemorySegmentContract(fused, requestedStorage)) {
            return fused;
        }
        return fused.withNumericContract(FusedNumericContractResolver.resolve(
                fused.getPlan(),
                FusedStorageKind.CPU_MEMORY_SEGMENT,
                FusedStorageKind.CPU_MEMORY_SEGMENT
        ));
    }

    static boolean usesMemorySegmentContract(FusedOperation fused, CpuStorageProfile requestedStorage) {
        if (requestedStorage != CpuStorageProfile.CPU_NATIVE) {
            return false;
        }
        return supportsMemorySegmentScalarExecution(fused);
    }

    private static boolean supportsMemorySegmentScalarExecution(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        FusedExpressionPlan plan = fused.getPlan();
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (!isSupportedSegmentDType(input.dataType()) || !isNativeSegmentBindableInput(input)) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (!isSupportedSegmentDType(node.outputType()) || !isSupportedSegmentScalarOp(node.opType())) {
                return false;
            }
        }
        return isSupportedSegmentDType(plan.outputNode().outputType());
    }

    private static boolean isSupportedSegmentDType(DataType dataType) {
        return switch (dataType) {
            case FLOAT32, FLOAT64, BFLOAT16, BOOL -> true;
            default -> false;
        };
    }

    private static boolean isNativeSegmentBindableInput(FusedExternalInputPlan input) {
        if (input.storageOffset() != 0) {
            return false;
        }
        return switch (input.accessKind()) {
            case DIRECT_CONTIGUOUS -> isDenseContiguousSource(input);
            case BROADCAST_STRIDED -> isDenseContiguousSource(input);
            case DIRECT_STRIDED, OFFSET_CONTIGUOUS, OFFSET_STRIDED -> false;
        };
    }

    private static boolean isDenseContiguousSource(FusedExternalInputPlan input) {
        return Arrays.equals(input.inputStrides(), TensorMetadata.computeStrides(input.inputShape()));
    }

    private static boolean isSupportedSegmentScalarOp(Operation.OpType opType) {
        if (opType == null) {
            return false;
        }
        if (opType == Operation.OpType.CONST_SCALAR || opType == Operation.OpType.NOOP) {
            return true;
        }
        return isLegacySupportedSegmentScalarOp(opType);
    }

    private static boolean isLegacySupportedSegmentScalarOp(Operation.OpType opType) {
        // Old CPU fused plans retain only canonical opType identity here.
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX,
                 GT, GE, LT, LE, EQ, NE,
                 LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                 WHERE, NEG, INV, LOG, EXP, FAST_EXP, TANH, FAST_TANH,
                 POW, POW_TENSOR, SQRT, ABS, MUL_SCALAR,
                 RELU, CLAMP_MIN, CLAMP_MAX, SIGMOID -> true;
            default -> false;
        };
    }
}
