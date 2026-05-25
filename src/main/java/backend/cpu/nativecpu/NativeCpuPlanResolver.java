package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuNodeExecutionPlan;
import backend.cpu.kernels.layout.plan.ResolvedBroadcastPlan;
import backend.cpu.kernels.layout.plan.ResolvedWhereBroadcastPlan;
import config.runtime.CpuStorageProfile;
import operations.Operation;
import operations.reduction.mean;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.sum;
import tensor.DataType;
import graph.compile.descriptor.CompiledTensorDescriptor;

import java.util.Arrays;
import java.util.List;

/**
 * Prepare-time native CPU route resolver.
 */
public final class NativeCpuPlanResolver {
    private NativeCpuPlanResolver() {
    }

    public static PreparedNativeCpuPlan resolve(
            Operation op,
            List<CompiledTensorDescriptor> inputs,
            DataType dataType,
            CpuNodeExecutionPlan plan,
            CpuStorageProfile cpuStorageProfile
    ) {
        CpuStorageProfile requestedStorage = cpuStorageProfile == null ? CpuStorageProfile.CPU_ARRAY : cpuStorageProfile;
        Operation.OpType opType = op == null ? Operation.OpType.UNKNOWN : op.opType();
        DataType safeDataType = dataType == null ? DataType.FLOAT64 : dataType;
        NativeCpuCoverageEntry coverage = NativeCpuCoverageMatrix.entryFor(opType, safeDataType);

        if (requestedStorage != CpuStorageProfile.CPU_NATIVE) {
            return PreparedNativeCpuPlan.none(requestedStorage,
                    "cpu-storage-profile-not-native:" + requestedStorage.name().toLowerCase());
        }
        if (plan == null) {
            return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "plan"));
        }
        if (usesMatMulProviderMemorySegment(opType, plan)) {
            return PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage);
        }
        if (opType == Operation.OpType.CAST) {
            if (!supportedCast(inputs, dataType)) {
                return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "dtype"));
            }
            return !plan.stridedPath() && denseInputs(inputs)
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (!coverage.nativeSupported()) {
            return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, coverage.fallbackReason());
        }
        if (isViewAlias(opType)) {
            return PreparedNativeCpuPlan.viewAlias(coverage, requestedStorage);
        }
        if (opType == Operation.OpType.WHERE && supportsNativeWhereDType(dataType)) {
            if (!whereNoBroadcast(plan)) {
                return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "broadcast"));
            }
            return denseInputs(inputs)
                    ? PreparedNativeCpuPlan.conditionArrayInputNativeOutput(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (isCompare(opType) && dataType == DataType.BOOL) {
            if (!noBroadcast(plan)) {
                return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "broadcast"));
            }
            return denseInputs(inputs)
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (isBoolLogical(opType) && dataType == DataType.BOOL) {
            if (isBoolLogicalBinary(opType) && !noBroadcast(plan)) {
                return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "broadcast"));
            }
            return denseInputs(inputs)
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (opType == Operation.OpType.CONTIGUOUS) {
            if (!supportsNativeStorageDType(dataType)) {
                return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, coverage.fallbackReason());
            }
            return denseInputs(inputs)
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (isBoolReduction(opType) && dataType == DataType.BOOL) {
            return denseInputs(inputs) && reductionDimension(op) >= -1
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (isFloatReduction(opType)) {
            return supportsNativeReductionDType(dataType, opType) && !plan.stridedPath() && denseInputs(inputs) && reductionDimension(op) >= -1
                    ? PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage)
                    : PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (plan.stridedPath()) {
            return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, ineligible(opType, "strided"));
        }
        if (supportsNativeElementwise(opType, dataType, plan) && denseInputs(inputs)) {
            return PreparedNativeCpuPlan.nativeExecutable(coverage, requestedStorage);
        }
        return PreparedNativeCpuPlan.fallbackOnly(coverage, requestedStorage, fallbackReason(coverage, opType, plan));
    }

    private static boolean usesMatMulProviderMemorySegment(Operation.OpType opType, CpuNodeExecutionPlan plan) {
        return (opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR)
                && plan != null
                && plan.matMulHints() != null
                && plan.matMulHints().usesOpenBlasMemorySegment();
    }

    private static boolean supportsNativeElementwise(Operation.OpType opType, DataType dataType, CpuNodeExecutionPlan plan) {
        if (!supportsNativeElementwiseDType(dataType)) {
            return false;
        }
        if (isNativeUnaryOp(opType, dataType)) {
            return true;
        }
        if (dataType == DataType.FLOAT32 && opType == Operation.OpType.ADD) {
            ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
            return broadcastPlan == null || broadcastPlan.isNoBroadcast() || isLastDimBiasBroadcast(broadcastPlan);
        }
        if (isNativeBinaryOp(opType, dataType)) {
            return noBroadcast(plan);
        }
        return false;
    }

    private static boolean denseInputs(List<CompiledTensorDescriptor> inputs) {
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        for (CompiledTensorDescriptor input : inputs) {
            if (input == null || !input.contiguous() || input.hasStorageOffset()) {
                return false;
            }
        }
        return true;
    }

    private static String fallbackReason(NativeCpuCoverageEntry coverage, Operation.OpType opType, CpuNodeExecutionPlan plan) {
        if (coverage != null && !coverage.fallbackReason().isBlank()) {
            return coverage.fallbackReason();
        }
        if (isBinaryLike(opType) && !noBroadcast(plan)) {
            return ineligible(opType, "broadcast");
        }
        if (opType == Operation.OpType.WHERE && !whereNoBroadcast(plan)) {
            return ineligible(opType, "broadcast");
        }
        return "native-kernel-unsupported:" + opLabel(opType);
    }

    private static boolean supportsNativeElementwiseDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportsNativeStorageDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportsNativeReductionDType(DataType dataType, Operation.OpType opType) {
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
        }
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static boolean supportsNativeWhereDType(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16;
    }

    private static boolean supportedCast(List<CompiledTensorDescriptor> inputs, DataType outputType) {
        if (inputs == null || inputs.size() != 1 || inputs.getFirst() == null) {
            return false;
        }
        DataType inputType = inputs.getFirst().dataType();
        return inputType == DataType.FLOAT32 && outputType == DataType.BFLOAT16
                || inputType == DataType.BFLOAT16 && outputType == DataType.FLOAT32;
    }

    private static boolean isNativeUnaryOp(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.FLOAT64) {
            return opType == Operation.OpType.NEG
                    || opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.LOG
                    || opType == Operation.OpType.EXP
                    || opType == Operation.OpType.FAST_EXP
                    || opType == Operation.OpType.SQRT
                    || opType == Operation.OpType.ABS
                    || opType == Operation.OpType.FLOOR
                    || opType == Operation.OpType.CEIL
                    || opType == Operation.OpType.SIGN
                    || opType == Operation.OpType.POW
                    || opType == Operation.OpType.TANH
                    || opType == Operation.OpType.FAST_TANH
                    || opType == Operation.OpType.SIGMOID
                    || opType == Operation.OpType.INV;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.NEG
                    || opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.ABS;
        }
        return dataType == DataType.FLOAT32
                && (opType == Operation.OpType.NEG
                || opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.LOG
                || opType == Operation.OpType.EXP
                || opType == Operation.OpType.FAST_EXP
                || opType == Operation.OpType.SQRT
                || opType == Operation.OpType.ABS
                || opType == Operation.OpType.FLOOR
                || opType == Operation.OpType.CEIL
                || opType == Operation.OpType.SIGN
                || opType == Operation.OpType.POW
                || opType == Operation.OpType.TANH
                || opType == Operation.OpType.FAST_TANH
                || opType == Operation.OpType.SIGMOID);
    }

    private static boolean isNativeBinaryOp(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.FLOAT64) {
            return opType == Operation.OpType.ADD
                    || opType == Operation.OpType.SUB
                    || opType == Operation.OpType.MUL
                    || opType == Operation.OpType.DIV
                    || opType == Operation.OpType.MIN
                    || opType == Operation.OpType.MAX
                    || opType == Operation.OpType.POW_TENSOR;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.ADD
                    || opType == Operation.OpType.SUB
                    || opType == Operation.OpType.MUL
                    || opType == Operation.OpType.DIV
                    || opType == Operation.OpType.MIN
                    || opType == Operation.OpType.MAX;
        }
        return dataType == DataType.FLOAT32
                && (opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX
                || opType == Operation.OpType.POW_TENSOR);
    }

    private static boolean isBinaryLike(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX
                || opType == Operation.OpType.POW_TENSOR;
    }

    private static boolean isCompare(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static boolean isBoolLogical(Operation.OpType opType) {
        return isBoolLogicalBinary(opType) || opType == Operation.OpType.LOGICAL_NOT;
    }

    private static boolean isBoolLogicalBinary(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND || opType == Operation.OpType.LOGICAL_OR;
    }

    private static boolean isBoolReduction(Operation.OpType opType) {
        return opType == Operation.OpType.REDUCE_ALL || opType == Operation.OpType.REDUCE_ANY;
    }

    private static boolean isFloatReduction(Operation.OpType opType) {
        return opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX;
    }

    private static boolean isViewAlias(Operation.OpType opType) {
        return opType == Operation.OpType.NOOP
                || opType == Operation.OpType.RESHAPE
                || opType == Operation.OpType.PERMUTE
                || opType == Operation.OpType.SELECT
                || opType == Operation.OpType.SLICE
                || opType == Operation.OpType.EXPAND_DIMS
                || opType == Operation.OpType.SQUEEZE;
    }

    private static boolean noBroadcast(CpuNodeExecutionPlan plan) {
        ResolvedBroadcastPlan broadcastPlan = plan.broadcastPlan();
        return broadcastPlan == null || broadcastPlan.isNoBroadcast();
    }

    private static boolean whereNoBroadcast(CpuNodeExecutionPlan plan) {
        ResolvedWhereBroadcastPlan whereBroadcastPlan = plan.whereBroadcastPlan();
        return whereBroadcastPlan == null || whereBroadcastPlan.isNoBroadcast();
    }

    private static boolean isLastDimBiasBroadcast(ResolvedBroadcastPlan broadcastPlan) {
        if (broadcastPlan == null || broadcastPlan.isNoBroadcast()) {
            return false;
        }
        return isFullOutputSide(broadcastPlan.aEffStrides(), broadcastPlan.outStrides())
                && isLastDimBiasSide(broadcastPlan.bEffStrides(), broadcastPlan.outShape())
                || isLastDimBiasSide(broadcastPlan.aEffStrides(), broadcastPlan.outShape())
                && isFullOutputSide(broadcastPlan.bEffStrides(), broadcastPlan.outStrides());
    }

    private static boolean isFullOutputSide(int[] effectiveStrides, int[] outStrides) {
        return Arrays.equals(effectiveStrides, outStrides);
    }

    private static boolean isLastDimBiasSide(int[] effectiveStrides, int[] outShape) {
        if (effectiveStrides == null || outShape == null || effectiveStrides.length != outShape.length || outShape.length < 2) {
            return false;
        }
        int last = effectiveStrides.length - 1;
        if (outShape[last] <= 0 || effectiveStrides[last] != 1) {
            return false;
        }
        for (int dim = 0; dim < last; dim++) {
            if (effectiveStrides[dim] != 0) {
                return false;
            }
        }
        return true;
    }

    private static int reductionDimension(Operation op) {
        if (op instanceof sum sumOp) {
            return sumOp.getDimension();
        }
        if (op instanceof mean meanOp) {
            return meanOp.getDimension();
        }
        if (op instanceof reduceAll reduceAllOp) {
            return reduceAllOp.getDimension();
        }
        if (op instanceof reduceAny reduceAnyOp) {
            return reduceAnyOp.getDimension();
        }
        if (op instanceof reduceMin reduceMinOp) {
            return reduceMinOp.getDimension();
        }
        if (op instanceof reduceMax reduceMaxOp) {
            return reduceMaxOp.getDimension();
        }
        return -2;
    }

    private static String ineligible(Operation.OpType opType, String reason) {
        return "native-kernel-ineligible:" + opLabel(opType) + "-" + reason;
    }

    private static String opLabel(Operation.OpType opType) {
        return opType == null ? "unknown" : opType.name().toLowerCase();
    }
}
