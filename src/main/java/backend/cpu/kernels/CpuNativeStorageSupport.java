package backend.cpu.kernels;

import operations.Operation;
import tensor.DataType;

public final class CpuNativeStorageSupport {
    public enum Status {
        NATIVE_CORRECT_BUT_SLOW,
        NATIVE_UNSUPPORTED,
        ARRAY_ONLY,
        VIEW_ONLY,
        LIBRARY_PROVIDER
    }

    public enum Family {
        OPENBLAS_NATIVE_SEGMENT,
        SEGMENT_SCALAR,
        NATIVE_MICROKERNEL,
        VIEW_ONLY,
        ARRAY_ONLY
    }

    private CpuNativeStorageSupport() {
    }

    public static boolean nativeRegionSupported(Operation.OpType opType, DataType dataType) {
        return providerRoute(opType, dataType)
                || viewAlias(opType, dataType)
                || consumesNativeInputs(opType, dataType)
                || writesNativeOutput(opType, dataType);
    }

    public static boolean consumesNativeInputs(Operation.OpType opType, DataType dataType) {
        return supportsNativeBinary(opType, dataType)
                || supportsNativeUnary(opType, dataType)
                || supportsNativeWhere(opType, dataType)
                || supportsNativeCompare(opType)
                || supportsNativeBoolLogical(opType, dataType)
                || supportsNativeReduction(opType, dataType)
                || supportsNativeLayout(opType, dataType)
                || viewAlias(opType, dataType);
    }

    public static boolean writesNativeOutput(Operation.OpType opType, DataType dataType) {
        return supportsNativeBinary(opType, dataType)
                || supportsNativeUnary(opType, dataType)
                || supportsNativeWhere(opType, dataType)
                || supportsNativeBoolLogical(opType, dataType)
                || supportsNativeReduction(opType, dataType)
                || supportsNativeLayout(opType, dataType)
                || viewAlias(opType, dataType);
    }

    public static boolean preservesNativeStorage(Operation.OpType opType, DataType dataType) {
        return dataType != DataType.BOOL && nativeRegionSupported(opType, dataType);
    }

    /**
     * AUTO may keep native storage only for provider-backed kernels or metadata-only views.
     * Segment scalar kernels are CPU_NATIVE-only until a benchmark-backed promotion is added here.
     */
    public static boolean autoNativeRegionEligible(Operation.OpType opType, DataType dataType) {
        return providerRoute(opType, dataType) || viewAlias(opType, dataType);
    }

    public static boolean providerRoute(Operation.OpType opType, DataType dataType) {
        return (opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR)
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16);
    }

    public static boolean viewAlias(Operation.OpType opType, DataType dataType) {
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            return false;
        }
        return opType == Operation.OpType.NOOP
                || opType == Operation.OpType.RESHAPE
                || opType == Operation.OpType.PERMUTE
                || opType == Operation.OpType.EXPAND
                || opType == Operation.OpType.SELECT
                || opType == Operation.OpType.SLICE
                || opType == Operation.OpType.EXPAND_DIMS
                || opType == Operation.OpType.SQUEEZE;
    }

    public static Status status(Operation.OpType opType, DataType dataType) {
        if (providerRoute(opType, dataType)) {
            return Status.LIBRARY_PROVIDER;
        }
        if (viewAlias(opType, dataType)) {
            return Status.VIEW_ONLY;
        }
        if (nativeRegionSupported(opType, dataType)) {
            return Status.NATIVE_CORRECT_BUT_SLOW;
        }
        if (dataType == DataType.INT32 || dataType == DataType.INT64) {
            return Status.ARRAY_ONLY;
        }
        return Status.NATIVE_UNSUPPORTED;
    }

    public static Family family(Operation.OpType opType, DataType dataType) {
        if (providerRoute(opType, dataType)) {
            return Family.OPENBLAS_NATIVE_SEGMENT;
        }
        if (viewAlias(opType, dataType)) {
            return Family.VIEW_ONLY;
        }
        if (opType == Operation.OpType.CONTIGUOUS) {
            return Family.NATIVE_MICROKERNEL;
        }
        if (nativeRegionSupported(opType, dataType)) {
            return Family.SEGMENT_SCALAR;
        }
        return Family.ARRAY_ONLY;
    }

    public static String unsupportedReason(Operation.OpType opType, DataType dataType) {
        if (nativeRegionSupported(opType, dataType)) {
            return "";
        }
        String label = opType == null ? "unknown" : opType.name().toLowerCase();
        if (opType == Operation.OpType.ARGMAX) {
            return "native-argmax-index-output-unsupported";
        }
        if (opType == Operation.OpType.SOFTMAX || opType == Operation.OpType.LOG_SOFTMAX) {
            return "native-softmax-scalar-loop-slower-than-array";
        }
        if (dataType == DataType.BFLOAT16
                && (opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX)) {
            return "native-bf16-reduce-minmax-output-policy-unsupported";
        }
        if (dataType != DataType.FLOAT32
                && dataType != DataType.FLOAT64
                && dataType != DataType.BFLOAT16
                && dataType != DataType.BOOL) {
            return "native-storage-dtype-unsupported:" + dataType.name().toLowerCase();
        }
        return "native-kernel-unsupported:" + label;
    }

    public static boolean supportsNativeBinary(Operation.OpType opType, DataType dataType) {
        if (opType == Operation.OpType.POW_TENSOR) {
            return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16)
                && (opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX);
    }

    public static boolean supportsNativeUnary(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.ABS;
        }
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64) {
            return false;
        }
        return opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.NEG
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
                || dataType == DataType.FLOAT64 && opType == Operation.OpType.INV;
    }

    public static boolean supportsNativeWhere(Operation.OpType opType, DataType dataType) {
        return opType == Operation.OpType.WHERE
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16);
    }

    public static boolean supportsNativeCompare(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    public static boolean supportsNativeBoolLogical(Operation.OpType opType, DataType dataType) {
        return dataType == DataType.BOOL
                && (opType == Operation.OpType.LOGICAL_AND
                || opType == Operation.OpType.LOGICAL_OR
                || opType == Operation.OpType.LOGICAL_NOT);
    }

    public static boolean supportsNativeReduction(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BOOL) {
            return opType == Operation.OpType.REDUCE_ALL || opType == Operation.OpType.REDUCE_ANY;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)
                && (opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX);
    }

    public static boolean supportsNativeLayout(Operation.OpType opType, DataType dataType) {
        if (opType == Operation.OpType.CAST) {
            return dataType == DataType.FLOAT32 || dataType == DataType.BFLOAT16;
        }
        return opType == Operation.OpType.CONTIGUOUS
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16);
    }
}
