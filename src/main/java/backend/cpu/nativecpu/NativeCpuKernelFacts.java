package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;

/**
 * Source-of-truth native CPU operation coverage table used before enabling chain-aware native planning.
 */
public final class NativeCpuKernelFacts {
    private static final EnumSet<DataType> FLOAT_NATIVE_STORAGE_DTYPES = EnumSet.of(
            DataType.FLOAT32,
            DataType.FLOAT64,
            DataType.BFLOAT16
    );
    private static final EnumSet<Operation.OpType> METADATA_VIEW_OPS = EnumSet.of(
            Operation.OpType.NOOP,
            Operation.OpType.RESHAPE,
            Operation.OpType.PERMUTE,
            Operation.OpType.EXPAND,
            Operation.OpType.SELECT,
            Operation.OpType.SLICE,
            Operation.OpType.EXPAND_DIMS,
            Operation.OpType.SQUEEZE
    );

    private NativeCpuKernelFacts() {
    }

    /**
     * Returns the native CPU coverage fact for an operation/dtype pair.
     */
    public static NativeCpuKernelFact factFor(Operation.OpType opType, DataType dataType) {
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(dataType, "dataType cannot be null");

        if (dataType == DataType.BOOL && supportsCompare(opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    "requires-dense-contiguous-compare"
            );
        }
        if (dataType == DataType.BOOL && supportsBoolLogical(opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    opType == Operation.OpType.LOGICAL_NOT
                            ? "requires-dense-contiguous"
                            : "requires-dense-contiguous-same-shape"
            );
        }
        if (dataType == DataType.BOOL && supportsBoolReduction(opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    "requires-dense-contiguous-reduction"
            );
        }
        if (opType == Operation.OpType.ARGMAX) {
            return unsupported(opType, dataType, "native-argmax-index-output-unsupported");
        }
        if (!FLOAT_NATIVE_STORAGE_DTYPES.contains(dataType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.ARRAY_ONLY,
                    NativeCpuKernelFamily.ARRAY_ONLY,
                    "native-storage-dtype-unsupported:" + dataType.name().toLowerCase()
            );
        }
        if (opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER,
                    NativeCpuKernelFamily.OPENBLAS_NATIVE_SEGMENT,
                    "requires-openblas-native-segment"
            );
        }
        if (supportsSegmentScalar(dataType, opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    isSameShapeSegmentScalar(opType)
                            ? "requires-dense-contiguous-same-shape"
                            : "requires-dense-contiguous"
            );
        }
        if (dataType == DataType.BFLOAT16 && isMinMaxReduction(opType)) {
            return unsupported(opType, dataType, "native-bf16-reduce-minmax-output-policy-unsupported");
        }
        if (supportsReduction(dataType, opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    "requires-dense-contiguous-reduction"
            );
        }
        if (isSoftmaxLike(opType)) {
            return unsupported(opType, dataType, "native-softmax-scalar-loop-slower-than-array");
        }
        if (opType == Operation.OpType.CAST && supportsNativeCastOutput(dataType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    "requires-dense-contiguous-cast"
            );
        }
        if (opType == Operation.OpType.CONTIGUOUS) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.NATIVE_MICROKERNEL,
                    "requires-dense-contiguous-copy"
            );
        }
        if (METADATA_VIEW_OPS.contains(opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.VIEW_ONLY,
                    NativeCpuKernelFamily.VIEW_ONLY,
                    "metadata-only-native-view"
            );
        }
        if (opType == Operation.OpType.UNKNOWN) {
            return unsupported(opType, dataType, "native-kernel-unknown-op");
        }
        return unsupported(opType, dataType, "native-kernel-unsupported:" + opType.name().toLowerCase());
    }

    /**
     * Returns facts for all stable operation identifiers for one dtype.
     */
    public static List<NativeCpuKernelFact> factsFor(DataType dataType) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        List<NativeCpuKernelFact> facts = new ArrayList<>();
        for (Operation.OpType opType : Operation.OpType.values()) {
            facts.add(factFor(opType, dataType));
        }
        return List.copyOf(facts);
    }

    /**
     * Returns whether a native segment planner may keep this operation inside a native CPU chain.
     */
    public static boolean preservesNativeStorage(Operation.OpType opType, DataType dataType) {
        return factFor(opType, dataType).preservesNativeStorage();
    }

    private static boolean supportsSegmentScalar(DataType dataType, Operation.OpType opType) {
        if (dataType == DataType.FLOAT64) {
            return opType == Operation.OpType.ADD
                    || opType == Operation.OpType.SUB
                    || opType == Operation.OpType.MUL
                    || opType == Operation.OpType.DIV
                    || opType == Operation.OpType.MIN
                    || opType == Operation.OpType.MAX
                    || opType == Operation.OpType.POW_TENSOR
                    || opType == Operation.OpType.MUL_SCALAR
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
                    || opType == Operation.OpType.INV
                    || opType == Operation.OpType.WHERE;
        }
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.ADD
                    || opType == Operation.OpType.SUB
                    || opType == Operation.OpType.MUL
                    || opType == Operation.OpType.DIV
                    || opType == Operation.OpType.MIN
                    || opType == Operation.OpType.MAX
                    || opType == Operation.OpType.MUL_SCALAR
                    || opType == Operation.OpType.NEG
                    || opType == Operation.OpType.RELU
                    || opType == Operation.OpType.CLAMP_MIN
                    || opType == Operation.OpType.CLAMP_MAX
                    || opType == Operation.OpType.ABS
                    || opType == Operation.OpType.WHERE;
        }
        return dataType == DataType.FLOAT32
                && (opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX
                || opType == Operation.OpType.POW_TENSOR
                || opType == Operation.OpType.MUL_SCALAR
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
                || opType == Operation.OpType.WHERE);
    }

    private static boolean isSameShapeSegmentScalar(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX
                || opType == Operation.OpType.POW_TENSOR
                || opType == Operation.OpType.WHERE;
    }

    private static boolean supportsReduction(DataType dataType, Operation.OpType opType) {
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.SUM || opType == Operation.OpType.MEAN;
        }
        return (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64)
                && (opType == Operation.OpType.SUM
                || opType == Operation.OpType.MEAN
                || opType == Operation.OpType.REDUCE_MIN
                || opType == Operation.OpType.REDUCE_MAX);
    }

    private static boolean isMinMaxReduction(Operation.OpType opType) {
        return opType == Operation.OpType.REDUCE_MIN || opType == Operation.OpType.REDUCE_MAX;
    }

    private static boolean isSoftmaxLike(Operation.OpType opType) {
        return opType == Operation.OpType.SOFTMAX || opType == Operation.OpType.LOG_SOFTMAX;
    }

    private static boolean supportsCompare(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static boolean supportsBoolLogical(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND
                || opType == Operation.OpType.LOGICAL_OR
                || opType == Operation.OpType.LOGICAL_NOT;
    }

    private static boolean supportsBoolReduction(Operation.OpType opType) {
        return opType == Operation.OpType.REDUCE_ALL
                || opType == Operation.OpType.REDUCE_ANY;
    }

    private static boolean supportsNativeCastOutput(DataType dataType) {
        return dataType == DataType.FLOAT32
                || dataType == DataType.BFLOAT16;
    }

    private static NativeCpuKernelFact unsupported(Operation.OpType opType, DataType dataType, String reason) {
        return new NativeCpuKernelFact(
                opType,
                dataType,
                NativeCpuKernelPerformanceStatus.NATIVE_UNSUPPORTED,
                NativeCpuKernelFamily.ARRAY_ONLY,
                reason
        );
    }
}
