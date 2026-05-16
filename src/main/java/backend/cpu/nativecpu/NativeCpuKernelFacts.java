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

        if (!FLOAT_NATIVE_STORAGE_DTYPES.contains(dataType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.ARRAY_ONLY,
                    NativeCpuKernelFamily.ARRAY_ONLY,
                    "native-storage-dtype-unsupported:" + dataType.name().toLowerCase()
            );
        }
        if (opType == Operation.OpType.MATMUL) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER,
                    NativeCpuKernelFamily.OPENBLAS_NATIVE_SEGMENT,
                    "requires-openblas-native-segment"
            );
        }
        if (dataType == DataType.FLOAT32 && supportsF32SegmentScalar(opType)) {
            return new NativeCpuKernelFact(
                    opType,
                    dataType,
                    NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW,
                    NativeCpuKernelFamily.SEGMENT_SCALAR,
                    isSameShapeF32SegmentScalar(opType)
                            ? "requires-dense-contiguous-same-shape"
                            : "requires-dense-contiguous"
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

    private static boolean supportsF32SegmentScalar(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.NEG
                || opType == Operation.OpType.RELU;
    }

    private static boolean isSameShapeF32SegmentScalar(Operation.OpType opType) {
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV;
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
