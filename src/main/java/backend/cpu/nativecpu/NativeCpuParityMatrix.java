package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Executable native CPU parity matrix derived from the native CPU coverage facts.
 */
public final class NativeCpuParityMatrix {
    private static final List<NativeCpuParityEntry> ENTRIES = buildEntries();
    private static final Map<DataType, Map<Operation.OpType, NativeCpuParityEntry>> BY_DTYPE_AND_OP =
            indexEntries(ENTRIES);

    private NativeCpuParityMatrix() {
    }

    public static List<NativeCpuParityEntry> entries() {
        return ENTRIES;
    }

    public static List<NativeCpuParityEntry> entriesFor(DataType dataType) {
        if (dataType == null) {
            return List.of();
        }
        return ENTRIES.stream()
                .filter(entry -> entry.dataType() == dataType)
                .toList();
    }

    public static NativeCpuParityEntry entryFor(Operation.OpType opType, DataType dataType) {
        Operation.OpType safeOpType = opType == null ? Operation.OpType.UNKNOWN : opType;
        DataType safeDataType = dataType == null ? DataType.FLOAT64 : dataType;
        Map<Operation.OpType, NativeCpuParityEntry> byOp = BY_DTYPE_AND_OP.get(safeDataType);
        NativeCpuParityEntry entry = byOp == null ? null : byOp.get(safeOpType);
        return entry == null ? entryFromCoverage(NativeCpuCoverageMatrix.entryFor(safeOpType, safeDataType)) : entry;
    }

    public static boolean isAutoEligible(Operation.OpType opType, DataType dataType) {
        return entryFor(opType, dataType).autoEligible();
    }

    public static String renderCsv() {
        StringBuilder out = new StringBuilder();
        out.append("opType,dataType,logicalOperationDefined,status,family,autoEligible,storagePaths,layoutCapabilities,resultResidencies,reason\n");
        for (NativeCpuParityEntry entry : ENTRIES) {
            out.append(entry.reportRow()).append('\n');
        }
        return out.toString();
    }

    private static List<NativeCpuParityEntry> buildEntries() {
        ArrayList<NativeCpuParityEntry> entries = new ArrayList<>();
        for (NativeCpuCoverageEntry coverage : NativeCpuCoverageMatrix.entries()) {
            entries.add(entryFromCoverage(coverage));
        }
        return List.copyOf(entries);
    }

    private static NativeCpuParityEntry entryFromCoverage(NativeCpuCoverageEntry coverage) {
        EnumSet<NativeCpuStoragePath> storagePaths = EnumSet.noneOf(NativeCpuStoragePath.class);
        EnumSet<NativeCpuLayoutCapability> layoutCapabilities = EnumSet.noneOf(NativeCpuLayoutCapability.class);
        EnumSet<NativeCpuResultResidency> resultResidencies = EnumSet.noneOf(NativeCpuResultResidency.class);

        boolean logicalOperationDefined = coverage.opType() != Operation.OpType.UNKNOWN;
        if (logicalOperationDefined) {
            storagePaths.add(NativeCpuStoragePath.CPU_ARRAY_DENSE);
            storagePaths.add(NativeCpuStoragePath.CPU_ARRAY_STRIDED);
            layoutCapabilities.add(NativeCpuLayoutCapability.DENSE);
        }

        switch (coverage.status()) {
            case LIBRARY_PROVIDER -> {
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_REGION_PROVIDER);
                layoutCapabilities.add(NativeCpuLayoutCapability.DENSE);
                resultResidencies.add(NativeCpuResultResidency.CPU_NATIVE);
            }
            case VIEW_ONLY -> {
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_REGION_VIEW_ALIAS);
                addViewLayoutCapabilities(coverage.opType(), layoutCapabilities);
                resultResidencies.add(NativeCpuResultResidency.VIEW_ALIAS);
            }
            case NATIVE_FAST, NATIVE_CORRECT_BUT_SLOW -> {
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_SINGLE_DENSE);
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_REGION_DENSE);
                addNativeLayoutCapabilities(coverage, storagePaths, layoutCapabilities);
                if (isNativeBoolMaskOp(coverage.opType()) && coverage.dataType() == DataType.BOOL) {
                    resultResidencies.add(NativeCpuResultResidency.BOOL_MASK_ARRAY);
                    resultResidencies.add(NativeCpuResultResidency.BOOL_MASK_NATIVE);
                } else {
                    resultResidencies.add(resultResidencyFor(coverage));
                }
            }
            case ARRAY_ONLY, NATIVE_UNSUPPORTED -> resultResidencies.add(NativeCpuResultResidency.CPU_ARRAY);
        }

        if (resultResidencies.isEmpty()) {
            resultResidencies.add(NativeCpuResultResidency.CPU_ARRAY);
        }
        return new NativeCpuParityEntry(
                coverage.opType(),
                coverage.dataType(),
                logicalOperationDefined,
                storagePaths,
                layoutCapabilities,
                resultResidencies,
                coverage.status(),
                coverage.family(),
                autoEligible(coverage.status()),
                parityReason(coverage)
        );
    }

    private static void addViewLayoutCapabilities(
            Operation.OpType opType,
            Set<NativeCpuLayoutCapability> layoutCapabilities
    ) {
        switch (opType) {
            case SELECT, SLICE -> {
                layoutCapabilities.add(NativeCpuLayoutCapability.SELECT_SLICE_OFFSET_VIEW);
                layoutCapabilities.add(NativeCpuLayoutCapability.SAME_SHAPE_STRIDED_READ);
            }
            case EXPAND -> {
                layoutCapabilities.add(NativeCpuLayoutCapability.ZERO_STRIDE_BROADCAST_READ);
                layoutCapabilities.add(NativeCpuLayoutCapability.LAST_DIM_BIAS_BROADCAST);
            }
            default -> layoutCapabilities.add(NativeCpuLayoutCapability.TRANSPOSE_PERMUTE_READ_VIEW);
        }
    }

    private static void addNativeLayoutCapabilities(
            NativeCpuCoverageEntry coverage,
            Set<NativeCpuStoragePath> storagePaths,
            Set<NativeCpuLayoutCapability> layoutCapabilities
    ) {
        layoutCapabilities.add(NativeCpuLayoutCapability.DENSE);
        layoutCapabilities.add(NativeCpuLayoutCapability.OFFSET_CONTIGUOUS);
        if (coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_CORRECT_BUT_SLOW
                || coverage.status() == NativeCpuKernelPerformanceStatus.NATIVE_FAST) {
            if (supportsRegionStridedRead(coverage.opType(), coverage.dataType())) {
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_REGION_STRIDED);
                layoutCapabilities.add(NativeCpuLayoutCapability.SAME_SHAPE_STRIDED_READ);
                layoutCapabilities.add(NativeCpuLayoutCapability.TRANSPOSE_PERMUTE_READ_VIEW);
            }
            if (supportsRegionBroadcastRead(coverage.opType(), coverage.dataType())) {
                storagePaths.add(NativeCpuStoragePath.CPU_NATIVE_REGION_BROADCAST);
                layoutCapabilities.add(NativeCpuLayoutCapability.ZERO_STRIDE_BROADCAST_READ);
                layoutCapabilities.add(NativeCpuLayoutCapability.LAST_DIM_BIAS_BROADCAST);
            }
        }
        if (coverage.opType() == Operation.OpType.CONTIGUOUS) {
            layoutCapabilities.add(NativeCpuLayoutCapability.DENSE_MATERIALIZATION);
        }
    }

    private static boolean supportsRegionStridedRead(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BOOL && isNativeBoolMaskOp(opType)) {
            return true;
        }
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            return false;
        }
        return supportsSegmentUnary(opType, dataType)
                || supportsSegmentBinary(opType, dataType)
                || supportsSegmentCompare(opType)
                || supportsSegmentWhere(opType, dataType)
                || supportsSegmentReduction(opType, dataType)
                || opType == Operation.OpType.CONTIGUOUS;
    }

    private static boolean supportsRegionBroadcastRead(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BOOL && (supportsSegmentCompare(opType) || isBoolLogicalBinary(opType))) {
            return true;
        }
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            return false;
        }
        return supportsSegmentBinary(opType, dataType)
                || supportsSegmentCompare(opType)
                || supportsSegmentWhere(opType, dataType);
    }

    private static boolean supportsSegmentUnary(Operation.OpType opType, DataType dataType) {
        if (dataType == DataType.BFLOAT16) {
            return opType == Operation.OpType.NEG
                || opType == Operation.OpType.MUL_SCALAR
                || opType == Operation.OpType.RELU
                || opType == Operation.OpType.CLAMP_MIN
                || opType == Operation.OpType.CLAMP_MAX
                || opType == Operation.OpType.ABS;
        }
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64
                ? opType == Operation.OpType.NEG
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
                || (dataType == DataType.FLOAT64 && opType == Operation.OpType.INV)
                : false;
    }

    private static boolean supportsSegmentBinary(Operation.OpType opType, DataType dataType) {
        if (dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64 && dataType != DataType.BFLOAT16) {
            return false;
        }
        if (opType == Operation.OpType.POW_TENSOR) {
            return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
        }
        return opType == Operation.OpType.ADD
                || opType == Operation.OpType.SUB
                || opType == Operation.OpType.MUL
                || opType == Operation.OpType.DIV
                || opType == Operation.OpType.MIN
                || opType == Operation.OpType.MAX;
    }

    private static boolean supportsSegmentCompare(Operation.OpType opType) {
        return opType == Operation.OpType.GT
                || opType == Operation.OpType.GE
                || opType == Operation.OpType.LT
                || opType == Operation.OpType.LE
                || opType == Operation.OpType.EQ
                || opType == Operation.OpType.NE;
    }

    private static boolean supportsSegmentWhere(Operation.OpType opType, DataType dataType) {
        return opType == Operation.OpType.WHERE
                && (dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64 || dataType == DataType.BFLOAT16);
    }

    private static boolean supportsSegmentReduction(Operation.OpType opType, DataType dataType) {
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

    private static boolean isNativeBoolMaskOp(Operation.OpType opType) {
        return supportsSegmentCompare(opType)
                || opType == Operation.OpType.LOGICAL_AND
                || opType == Operation.OpType.LOGICAL_OR
                || opType == Operation.OpType.LOGICAL_NOT
                || opType == Operation.OpType.REDUCE_ALL
                || opType == Operation.OpType.REDUCE_ANY;
    }

    private static boolean isBoolLogicalBinary(Operation.OpType opType) {
        return opType == Operation.OpType.LOGICAL_AND || opType == Operation.OpType.LOGICAL_OR;
    }

    private static NativeCpuResultResidency resultResidencyFor(NativeCpuCoverageEntry coverage) {
        return coverage.preservesNativeStorage()
                ? NativeCpuResultResidency.CPU_NATIVE
                : NativeCpuResultResidency.CPU_ARRAY;
    }

    private static String parityReason(NativeCpuCoverageEntry coverage) {
        if (!coverage.fallbackReason().isBlank()) {
            return coverage.fallbackReason();
        }
        return switch (coverage.status()) {
            case LIBRARY_PROVIDER -> "provider-backed-native-region";
            case VIEW_ONLY -> "metadata-only-native-view";
            case NATIVE_FAST -> "native-fast";
            case NATIVE_CORRECT_BUT_SLOW -> "native-correct-but-slow";
            case ARRAY_ONLY -> "array-only";
            case NATIVE_UNSUPPORTED -> "native-unsupported";
        };
    }

    private static boolean autoEligible(NativeCpuKernelPerformanceStatus status) {
        return status == NativeCpuKernelPerformanceStatus.NATIVE_FAST
                || status == NativeCpuKernelPerformanceStatus.LIBRARY_PROVIDER
                || status == NativeCpuKernelPerformanceStatus.VIEW_ONLY;
    }

    private static Map<DataType, Map<Operation.OpType, NativeCpuParityEntry>> indexEntries(List<NativeCpuParityEntry> entries) {
        EnumMap<DataType, Map<Operation.OpType, NativeCpuParityEntry>> byDtype = new EnumMap<>(DataType.class);
        for (NativeCpuParityEntry entry : entries) {
            byDtype.computeIfAbsent(entry.dataType(), ignored -> new EnumMap<>(Operation.OpType.class))
                    .put(entry.opType(), entry);
        }
        byDtype.replaceAll((ignored, byOp) -> Map.copyOf(byOp));
        return Map.copyOf(byDtype);
    }
}
