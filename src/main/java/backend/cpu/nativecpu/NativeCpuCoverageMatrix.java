package backend.cpu.nativecpu;

import operations.Operation;
import tensor.DataType;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

/**
 * Matrix view of native CPU operation coverage derived from {@link NativeCpuKernelFacts}.
 */
public final class NativeCpuCoverageMatrix {
    private static final List<NativeCpuCoverageEntry> ENTRIES = buildEntries();
    private static final Map<DataType, Map<Operation.OpType, NativeCpuCoverageEntry>> BY_DTYPE_AND_OP =
            indexEntries(ENTRIES);

    private NativeCpuCoverageMatrix() {
    }

    /**
     * Returns one matrix row for every operation/dtype pair.
     */
    public static List<NativeCpuCoverageEntry> entries() {
        return ENTRIES;
    }

    /**
     * Returns all operation rows for one dtype.
     */
    public static List<NativeCpuCoverageEntry> entriesFor(DataType dataType) {
        if (dataType == null) {
            return List.of();
        }
        return ENTRIES.stream()
                .filter(entry -> entry.dataType() == dataType)
                .toList();
    }

    /**
     * Returns the matrix row for one operation/dtype pair.
     */
    public static NativeCpuCoverageEntry entryFor(Operation.OpType opType, DataType dataType) {
        Operation.OpType safeOpType = opType == null ? Operation.OpType.UNKNOWN : opType;
        DataType safeDataType = dataType == null ? DataType.FLOAT64 : dataType;
        Map<Operation.OpType, NativeCpuCoverageEntry> byOp = BY_DTYPE_AND_OP.get(safeDataType);
        NativeCpuCoverageEntry entry = byOp == null ? null : byOp.get(safeOpType);
        if (entry != null) {
            return entry;
        }
        NativeCpuKernelFact fact = NativeCpuKernelFacts.factFor(safeOpType, safeDataType);
        return entryFromFact(fact);
    }

    /**
     * Returns whether native CPU storage is supported for one operation/dtype pair.
     */
    public static boolean isNativeSupported(Operation.OpType opType, DataType dataType) {
        return entryFor(opType, dataType).nativeSupported();
    }

    private static List<NativeCpuCoverageEntry> buildEntries() {
        ArrayList<NativeCpuCoverageEntry> entries = new ArrayList<>();
        for (DataType dataType : DataType.values()) {
            for (Operation.OpType opType : Operation.OpType.values()) {
                entries.add(entryFromFact(NativeCpuKernelFacts.factFor(opType, dataType)));
            }
        }
        return List.copyOf(entries);
    }

    private static NativeCpuCoverageEntry entryFromFact(NativeCpuKernelFact fact) {
        boolean nativeSupported = fact.nativeComputeEligible()
                || fact.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY;
        return new NativeCpuCoverageEntry(
                fact.opType(),
                fact.dataType(),
                layoutScope(fact),
                fact.status(),
                fact.family(),
                nativeSupported,
                fact.preservesNativeStorage(),
                nativeSupported ? "" : fact.reason()
        );
    }

    private static NativeCpuCoverageLayoutScope layoutScope(NativeCpuKernelFact fact) {
        if (fact.status() == NativeCpuKernelPerformanceStatus.VIEW_ONLY) {
            return NativeCpuCoverageLayoutScope.VIEW_ONLY;
        }
        if (fact.status() == NativeCpuKernelPerformanceStatus.ARRAY_ONLY
                || fact.dataType() == DataType.INT32
                || fact.dataType() == DataType.INT64) {
            return NativeCpuCoverageLayoutScope.ARRAY_ONLY;
        }
        if (isUnsupportedLayoutOp(fact.opType())) {
            return NativeCpuCoverageLayoutScope.STRIDED_UNSUPPORTED;
        }
        if (fact.nativeComputeEligible()) {
            return NativeCpuCoverageLayoutScope.DENSE_CONTIGUOUS;
        }
        return NativeCpuCoverageLayoutScope.ARRAY_ONLY;
    }

    private static boolean isUnsupportedLayoutOp(Operation.OpType opType) {
        return opType == Operation.OpType.SELECT
                || opType == Operation.OpType.SLICE
                || opType == Operation.OpType.PERMUTE
                || opType == Operation.OpType.EXPAND;
    }

    private static Map<DataType, Map<Operation.OpType, NativeCpuCoverageEntry>> indexEntries(List<NativeCpuCoverageEntry> entries) {
        EnumMap<DataType, Map<Operation.OpType, NativeCpuCoverageEntry>> byDtype = new EnumMap<>(DataType.class);
        for (NativeCpuCoverageEntry entry : entries) {
            byDtype.computeIfAbsent(entry.dataType(), ignored -> new EnumMap<>(Operation.OpType.class))
                    .put(entry.opType(), entry);
        }
        byDtype.replaceAll((ignored, byOp) -> Map.copyOf(byOp));
        return Map.copyOf(byDtype);
    }
}
