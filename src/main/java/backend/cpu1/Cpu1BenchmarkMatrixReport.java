package backend.cpu1;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;

/**
 * Central inventory for checked-in cpu1 benchmark coverage.
 */
public final class Cpu1BenchmarkMatrixReport {
    private static final List<BenchmarkEntry> MATRIX_ENTRIES = entries(
            covered(
                    "elementwise",
                    "F32 MLP hot path array vs memory segment",
                    "backend.cpu1.Cpu1ElementwiseSegmentBenchmarkTest",
                    "benchmarkF32MlpHotPathArrayVsMemorySegment",
                    "Canonical @Tag(\"benchmark\") owner for elementwise hot-path storage residency."
            ),
            covered(
                    "elementwise",
                    "F32 cheap vector ops array vs memory segment",
                    "backend.cpu1.Cpu1ElementwiseSegmentBenchmarkTest",
                    "benchmarkF32CheapOpsVectorArrayVsMemorySegment",
                    "Canonical @Tag(\"benchmark\") owner for F32 elementwise vector storage comparison."
            ),
            covered(
                    "elementwise",
                    "F64 cheap vector ops array vs memory segment",
                    "backend.cpu1.Cpu1ElementwiseSegmentBenchmarkTest",
                    "benchmarkF64CheapOpsVectorArrayVsMemorySegment",
                    "Canonical @Tag(\"benchmark\") owner for F64 elementwise vector storage comparison."
            ),
            covered(
                    "reduction",
                    "large scalar SUM/MEAN single thread vs parallel partials",
                    "backend.cpu1.Cpu1ReductionBenchmarkTest",
                    "benchmarkScalarLargeSumMeanSingleThreadVsParallelPartials",
                    "Canonical @Tag(\"benchmark\") owner for scalar large reduction launch policy."
            ),
            covered(
                    "reduction",
                    "SOFTMAX/LOG_SOFTMAX group width",
                    "backend.cpu1.Cpu1ReductionSoftmaxBenchmarkTest",
                    "benchmarkSoftmaxAndLogSoftmaxGroupWidth",
                    "Canonical @Tag(\"benchmark\") owner for softmax/logSoftmax group width."
            ),
            covered(
                    "layout",
                    "dense multi-axis TILE against generic scalar copy",
                    "backend.cpu1.Cpu1LayoutTileBenchmarkTest",
                    "benchmarkDenseMultiAxisTileAgainstGenericScalar",
                    "Canonical @Tag(\"benchmark\") owner for dense tile layout route."
            ),
            covered(
                    "matmul",
                    "F32 dense MATMUL scalar/vector/OpenBLAS routes",
                    "backend.cpu1.Cpu1MatmulBenchmarkTest",
                    "benchmarkF32DenseMatmulExplicitScalarVectorAndOpenBlasRoutes",
                    "Canonical @Tag(\"benchmark\") owner for F32 matmul route comparison."
            ),
            covered(
                    "matmul",
                    "F64 dense MATMUL scalar/vector/OpenBLAS routes",
                    "backend.cpu1.Cpu1MatmulBenchmarkTest",
                    "benchmarkF64DenseMatmulExplicitScalarVectorAndOpenBlasRoutes",
                    "Canonical @Tag(\"benchmark\") owner for F64 matmul route comparison."
            ),
            covered(
                    "matmul",
                    "F32 three-layer MLP array vs all-native segment",
                    "backend.cpu1.Cpu1MlpBenchmarkTest",
                    "benchmarkF32ThreeLayerMlpArrayVsAllNativeSegment",
                    "Canonical @Tag(\"benchmark\") owner for MLP route residency comparison."
            ),
            covered(
                    "matmul",
                    "F32 large three-layer MLP array vs all-native segment",
                    "backend.cpu1.Cpu1MlpBenchmarkTest",
                    "benchmarkF32LargeThreeLayerMlpArrayVsAllNativeSegment",
                    "Canonical @Tag(\"benchmark\") owner for larger MLP route residency comparison."
            ),
            covered(
                    "matmul",
                    "F32 three-layer MLP OpenBLAS thread counts",
                    "backend.cpu1.Cpu1MlpBenchmarkTest",
                    "benchmarkF32ThreeLayerMlpOpenBlasThreadCounts",
                    "Canonical @Tag(\"benchmark\") owner for OpenBLAS thread-count sensitivity."
            ),
            covered(
                    "matmul",
                    "F32 matmul-activation-matmul chain array vs native segment",
                    "backend.cpu1.Cpu1MlpBenchmarkTest",
                    "benchmarkF32MatmulActivationMatmulChainArrayVsNativeSegment",
                    "Canonical @Tag(\"benchmark\") owner for fused MLP-style graph route comparison."
            ),
            covered(
                    "attention-backward",
                    "dense SDPA backward array/segment scalar/vector routes",
                    "backend.cpu1.Cpu1AttentionBackwardBenchmarkTest",
                    "benchmarkDenseSdpaBackwardArrayScalarVectorAndSegmentScalarVector",
                    "Canonical @Tag(\"benchmark\") owner for dense F32/F64 SDPA backward route comparison."
            ),
            deferred(
                    "loss",
                    "cross entropy indices array vs memory segment performance",
                    "No canonical @Tag(\"benchmark\") loss benchmark owner is checked in yet; dense correctness is covered by targeted parity tests."
            ),
            deferred(
                    "materialization",
                    "broad strided/view materialization performance matrix",
                    "Deferred to todo/118-cpu1-graph-input-materialization-plan.md."
            ),
            deferred(
                    "matmul",
                    "backend-neutral MATMUL_EPILOGUE IR performance matrix",
                    "Deferred to todo/119-general-matmul-epilogue-ir-plan.md."
            ),
            deferred(
                    "attention-backward",
                    "BF16 attention backward route performance",
                    "Deferred until BF16 attention backward parity work is implemented."
            ),
            deferred(
                    "attention",
                    "blocked/tiled attention optimization matrix",
                    "Deferred as an optimization scope beyond dense direct route parity."
            ),
            deferred(
                    "index",
                    "deterministic parallel scatter performance matrix",
                    "Deferred until deterministic parallel scatter is designed and implemented."
            )
    );

    private final List<BenchmarkEntry> entries;

    private Cpu1BenchmarkMatrixReport(List<BenchmarkEntry> entries) {
        this.entries = List.copyOf(entries);
    }

    public static Cpu1BenchmarkMatrixReport current() {
        return new Cpu1BenchmarkMatrixReport(MATRIX_ENTRIES);
    }

    public List<BenchmarkEntry> entries() {
        return entries;
    }

    public List<BenchmarkEntry> coveredEntries() {
        return entries.stream()
                .filter(BenchmarkEntry::covered)
                .toList();
    }

    public List<BenchmarkEntry> deferredEntries() {
        return entries.stream()
                .filter(BenchmarkEntry::deferred)
                .toList();
    }

    public List<BenchmarkEntry> coveredEntriesWithoutOwners() {
        return coveredEntries().stream()
                .filter(entry -> isBlank(entry.ownerClassName()) || isBlank(entry.ownerMethodName()))
                .toList();
    }

    public List<String> canonicalBenchmarkOwnerClassNames() {
        TreeSet<String> names = new TreeSet<>();
        for (BenchmarkEntry entry : coveredEntries()) {
            names.add(entry.ownerClassName());
        }
        return List.copyOf(names);
    }

    public String gateReport() {
        return String.join("\n",
                "cpu1 benchmark matrix gate:",
                "  coveredEntries=" + format(coveredEntries()),
                "  coveredOwnerClasses=" + canonicalBenchmarkOwnerClassNames(),
                "  coveredEntriesWithoutOwners=" + format(coveredEntriesWithoutOwners()),
                "  deferredEntries=" + format(deferredEntries())
        );
    }

    public enum Status {
        COVERED,
        DEFERRED
    }

    public record BenchmarkEntry(
            String family,
            String scenario,
            Status status,
            String ownerClassName,
            String ownerMethodName,
            String note
    ) {
        public BenchmarkEntry {
            family = requireText(family, "family");
            scenario = requireText(scenario, "scenario");
            status = Objects.requireNonNull(status, "status");
            note = requireText(note, "note");
            if (status == Status.COVERED) {
                ownerClassName = requireText(ownerClassName, "ownerClassName");
                ownerMethodName = requireText(ownerMethodName, "ownerMethodName");
            } else {
                ownerClassName = emptyToNull(ownerClassName);
                ownerMethodName = emptyToNull(ownerMethodName);
            }
        }

        public boolean covered() {
            return status == Status.COVERED;
        }

        public boolean deferred() {
            return status == Status.DEFERRED;
        }

        public String key() {
            return family + "/" + scenario;
        }
    }

    private static BenchmarkEntry covered(
            String family,
            String scenario,
            String ownerClassName,
            String ownerMethodName,
            String note
    ) {
        return new BenchmarkEntry(family, scenario, Status.COVERED, ownerClassName, ownerMethodName, note);
    }

    private static BenchmarkEntry deferred(String family, String scenario, String reason) {
        return new BenchmarkEntry(family, scenario, Status.DEFERRED, null, null, reason);
    }

    private static List<BenchmarkEntry> entries(BenchmarkEntry... entries) {
        List<BenchmarkEntry> list = List.of(entries);
        assertUniqueKeys(list);
        return list;
    }

    private static void assertUniqueKeys(Collection<BenchmarkEntry> entries) {
        TreeSet<String> keys = new TreeSet<>();
        for (BenchmarkEntry entry : entries) {
            if (!keys.add(entry.key())) {
                throw new IllegalArgumentException("Duplicate cpu1 benchmark matrix entry: " + entry.key());
            }
        }
    }

    private static String format(Collection<BenchmarkEntry> entries) {
        List<String> formatted = new ArrayList<>();
        entries.stream()
                .sorted(Comparator.comparing(BenchmarkEntry::key))
                .forEach(entry -> formatted.add(format(entry)));
        return formatted.toString();
    }

    private static String format(BenchmarkEntry entry) {
        if (entry.covered()) {
            return entry.key() + " -> " + entry.ownerClassName() + "#" + entry.ownerMethodName()
                    + " (" + entry.note() + ")";
        }
        return entry.key() + " -> DEFERRED (" + entry.note() + ")";
    }

    private static String requireText(String value, String fieldName) {
        if (isBlank(value)) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return value;
    }

    private static String emptyToNull(String value) {
        return isBlank(value) ? null : value;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }
}
