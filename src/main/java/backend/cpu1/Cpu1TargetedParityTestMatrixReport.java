package backend.cpu1;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.TreeSet;

/**
 * Central inventory for cpu1 family-level targeted parity/contract test ownership.
 */
public final class Cpu1TargetedParityTestMatrixReport {
    private static final List<FamilyEntry> MATRIX_ENTRIES = entries(
            targeted(
                    "elementwise",
                    "elementwise prepared runtime parity",
                    List.of("backend.cpu1.Cpu1ExecutionContractTest"),
                    "Covers scalar/vector storage access, strided input views, runtime binding, and trace metadata."
            ),
            targeted(
                    "dtype",
                    "CAST prepared runtime parity",
                    List.of("backend.cpu1.Cpu1DTypeExecutionContractTest"),
                    "Covers array, strided, and memory segment dtype conversion contracts."
            ),
            targeted(
                    "layout",
                    "layout/view/materializing operation parity",
                    List.of("backend.cpu1.Cpu1LayoutExecutionContractTest"),
                    "Covers aliasing layout ops, materializing layout ops, memory segment layout outputs, and trace metadata."
            ),
            targeted(
                    "reduction",
                    "reduction and softmax prepared runtime parity",
                    List.of("backend.cpu1.Cpu1ReductionExecutionContractTest"),
                    "Covers sum/mean/min/max/prod/all/any/argmax/cumsum/softmax/logSoftmax contracts."
            ),
            targeted(
                    "index",
                    "gather/scatter family dense direct parity",
                    List.of("backend.cpu1.Cpu1GatherExecutionContractTest"),
                    "Covers gather, gatherAxis, gatherNd, takeAlongAxis, scatterAdd, scatterAxisAdd, scatterElements, and scatterNd."
            ),
            targeted(
                    "matmul",
                    "MATMUL and LINEAR dense prepared runtime parity",
                    List.of(
                            "backend.cpu1.Cpu1MatmulExecutionContractTest",
                            "backend.cpu1.Cpu1LinearExecutionContractTest"
                    ),
                    "Covers dense matmul route contracts and the current dense linear epilogue subset."
            ),
            targeted(
                    "attention",
                    "SDPA forward prepared runtime parity",
                    List.of("backend.cpu1.Cpu1AttentionExecutionContractTest"),
                    "Covers dense scaled dot product attention forward route contracts."
            ),
            targeted(
                    "loss",
                    "NLL/CrossEntropy/MSE loss prepared runtime parity",
                    List.of(
                            "backend.cpu1.Cpu1NllLossExecutionContractTest",
                            "backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest",
                            "backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest",
                            "backend.cpu1.Cpu1MseLossExecutionContractTest",
                            "backend.cpu1.Cpu1LossMaterializationExecutionContractTest"
                    ),
                    "Covers dense loss route contracts across index and dense target variants."
            ),
            targeted(
                    "normalization",
                    "LayerNorm and RMSNorm dense prepared runtime parity",
                    List.of(
                            "backend.cpu1.Cpu1LayerNormExecutionContractTest",
                            "backend.cpu1.Cpu1RmsNormExecutionContractTest"
                    ),
                    "Covers dense normalization route contracts for Java arrays and memory segments."
            ),
            targeted(
                    "pool2d",
                    "MAX_POOL2D and AVG_POOL2D dense prepared runtime parity",
                    List.of("backend.cpu1.Cpu1Pool2dExecutionContractTest"),
                    "Covers dense pool2d route contracts."
            ),
            targeted(
                    "conv2d",
                    "CONV2D dense direct correctness/fallback parity",
                    List.of("backend.cpu1.Cpu1Conv2dExecutionContractTest"),
                    "Covers dense conv2d direct route correctness and explicit unsupported strided route rejection."
            ),
            targeted(
                    "attention-backward",
                    "SDPA backward specialized graph/backward route",
                    List.of("backend.cpu1.Cpu1AttentionBackwardExecutionContractTest"),
                    "Not an old CPU direct route, but cpu1 keeps a targeted contract owner for the specialized backward route."
            ),
            targeted(
                    "fused",
                    "graph-lowered fused elementwise route",
                    List.of(
                            "backend.cpu1.BackendPrepareDispatcherCpu1FusedRouteTest",
                            "backend.cpu1.Cpu1FusedElementwisePreparerTest",
                            "backend.cpu1.Cpu1FusedGeneratedExecutionTest",
                            "backend.cpu1.Cpu1FusedCodegenContractAlignmentTest",
                            "backend.cpu1.fused.Cpu1FusedGeneratedSupportTest",
                            "backend.cpu1.fused.Cpu1FusedIrBuilderTest"
                    ),
                    "FUSED is intentionally graph-lowered/not-direct in Cpu1CoverageReport, with dedicated route/codegen tests."
            ),
            deferred(
                    "materialization",
                    "broad strided/view materialization parity",
                    "Deferred to todo/118-cpu1-graph-input-materialization-plan.md."
            ),
            deferred(
                    "matmul",
                    "backend-neutral MATMUL_EPILOGUE IR parity",
                    "Deferred to todo/119-general-matmul-epilogue-ir-plan.md."
            ),
            deferred(
                    "attention-backward",
                    "BF16 attention backward parity",
                    "Deferred until BF16 attention backward support is implemented."
            ),
            deferred(
                    "attention",
                    "blocked/tiled attention optimization parity",
                    "Deferred as an optimization scope beyond dense direct route parity."
            ),
            deferred(
                    "index",
                    "deterministic parallel scatter parity",
                    "Deferred until deterministic parallel scatter is designed and implemented."
            ),
            nonGoal(
                    "readiness",
                    "actual benchmark performance numbers/default-route enablement",
                    "Tracked by plan 117 benchmark/default-readiness evidence, not by targeted parity ownership."
            )
    );

    private final List<String> requiredRouteFamilies;
    private final List<FamilyEntry> entries;
    private final List<String> missingRequiredRouteFamilies;

    private Cpu1TargetedParityTestMatrixReport(
            List<String> requiredRouteFamilies,
            List<FamilyEntry> entries,
            List<String> missingRequiredRouteFamilies
    ) {
        this.requiredRouteFamilies = List.copyOf(requiredRouteFamilies);
        this.entries = List.copyOf(entries);
        this.missingRequiredRouteFamilies = List.copyOf(missingRequiredRouteFamilies);
    }

    public static Cpu1TargetedParityTestMatrixReport current() {
        return from(Cpu1CoverageReport.current());
    }

    public static Cpu1TargetedParityTestMatrixReport from(Cpu1CoverageReport coverageReport) {
        TreeSet<String> requiredRouteFamilies = new TreeSet<>(coverageReport.cpu1PreparedFamilyRoutes().values());
        TreeSet<String> classifiedFamilies = new TreeSet<>();
        for (FamilyEntry entry : MATRIX_ENTRIES) {
            classifiedFamilies.add(entry.family());
        }

        List<String> missingRequiredRouteFamilies = requiredRouteFamilies.stream()
                .filter(family -> !classifiedFamilies.contains(family))
                .toList();

        return new Cpu1TargetedParityTestMatrixReport(
                List.copyOf(requiredRouteFamilies),
                MATRIX_ENTRIES,
                missingRequiredRouteFamilies
        );
    }

    public List<String> requiredRouteFamilies() {
        return requiredRouteFamilies;
    }

    public List<FamilyEntry> entries() {
        return entries;
    }

    public List<FamilyEntry> targetedOwnerEntries() {
        return entries.stream()
                .filter(FamilyEntry::targetedOwner)
                .toList();
    }

    public List<FamilyEntry> deferredEntries() {
        return entries.stream()
                .filter(FamilyEntry::deferred)
                .toList();
    }

    public List<FamilyEntry> nonGoalEntries() {
        return entries.stream()
                .filter(FamilyEntry::nonGoal)
                .toList();
    }

    public List<String> missingRequiredRouteFamilies() {
        return missingRequiredRouteFamilies;
    }

    public List<String> targetedOwnerClassNames() {
        TreeSet<String> names = new TreeSet<>();
        for (FamilyEntry entry : targetedOwnerEntries()) {
            names.addAll(entry.ownerClassNames());
        }
        return List.copyOf(names);
    }

    public String gateReport() {
        return String.join("\n",
                "cpu1 targeted parity test matrix gate:",
                "  requiredRouteFamilies=" + requiredRouteFamilies,
                "  targetedOwnerEntries=" + format(targetedOwnerEntries()),
                "  targetedOwnerClasses=" + targetedOwnerClassNames(),
                "  missingRequiredRouteFamilies=" + missingRequiredRouteFamilies,
                "  deferredEntries=" + format(deferredEntries()),
                "  nonGoalEntries=" + format(nonGoalEntries())
        );
    }

    public enum Status {
        TARGETED_TEST_OWNER,
        DEFERRED,
        NON_GOAL
    }

    public record FamilyEntry(
            String family,
            String scope,
            Status status,
            List<String> ownerClassNames,
            String note
    ) {
        public FamilyEntry {
            family = requireText(family, "family");
            scope = requireText(scope, "scope");
            status = Objects.requireNonNull(status, "status");
            ownerClassNames = List.copyOf(ownerClassNames == null ? List.of() : ownerClassNames);
            note = requireText(note, "note");
            if (status == Status.TARGETED_TEST_OWNER && ownerClassNames.isEmpty()) {
                throw new IllegalArgumentException("targeted parity entry must have at least one owner: " + family);
            }
        }

        public boolean targetedOwner() {
            return status == Status.TARGETED_TEST_OWNER;
        }

        public boolean deferred() {
            return status == Status.DEFERRED;
        }

        public boolean nonGoal() {
            return status == Status.NON_GOAL;
        }

        public String key() {
            return family + "/" + scope;
        }
    }

    private static FamilyEntry targeted(String family, String scope, List<String> ownerClassNames, String note) {
        return new FamilyEntry(family, scope, Status.TARGETED_TEST_OWNER, ownerClassNames, note);
    }

    private static FamilyEntry deferred(String family, String scope, String note) {
        return new FamilyEntry(family, scope, Status.DEFERRED, List.of(), note);
    }

    private static FamilyEntry nonGoal(String family, String scope, String note) {
        return new FamilyEntry(family, scope, Status.NON_GOAL, List.of(), note);
    }

    private static List<FamilyEntry> entries(FamilyEntry... entries) {
        List<FamilyEntry> list = List.of(entries);
        assertUniqueKeys(list);
        return list;
    }

    private static void assertUniqueKeys(Collection<FamilyEntry> entries) {
        TreeSet<String> keys = new TreeSet<>();
        for (FamilyEntry entry : entries) {
            if (!keys.add(entry.key())) {
                throw new IllegalArgumentException("Duplicate cpu1 targeted parity test matrix entry: " + entry.key());
            }
        }
    }

    private static String format(Collection<FamilyEntry> entries) {
        List<String> formatted = new ArrayList<>();
        entries.stream()
                .sorted(java.util.Comparator.comparing(FamilyEntry::key))
                .forEach(entry -> formatted.add(format(entry)));
        return formatted.toString();
    }

    private static String format(FamilyEntry entry) {
        if (entry.targetedOwner()) {
            return entry.key() + " -> " + entry.ownerClassNames() + " (" + entry.note() + ")";
        }
        return entry.key() + " -> " + entry.status() + " (" + entry.note() + ")";
    }

    private static String requireText(String value, String fieldName) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(fieldName + " must not be blank");
        }
        return value;
    }
}
