package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Closed immutable CPU-private facts explaining bounded fusion legality, profitability, and final
 * selection. These values are cold preparation metadata. They contain no graph identity, carrier
 * object, physical resource, generated class, cache state, measurement, or Runtime input and do
 * not participate in generated-artifact identity.
 */
public sealed interface CpuFusionDecision permits CpuFusionDecision.LegalCandidate,
        CpuFusionDecision.LegalityRejection, CpuFusionDecision.ProfitabilityRejection,
        CpuFusionDecision.Selection {

    /**
     * Closed graph-identity-free role of one materialized boundary position. Publication is kept
     * distinct from an unpublished partition write so plan validation can recompute that boundary
     * from the retained logical-memory publication projection rather than trusting this fact.
     */
    enum BoundaryRole {
        /** Value read from outside the selected partition. */ EXTERNAL_READ,
        /** Value materialized between two selected units. */ CROSS_UNIT,
        /** Unpublished value written by the partition. */ PARTITION_WRITE,
        /** Partition write required at the graph publication boundary. */ PUBLICATION
    }

    /** Closed selected execution strategy retained by candidate identity. */
    enum Strategy {
        /** Scalar compute on the invoking thread. */ SCALAR,
        /** Vector compute on the invoking thread. */ VECTOR,
        /** Scalar compute over deterministic parallel ranges. */ PARALLEL_SCALAR,
        /** Vector compute over deterministic parallel ranges. */ PARALLEL_VECTOR
    }

    /** Closed fused/split unit topology role. */
    enum UnitTopology {
        /** Established family unit that enumeration cannot contract. */ INDIVISIBLE,
        /** One-node ordinary pointwise unit. */ SPLIT_POINTWISE,
        /** Multi-node ordinary pointwise contraction. */ FUSED_POINTWISE
    }

    /** Closed workspace role retained without requirement or slot identity. */
    enum WorkspaceRole {
        /** Contiguous external-read copy. */ MATERIALIZATION,
        /** Exact scatter-product accumulation state. */ SCATTER_PRODUCT,
        /** Stable-ordering index state. */ ORDERING_INDICES,
        /** Exact aggregate, reduction, normalization, or statistic state. */ AGGREGATE_EXACT_STATE,
        /** Per-row scaled-dot-product attention score state. */ ATTENTION_ROW_STATE
    }

    /** Closed attempted contraction relation. */
    enum PairKind {
        /** Producer-to-consumer contraction. */ VERTICAL,
        /** Dependency-independent same-domain contraction. */ HORIZONTAL
    }

    /** Closed hard fact that made an attempted contraction illegal. */
    enum HardFact {
        /** Operation semantics forbid contraction. */ SEMANTICS,
        /** Graph publication forbids virtualizing the edge. */ PUBLICATION,
        /** Multiple consumers require materialization. */ FAN_OUT,
        /** State transition or random advancement must remain indivisible. */ STATE_OR_RANDOM,
        /** Numerical traversal or accumulation order must remain indivisible. */ NUMERICAL_ORDER,
        /** Access, alias, layout, or write proof is incomplete. */ ACCESS,
        /** Contraction would violate dependency order. */ DEPENDENCY,
        /** The combined occurrence has no supported lowering. */ LOWERING,
        /** The combined lowering has no admitted pointwise route. */ ROUTE,
        /** A hard structural or resource ceiling would be exceeded. */ BUDGET
    }

    /** Closed legality-rejection reason. */
    enum LegalityReason {
        /** The operation-family boundary is not contractible. */ SEMANTIC_BARRIER,
        /** The connecting value is a graph publication. */ PUBLICATION_BARRIER,
        /** The connecting value has more than one consumer. */ FAN_OUT_BARRIER,
        /** State or random semantics require one indivisible occurrence. */ STATE_OR_RANDOM_BARRIER,
        /** Numerical-order semantics require one indivisible occurrence. */ NUMERICAL_ORDER_BARRIER,
        /** Alias or normalized-access compatibility was not proved. */ ALIAS_OR_ACCESS_UNPROVED,
        /** The proposed contraction would create an invalid dependency. */ DEPENDENCY_CYCLE,
        /** Current CPU lowering does not support the combined occurrence. */ UNSUPPORTED_LOWERING,
        /** The combined occurrence is not ordinary pointwise IR. */ ROUTE_INELIGIBLE,
        /** A CPU 0008B hard ceiling was exceeded. */ HARD_BUDGET_EXCEEDED
    }

    /** Closed profitability-rejection reason. */
    enum ProfitabilityReason {
        /** Candidate improves split by fewer than the required 32 points. */ INSUFFICIENT_MARGIN,
        /** Candidate exceeds the 48-unit profitability-only code ceiling. */ CODE_SIZE_PRESSURE,
        /** Candidate exceeds the 12-value profitability-only liveness ceiling. */ LIVE_VALUE_PRESSURE,
        /** Candidate increases cross-unit materialized bytes. */ MATERIALIZATION_COST,
        /** Best comparable fused alternative ties canonical split. */ SAFE_SPLIT_TIE,
        /** Checked score input is absent or overflowed. */ UNCERTAIN_INPUT
    }

    /** Closed final selection reason. */
    enum SelectionReason {
        /** Best comparable fusion clears every guardrail and the margin. */ PROFITABLE_FUSION,
        /** No comparable alternative clears the required margin. */ CANONICAL_SPLIT,
        /** Best comparable fused alternative ties split. */ TIE_FALLBACK,
        /** At least one candidate score is uncertain. */ UNCERTAINTY_FALLBACK,
        /** Candidate or pair ceiling prevented complete enumeration. */ ENUMERATION_BUDGET_FALLBACK
    }

    /**
     * Typed hexadecimal structural key represented as numeric octets rather than dispatch text.
     *
     * @param octets non-empty unsigned byte values; copied defensively
     */
    record StructuralKey(List<Integer> octets) {
        /** Validates and snapshots one structural key. */
        public StructuralKey {
            octets = List.copyOf(octets);
            if (octets.isEmpty() || octets.stream().anyMatch(value -> value == null
                    || value < 0 || value > 255)) {
                throw new IllegalArgumentException("CPU structural-key octets are invalid");
            }
        }

        /**
         * Decodes the existing hexadecimal portable-IR key into typed octets.
         *
         * @param hexadecimal non-null even-length hexadecimal key
         * @return non-null typed immutable key
         * @throws IllegalArgumentException if the text is empty, odd-length, or non-hexadecimal
         */
        public static StructuralKey fromHex(String hexadecimal) {
            Objects.requireNonNull(hexadecimal, "hexadecimal");
            if (hexadecimal.isEmpty() || (hexadecimal.length() & 1) != 0) {
                throw new IllegalArgumentException("CPU structural key is not even hexadecimal");
            }
            var values = new java.util.ArrayList<Integer>(hexadecimal.length() / 2);
            for (int index = 0; index < hexadecimal.length(); index += 2) {
                int high = Character.digit(hexadecimal.charAt(index), 16);
                int low = Character.digit(hexadecimal.charAt(index + 1), 16);
                if (high < 0 || low < 0) throw new IllegalArgumentException(
                        "CPU structural key is not hexadecimal");
                values.add((high << 4) | low);
            }
            return new StructuralKey(values);
        }
    }

    /**
     * Relative materialized-boundary identity and exact referenced geometry.
     *
     * @param relativeBoundaryPosition stable complete-candidate boundary position
     * @param unitBoundaryPosition stable position within its unit
     * @param role external-read, cross-unit, or publication role
     * @param regime normalized access regime
     * @param referencedBytes checked referenced bytes
     * @param byteAlignment positive exact alignment
     */
    record BoundaryFact(int relativeBoundaryPosition, int unitBoundaryPosition, BoundaryRole role,
            CpuAccessPlan.Regime regime, long referencedBytes, long byteAlignment) {
        /** Validates one relative materialized-boundary fact. */
        public BoundaryFact {
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(regime, "regime");
            if (relativeBoundaryPosition < 0 || unitBoundaryPosition < 0 || referencedBytes < 0
                    || byteAlignment <= 0) {
                throw new IllegalArgumentException("CPU candidate boundary fact is invalid");
            }
        }
    }

    /**
     * Graph-identity-free exact workspace geometry for one unit.
     *
     * @param role closed workspace role
     * @param byteSize non-negative exact byte size
     * @param byteAlignment positive exact alignment
     */
    record WorkspaceFact(WorkspaceRole role, long byteSize, long byteAlignment) {
        /** Validates exact workspace geometry. */
        public WorkspaceFact {
            Objects.requireNonNull(role, "role");
            if (byteSize < 0 || byteAlignment <= 0) throw new IllegalArgumentException(
                    "CPU candidate workspace fact is invalid");
        }
    }

    /**
     * Stable identity of one selected unit.
     *
     * @param memberNodePositions stable relative partition positions
     * @param dependencyUnitPositions strictly earlier unit positions
     * @param portableIrStructuralKey typed portable-IR key
     * @param specialization exact existing generated specialization
     * @param strategy exact selected strategy
     * @param boundaries ordered relative materialized-boundary facts
     * @param workspace optional exact workspace fact
     * @param topology indivisible, split-pointwise, or fused-pointwise role
     */
    record UnitIdentity(List<Integer> memberNodePositions, List<Integer> dependencyUnitPositions,
            StructuralKey portableIrStructuralKey, CpuKernelSpecialization specialization,
            Strategy strategy, List<BoundaryFact> boundaries, Optional<WorkspaceFact> workspace,
            UnitTopology topology) {
        /** Validates and snapshots one complete unit identity. */
        public UnitIdentity {
            memberNodePositions = orderedNonNegative(memberNodePositions, false);
            dependencyUnitPositions = orderedNonNegative(dependencyUnitPositions, true);
            Objects.requireNonNull(portableIrStructuralKey, "portableIrStructuralKey");
            Objects.requireNonNull(specialization, "specialization");
            Objects.requireNonNull(strategy, "strategy");
            boundaries = List.copyOf(boundaries);
            workspace = Objects.requireNonNull(workspace, "workspace");
            Objects.requireNonNull(topology, "topology");
            if (boundaries.stream().anyMatch(Objects::isNull)) throw new NullPointerException(
                    "CPU candidate boundary fact is null");
        }
    }

    /**
     * Stable identity of one complete candidate topology.
     *
     * @param units one through eight units in stable topological order; copied defensively
     */
    record CandidateIdentity(List<UnitIdentity> units) {
        /** Validates and snapshots a complete candidate identity. */
        public CandidateIdentity {
            units = List.copyOf(units);
            if (units.isEmpty() || units.size() > 8 || units.stream().anyMatch(Objects::isNull)) {
                throw new IllegalArgumentException("CPU candidate identity is not one through eight units");
            }
            for (int index = 0; index < units.size(); index++) {
                int unitIndex = index;
                if (units.get(index).dependencyUnitPositions().stream()
                        .anyMatch(value -> value >= unitIndex)) {
                    throw new IllegalArgumentException("CPU candidate dependencies are not stable");
                }
            }
            var members = new java.util.BitSet();
            int boundaryPositions = 0;
            int workspaceFacts = 0;
            for (UnitIdentity unit : units) {
                if (unit.boundaries().size() > 16) throw new IllegalArgumentException(
                        "CPU candidate unit boundary facts exceed the hard ceiling");
                for (int member : unit.memberNodePositions()) {
                    if (member >= 8 || members.get(member)) throw new IllegalArgumentException(
                            "CPU candidate member positions overlap or exceed the partition");
                    members.set(member);
                }
                boundaryPositions = Math.addExact(boundaryPositions, unit.boundaries().size());
                if (unit.workspace().isPresent()) workspaceFacts++;
            }
            if (members.nextClearBit(0) != members.length() || boundaryPositions > 64
                    || workspaceFacts > 8) {
                throw new IllegalArgumentException(
                        "CPU candidate coverage or resource facts exceed their ceiling");
            }
        }
    }

    /**
     * Exact non-negative integer profitability score decomposition. The score is a deterministic
     * structural heuristic, not elapsed time or a claim that a lower score universally executes
     * faster.
     *
     * @param unitCost cost contributed by final unit count
     * @param materializationCost cost contributed by cross-unit materialization
     * @param structuralCost cost contributed by pointwise code size, indexing, and liveness
     * @param familyCost cost contributed by indivisible non-pointwise families
     * @param totalScore checked sum of the preceding four components
     */
    record Score(long unitCost, long materializationCost, long structuralCost, long familyCost,
            long totalScore) {
        /** Validates the checked non-negative score sum. */
        public Score {
            if (unitCost < 0 || materializationCost < 0 || structuralCost < 0 || familyCost < 0
                    || totalScore != Math.addExact(Math.addExact(unitCost, materializationCost),
                            Math.addExact(structuralCost, familyCost))) {
                throw new IllegalArgumentException("CPU profitability score is inconsistent");
            }
        }
    }

    /**
     * Complete resource, hard-budget, and profitability inputs retained for one legal candidate.
     *
     * @param unitCount one through eight final units
     * @param materializedBoundaryPositions at most 64 ordered positions
     * @param workspaceFacts at most eight workspaces
     * @param crossUnitMaterializedBytes checked total cross-unit referenced bytes
     * @param maximumGeneratedCodeSizeUnits maximum pointwise code-size units
     * @param maximumSimultaneouslyLiveValues maximum pointwise live values
     * @param pointwiseUnitCount ordinary pointwise units
     * @param indivisibleNonPointwiseUnitCount established family units
     */
    record CandidateFacts(int unitCount, int materializedBoundaryPositions, int workspaceFacts,
            long crossUnitMaterializedBytes, int maximumGeneratedCodeSizeUnits,
            int maximumSimultaneouslyLiveValues, int pointwiseUnitCount,
            int indivisibleNonPointwiseUnitCount) {
        /** Validates all bounded complete-candidate facts. */
        public CandidateFacts {
            if (unitCount < 1 || unitCount > 8 || materializedBoundaryPositions < 0
                    || materializedBoundaryPositions > 64 || workspaceFacts < 0
                    || workspaceFacts > 8 || crossUnitMaterializedBytes < 0
                    || maximumGeneratedCodeSizeUnits < 0
                    || maximumSimultaneouslyLiveValues < 0 || pointwiseUnitCount < 0
                    || indivisibleNonPointwiseUnitCount < 0
                    || pointwiseUnitCount + indivisibleNonPointwiseUnitCount != unitCount) {
                throw new IllegalArgumentException("CPU complete-candidate facts are invalid");
            }
        }
    }

    /**
     * One retained legal complete candidate and its exact deterministic score.
     *
     * @param identity non-null graph-identity-free complete topology identity
     * @param facts non-null complete resource and structural facts agreeing with {@code identity}
     * @param score non-null optional checked score; empty means profitability is uncertain
     * @param stableRank non-negative deterministic rank among all admitted legal candidates
     * @param canonicalSplit whether this is the safe maximally split selection fallback
     * @param compatibilityBaseline whether this is the exact retained CPU 0008B decomposition
     */
    record LegalCandidate(CandidateIdentity identity, CandidateFacts facts, Optional<Score> score,
            int stableRank, boolean canonicalSplit, boolean compatibilityBaseline)
            implements CpuFusionDecision {
        /** Validates one ranked legal candidate. */
        public LegalCandidate {
            Objects.requireNonNull(identity, "identity");
            Objects.requireNonNull(facts, "facts");
            score = Objects.requireNonNull(score, "score");
            int boundaries = identity.units().stream().mapToInt(unit -> unit.boundaries().size()).sum();
            int workspaces = (int) identity.units().stream().filter(unit -> unit.workspace().isPresent()).count();
            int pointwise = (int) identity.units().stream()
                    .filter(unit -> unit.topology() != UnitTopology.INDIVISIBLE).count();
            long crossBytes = 0;
            var countedCrossPositions = new java.util.BitSet();
            for (UnitIdentity unit : identity.units()) for (BoundaryFact boundary : unit.boundaries()) {
                if (boundary.role() == BoundaryRole.CROSS_UNIT
                        && !countedCrossPositions.get(boundary.relativeBoundaryPosition())) {
                    countedCrossPositions.set(boundary.relativeBoundaryPosition());
                    crossBytes = Math.addExact(crossBytes, boundary.referencedBytes());
                }
            }
            if (stableRank < 0 || identity.units().size() != facts.unitCount()
                    || boundaries != facts.materializedBoundaryPositions()
                    || workspaces != facts.workspaceFacts()
                    || pointwise != facts.pointwiseUnitCount()
                    || crossBytes != facts.crossUnitMaterializedBytes()
                    || score.stream().anyMatch(value -> value.unitCost() != 64L * facts.unitCount()
                        || value.familyCost() != 32L * facts.indivisibleNonPointwiseUnitCount())) {
                throw new IllegalArgumentException("CPU legal-candidate rank or unit count disagrees");
            }
        }
    }

    /**
     * Relative attempted source-unit pair.
     *
     * @param leftUnitPosition non-negative position in the source topology
     * @param rightUnitPosition distinct non-negative position in the source topology
     * @param kind non-null vertical or horizontal contraction relation
     */
    record AttemptedPair(int leftUnitPosition, int rightUnitPosition, PairKind kind) {
        /** Validates the distinct non-negative pair positions. */
        public AttemptedPair {
            Objects.requireNonNull(kind, "kind");
            if (leftUnitPosition < 0 || rightUnitPosition < 0
                    || leftUnitPosition == rightUnitPosition) {
                throw new IllegalArgumentException("CPU attempted pair is invalid");
            }
        }
    }

    /**
     * One failed hard-legality contraction fact. Legality is decided before and independently of
     * profitability; a rejected pair cannot be admitted by a favorable score.
     *
     * @param sourceTopology non-null complete topology against which the pair was attempted
     * @param attemptedPair non-null pair whose positions must exist in {@code sourceTopology}
     * @param failedHardFact non-null category of correctness or resource proof that failed
     * @param reason non-null precise closed rejection reason
     */
    record LegalityRejection(CandidateIdentity sourceTopology, AttemptedPair attemptedPair,
            HardFact failedHardFact, LegalityReason reason) implements CpuFusionDecision {
        /** Validates a source-relative closed legality rejection. */
        public LegalityRejection {
            Objects.requireNonNull(sourceTopology, "sourceTopology");
            Objects.requireNonNull(attemptedPair, "attemptedPair");
            Objects.requireNonNull(failedHardFact, "failedHardFact");
            Objects.requireNonNull(reason, "reason");
            if (attemptedPair.leftUnitPosition() >= sourceTopology.units().size()
                    || attemptedPair.rightUnitPosition() >= sourceTopology.units().size()) {
                throw new IllegalArgumentException("CPU legality rejection pair is outside source");
            }
        }
    }

    /**
     * One legal complete candidate rejected only by profitability policy.
     *
     * @param candidate non-null identity of the legal candidate
     * @param reason non-null profitability-only rejection reason
     * @param canonicalSplitScore non-negative comparison score for canonical split
     * @param candidateScore non-negative candidate score, or zero when the input is uncertain
     * @param requiredMargin non-negative improvement required to replace canonical split
     */
    record ProfitabilityRejection(CandidateIdentity candidate, ProfitabilityReason reason,
            long canonicalSplitScore, long candidateScore, long requiredMargin)
            implements CpuFusionDecision {
        /** Validates one closed profitability rejection. */
        public ProfitabilityRejection {
            Objects.requireNonNull(candidate, "candidate");
            Objects.requireNonNull(reason, "reason");
            if (canonicalSplitScore < 0 || candidateScore < 0 || requiredMargin < 0) {
                throw new IllegalArgumentException("CPU profitability rejection score is invalid");
            }
        }
    }

    /**
     * Final selected identity, comparison roles, rank, and fallback or profitability reason.
     * Incomplete enumeration, any uncertain score, and a tie involving the best comparable
     * alternative select canonical split. A tie belonging only to a non-winning candidate cannot
     * suppress a different strictly profitable winner.
     *
     * @param selected non-null selected complete candidate identity
     * @param canonicalSplit non-null safe split fallback identity
     * @param compatibilityBaseline non-null exact retained CPU 0008B comparison identity
     * @param stableRank non-negative deterministic rank of {@code selected}
     * @param selectedScore non-null optional selected score; empty means uncertain
     * @param canonicalSplitScore non-null optional split score; empty means uncertain
     * @param achievedMargin non-null optional split-minus-selected score difference
     * @param reason non-null profitability or safe-fallback selection reason
     */
    record Selection(CandidateIdentity selected, CandidateIdentity canonicalSplit,
            CandidateIdentity compatibilityBaseline, int stableRank, Optional<Score> selectedScore,
            Optional<Score> canonicalSplitScore, Optional<Long> achievedMargin,
            SelectionReason reason)
            implements CpuFusionDecision {
        /** Validates one final immutable selection fact. */
        public Selection {
            Objects.requireNonNull(selected, "selected");
            Objects.requireNonNull(canonicalSplit, "canonicalSplit");
            Objects.requireNonNull(compatibilityBaseline, "compatibilityBaseline");
            selectedScore = Objects.requireNonNull(selectedScore, "selectedScore");
            canonicalSplitScore = Objects.requireNonNull(canonicalSplitScore,
                    "canonicalSplitScore");
            achievedMargin = Objects.requireNonNull(achievedMargin, "achievedMargin");
            Objects.requireNonNull(reason, "reason");
            if (stableRank < 0 || selectedScore.stream().anyMatch(score -> score.totalScore() < 0)
                    || canonicalSplitScore.stream().anyMatch(score -> score.totalScore() < 0)) {
                throw new IllegalArgumentException("CPU selected score or rank is invalid");
            }
        }
    }

    private static List<Integer> orderedNonNegative(List<Integer> values, boolean allowEmpty) {
        values = List.copyOf(values);
        if (!allowEmpty && values.isEmpty()) throw new IllegalArgumentException(
                "CPU identity position list is empty");
        int previous = -1;
        for (Integer value : values) {
            if (value == null || value < 0 || value <= previous) throw new IllegalArgumentException(
                    "CPU identity positions are not strictly increasing");
            previous = value;
        }
        return values;
    }
}
