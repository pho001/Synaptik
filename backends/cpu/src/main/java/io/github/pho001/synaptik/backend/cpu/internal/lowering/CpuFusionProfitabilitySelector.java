package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.BoundaryFact;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.BoundaryRole;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.CandidateFacts;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.CandidateIdentity;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.LegalCandidate;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.ProfitabilityReason;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.Score;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.SelectionReason;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.StructuralKey;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.UnitIdentity;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.UnitTopology;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision.WorkspaceFact;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Stateless CPU-private owner of complete-topology identity, deterministic integer profitability
 * ranking, and safe split fallback. It reads only immutable cold analysis facts and performs no
 * measurement, cache access, environment lookup, persistence, generated-code selection, or
 * Runtime work. The unchanged margin and structural-pressure constants are also the required
 * topology guardrails when {@link CpuRepresentationPlanner} composes representation costs with
 * these already-legal topology scores. Boundary roles map exact shared partition-DAG producer and
 * consumer occurrences through each already-selected CPU unit membership. Candidate identity,
 * score, and unit accounting remain CPU-owned, while publication remains a logical-memory fact.
 */
public final class CpuFusionProfitabilitySelector {
    /** Required score improvement before a fused candidate may replace canonical split. */
    static final long REQUIRED_MARGIN = 32;
    /** Profitability-only pointwise generated-code pressure ceiling. */
    static final int MAX_PROFITABLE_CODE_UNITS = 48;
    /** Profitability-only pointwise live-value pressure ceiling. */
    static final int MAX_PROFITABLE_LIVE_VALUES = 12;
    /** Maximum retained decision facts for one admitted partition. */
    static final int MAX_DECISION_FACTS = 384;

    /** Creates a stateless selector that owns no mutable state or external resources. */
    public CpuFusionProfitabilitySelector() { }

    /**
     * One fully prepared legal complete topology supplied for identity and ranking.
     *
     * @param topology non-null complete decomposer topology
     * @param plan non-null plan derived from exactly {@code topology}, with no decision facts yet
     */
    public record Candidate(List<CpuPartitionDagDecomposer.Unit> topology,
            CpuPartitionPreparationPlan plan) {
        /**
         * Validates and snapshots one complete candidate.
         *
         * @throws NullPointerException if {@code topology}, an element, or {@code plan} is null
         * @throws IllegalArgumentException if the topology is empty, differs from the plan's
         *     unit membership or dependencies, or the plan already contains decision facts
         */
        public Candidate {
            topology = List.copyOf(topology);
            Objects.requireNonNull(plan, "plan");
            if (topology.isEmpty() || topology.size() != plan.units().size()
                    || !plan.fusionDecisions().isEmpty()) {
                throw new IllegalArgumentException("CPU candidate topology and plan disagree");
            }
            for (int index = 0; index < topology.size(); index++) {
                if (!topology.get(index).memberNodeOrdinals()
                        .equals(plan.units().get(index).memberNodeOrdinals())
                        || !topology.get(index).dependencies()
                            .equals(plan.units().get(index).dependencies())) {
                    throw new IllegalArgumentException("CPU candidate unit topology disagrees");
                }
            }
        }
    }

    /**
     * Selected complete plan input and its ordered immutable decision facts.
     *
     * @param selected selected candidate supplied by the caller
     * @param decisions ordered legal, legality, profitability, then selection facts
     */
    public record Result(Candidate selected, List<CpuFusionDecision> decisions) {
        /**
         * Validates and snapshots one selector result.
         *
         * @throws NullPointerException if {@code selected}, {@code decisions}, or a decision is
         *     null
         * @throws IllegalArgumentException if the fact list is empty, exceeds 384 entries, or
         *     does not end in exactly the required selection shape
         */
        public Result {
            Objects.requireNonNull(selected, "selected");
            decisions = List.copyOf(decisions);
            if (decisions.isEmpty() || decisions.size() > MAX_DECISION_FACTS
                    || !(decisions.getLast() instanceof CpuFusionDecision.Selection)) {
                throw new IllegalArgumentException("CPU selector result facts are incomplete");
            }
        }
    }

    /**
     * Ranks every admitted legal complete topology and selects a safe complete plan.
     *
     * @param context non-null complete CPU analysis context carrying the shared partition-local
     *     DAG and matching logical-memory facts
     * @param enumeration non-null complete or budget-incomplete bounded enumeration
     * @param candidates non-null candidate plans aligned with enumeration discovery order
     * @return selected candidate and ordered closed decision facts
     * @throws NullPointerException if an argument or candidate element is {@code null}
     * @throws IllegalArgumentException if discovery order, candidate topology, required
     *     canonical/baseline candidates, retained facts, or recomputed identities disagree
     * @throws ArithmeticException if final exact comparison arithmetic overflows
     */
    public Result select(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionDagDecomposer.Enumeration enumeration, List<Candidate> candidates) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(enumeration, "enumeration");
        candidates = List.copyOf(candidates);
        if (candidates.size() != enumeration.candidates().size()
                || candidates.isEmpty() || candidates.size() > 64) {
            throw new IllegalArgumentException("CPU ranked candidates do not match enumeration");
        }
        for (int index = 0; index < candidates.size(); index++) {
            if (!sameTopology(candidates.get(index).topology(), enumeration.candidates().get(index))) {
                throw new IllegalArgumentException("CPU candidate discovery order changed");
            }
        }
        Candidate split = candidateFor(enumeration.canonicalSplit(), candidates);
        Candidate baseline = candidateFor(enumeration.compatibilityBaseline(), candidates);
        var assessed = candidates.stream().map(candidate -> assess(context, candidate)).toList();
        Assessed splitAssessed = assessed.get(candidates.indexOf(split));
        Assessed baselineAssessed = assessed.get(candidates.indexOf(baseline));
        var ranked = new ArrayList<>(assessed);
        ranked.sort(ASSESSMENT_ORDER);
        var legalCandidates = new ArrayList<LegalCandidate>();
        for (Assessed value : assessed) legalCandidates.add(new LegalCandidate(value.identity(),
                value.facts(), value.score(), rankOf(ranked, value.candidate()),
                value.candidate() == split, value.candidate() == baseline));
        int selectedIndex = selectedCandidateIndex(legalCandidates, enumeration.complete());
        Candidate selected = candidates.get(selectedIndex);
        Assessed selectedAssessed = assessed.get(selectedIndex);
        boolean uncertain = assessed.stream().anyMatch(value -> value.score().isEmpty());
        Optional<CandidateIdentity> tiedBest = selected == split && enumeration.complete()
                && !uncertain ? tiedBestIdentity(legalCandidates) : Optional.empty();
        SelectionReason selectionReason = !enumeration.complete()
                ? SelectionReason.ENUMERATION_BUDGET_FALLBACK
                : uncertain ? SelectionReason.UNCERTAINTY_FALLBACK
                : selected != split ? SelectionReason.PROFITABLE_FUSION
                : tiedBest.isPresent() ? SelectionReason.TIE_FALLBACK
                : SelectionReason.CANONICAL_SPLIT;
        var decisions = new ArrayList<CpuFusionDecision>(legalCandidates);
        appendLegality(decisions, enumeration, assessed);
        appendProfitability(decisions, assessed, splitAssessed, selectedAssessed, tiedBest);
        Optional<Long> margin = selectedAssessed.score().isPresent() && splitAssessed.score().isPresent()
                ? Optional.of(Math.subtractExact(splitAssessed.score().orElseThrow().totalScore(),
                        selectedAssessed.score().orElseThrow().totalScore())) : Optional.empty();
        decisions.add(new CpuFusionDecision.Selection(selectedAssessed.identity(),
                splitAssessed.identity(), baselineAssessed.identity(),
                rankOf(ranked, selected), selectedAssessed.score(), splitAssessed.score(),
                margin, selectionReason));
        if (decisions.size() > MAX_DECISION_FACTS) throw new IllegalArgumentException(
                "CPU decision facts exceed the complete partition ceiling");
        return new Result(selected, decisions);
    }

    /**
     * Applies the closed deterministic selection policy to already ranked typed legal facts.
     * This narrow operation-free seam exists so focused tests can prove that a non-winning tie
     * cannot suppress a different strictly profitable winner.
     *
     * @param candidates non-empty ranked legal facts containing exactly one canonical split
     * @param enumerationComplete whether the full bounded candidate set was enumerated
     * @return input-list index of the selected candidate
     * @throws NullPointerException if {@code candidates} or an element is {@code null}
     * @throws IllegalArgumentException if canonical split is missing or duplicated
     */
    static int selectedCandidateIndex(List<LegalCandidate> candidates,
            boolean enumerationComplete) {
        candidates = List.copyOf(candidates);
        int splitIndex = -1;
        for (int index = 0; index < candidates.size(); index++) {
            if (!candidates.get(index).canonicalSplit()) continue;
            if (splitIndex >= 0) throw new IllegalArgumentException(
                    "CPU policy facts contain multiple canonical splits");
            splitIndex = index;
        }
        if (splitIndex < 0) throw new IllegalArgumentException(
                "CPU policy facts omit canonical split");
        if (!enumerationComplete || candidates.stream().anyMatch(value -> value.score().isEmpty())) {
            return splitIndex;
        }
        LegalCandidate split = candidates.get(splitIndex);
        LegalCandidate best = candidates.stream().filter(value -> !value.canonicalSplit())
                .filter(value -> comparable(value, split))
                .min(Comparator.comparingInt(LegalCandidate::stableRank)).orElse(null);
        if (best == null || !profitable(best, split)) return splitIndex;
        return candidates.indexOf(best);
    }

    /**
     * Recomputes the exact candidate identity used by plan validation.
     *
     * @param context complete non-null context carrying the shared partition-local DAG and
     *     matching logical-memory facts
     * @param candidate complete non-null candidate
     * @return graph-identity-free immutable identity
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if the candidate and complete resource view disagree
     */
    public static CandidateIdentity identityOf(PrepareContext<CpuPartitionAnalysisInputs> context,
            Candidate candidate) {
        return assess(context, candidate).identity();
    }

    private static Assessed assess(PrepareContext<CpuPartitionAnalysisInputs> context,
            Candidate candidate) {
        var unitIdentities = new ArrayList<UnitIdentity>();
        int boundaryPositions = 0;
        int workspaces = 0;
        int pointwise = 0;
        int families = 0;
        int maximumCode = 0;
        int maximumLive = 0;
        long materializedBytes = 0;
        boolean uncertain = false;
        for (int unitIndex = 0; unitIndex < candidate.plan().units().size(); unitIndex++) {
            var unit = candidate.plan().units().get(unitIndex);
            var topology = candidate.topology().get(unitIndex);
            boolean ordinaryPointwise=unit.portablePlan().portableKernelIr() instanceof CpuKernelIr
                    &&unit.portablePlan().specialization().matmulIr().isEmpty();
            if (ordinaryPointwise) pointwise++; else families++;
            if (ordinaryPointwise) {
                try {
                    var structural = CpuPartitionDagDecomposer.structuralFacts(
                            (CpuKernelIr) unit.portablePlan().portableKernelIr());
                    maximumCode = Math.max(maximumCode, structural.generatedCodeSizeUnits());
                    maximumLive = Math.max(maximumLive, structural.simultaneouslyLiveValues());
                } catch (ArithmeticException invalid) {
                    uncertain = true;
                }
            }
            var boundaries = new ArrayList<BoundaryFact>();
            for (int local = 0; local < unit.boundaryValues().size(); local++) {
                ValueId value = unit.boundaryValues().get(local);
                int relative = candidate.plan().boundaryValues().indexOf(value);
                if (relative < 0) throw new IllegalArgumentException(
                        "CPU unit boundary is absent from complete resource view");
                var declaration = candidate.plan().bufferDeclarations().get(relative);
                BoundaryRole role = boundaryRole(context, candidate.topology(), value);
                boundaries.add(new BoundaryFact(relative, local, role,
                        unit.accessBindings().get(local).plan().regime(), declaration.byteSize(),
                        declaration.byteAlignment()));
                boundaryPositions = Math.addExact(boundaryPositions, 1);
            }
            Optional<WorkspaceFact> workspace = unit.runtimeFacts().workspaceDeclaration().map(value ->
                    new WorkspaceFact(workspaceRole(unit.runtimeFacts().workspaceUse()),
                            value.byteSize(), value.byteAlignment()));
            if (workspace.isPresent()) workspaces = Math.addExact(workspaces, 1);
            boolean vector = unit.executionStrategy().compute()
                    == CpuPartitionPreparationPlan.ExecutionStrategy.Compute.VECTOR;
            boolean parallel = unit.executionStrategy().orchestration()
                    == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL;
            CpuFusionDecision.Strategy strategy = vector
                    ? parallel ? CpuFusionDecision.Strategy.PARALLEL_VECTOR
                            : CpuFusionDecision.Strategy.VECTOR
                    : parallel ? CpuFusionDecision.Strategy.PARALLEL_SCALAR
                            : CpuFusionDecision.Strategy.SCALAR;
            UnitTopology unitTopology = !ordinaryPointwise ? UnitTopology.INDIVISIBLE
                    : topology.memberNodeOrdinals().size() == 1
                        ? UnitTopology.SPLIT_POINTWISE : UnitTopology.FUSED_POINTWISE;
            unitIdentities.add(new UnitIdentity(topology.memberNodeOrdinals(),
                    topology.dependencies(), StructuralKey.fromHex(
                            unit.portablePlan().portableKernelIr().structuralKey()),
                    unit.portablePlan().specialization(), strategy, boundaries, workspace,
                    unitTopology));
        }
        for (int position = 0; position < candidate.plan().boundaryValues().size(); position++) {
            ValueId value = candidate.plan().boundaryValues().get(position);
            if (boundaryRole(context, candidate.topology(), value) == BoundaryRole.CROSS_UNIT) {
                try {
                    materializedBytes = Math.addExact(materializedBytes,
                            candidate.plan().bufferDeclarations().get(position).byteSize());
                } catch (ArithmeticException overflow) {
                    uncertain = true;
                }
            }
        }
        CandidateFacts facts = new CandidateFacts(candidate.plan().units().size(), boundaryPositions,
                workspaces, uncertain ? 0 : materializedBytes, maximumCode, maximumLive,
                pointwise, families);
        Optional<Score> score;
        try {
            long unitCost = Math.multiplyExact(64L, facts.unitCount());
            long materializationCost = 0;
            for (int position = 0; position < candidate.plan().boundaryValues().size(); position++) {
                ValueId value = candidate.plan().boundaryValues().get(position);
                if (boundaryRole(context, candidate.topology(), value) != BoundaryRole.CROSS_UNIT) continue;
                long bytes = candidate.plan().bufferDeclarations().get(position).byteSize();
                long pages = bytes == 0 ? 0 : Math.addExact(1, Math.floorDiv(bytes - 1, 4096));
                materializationCost = Math.addExact(materializationCost,
                        Math.addExact(16, Math.min(4096, pages)));
            }
            long structuralCost = 0;
            for (var unit : candidate.plan().units()) {
                if (!(unit.portablePlan().portableKernelIr() instanceof CpuKernelIr ir)) continue;
                var structural = CpuPartitionDagDecomposer.structuralFacts(ir);
                structuralCost = Math.addExact(structuralCost, Math.addExact(
                        structural.generatedCodeSizeUnits(), Math.addExact(
                                structural.indexingComplexityUnits(), Math.multiplyExact(8L,
                                    Math.max(0, structural.simultaneouslyLiveValues() - 8)))));
            }
            long familyCost = Math.multiplyExact(32L, families);
            score = uncertain ? Optional.empty() : Optional.of(new Score(unitCost,
                    materializationCost, structuralCost, familyCost,
                    Math.addExact(Math.addExact(unitCost, materializationCost),
                            Math.addExact(structuralCost, familyCost))));
        } catch (ArithmeticException invalid) {
            score = Optional.empty();
        }
        return new Assessed(candidate, new CandidateIdentity(unitIdentities), facts, score);
    }

    private static BoundaryRole boundaryRole(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<CpuPartitionDagDecomposer.Unit> topology, ValueId value) {
        PartitionDag dag = context.partitionDag();
        int producer = dag.producer(value).map(occurrence ->
                unitIndex(topology, occurrence.node())).orElse(-1);
        boolean consumed = dag.consumers(value).stream()
                .anyMatch(occurrence -> unitIndex(topology, occurrence.node()) >= 0);
        if (producer >= 0 && consumed) return BoundaryRole.CROSS_UNIT;
        boolean publication = context.memoryRequirements().stream()
                .filter(LogicalMemoryRequirement::graphOutput)
                .anyMatch(requirement -> requirement.valueId().equals(value));
        return producer >= 0 ? publication ? BoundaryRole.PUBLICATION : BoundaryRole.PARTITION_WRITE
                : BoundaryRole.EXTERNAL_READ;
    }

    private static int unitIndex(List<CpuPartitionDagDecomposer.Unit> topology,
            io.github.pho001.synaptik.model.graph.CompiledNode node) {
        for (int index = 0; index < topology.size(); index++) {
            if (topology.get(index).nodes().stream().anyMatch(member -> member == node)) {
                return index;
            }
        }
        return -1;
    }

    private static CpuFusionDecision.WorkspaceRole workspaceRole(
            CpuPartitionPreparationPlan.WorkspaceUse use) {
        return switch (use) {
            case MATERIALIZATION -> CpuFusionDecision.WorkspaceRole.MATERIALIZATION;
            case SCATTER_PRODUCT -> CpuFusionDecision.WorkspaceRole.SCATTER_PRODUCT;
            case ORDERING_INDICES -> CpuFusionDecision.WorkspaceRole.ORDERING_INDICES;
            case AGGREGATE_EXACT_STATE -> CpuFusionDecision.WorkspaceRole.AGGREGATE_EXACT_STATE;
            case ATTENTION_ROW_STATE -> CpuFusionDecision.WorkspaceRole.ATTENTION_ROW_STATE;
            case NONE -> throw new IllegalArgumentException("workspace has no closed role");
        };
    }

    private static boolean comparable(LegalCandidate value, LegalCandidate split) {
        return value.score().isPresent()
                && value.facts().maximumGeneratedCodeSizeUnits() <= MAX_PROFITABLE_CODE_UNITS
                && value.facts().maximumSimultaneouslyLiveValues() <= MAX_PROFITABLE_LIVE_VALUES
                && value.facts().crossUnitMaterializedBytes()
                    <= split.facts().crossUnitMaterializedBytes();
    }

    private static boolean profitable(LegalCandidate value, LegalCandidate split) {
        return comparable(value, split) && split.score().isPresent()
                && Math.subtractExact(split.score().orElseThrow().totalScore(),
                    value.score().orElseThrow().totalScore()) >= REQUIRED_MARGIN;
    }

    private static boolean samePrimaryRank(LegalCandidate left, LegalCandidate right) {
        return left.score().orElseThrow().totalScore() == right.score().orElseThrow().totalScore()
                && left.facts().crossUnitMaterializedBytes()
                    == right.facts().crossUnitMaterializedBytes()
                && left.facts().maximumGeneratedCodeSizeUnits()
                    == right.facts().maximumGeneratedCodeSizeUnits()
                && left.facts().maximumSimultaneouslyLiveValues()
                    == right.facts().maximumSimultaneouslyLiveValues();
    }

    private static Optional<CandidateIdentity> tiedBestIdentity(
            List<LegalCandidate> candidates) {
        LegalCandidate split = candidates.stream().filter(LegalCandidate::canonicalSplit)
                .findFirst().orElseThrow();
        LegalCandidate best = candidates.stream().filter(value -> !value.canonicalSplit())
                .filter(value -> comparable(value, split))
                .min(Comparator.comparingInt(LegalCandidate::stableRank)).orElse(null);
        return best != null && best.facts().unitCount() < split.facts().unitCount()
                && samePrimaryRank(best, split) ? Optional.of(best.identity()) : Optional.empty();
    }

    private static void appendLegality(List<CpuFusionDecision> decisions,
            CpuPartitionDagDecomposer.Enumeration enumeration, List<Assessed> assessed) {
        for (var attempt : enumeration.attempts()) {
            if (attempt.rejection().isEmpty()) continue;
            Assessed source = assessed.stream().filter(value -> topologyIdentity(value.candidate())
                    .equals(attempt.source())).findFirst().orElseThrow(() ->
                    new IllegalArgumentException("legality source topology was not ranked"));
            var reason = CpuFusionDecision.LegalityReason.valueOf(
                    attempt.rejection().orElseThrow().name());
            var hardFact = switch (reason) {
                case SEMANTIC_BARRIER -> CpuFusionDecision.HardFact.SEMANTICS;
                case PUBLICATION_BARRIER -> CpuFusionDecision.HardFact.PUBLICATION;
                case FAN_OUT_BARRIER -> CpuFusionDecision.HardFact.FAN_OUT;
                case STATE_OR_RANDOM_BARRIER -> CpuFusionDecision.HardFact.STATE_OR_RANDOM;
                case NUMERICAL_ORDER_BARRIER -> CpuFusionDecision.HardFact.NUMERICAL_ORDER;
                case ALIAS_OR_ACCESS_UNPROVED -> CpuFusionDecision.HardFact.ACCESS;
                case DEPENDENCY_CYCLE -> CpuFusionDecision.HardFact.DEPENDENCY;
                case UNSUPPORTED_LOWERING -> CpuFusionDecision.HardFact.LOWERING;
                case ROUTE_INELIGIBLE -> CpuFusionDecision.HardFact.ROUTE;
                case HARD_BUDGET_EXCEEDED -> CpuFusionDecision.HardFact.BUDGET;
            };
            decisions.add(new CpuFusionDecision.LegalityRejection(source.identity(),
                    new CpuFusionDecision.AttemptedPair(attempt.leftUnit(), attempt.rightUnit(),
                            CpuFusionDecision.PairKind.valueOf(attempt.kind().name())),
                    hardFact, reason));
        }
    }

    private static void appendProfitability(List<CpuFusionDecision> decisions,
            List<Assessed> assessed, Assessed split, Assessed selected,
            Optional<CandidateIdentity> tiedBest) {
        long splitScore = split.score().map(Score::totalScore).orElse(0L);
        for (Assessed value : assessed) {
            if (value == selected || value == split) continue;
            ProfitabilityReason reason;
            if (value.score().isEmpty() || split.score().isEmpty()) {
                reason = ProfitabilityReason.UNCERTAIN_INPUT;
            } else if (tiedBest.filter(value.identity()::equals).isPresent()) {
                reason = ProfitabilityReason.SAFE_SPLIT_TIE;
            } else if (value.facts().maximumGeneratedCodeSizeUnits()
                    > MAX_PROFITABLE_CODE_UNITS) {
                reason = ProfitabilityReason.CODE_SIZE_PRESSURE;
            } else if (value.facts().maximumSimultaneouslyLiveValues()
                    > MAX_PROFITABLE_LIVE_VALUES) {
                reason = ProfitabilityReason.LIVE_VALUE_PRESSURE;
            } else if (value.facts().crossUnitMaterializedBytes()
                    > split.facts().crossUnitMaterializedBytes()) {
                reason = ProfitabilityReason.MATERIALIZATION_COST;
            } else {
                reason = ProfitabilityReason.INSUFFICIENT_MARGIN;
            }
            decisions.add(new CpuFusionDecision.ProfitabilityRejection(value.identity(), reason,
                    splitScore, value.score().map(Score::totalScore).orElse(0L), REQUIRED_MARGIN));
        }
    }

    private static Candidate candidateFor(List<CpuPartitionDagDecomposer.Unit> topology,
            List<Candidate> candidates) {
        return candidates.stream().filter(candidate -> sameTopology(candidate.topology(), topology))
                .findFirst().orElseThrow(() -> new IllegalArgumentException(
                        "required CPU comparison candidate is absent"));
    }

    private static int rankOf(List<Assessed> ranked, Candidate candidate) {
        for (int index = 0; index < ranked.size(); index++) {
            if (ranked.get(index).candidate() == candidate) return index;
        }
        throw new IllegalArgumentException("ranked CPU candidate is absent");
    }

    private static boolean sameTopology(List<CpuPartitionDagDecomposer.Unit> left,
            List<CpuPartitionDagDecomposer.Unit> right) {
        return left.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList()
                .equals(right.stream().map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList());
    }

    private static CpuPartitionDagDecomposer.TopologyIdentity topologyIdentity(Candidate candidate) {
        return new CpuPartitionDagDecomposer.TopologyIdentity(candidate.topology().stream()
                .map(CpuPartitionDagDecomposer.Unit::memberNodeOrdinals).toList());
    }

    private static int compareIdentity(CandidateIdentity left, CandidateIdentity right) {
        int compared = Integer.compare(left.units().size(), right.units().size());
        if (compared != 0) return compared;
        for (int unit = 0; unit < left.units().size(); unit++) {
            UnitIdentity a = left.units().get(unit);
            UnitIdentity b = right.units().get(unit);
            compared = compareIntegers(a.memberNodePositions(), b.memberNodePositions());
            if (compared != 0) return compared;
            compared = compareIntegers(a.portableIrStructuralKey().octets(),
                    b.portableIrStructuralKey().octets());
            if (compared != 0) return compared;
            compared = compareIntegers(a.dependencyUnitPositions(), b.dependencyUnitPositions());
            if (compared != 0) return compared;
            compared = compareBytes(a.specialization().compatibilityBytes(),
                    b.specialization().compatibilityBytes());
            if (compared != 0) return compared;
            compared = Integer.compare(a.strategy().ordinal(), b.strategy().ordinal());
            if (compared != 0) return compared;
            compared = compareBoundaries(a.boundaries(), b.boundaries());
            if (compared != 0) return compared;
            compared = compareWorkspace(a.workspace(), b.workspace());
            if (compared != 0) return compared;
            compared = Integer.compare(a.topology().ordinal(), b.topology().ordinal());
            if (compared != 0) return compared;
        }
        return 0;
    }

    private static int compareIntegers(List<Integer> left, List<Integer> right) {
        int size = Math.min(left.size(), right.size());
        for (int index = 0; index < size; index++) {
            int compared = Integer.compare(left.get(index), right.get(index));
            if (compared != 0) return compared;
        }
        return Integer.compare(left.size(), right.size());
    }

    private static int compareBytes(byte[] left, byte[] right) {
        int size = Math.min(left.length, right.length);
        for (int index = 0; index < size; index++) {
            int compared = Integer.compare(Byte.toUnsignedInt(left[index]),
                    Byte.toUnsignedInt(right[index]));
            if (compared != 0) return compared;
        }
        return Integer.compare(left.length, right.length);
    }

    private static int compareBoundaries(List<BoundaryFact> left, List<BoundaryFact> right) {
        int size = Math.min(left.size(), right.size());
        for (int index = 0; index < size; index++) {
            BoundaryFact a = left.get(index);
            BoundaryFact b = right.get(index);
            int compared = Integer.compare(a.relativeBoundaryPosition(),
                    b.relativeBoundaryPosition());
            if (compared == 0) compared = Integer.compare(a.unitBoundaryPosition(),
                    b.unitBoundaryPosition());
            if (compared == 0) compared = Integer.compare(a.role().ordinal(), b.role().ordinal());
            if (compared == 0) compared = Integer.compare(a.regime().ordinal(),
                    b.regime().ordinal());
            if (compared == 0) compared = Long.compare(a.referencedBytes(), b.referencedBytes());
            if (compared == 0) compared = Long.compare(a.byteAlignment(), b.byteAlignment());
            if (compared != 0) return compared;
        }
        return Integer.compare(left.size(), right.size());
    }

    private static int compareWorkspace(Optional<WorkspaceFact> left,
            Optional<WorkspaceFact> right) {
        if (left.isEmpty() != right.isEmpty()) return left.isPresent() ? 1 : -1;
        if (left.isEmpty()) return 0;
        WorkspaceFact a = left.orElseThrow();
        WorkspaceFact b = right.orElseThrow();
        int compared = Integer.compare(a.role().ordinal(), b.role().ordinal());
        if (compared == 0) compared = Long.compare(a.byteSize(), b.byteSize());
        return compared != 0 ? compared : Long.compare(a.byteAlignment(), b.byteAlignment());
    }

    private static final Comparator<Assessed> ASSESSMENT_ORDER = (left, right) -> {
        if (left.score().isEmpty() != right.score().isEmpty()) return left.score().isPresent() ? -1 : 1;
        if (left.score().isPresent()) {
            int compared = Long.compare(left.score().orElseThrow().totalScore(),
                    right.score().orElseThrow().totalScore());
            if (compared != 0) return compared;
        }
        int compared = Long.compare(left.facts().crossUnitMaterializedBytes(),
                right.facts().crossUnitMaterializedBytes());
        if (compared != 0) return compared;
        compared = Integer.compare(left.facts().maximumGeneratedCodeSizeUnits(),
                right.facts().maximumGeneratedCodeSizeUnits());
        if (compared != 0) return compared;
        compared = Integer.compare(left.facts().maximumSimultaneouslyLiveValues(),
                right.facts().maximumSimultaneouslyLiveValues());
        if (compared != 0) return compared;
        compared = Integer.compare(right.facts().unitCount(), left.facts().unitCount());
        return compared != 0 ? compared : compareIdentity(left.identity(), right.identity());
    };

    private record Assessed(Candidate candidate, CandidateIdentity identity, CandidateFacts facts,
            Optional<Score> score) { }

    /**
     * Immutable operation-free topology probe used only by focused ceiling tests.
     *
     * @param candidates non-null candidate identities; copied defensively
     * @param attempts non-null typed attempts; copied defensively
     * @param untestedCandidateRemains whether candidate 65 would otherwise be admitted
     * @param untestedPairRemains whether attempt 257 would otherwise be examined
     */
    record TopologyProbe(List<CandidateIdentity> candidates, List<TopologyAttempt> attempts,
            boolean untestedCandidateRemains, boolean untestedPairRemains) {
        /** Snapshots one typed ceiling probe. */
        TopologyProbe {
            candidates = List.copyOf(candidates);
            attempts = List.copyOf(attempts);
            if (candidates.stream().anyMatch(Objects::isNull)
                    || attempts.stream().anyMatch(Objects::isNull)) {
                throw new NullPointerException("CPU topology probe contains null facts");
            }
        }
    }

    /**
     * One operation-free source-relative pair in the typed ceiling probe.
     *
     * @param source non-null source topology identity
     * @param pair non-null source-relative attempted pair
     */
    record TopologyAttempt(CandidateIdentity source,
            CpuFusionDecision.AttemptedPair pair) {
        /** Validates one typed attempted topology pair. */
        TopologyAttempt {
            Objects.requireNonNull(source, "source");
            Objects.requireNonNull(pair, "pair");
        }
    }

    /**
     * Reports whether a probe proves complete enumeration within both ceilings.
     *
     * @param probe non-null typed probe
     * @return {@code true} only when at most 64 candidates and 256 attempts are retained and no
     *     pending candidate or pair is hidden exactly at a ceiling
     * @throws NullPointerException if {@code probe} is {@code null}
     */
    static boolean completeWithinBudgets(TopologyProbe probe) {
        Objects.requireNonNull(probe, "probe");
        return probe.candidates().size() <= 64 && probe.attempts().size() <= 256
                && !(probe.candidates().size() == 64 && probe.untestedCandidateRemains())
                && !(probe.attempts().size() == 256 && probe.untestedPairRemains());
    }
}
