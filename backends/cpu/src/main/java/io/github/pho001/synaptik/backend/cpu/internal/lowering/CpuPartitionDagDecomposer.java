package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Deterministically decomposes one complete CPU-owned partition DAG into bounded computation
 * units. Complete-partition order, producer, consumer, edge, and port-occurrence facts come from
 * {@link PrepareContext#partitionDag()}; unit membership, contracted topology, lowering, and
 * candidate dependencies and their unit-index accounting remain CPU-owned. Established affine
 * and numerical-family lowerings
 * remain indivisible seeds; only the ordinary pointwise IR is contracted, and a rejected
 * contraction leaves the split topology unchanged.
 */
public final class CpuPartitionDagDecomposer {
    /** Maximum compiled nodes admitted by one complete CPU partition. */
    static final int MAX_NODES = 8;
    /** Maximum contraction pairs examined for one unchanged analysis. */
    static final int MAX_ATTEMPTS = 28;
    /** Maximum materialized input/output values in a newly contracted pointwise unit. */
    static final int MAX_BOUNDARIES = 16;
    /** Maximum simultaneously live pointwise IR values after contraction. */
    static final int MAX_LIVE_VALUES = 16;
    /** Maximum summed structural indexing-complexity units after contraction. */
    static final int MAX_INDEXING_UNITS = 32;
    /** Maximum structural generated-code estimate for a contracted pointwise unit. */
    static final int MAX_CODE_UNITS = 64;
    /** Maximum distinct complete topologies admitted for profitability ranking. */
    static final int MAX_CANDIDATES = 64;
    /** Maximum source-topology/pair attempts admitted by complete enumeration. */
    static final int MAX_ENUMERATION_ATTEMPTS = 256;

    /** Creates a stateless cold-analysis decomposition boundary with no retained graph state. */
    public CpuPartitionDagDecomposer() { }

    /**
     * One immutable finalized unit in stable topological order.
     *
     * @param nodes non-null, non-empty compiled occurrences in original stable order; copied
     *     defensively and retained only during cold analysis
     * @param lowering non-null route-neutral lowering for exactly {@code nodes}
     * @param dependencies non-null direct producer-unit indices; copied defensively
     * @param memberNodeOrdinals non-null original partition ordinals aligned with {@code nodes};
     *     copied defensively
     */
    public record Unit(List<CompiledNode> nodes, CpuPartitionLowering.LoweredPartition lowering,
            List<Integer> dependencies, List<Integer> memberNodeOrdinals) {
        /**
         * Validates and snapshots one selected unit.
         *
         * @throws NullPointerException if a required list, element, or lowering is {@code null}
         * @throws IllegalArgumentException if the unit is empty, member cardinality differs, or
         *     an index is negative
         */
        public Unit {
            nodes = List.copyOf(nodes);
            Objects.requireNonNull(lowering, "lowering");
            dependencies = List.copyOf(dependencies);
            memberNodeOrdinals = List.copyOf(memberNodeOrdinals);
            if (nodes.isEmpty() || nodes.size() != memberNodeOrdinals.size()
                    || dependencies.stream().anyMatch(value -> value == null || value < 0)
                    || memberNodeOrdinals.stream().anyMatch(value -> value == null || value < 0)) {
                throw new IllegalArgumentException("CPU DAG unit facts disagree");
            }
        }

        /** Reports whether this unit may participate in 0008B pointwise contraction.
         * @return {@code true} exactly when the selected portable IR is ordinary pointwise IR
         */
        public boolean pointwise() { return lowering.portableKernelIr() instanceof CpuKernelIr; }
    }

    /** Closed contraction relation examined by complete bounded enumeration. */
    public enum PairKind {
        /** Producer-to-consumer contraction. */ VERTICAL,
        /** Dependency-independent same-domain contraction. */ HORIZONTAL
    }

    /** Closed hard-legality result for one rejected contraction attempt. */
    public enum RejectionReason {
        /** Operation-family semantics forbid contraction. */ SEMANTIC_BARRIER,
        /** The connecting value is a graph publication. */ PUBLICATION_BARRIER,
        /** The connecting value has multiple consumers. */ FAN_OUT_BARRIER,
        /** State transition or random advancement must remain indivisible. */ STATE_OR_RANDOM_BARRIER,
        /** Numerical traversal or accumulation order must remain indivisible. */ NUMERICAL_ORDER_BARRIER,
        /** Alias or normalized-access compatibility was not proved. */ ALIAS_OR_ACCESS_UNPROVED,
        /** Contraction would violate dependency order. */ DEPENDENCY_CYCLE,
        /** Current lowering does not support the combined occurrence. */ UNSUPPORTED_LOWERING,
        /** The combined lowering is not an admitted pointwise route. */ ROUTE_INELIGIBLE,
        /** A hard structural or resource ceiling would be exceeded. */ HARD_BUDGET_EXCEEDED
    }

    /**
     * Stable graph-identity-free source topology used to identify one attempted pair.
     *
     * @param memberNodeOrdinals ordered unit member positions; copied defensively
     */
    public record TopologyIdentity(List<List<Integer>> memberNodeOrdinals) {
        /**
         * Validates and snapshots the complete unit membership.
         *
         * @throws NullPointerException if the outer list, an inner list, or a member is null
         * @throws IllegalArgumentException if there are no units, more than eight units, or an
         *     empty unit membership
         */
        public TopologyIdentity {
            memberNodeOrdinals = memberNodeOrdinals.stream().map(List::copyOf).toList();
            if (memberNodeOrdinals.isEmpty() || memberNodeOrdinals.size() > MAX_NODES
                    || memberNodeOrdinals.stream().anyMatch(List::isEmpty)) {
                throw new IllegalArgumentException("CPU topology identity is invalid");
            }
        }
    }

    /**
     * One distinct source-topology pair attempt.
     *
     * @param source complete source topology
     * @param leftUnit stable source-unit position
     * @param rightUnit stable source-unit position
     * @param kind vertical or horizontal relation
     * @param rejection empty exactly when contraction produced a legal complete topology
     */
    public record Attempt(TopologyIdentity source, int leftUnit, int rightUnit, PairKind kind,
            Optional<RejectionReason> rejection) {
        /**
         * Validates the immutable attempted-pair fact.
         *
         * @throws NullPointerException if {@code source}, {@code kind}, or {@code rejection} is
         *     null
         * @throws IllegalArgumentException if either position is outside {@code source} or both
         *     positions are equal
         */
        public Attempt {
            Objects.requireNonNull(source, "source");
            Objects.requireNonNull(kind, "kind");
            rejection = Objects.requireNonNull(rejection, "rejection");
            if (leftUnit < 0 || rightUnit < 0 || leftUnit == rightUnit
                    || leftUnit >= source.memberNodeOrdinals().size()
                    || rightUnit >= source.memberNodeOrdinals().size()) {
                throw new IllegalArgumentException("CPU attempted pair is outside its topology");
            }
        }
    }

    /**
     * Complete bounded enumeration result. Candidate zero is always canonical split; the exact
     * compatibility baseline is always present. An incomplete result is inspection-only and must
     * select canonical split.
     *
     * @param canonicalSplit immutable safe split topology
     * @param compatibilityBaseline exact unchanged {@link #decompose} result
     * @param candidates distinct complete legal topologies in breadth-first discovery order
     * @param attempts distinct source/pair attempts in deterministic order
     * @param complete whether every reachable pair was examined within both ceilings
     */
    public record Enumeration(List<Unit> canonicalSplit, List<Unit> compatibilityBaseline,
            List<List<Unit>> candidates, List<Attempt> attempts, boolean complete) {
        /**
         * Validates and snapshots one enumeration result.
         *
         * @throws NullPointerException if a required list or retained element is {@code null}
         * @throws IllegalArgumentException if a ceiling is exceeded, canonical split is not the
         *     first candidate, or the exact compatibility baseline is absent
         */
        public Enumeration {
            canonicalSplit = List.copyOf(canonicalSplit);
            compatibilityBaseline = List.copyOf(compatibilityBaseline);
            candidates = candidates.stream().map(List::copyOf).toList();
            attempts = List.copyOf(attempts);
            if (canonicalSplit.isEmpty() || compatibilityBaseline.isEmpty()
                    || candidates.isEmpty() || candidates.size() > MAX_CANDIDATES
                    || attempts.size() > MAX_ENUMERATION_ATTEMPTS
                    || !sameMembership(candidates.getFirst(), canonicalSplit)
                    || !containsMembership(candidates, compatibilityBaseline)) {
                throw new IllegalArgumentException("CPU complete topology enumeration disagrees");
            }
        }
    }

    /**
     * Exact shared 0008B pointwise structural calculation.
     *
     * @param materializedBoundaries non-negative count of non-virtual IR boundaries
     * @param indexingComplexityUnits non-negative checked access-complexity sum
     * @param simultaneouslyLiveValues non-negative maximum live-value count
     * @param generatedCodeSizeUnits non-negative structural code-size estimate, not byte length
     */
    public record StructuralFacts(int materializedBoundaries, int indexingComplexityUnits,
            int simultaneouslyLiveValues, int generatedCodeSizeUnits) {
        /** Validates non-negative checked structural facts. */
        public StructuralFacts {
            if (materializedBoundaries < 0 || indexingComplexityUnits < 0
                    || simultaneouslyLiveValues < 0 || generatedCodeSizeUnits < 0) {
                throw new IllegalArgumentException("CPU pointwise structural facts are negative");
            }
        }
    }

    /**
     * Enumerates every complete topology reachable through the unchanged 0008B contraction
     * grammar. Recognition-associated baseline units are immutable barriers.
     *
     * @param context complete non-null CPU analysis context whose shared partition DAG supplies
     *     stable node order and exact producer, consumer, edge, and port occurrences
     * @param lowering non-null current lowering owner
     * @param recognition non-null immutable 0008C facts associated with the exact baseline
     * @return bounded deterministic complete-topology enumeration
     * @throws NullPointerException if an argument or recognition fact is {@code null}
     * @throws IllegalArgumentException if the complete context, recognition association,
     *     canonical seed topology, or retained compatibility baseline is inconsistent
     * @throws ArithmeticException if exact topology or lowering arithmetic overflows
     */
    public Enumeration enumerate(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, List<CpuSpecializedSubgraph> recognition) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(lowering, "lowering");
        recognition = List.copyOf(recognition);
        validate(context);
        PartitionDag dag = context.partitionDag();
        var ordinals = ordinals(dag);
        List<Unit> baseline = decompose(context, lowering);
        var lockedBaselineUnits = new LinkedHashSet<Integer>();
        recognition.forEach(fact -> lockedBaselineUnits.addAll(fact.baselineUnitIndices()));
        if (lockedBaselineUnits.stream().anyMatch(index -> index < 0 || index >= baseline.size())) {
            throw new IllegalArgumentException("recognition references an invalid baseline unit");
        }
        List<MutableUnit> splitMutable = canonicalSeeds(context, lowering, ordinals, baseline,
                lockedBaselineUnits);
        List<Unit> split = finish(splitMutable, ordinals, dag);
        var queue = new java.util.ArrayDeque<List<MutableUnit>>();
        queue.add(copyTopology(splitMutable));
        var candidates = new ArrayList<List<Unit>>();
        candidates.add(split);
        var seen = new LinkedHashSet<TopologyIdentity>();
        seen.add(identity(split));
        for(CpuSpecializedSubgraph fact:recognition) {
            if(!(fact instanceof CpuSpecializedSubgraph.MatmulEpilogue)
                    ||fact.disposition()!=CpuSpecializedSubgraph.ExecutionDisposition.EXECUTABLE_ALTERNATIVES)
                continue;
            var memberSet=new LinkedHashSet<>(fact.memberNodeOrdinals());
            List<CompiledNode> fusedNodes=fact.memberNodeOrdinals().stream()
                    .map(dag.nodes()::get).toList();
            var fusedLowering=new CpuMatmulLowering().lower(
                    project(context,fusedNodes,context.backendInputs()),
                    (CpuSpecializedSubgraph.MatmulEpilogue)fact);
            if(fusedLowering.matmulIr().isEmpty())
                throw new IllegalArgumentException("MATMUL fused alternative did not lower to MATMUL IR");
            List<MutableUnit> fusedTopology=copyTopology(splitMutable);
            var covered=new LinkedHashSet<Integer>();
            fusedTopology.removeIf(unit->{List<Integer> members=unit.nodes.stream().map(ordinals::get).toList();
                boolean selected=members.stream().anyMatch(memberSet::contains);
                if(selected)covered.addAll(members);return selected;});
            if(!covered.equals(memberSet))throw new IllegalArgumentException(
                    "MATMUL recognition does not match canonical split members");
            fusedTopology.add(new MutableUnit(fusedNodes,fusedLowering));
            List<Unit> finished=finish(fusedTopology,ordinals,dag);
            if(seen.add(identity(finished))){candidates.add(finished);queue.addLast(fusedTopology);}
        }
        var attempts = new ArrayList<Attempt>();
        boolean complete = true;
        enumeration: while (!queue.isEmpty()) {
            List<MutableUnit> source = queue.removeFirst();
            List<MutableUnit> ordered = topological(source, ordinals, dag);
            TopologyIdentity sourceIdentity = identity(finish(source, ordinals, dag));
            var pairs = enumerationPairs(dag, ordered);
            for (Pair pair : pairs) {
                if (attempts.size() == MAX_ENUMERATION_ATTEMPTS) {
                    complete = false;
                    break enumeration;
                }
                Contraction result = attempt(context, lowering, source, pair.left(), pair.right(),
                        pair.kind(), ordinals, baseline, lockedBaselineUnits);
                int leftIndex = ordered.indexOf(pair.left());
                int rightIndex = ordered.indexOf(pair.right());
                attempts.add(new Attempt(sourceIdentity, leftIndex, rightIndex, pair.kind(),
                        Optional.ofNullable(result.rejection())));
                if (result.unit() == null) continue;
                List<MutableUnit> next = copyTopology(source);
                int left = memberIndex(next, pair.left());
                int right = memberIndex(next, pair.right());
                replace(next, left, right, result.unit());
                List<Unit> finished = finish(next, ordinals, dag);
                TopologyIdentity candidateIdentity = identity(finished);
                if (!seen.add(candidateIdentity)) continue;
                if (candidates.size() == MAX_CANDIDATES) {
                    complete = false;
                    break enumeration;
                }
                candidates.add(finished);
                queue.addLast(next);
            }
        }
        if (candidates.stream().noneMatch(candidate -> sameMembership(candidate, baseline))) {
            if (complete) throw new IllegalArgumentException(
                    "exact CPU 0008B baseline is not reachable");
            if (candidates.size() == MAX_CANDIDATES) candidates.set(candidates.size() - 1, baseline);
            else candidates.add(baseline);
        }
        return new Enumeration(split, baseline, candidates, attempts, complete);
    }

    /**
     * Builds the maximally split supported baseline and applies bounded deterministic pointwise
     * contractions.
     *
     * @param context complete non-null CPU analysis context whose shared partition DAG supplies
     *     stable node order and exact producer, consumer, edge, and port occurrences
     * @param lowering current non-null family lowering owner
     * @return one through eight immutable units in stable topological order
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if ownership, topology, projection, seed support, or a
     *     selected unit contract is invalid
     * @throws ArithmeticException if exact lowering or budget arithmetic overflows
     */
    public List<Unit> decompose(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(lowering, "lowering");
        validate(context);
        PartitionDag dag = context.partitionDag();
        var ordinals = ordinals(dag);
        var working = seeds(context, lowering, ordinals);
        int attempts = 0;
        boolean changed;
        do {
            changed = false;
            outer: for (int left = 0; left < working.size(); left++) {
                for (int right = 0; right < working.size(); right++) {
                    if (left == right) continue;
                    if (attempts++ >= MAX_ATTEMPTS) break outer;
                    if (vertical(context, working.get(left), working.get(right))) {
                        MutableUnit fused = contract(context, lowering, working.get(left),
                                working.get(right), ordinals);
                        if (fused != null) {
                            replace(working, left, right, fused);
                            changed = true;
                            break outer;
                        }
                    }
                }
            }
        } while (changed && attempts < MAX_ATTEMPTS);

        changed = true;
        while (changed && attempts < MAX_ATTEMPTS) {
            changed = false;
            List<MutableUnit> ordered = topological(working, ordinals, dag);
            outer: for (int left = 0; left < ordered.size(); left++) {
                for (int right = left + 1; right < ordered.size(); right++) {
                    if (attempts++ >= MAX_ATTEMPTS) break outer;
                    MutableUnit a = ordered.get(left);
                    MutableUnit b = ordered.get(right);
                    if (horizontal(dag, working, a, b)) {
                        MutableUnit fused = contract(context, lowering, a, b, ordinals);
                        if (fused != null) {
                            replace(working, working.indexOf(a), working.indexOf(b), fused);
                            changed = true;
                            break outer;
                        }
                    }
                }
            }
        }
        return finish(working, ordinals, dag);
    }

    /**
     * Projects one selected unit without inventing graph values or changing the outer owner.
     * Membership and outside-consumer facts are filtered from the source partition DAG, and the
     * returned context contains exactly the selected nodes in their original stable order. The
     * returned context is immutable analysis state and owns no physical resources.
     *
     * @param source non-null complete partition context carrying the source shared partition DAG
     * @param nodes non-null, non-empty identity-preserving subset of source DAG nodes in original
     *     stable order
     * @param inputs non-null CPU-private analysis inputs for the selected unit
     * @return a non-null unit-scoped context with projected memory/publication facts
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if the selected projection violates {@code PrepareContext}
     *     invariants
     */
    public PrepareContext<CpuPartitionAnalysisInputs> unitContext(
            PrepareContext<CpuPartitionAnalysisInputs> source, List<CompiledNode> nodes,
            CpuPartitionAnalysisInputs inputs) {
        return project(source, nodes, inputs);
    }

    private static void validate(PrepareContext<CpuPartitionAnalysisInputs> context) {
        if (!context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)
                || context.partitionDag().nodes().isEmpty()
                || context.partitionDag().nodes().size() > MAX_NODES) {
            throw new IllegalArgumentException("CPU partition requires one through eight nodes");
        }
        var values = new HashSet<ValueId>();
        context.values().forEach(value -> values.add(value.id()));
        var memory = new HashSet<ValueId>();
        context.memoryRequirements().forEach(value -> memory.add(value.valueId()));
        for (CompiledNode node : context.partitionDag().nodes()) {
            for (ValueId input : node.inputs()) {
                if (!values.contains(input) || !memory.contains(input))
                    throw new IllegalArgumentException("CPU input fact is not projected");
            }
            for (ValueId output : node.outputs()) {
                if (!values.contains(output) || !memory.contains(output)) {
                    throw new IllegalArgumentException("CPU output fact is not projected");
                }
            }
        }
    }

    private static List<MutableUnit> seeds(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, Map<CompiledNode, Integer> ordinals) {
        var result = new ArrayList<MutableUnit>();
        List<CompiledNode> partitionNodes = context.partitionDag().nodes();
        int index = 0;
        while (index < partitionNodes.size()) {
            MutableUnit selected = null;
            IllegalArgumentException lastFailure = null;
            for (int end = partitionNodes.size(); end > index; end--) {
                List<CompiledNode> nodes = partitionNodes.subList(index, end);
                if (nodes.size() > 1 && !establishedMultiNodeCandidate(nodes)) continue;
                try {
                    var lowered = lowering.lower(project(context, nodes,
                            context.backendInputs()));
                    selected = new MutableUnit(nodes, lowered);
                    break;
                } catch (IllegalArgumentException unsupported) {
                    lastFailure = unsupported;
                    // A shorter current-family occurrence may still be the exact seed.
                }
            }
            if (selected == null) throw new IllegalArgumentException(
                    "CPU partition contains an independently unsupported node at ordinal " + index,
                    lastFailure);
            result.add(selected);
            index += selected.nodes.size();
        }
        return result;
    }

    private static boolean establishedMultiNodeCandidate(List<CompiledNode> nodes) {
        boolean affine = nodes.stream().allMatch(node -> {
            Object kind = node.operation().kind();
            return kind instanceof io.github.pho001.synaptik.model.operation.layout.ContiguousKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.index.SelectKind
                    || kind == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE;
        });
        Object first = nodes.getFirst().operation().kind();
        return affine || first == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.EXPAND_DIMS
                || first == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D;
    }

    private static boolean vertical(PrepareContext<CpuPartitionAnalysisInputs> context,
            MutableUnit producer, MutableUnit consumer) {
        if (!producer.pointwise() || !consumer.pointwise()) return false;
        var graphOutputs = new HashSet<ValueId>();
        context.memoryRequirements().stream().filter(LogicalMemoryRequirement::graphOutput)
                .forEach(value -> graphOutputs.add(value.valueId()));
        for (CompiledNode node : producer.nodes) for (ValueId output : node.outputs()) {
            List<PartitionDag.ConsumerOccurrence> uses = context.partitionDag().consumers(output);
            if (uses.size() == 1 && !graphOutputs.contains(output)
                    && containsNode(consumer, uses.getFirst().node())) return true;
        }
        return false;
    }

    private static boolean horizontal(PartitionDag dag, List<MutableUnit> all,
            MutableUnit left, MutableUnit right) {
        if (!left.pointwise() || !right.pointwise()) return false;
        var leftDependencies = producers(dag, all, left);
        var rightDependencies = producers(dag, all, right);
        if (!leftDependencies.equals(rightDependencies)) return false;
        return !hasEdge(dag, left, right) && !hasEdge(dag, right, left)
                && java.util.Arrays.equals(left.lowering.extents(), right.lowering.extents());
    }

    private static MutableUnit contract(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, MutableUnit left, MutableUnit right,
            Map<CompiledNode, Integer> ordinals) {
        return contractResult(context, lowering, left, right, ordinals).unit();
    }

    private static Contraction contractResult(
            PrepareContext<CpuPartitionAnalysisInputs> context, CpuPartitionLowering lowering,
            MutableUnit left, MutableUnit right, Map<CompiledNode, Integer> ordinals) {
        var nodes = new ArrayList<CompiledNode>(left.nodes);
        nodes.addAll(right.nodes);
        nodes.sort(Comparator.comparingInt(ordinals::get));
        if (nodes.size() > MAX_NODES) return new Contraction(null,
                RejectionReason.HARD_BUDGET_EXCEEDED);
        try {
            var lowered = lowering.lower(project(context, nodes, context.backendInputs()));
            if (!(lowered.portableKernelIr() instanceof CpuKernelIr ir)) return new Contraction(
                    null, RejectionReason.ROUTE_INELIGIBLE);
            if (!withinBudgets(ir)) return new Contraction(null,
                    RejectionReason.HARD_BUDGET_EXCEEDED);
            return new Contraction(new MutableUnit(nodes, lowered), null);
        } catch (ArithmeticException overflow) {
            return new Contraction(null, RejectionReason.HARD_BUDGET_EXCEEDED);
        } catch (IllegalArgumentException unsupported) {
            return new Contraction(null, RejectionReason.UNSUPPORTED_LOWERING);
        }
    }

    /**
     * Applies every pointwise-contraction resource ceiling to an already validated canonical IR.
     * Arithmetic overflow is reported to the caller so contraction fails closed.
     *
     * @param ir non-null proposed pointwise contraction IR
     * @return {@code true} exactly when every materialized-boundary, indexing, liveness, and
     *     generated-code structural ceiling is satisfied
     * @throws NullPointerException if {@code ir} is {@code null}
     * @throws ArithmeticException if exact indexing arithmetic overflows
     */
    static boolean withinBudgets(CpuKernelIr ir) {
        StructuralFacts facts = structuralFacts(ir);
        return facts.materializedBoundaries() <= MAX_BOUNDARIES
                && facts.indexingComplexityUnits() <= MAX_INDEXING_UNITS
                && facts.simultaneouslyLiveValues() <= MAX_LIVE_VALUES
                && facts.generatedCodeSizeUnits() <= MAX_CODE_UNITS;
    }

    /**
     * Computes the shared checked 0008B structural quantities used by hard legality and 0008D
     * profitability. Keeping this calculation here prevents the heuristic from drifting from the
     * established contraction proof.
     *
     * @param ir non-null validated pointwise IR
     * @return exact non-negative boundary, indexing, liveness, and code-size facts
     * @throws ArithmeticException if checked structural arithmetic exceeds {@code int}
     */
    public static StructuralFacts structuralFacts(CpuKernelIr ir) {
        Objects.requireNonNull(ir, "ir");
        int boundaries = Math.toIntExact(ir.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).count());
        int indexing = 0;
        for (CpuKernelIr.Value value : ir.values()) {
            if (value.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
            indexing = Math.addExact(indexing, switch (value.accessPlan().regime()) {
                case DENSE_LINEAR, SCALAR_ALL_ZERO -> 1;
                case LAST_AXIS_BIAS -> 2;
                case BLOCK_OUTER -> 3;
                case GENERAL_ODOMETER -> 4;
            });
        }
        int liveMaximum = liveMaximum(ir);
        long virtuals = ir.values().stream().filter(value -> value.kind() == CpuKernelIr.Value.Kind.VIRTUAL).count();
        int code = Math.toIntExact(Math.addExact(Math.addExact(Math.addExact(8L,
                Math.multiplyExact(4L, ir.instructions().size())),
                Math.multiplyExact(3L, ir.stores().size())),
                Math.addExact(indexing, virtuals)));
        return new StructuralFacts(boundaries, indexing, liveMaximum, code);
    }

    /**
     * Computes the exact maximum live-value count at stable instruction and store events.
     * External inputs are born only at first use, while every instruction result is born at its
     * definition even when it has no later consumer. Uses by the same instruction are deduplicated;
     * an inclusive final-use event lets inputs die immediately after their last instruction and
     * produced values die immediately after their last instruction or store.
     *
     * @param ir non-null validated canonical pointwise IR
     * @return non-negative maximum number of simultaneously live IR values
     * @throws NullPointerException if {@code ir} is {@code null}
     */
    static int liveMaximum(CpuKernelIr ir) {
        Objects.requireNonNull(ir, "ir");
        var first = new int[ir.values().size()];
        var last = new int[ir.values().size()];
        java.util.Arrays.fill(first, Integer.MAX_VALUE);
        java.util.Arrays.fill(last, -1);
        for (int i = 0; i < ir.instructions().size(); i++) {
            CpuKernelIr.Instruction instruction = ir.instructions().get(i);
            for (int input : new LinkedHashSet<>(instruction.inputs())) {
                if (ir.values().get(input).kind() == CpuKernelIr.Value.Kind.INPUT) {
                    first[input] = Math.min(first[input], i);
                }
                last[input] = i;
            }
            first[instruction.output()] = Math.min(first[instruction.output()], i);
            last[instruction.output()] = Math.max(last[instruction.output()], i);
        }
        for (CpuKernelIr.Store store : ir.stores()) {
            int position = ir.instructions().size();
            first[store.value()] = Math.min(first[store.value()], position);
            last[store.value()] = position;
        }
        int maximum = 0;
        for (int position = 0; position <= ir.instructions().size(); position++) {
            int live = 0;
            for (int ordinal = 0; ordinal < ir.values().size(); ordinal++) {
                if (first[ordinal] <= position && last[ordinal] >= position) live++;
            }
            maximum = Math.max(maximum, live);
        }
        return maximum;
    }

    private static void replace(List<MutableUnit> units, int left, int right, MutableUnit fused) {
        int high = Math.max(left, right);
        int low = Math.min(left, right);
        units.remove(high);
        units.remove(low);
        units.add(low, fused);
    }

    private static List<MutableUnit> canonicalSeeds(
            PrepareContext<CpuPartitionAnalysisInputs> context, CpuPartitionLowering lowering,
            Map<CompiledNode, Integer> ordinals, List<Unit> baseline,
            Set<Integer> lockedBaselineUnits) {
        if (lockedBaselineUnits.isEmpty()) return seeds(context, lowering, ordinals);
        var lockedByFirst = new HashMap<Integer, Unit>();
        for (int index : lockedBaselineUnits) {
            Unit unit = baseline.get(index);
            lockedByFirst.put(unit.memberNodeOrdinals().getFirst(), unit);
        }
        List<MutableUnit> ordinary = seeds(context, lowering, ordinals);
        var ordinaryByFirst = new HashMap<Integer, MutableUnit>();
        ordinary.forEach(unit -> ordinaryByFirst.put(ordinals.get(unit.nodes.getFirst()), unit));
        var result = new ArrayList<MutableUnit>();
        int ordinal = 0;
        while (ordinal < context.partitionDag().nodes().size()) {
            Unit locked = lockedByFirst.get(ordinal);
            if (locked != null) {
                result.add(new MutableUnit(locked.nodes(), locked.lowering()));
                ordinal = Math.addExact(ordinal, locked.nodes().size());
                continue;
            }
            MutableUnit seed = ordinaryByFirst.get(ordinal);
            if (seed == null) throw new IllegalArgumentException(
                    "recognition barrier cuts through an established seed");
            boolean overlapsLocked = seed.nodes.stream().map(ordinals::get)
                    .anyMatch(value -> lockedBaselineUnits.stream()
                            .map(baseline::get).flatMap(unit -> unit.memberNodeOrdinals().stream())
                            .anyMatch(value::equals));
            if (overlapsLocked) throw new IllegalArgumentException(
                    "recognition baseline association disagrees with canonical seeds");
            result.add(seed);
            ordinal = Math.addExact(ordinal, seed.nodes.size());
        }
        return result;
    }

    private static List<Pair> enumerationPairs(PartitionDag dag, List<MutableUnit> ordered) {
        var result = new ArrayList<Pair>();
        for (MutableUnit producer : ordered) for (MutableUnit consumer : ordered) {
            if (producer == consumer) continue;
            if (hasEdge(dag, producer, consumer)) {
                result.add(new Pair(producer, consumer, PairKind.VERTICAL));
            }
        }
        for (int left = 0; left < ordered.size(); left++) {
            for (int right = left + 1; right < ordered.size(); right++) {
                MutableUnit a = ordered.get(left);
                MutableUnit b = ordered.get(right);
                boolean edge = result.stream().anyMatch(pair -> pair.left() == a && pair.right() == b
                        || pair.left() == b && pair.right() == a);
                if (!edge) result.add(new Pair(a, b, PairKind.HORIZONTAL));
            }
        }
        return result;
    }

    private static Contraction attempt(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, List<MutableUnit> all, MutableUnit left,
            MutableUnit right, PairKind kind, Map<CompiledNode, Integer> ordinals,
            List<Unit> baseline, Set<Integer> lockedBaselineUnits) {
        if (locked(left, ordinals, baseline, lockedBaselineUnits)
                || locked(right, ordinals, baseline, lockedBaselineUnits)) {
            return new Contraction(null, RejectionReason.SEMANTIC_BARRIER);
        }
        if (!left.pointwise() || !right.pointwise()) {
            return new Contraction(null, barrier(left, right));
        }
        if (kind == PairKind.VERTICAL) {
            var graphOutputs = new HashSet<ValueId>();
            context.memoryRequirements().stream().filter(LogicalMemoryRequirement::graphOutput)
                    .forEach(value -> graphOutputs.add(value.valueId()));
            var connecting = outputs(left).stream().filter(output -> context.partitionDag()
                    .consumers(output).stream().anyMatch(occurrence ->
                            containsNode(right, occurrence.node()))).toList();
            if (connecting.isEmpty()) return new Contraction(null,
                    RejectionReason.ROUTE_INELIGIBLE);
            if (connecting.stream().anyMatch(graphOutputs::contains)) return new Contraction(null,
                    RejectionReason.PUBLICATION_BARRIER);
            for (ValueId output : connecting) {
                if (context.partitionDag().consumers(output).size() != 1) {
                    return new Contraction(null, RejectionReason.FAN_OUT_BARRIER);
                }
            }
        } else {
            if (!horizontal(context.partitionDag(), all, left, right)) {
                return new Contraction(null,
                    RejectionReason.ROUTE_INELIGIBLE);
            }
        }
        return contractResult(context, lowering, left, right, ordinals);
    }

    private static boolean locked(MutableUnit unit, Map<CompiledNode, Integer> ordinals,
            List<Unit> baseline, Set<Integer> lockedBaselineUnits) {
        List<Integer> members = unit.nodes.stream().map(ordinals::get).toList();
        return lockedBaselineUnits.stream().map(baseline::get)
                .map(Unit::memberNodeOrdinals).anyMatch(members::equals);
    }

    private static RejectionReason barrier(MutableUnit left, MutableUnit right) {
        var forms = java.util.stream.Stream.of(left, right)
                .map(unit -> unit.lowering.portableKernelIr()).toList();
        if (forms.stream().anyMatch(CpuRandomIr.class::isInstance)) {
            return RejectionReason.STATE_OR_RANDOM_BARRIER;
        }
        if (forms.stream().anyMatch(CpuPartitionDagDecomposer::numericalBarrier)) {
            return RejectionReason.NUMERICAL_ORDER_BARRIER;
        }
        return RejectionReason.SEMANTIC_BARRIER;
    }

    private static boolean numericalBarrier(CpuPortableKernelIr form) {
        return form instanceof CpuAggregateIr || form instanceof CpuAdvancedReductionIr
                || form instanceof CpuArgExtremaIr || form instanceof CpuMaskedReductionIr
                || form instanceof CpuFoldIr || form instanceof CpuScanIr
                || form instanceof CpuOrderingIr || form instanceof CpuSoftmaxIr
                || form instanceof CpuTrailingNormalizationIr
                || form instanceof CpuBatchNormInferenceIr
                || form instanceof CpuBatchNormTrainingIr || form instanceof CpuConv2dIr
                || form instanceof CpuConv3dIr;
    }

    private static int memberIndex(List<MutableUnit> topology, MutableUnit source) {
        List<CompiledNode> members = source.nodes;
        for (int index = 0; index < topology.size(); index++) {
            if (topology.get(index).nodes.equals(members)) return index;
        }
        throw new IllegalArgumentException("enumeration source pair is not retained");
    }

    private static List<MutableUnit> copyTopology(List<MutableUnit> source) {
        return source.stream().map(unit -> new MutableUnit(unit.nodes, unit.lowering)).toList()
                .stream().collect(java.util.stream.Collectors.toCollection(ArrayList::new));
    }

    private static TopologyIdentity identity(List<Unit> units) {
        return new TopologyIdentity(units.stream().map(Unit::memberNodeOrdinals).toList());
    }

    private static boolean sameMembership(List<Unit> left, List<Unit> right) {
        return left.stream().map(Unit::memberNodeOrdinals).toList()
                .equals(right.stream().map(Unit::memberNodeOrdinals).toList());
    }

    private static boolean containsMembership(List<List<Unit>> candidates, List<Unit> expected) {
        for (List<Unit> candidate : candidates) if (sameMembership(candidate, expected)) return true;
        return false;
    }

    private record Pair(MutableUnit left, MutableUnit right, PairKind kind) { }
    private record Contraction(MutableUnit unit, RejectionReason rejection) { }

    private static List<Unit> finish(List<MutableUnit> units,
            Map<CompiledNode, Integer> ordinals, PartitionDag dag) {
        List<MutableUnit> ordered = topological(units, ordinals, dag);
        var indices = new HashMap<MutableUnit, Integer>();
        for (int i = 0; i < ordered.size(); i++) indices.put(ordered.get(i), i);
        var result = new ArrayList<Unit>();
        for (MutableUnit unit : ordered) {
            List<Integer> dependencies = producers(dag, units, unit).stream().map(indices::get)
                    .sorted().toList();
            if (dependencies.stream().anyMatch(value -> value >= indices.get(unit)))
                throw new IllegalArgumentException("CPU unit topology is cyclic");
            result.add(new Unit(unit.nodes, unit.lowering, dependencies,
                    unit.nodes.stream().map(ordinals::get).toList()));
        }
        return List.copyOf(result);
    }

    private static List<MutableUnit> topological(List<MutableUnit> units,
            Map<CompiledNode, Integer> ordinals, PartitionDag dag) {
        var remaining = new ArrayList<>(units);
        var result = new ArrayList<MutableUnit>();
        while (!remaining.isEmpty()) {
            MutableUnit ready = remaining.stream().filter(unit ->
                            result.containsAll(producers(dag, units, unit)))
                    .min(Comparator.comparingInt(unit -> unit.nodes.stream().mapToInt(ordinals::get).min().orElseThrow()))
                    .orElseThrow(() -> new IllegalArgumentException("CPU unit topology is cyclic"));
            result.add(ready);
            remaining.remove(ready);
        }
        return result;
    }

    private static LinkedHashSet<MutableUnit> producers(PartitionDag dag,
            List<MutableUnit> units, MutableUnit target) {
        var result = new LinkedHashSet<MutableUnit>();
        for (MutableUnit unit : units) {
            if (unit != target && hasEdge(dag, unit, target)) result.add(unit);
        }
        return result;
    }

    private static boolean hasEdge(PartitionDag dag, MutableUnit producer,
            MutableUnit consumer) {
        return dag.edges().stream().anyMatch(edge ->
                containsNode(producer, edge.producer().node())
                        && containsNode(consumer, edge.consumer().node()));
    }

    private static Map<CompiledNode, Integer> ordinals(PartitionDag dag) {
        var result = new IdentityHashMap<CompiledNode, Integer>();
        for (int index = 0; index < dag.nodes().size(); index++) {
            result.put(dag.nodes().get(index), index);
        }
        return result;
    }

    private static boolean containsNode(MutableUnit unit, CompiledNode node) {
        return unit.nodes.stream().anyMatch(member -> member == node);
    }

    private static LinkedHashSet<ValueId> outputs(MutableUnit unit) {
        return unit.nodes.stream().flatMap(node -> node.outputs().stream())
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> project(
            PrepareContext<CpuPartitionAnalysisInputs> source, List<CompiledNode> nodes,
            CpuPartitionAnalysisInputs inputs) {
        Set<CompiledNode> nodeSet = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        nodeSet.addAll(nodes);
        var partition = new PlannedPartition(source.partition().owner(),
                nodes.stream().map(CompiledNode::id).toList());
        var produced = source.values().stream().map(GraphValue::id)
                .filter(value -> source.partitionDag().producer(value)
                        .map(occurrence -> nodeSet.contains(occurrence.node())).orElse(false))
                .collect(java.util.stream.Collectors.toSet());
        var consumed = source.values().stream().map(GraphValue::id)
                .filter(value -> source.partitionDag().consumers(value).stream()
                        .anyMatch(occurrence -> nodeSet.contains(occurrence.node())))
                .collect(java.util.stream.Collectors.toSet());
        var original = new HashMap<ValueId, LogicalMemoryRequirement>();
        source.memoryRequirements().forEach(value -> original.put(value.valueId(), value));
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (GraphValue value : source.values()) {
            LogicalMemoryRequirement old = original.get(value.id());
            boolean outsideConsumer = source.partitionDag().consumers(value.id()).stream()
                    .anyMatch(occurrence -> !nodeSet.contains(occurrence.node()));
            boolean publication = old != null && old.graphOutput();
            requirements.add(new LogicalMemoryRequirement(value.id(), value.descriptor(),
                    produced.contains(value.id()) ? Optional.of(partition) : Optional.empty(),
                    consumed.contains(value.id()) ? List.of(partition) : List.of(),
                    publication || produced.contains(value.id()) && outsideConsumer));
        }
        var constants = new LinkedHashMap<ValueId, io.github.pho001.synaptik.model.datatype.ScalarValue>();
        source.constants().forEach((id, value) -> {
            if (consumed.contains(id) && !produced.contains(id)) constants.put(id, value);
        });
        return new PrepareContext<>(partition, nodes, source.values(), requirements,
                constants, inputs);
    }

    private static final class MutableUnit {
        private final List<CompiledNode> nodes;
        private final CpuPartitionLowering.LoweredPartition lowering;
        private MutableUnit(List<CompiledNode> nodes, CpuPartitionLowering.LoweredPartition lowering) {
            this.nodes = List.copyOf(nodes);
            this.lowering = lowering;
        }
        private boolean pointwise() { return lowering.portableKernelIr() instanceof CpuKernelIr; }
    }
}
