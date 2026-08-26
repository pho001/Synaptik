package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Deterministically decomposes one complete CPU-owned partition DAG into bounded computation
 * units. Established affine and numerical-family lowerings remain indivisible seeds; only the
 * ordinary pointwise IR is contracted, and a rejected contraction leaves the split topology
 * unchanged.
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

    /** Creates a stateless cold-analysis decomposition boundary. */
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

    /**
     * Builds the maximally split supported baseline and applies bounded deterministic pointwise
     * contractions.
     *
     * @param context complete non-null CPU analysis context
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
        var ordinals = new HashMap<CompiledNode, Integer>();
        for (int i = 0; i < context.nodes().size(); i++) ordinals.put(context.nodes().get(i), i);
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
            List<MutableUnit> ordered = topological(working, ordinals);
            outer: for (int left = 0; left < ordered.size(); left++) {
                for (int right = left + 1; right < ordered.size(); right++) {
                    if (attempts++ >= MAX_ATTEMPTS) break outer;
                    MutableUnit a = ordered.get(left);
                    MutableUnit b = ordered.get(right);
                    if (horizontal(working, a, b)) {
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
        return finish(working, ordinals);
    }

    /**
     * Projects one selected unit without inventing graph values or changing the outer owner.
     * The returned context is immutable analysis state and owns no physical resources.
     *
     * @param source non-null complete partition context
     * @param nodes non-null, non-empty selected node list in stable order
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
                || context.nodes().isEmpty() || context.nodes().size() > MAX_NODES) {
            throw new IllegalArgumentException("CPU partition requires one through eight nodes");
        }
        var nodeIds = new HashSet<>();
        var outputs = new HashSet<ValueId>();
        var values = new HashSet<ValueId>();
        context.values().forEach(value -> values.add(value.id()));
        var memory = new HashSet<ValueId>();
        context.memoryRequirements().forEach(value -> memory.add(value.valueId()));
        for (CompiledNode node : context.nodes()) {
            if (!nodeIds.add(node.id())) throw new IllegalArgumentException("duplicate CPU node identity");
            for (ValueId input : node.inputs()) {
                if (!values.contains(input) || !memory.contains(input))
                    throw new IllegalArgumentException("CPU input fact is not projected");
                if (outputs.contains(input)) continue;
                boolean laterProducer = context.nodes().stream().flatMap(value -> value.outputs().stream())
                        .anyMatch(input::equals);
                if (laterProducer) throw new IllegalArgumentException("CPU consumer precedes producer");
            }
            for (ValueId output : node.outputs()) {
                if (!outputs.add(output) || !values.contains(output) || !memory.contains(output))
                    throw new IllegalArgumentException("duplicate or missing CPU output fact");
            }
        }
    }

    private static List<MutableUnit> seeds(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, Map<CompiledNode, Integer> ordinals) {
        var result = new ArrayList<MutableUnit>();
        int index = 0;
        while (index < context.nodes().size()) {
            MutableUnit selected = null;
            IllegalArgumentException lastFailure = null;
            for (int end = context.nodes().size(); end > index; end--) {
                List<CompiledNode> nodes = context.nodes().subList(index, end);
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
            long uses = context.nodes().stream().flatMap(value -> value.inputs().stream())
                    .filter(output::equals).count();
            if (uses == 1 && !graphOutputs.contains(output)
                    && consumer.nodes.stream().flatMap(value -> value.inputs().stream())
                        .anyMatch(output::equals)) return true;
        }
        return false;
    }

    private static boolean horizontal(List<MutableUnit> all, MutableUnit left, MutableUnit right) {
        if (!left.pointwise() || !right.pointwise()) return false;
        var leftDependencies = producers(all, left);
        var rightDependencies = producers(all, right);
        if (!leftDependencies.equals(rightDependencies)) return false;
        var leftOutputs = outputs(left);
        var rightOutputs = outputs(right);
        return left.nodes.stream().flatMap(node -> node.inputs().stream()).noneMatch(rightOutputs::contains)
                && right.nodes.stream().flatMap(node -> node.inputs().stream()).noneMatch(leftOutputs::contains)
                && java.util.Arrays.equals(left.lowering.extents(), right.lowering.extents());
    }

    private static MutableUnit contract(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionLowering lowering, MutableUnit left, MutableUnit right,
            Map<CompiledNode, Integer> ordinals) {
        var nodes = new ArrayList<CompiledNode>(left.nodes);
        nodes.addAll(right.nodes);
        nodes.sort(Comparator.comparingInt(ordinals::get));
        if (nodes.size() > MAX_NODES) return null;
        try {
            var lowered = lowering.lower(project(context, nodes, context.backendInputs()));
            if (!(lowered.portableKernelIr() instanceof CpuKernelIr ir)) return null;
            if (!withinBudgets(ir)) return null;
            return new MutableUnit(nodes, lowered);
        } catch (IllegalArgumentException | ArithmeticException rejected) {
            return null;
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
        Objects.requireNonNull(ir, "ir");
        long boundaries = ir.values().stream().filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).count();
        if (boundaries > MAX_BOUNDARIES) return false;
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
        if (indexing > MAX_INDEXING_UNITS) return false;
        int liveMaximum = liveMaximum(ir);
        if (liveMaximum > MAX_LIVE_VALUES) return false;
        long virtuals = ir.values().stream().filter(value -> value.kind() == CpuKernelIr.Value.Kind.VIRTUAL).count();
        long code = 8L + 4L * ir.instructions().size() + 3L * ir.stores().size()
                + indexing + virtuals;
        return code <= MAX_CODE_UNITS;
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

    private static List<Unit> finish(List<MutableUnit> units,
            Map<CompiledNode, Integer> ordinals) {
        List<MutableUnit> ordered = topological(units, ordinals);
        var indices = new HashMap<MutableUnit, Integer>();
        for (int i = 0; i < ordered.size(); i++) indices.put(ordered.get(i), i);
        var result = new ArrayList<Unit>();
        for (MutableUnit unit : ordered) {
            List<Integer> dependencies = producers(units, unit).stream().map(indices::get)
                    .sorted().toList();
            if (dependencies.stream().anyMatch(value -> value >= indices.get(unit)))
                throw new IllegalArgumentException("CPU unit topology is cyclic");
            result.add(new Unit(unit.nodes, unit.lowering, dependencies,
                    unit.nodes.stream().map(ordinals::get).toList()));
        }
        return List.copyOf(result);
    }

    private static List<MutableUnit> topological(List<MutableUnit> units,
            Map<CompiledNode, Integer> ordinals) {
        var remaining = new ArrayList<>(units);
        var result = new ArrayList<MutableUnit>();
        while (!remaining.isEmpty()) {
            MutableUnit ready = remaining.stream().filter(unit -> result.containsAll(producers(units, unit)))
                    .min(Comparator.comparingInt(unit -> unit.nodes.stream().mapToInt(ordinals::get).min().orElseThrow()))
                    .orElseThrow(() -> new IllegalArgumentException("CPU unit topology is cyclic"));
            result.add(ready);
            remaining.remove(ready);
        }
        return result;
    }

    private static LinkedHashSet<MutableUnit> producers(List<MutableUnit> units, MutableUnit target) {
        var inputs = target.nodes.stream().flatMap(node -> node.inputs().stream())
                .collect(java.util.stream.Collectors.toSet());
        var result = new LinkedHashSet<MutableUnit>();
        for (MutableUnit unit : units) if (unit != target && outputs(unit).stream().anyMatch(inputs::contains))
            result.add(unit);
        return result;
    }

    private static LinkedHashSet<ValueId> outputs(MutableUnit unit) {
        return unit.nodes.stream().flatMap(node -> node.outputs().stream())
                .collect(java.util.stream.Collectors.toCollection(LinkedHashSet::new));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> project(
            PrepareContext<CpuPartitionAnalysisInputs> source, List<CompiledNode> nodes,
            CpuPartitionAnalysisInputs inputs) {
        var nodeSet = new HashSet<>(nodes);
        var partition = new PlannedPartition(source.partition().owner(),
                nodes.stream().map(CompiledNode::id).toList());
        var produced = nodes.stream().flatMap(node -> node.outputs().stream())
                .collect(java.util.stream.Collectors.toSet());
        var consumed = nodes.stream().flatMap(node -> node.inputs().stream())
                .collect(java.util.stream.Collectors.toSet());
        var original = new HashMap<ValueId, LogicalMemoryRequirement>();
        source.memoryRequirements().forEach(value -> original.put(value.valueId(), value));
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (GraphValue value : source.values()) {
            LogicalMemoryRequirement old = original.get(value.id());
            boolean outsideConsumer = source.nodes().stream().filter(node -> !nodeSet.contains(node))
                    .flatMap(node -> node.inputs().stream()).anyMatch(value.id()::equals);
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
