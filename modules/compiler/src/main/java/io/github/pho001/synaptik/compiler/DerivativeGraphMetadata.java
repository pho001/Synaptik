package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Retains compiler-owned derivative order and optional Scan activity for one exact compiled graph.
 *
 * <p>Order zero identifies original forward producers. Orders one and two identify producers
 * first owned by the corresponding reverse-mode stage. This immutable sidecar augments the
 * Model-owned forward/backward phase without retaining Tensor, provenance, request, formula,
 * runtime, backend, or execution state.</p>
 * <p>Activity keys name containing Scan occurrences, never body-local IDs. Each plan retains
 * that occurrence's exact local graph reference. Rebuilds must preserve or explicitly remap
 * these facts; the two-argument constructor intentionally creates no activity annotations.</p>
 */
public final class DerivativeGraphMetadata {
    private final CompiledGraphModel graph;
    private final Map<NodeId, Integer> derivativeOrderByNode;
    private final Map<NodeId, ScanActivityPlan> scanActivityByNode;

    /**
     * Creates and validates derivative order for every node in graph encounter order.
     *
     * @param graph non-null exact graph reference retained by this artifact
     * @param derivativeOrderByNode non-null encounter-ordered map with exactly one entry per
     *     graph node and values zero, one, or two
     * @throws NullPointerException if an argument, key, or value is {@code null}
     * @throws IllegalArgumentException if a key is missing, unknown, duplicated by encounter,
     *     out of graph order, has an unsupported order, or disagrees with node phase
     */
    public DerivativeGraphMetadata(
            CompiledGraphModel graph,
            Map<NodeId, Integer> derivativeOrderByNode) {
        this(graph, derivativeOrderByNode, Map.of());
    }

    /**
     * Creates exact-graph derivative metadata with independently scoped Scan activity.
     *
     * @param graph non-null exact containing graph
     * @param derivativeOrderByNode non-null complete graph-ordered derivative orders
     * @param scanActivityByNode non-null containing-Scan map, snapshotted in graph order
     * @throws NullPointerException if a component, key or value is null
     * @throws IllegalArgumentException if derivative orders, containing membership, exact body
     *     association, declared positions, next-presence bindings or ordered temporal ADDs are invalid
     */
    public DerivativeGraphMetadata(
            CompiledGraphModel graph,
            Map<NodeId, Integer> derivativeOrderByNode,
            Map<NodeId, ScanActivityPlan> scanActivityByNode) {
        this.graph = Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(derivativeOrderByNode, "derivativeOrderByNode");
        if (derivativeOrderByNode.size() != graph.nodes().size()) {
            throw new IllegalArgumentException(
                    "derivativeOrderByNode must contain exactly one entry per graph node");
        }
        List<Map.Entry<NodeId, Integer>> entries =
                new ArrayList<>(derivativeOrderByNode.entrySet());
        Set<NodeId> graphIds = new HashSet<>();
        graph.nodes().forEach(node -> graphIds.add(node.id()));
        for (int index = 0; index < entries.size(); index++) {
            Map.Entry<NodeId, Integer> entry = entries.get(index);
            NodeId nodeId = Objects.requireNonNull(
                    entry.getKey(), "derivativeOrderByNode key[" + index + "]");
            Integer order = Objects.requireNonNull(
                    entry.getValue(), "derivativeOrderByNode[" + nodeId + "]");
            if (!graphIds.contains(nodeId)) {
                throw new IllegalArgumentException(
                        "derivativeOrderByNode contains unknown " + nodeId);
            }
            NodeId expected = graph.nodes().get(index).id();
            if (!nodeId.equals(expected)) {
                throw new IllegalArgumentException(
                        "derivativeOrderByNode iteration order must match graph.nodes()");
            }
            if (order < 0 || order > 2) {
                throw new IllegalArgumentException(
                        "derivative order must be 0, 1, or 2");
            }
            GraphPhase expectedPhase = order == 0 ? GraphPhase.FORWARD : GraphPhase.BACKWARD;
            if (graph.nodePhases().get(nodeId) != expectedPhase) {
                throw new IllegalArgumentException(
                        "derivative order for " + nodeId + " disagrees with graph phase");
            }
        }
        this.derivativeOrderByNode =
                Collections.unmodifiableMap(new LinkedHashMap<>(derivativeOrderByNode));
        Objects.requireNonNull(scanActivityByNode, "scanActivityByNode");
        scanActivityByNode.forEach((id, plan) -> {
            Objects.requireNonNull(id, "scanActivityByNode key");
            Objects.requireNonNull(plan, "scanActivityByNode[" + id + "]");
        });
        LinkedHashMap<NodeId, ScanActivityPlan> activity = new LinkedHashMap<>();
        for (var node : graph.nodes()) {
            ScanActivityPlan plan = scanActivityByNode.get(node.id());
            if (plan == null) {
                continue;
            }
            if (node.scanBody().isEmpty()
                    || node.scanBody().orElseThrow().graph() != plan.graph()) {
                throw new IllegalArgumentException("activity must name its containing Scan's exact body");
            }
            ScanAttrs attrs = (ScanAttrs) node.operation().attrs();
            for (ScanActivityPlan.Carry carry : plan.carries()) {
                if (carry.numericCarryPosition() >= attrs.carryCount()) {
                    throw new IllegalArgumentException("activity carry position is not a numeric carry");
                }
            }
            for (int formal : plan.seedFormals()) {
                if (formal < attrs.carryCount()
                        || formal >= attrs.carryCount() + attrs.sequenceCount()) {
                    throw new IllegalArgumentException("activity seed must name a sequence formal");
                }
            }
            for (ScanActivityPlan.Accumulator accumulator : plan.accumulators()) {
                int position = accumulator.numericSharedCarryPosition();
                if (position >= attrs.carryCount()
                        || (attrs.lengthAware() && position < attrs.rowCarryCount())) {
                    throw new IllegalArgumentException("activity accumulator must name a shared numeric carry");
                }
            }
            validateActivityBindings(node.scanBody().orElseThrow(), plan);
            activity.put(node.id(), plan);
        }
        if (activity.size() != scanActivityByNode.size()) {
            throw new IllegalArgumentException("activity map contains an unknown containing node");
        }
        this.scanActivityByNode = Collections.unmodifiableMap(activity);
    }

    private static void validateActivityBindings(CompiledScanBody body, ScanActivityPlan plan) {
        Map<ValueId, CompiledNode> producers = new HashMap<>();
        for (CompiledNode node : body.graph().nodes()) {
            for (ValueId output : node.outputs()) {
                producers.put(output, node);
            }
        }
        Map<ValueId, List<Integer>> roots = new HashMap<>();
        for (int root = 0; root < plan.carries().size(); root++) {
            roots.put(body.graph().inputs().get(plan.carries().get(root).numericCarryPosition()), List.of(root));
        }
        for (int root = 0; root < plan.seedFormals().size(); root++) {
            roots.put(body.graph().inputs().get(plan.seedFormals().get(root)),
                    List.of(plan.carries().size() + root));
        }
        for (ScanActivityPlan.Carry carry : plan.carries()) {
            ValueId next = body.nextCarry().get(carry.numericCarryPosition());
            CompiledNode producer = producers.get(next);
            List<Integer> expected = producer == null ? roots.getOrDefault(next, List.of())
                    : plan.scopes().get(plan.nodeScopes().get(producer.id()));
            if (!plan.scopes().get(carry.nextScope()).equals(expected)) {
                throw new IllegalArgumentException("activity next scope disagrees with its exact next-carry binding");
            }
        }
        for (ScanActivityPlan.Accumulator accumulator : plan.accumulators()) {
            int position = accumulator.numericSharedCarryPosition();
            CompiledNode add = producers.get(body.nextCarry().get(position));
            if (add == null || add.operation().kind() != BinaryArithmeticKind.ADD
                    || add.inputs().size() != 2 || add.outputs().size() != 1
                    || !add.inputs().getFirst().equals(body.graph().inputs().get(position))
                    || plan.merges().containsKey(add.id())
                    || !plan.scopes().get(plan.nodeScopes().get(add.id()))
                            .equals(plan.scopes().get(accumulator.contributionScope()))) {
                throw new IllegalArgumentException(
                        "activity accumulator must retain its separate ordered old-plus-contribution ADD");
            }
        }
    }

    /**
     * Returns the exact graph reference owned by this sidecar.
     *
     * @return exact non-null graph supplied at construction
     */
    public CompiledGraphModel graph() {
        return graph;
    }

    /**
     * Returns immutable derivative orders in exact graph-node encounter order.
     *
     * @return non-null immutable map with one entry per graph node
     */
    public Map<NodeId, Integer> derivativeOrderByNode() {
        return derivativeOrderByNode;
    }

    /**
     * Returns immutable containing-Scan activity in exact outer graph-node order.
     *
     * @return non-null map whose plans name each keyed occurrence's exact local body graph
     */
    public Map<NodeId, ScanActivityPlan> scanActivityByNode() {
        return scanActivityByNode;
    }

    /**
     * Creates order-zero metadata for an exact forward-only graph.
     *
     * @param graph non-null graph whose nodes are all forward producers
     * @return non-null metadata owning the exact graph
     */
    static DerivativeGraphMetadata forwardOnly(CompiledGraphModel graph) {
        LinkedHashMap<NodeId, Integer> orders = new LinkedHashMap<>();
        graph.nodes().forEach(node -> orders.put(node.id(), 0));
        return new DerivativeGraphMetadata(graph, orders);
    }

    /**
     * Remaps source-node orders and containing-Scan activity onto one rebuilt graph.
     *
     * @param source non-null source metadata
     * @param graph non-null rebuilt graph
     * @param sourceNodeIds non-null source-node ID for each rebuilt node
     * @return non-null metadata owning {@code graph}
     * @throws NullPointerException if an argument or source ID is {@code null}
     * @throws IllegalArgumentException if membership or cardinality is inconsistent
     */
    static DerivativeGraphMetadata remap(
            DerivativeGraphMetadata source,
            CompiledGraphModel graph,
            List<NodeId> sourceNodeIds) {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(sourceNodeIds, "sourceNodeIds");
        if (sourceNodeIds.size() != graph.nodes().size()) {
            throw new IllegalArgumentException(
                    "sourceNodeIds size must equal rebuilt graph node count");
        }
        LinkedHashMap<NodeId, Integer> orders = new LinkedHashMap<>();
        LinkedHashMap<NodeId, ScanActivityPlan> activity = new LinkedHashMap<>();
        for (int index = 0; index < graph.nodes().size(); index++) {
            NodeId sourceId = Objects.requireNonNull(
                    sourceNodeIds.get(index), "sourceNodeIds[" + index + "]");
            Integer order = source.derivativeOrderByNode.get(sourceId);
            if (order == null) {
                throw new IllegalArgumentException("unknown source node " + sourceId);
            }
            orders.put(graph.nodes().get(index).id(), order);
            ScanActivityPlan plan = source.scanActivityByNode.get(sourceId);
            if (plan != null) {
                activity.put(graph.nodes().get(index).id(), plan);
            }
        }
        return new DerivativeGraphMetadata(graph, orders, activity);
    }
}
