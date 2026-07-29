package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.NodeId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Retains compiler-owned derivative order for every node of one exact compiled graph.
 *
 * <p>Order zero identifies original forward producers. Orders one and two identify producers
 * first owned by the corresponding reverse-mode stage. This immutable sidecar augments the
 * Model-owned forward/backward phase without retaining Tensor, provenance, request, formula,
 * runtime, backend, or execution state.</p>
 */
public final class DerivativeGraphMetadata {
    private final CompiledGraphModel graph;
    private final Map<NodeId, Integer> derivativeOrderByNode;

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
     * Remaps source-node orders onto one rebuilt graph in encounter order.
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
        for (int index = 0; index < graph.nodes().size(); index++) {
            NodeId sourceId = Objects.requireNonNull(
                    sourceNodeIds.get(index), "sourceNodeIds[" + index + "]");
            Integer order = source.derivativeOrderByNode.get(sourceId);
            if (order == null) {
                throw new IllegalArgumentException("unknown source node " + sourceId);
            }
            orders.put(graph.nodes().get(index).id(), order);
        }
        return new DerivativeGraphMetadata(graph, orders);
    }
}
