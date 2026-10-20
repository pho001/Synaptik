package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Invocation-local ordering facts for contracted CPU computation units, not a second node DAG.
 * The shared partition DAG remains the source of exact port occurrences; only inter-unit pairs
 * are deduplicated here. A fresh instance is required after every membership change. The narrow
 * edge-iterable boundary permits structural tests without instrumenting shared Prepare or
 * retaining mutable contraction state.
 */
final class CpuUnitTopology {
    private final List<Integer> order;
    private final List<List<Integer>> dependencies;
    private final List<List<Integer>> memberNodeOrdinals;

    /**
     * Snapshots membership by node identity and visits the supplied edges exactly once. Kahn
     * ordering always selects the ready unit with the minimum original node ordinal, including
     * newly ready units ahead of higher-ordinal roots. Final dependencies reuse these same facts.
     *
     * @param members non-null current disjoint, non-empty unit memberships; inspected, not mutated
     * @param ordinals non-null original partition ordinals for every exact member node
     * @param edges non-null shared DAG edge occurrences whose endpoints belong to these units;
     *     consumed once, without changing or deduplicating the original occurrences
     * @throws NullPointerException if a required argument, member, or ordinal is null or missing
     * @throws IllegalArgumentException if membership is empty/overlapping or contraction is cyclic
     */
    CpuUnitTopology(List<List<CompiledNode>> members, Map<CompiledNode, Integer> ordinals,
            Iterable<PartitionDag.Edge> edges) {
        int count = members.size();
        var membership = new IdentityHashMap<CompiledNode, Integer>();
        var minima = new int[count];
        var memberOrdinals = new ArrayList<List<Integer>>(count);
        var producers = new ArrayList<HashSet<Integer>>(count);
        var consumers = new ArrayList<List<Integer>>(count);
        var indegrees = new int[count];
        for (int unit = 0; unit < count; unit++) {
            int minimum = Integer.MAX_VALUE;
            var positions = new ArrayList<Integer>(members.get(unit).size());
            for (CompiledNode node : members.get(unit)) {
                if (membership.put(node, unit) != null) {
                    throw new IllegalArgumentException("CPU unit memberships overlap");
                }
                int ordinal = ordinals.get(node);
                positions.add(ordinal);
                minimum = Math.min(minimum, ordinal);
            }
            if (positions.isEmpty()) throw new IllegalArgumentException("CPU unit is empty");
            minima[unit] = minimum;
            memberOrdinals.add(List.copyOf(positions));
            producers.add(new HashSet<>());
            consumers.add(new ArrayList<>());
        }
        for (PartitionDag.Edge edge : edges) {
            int producer = membership.get(edge.producer().node());
            int consumer = membership.get(edge.consumer().node());
            if (producer != consumer && producers.get(consumer).add(producer)) {
                consumers.get(producer).add(consumer);
                indegrees[consumer]++;
            }
        }
        var ready = new PriorityQueue<Integer>(Comparator.comparingInt(unit -> minima[unit]));
        for (int unit = 0; unit < count; unit++) if (indegrees[unit] == 0) ready.add(unit);
        var ordered = new ArrayList<Integer>(count);
        var finalIndices = new int[count];
        while (!ready.isEmpty()) {
            int unit = ready.remove();
            finalIndices[unit] = ordered.size();
            ordered.add(unit);
            for (int consumer : consumers.get(unit)) {
                if (--indegrees[consumer] == 0) ready.add(consumer);
            }
        }
        if (ordered.size() != count) {
            throw new IllegalArgumentException("CPU unit topology is cyclic");
        }
        order = List.copyOf(ordered);
        var finalDependencies = new ArrayList<List<Integer>>(count);
        var finalMembers = new ArrayList<List<Integer>>(count);
        for (int unit : order) {
            finalDependencies.add(producers.get(unit).stream()
                    .map(producer -> finalIndices[producer]).sorted().toList());
            finalMembers.add(memberOrdinals.get(unit));
        }
        dependencies = List.copyOf(finalDependencies);
        memberNodeOrdinals = List.copyOf(finalMembers);
    }

    /**
     * Returns the original unit positions in stable topological order.
     *
     * @return non-null immutable ordered positions, each occurring exactly once
     */
    List<Integer> order() { return order; }

    /**
     * Returns direct producer positions in the final order, without revisiting graph edges.
     *
     * @param index zero-based final unit position
     * @return non-null immutable ascending unique producer indices, all less than {@code index}
     * @throws IndexOutOfBoundsException if {@code index} is outside the final order
     */
    List<Integer> dependencies(int index) { return dependencies.get(index); }

    /**
     * Returns the original member ordinals captured during the identity-membership pass.
     *
     * @param index zero-based final unit position
     * @return non-null immutable ordinals in the supplied membership order
     * @throws IndexOutOfBoundsException if {@code index} is outside the final order
     */
    List<Integer> memberNodeOrdinals(int index) { return memberNodeOrdinals.get(index); }
}
