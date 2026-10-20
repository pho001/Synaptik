package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

class CpuUnitTopologyTest {
    @Test void chainsDiamondsAndIndependentRootsMatchOldScanExactly() {
        compare(graph(new int[][] {{}, {0}, {1}, {2}, {3}}),
                List.of(List.of(4), List.of(2, 3), List.of(0, 1)));
        compare(graph(new int[][] {{}, {0}, {0}, {1, 2}}),
                List.of(List.of(3), List.of(2), List.of(1), List.of(0)));
        compare(graph(new int[][] {{}, {}, {}, {}, {}}),
                List.of(List.of(4), List.of(1, 3), List.of(0, 2)));
    }

    @Test void newlyReadyLowerOrdinalPreemptsReadyRootsWithInterleavedMemberships() {
        var dag = graph(new int[][] {{}, {}, {}, {1}});
        var members = List.of(List.of(0, 3), List.of(1), List.of(2));
        var topology = compare(dag, members);
        // Unit [0,3] is blocked by [1], but must precede already-ready root [2] afterward.
        assertEquals(List.of(1, 0, 2), topology.order());
        assertEquals(List.of(0), topology.dependencies(1));
        assertEquals(List.of(0, 3), topology.memberNodeOrdinals(1));
    }

    @Test void contractionCreatedCyclesRejectExactlyLikeOldScan() {
        var dag = graph(new int[][] {{}, {0}, {1}, {2}});
        compare(dag, List.of(List.of(0, 2), List.of(1), List.of(3)));
        compare(dag, List.of(List.of(0, 3), List.of(1, 2)));
    }

    @Test void boundedRandomContractionsMatchOldOrderDependenciesAndCycleFailures() {
        var random = new Random(27);
        for (int trial = 0; trial < 256; trial++) {
            int[][] parents = new int[7][];
            parents[0] = new int[0];
            for (int node = 1; node < parents.length; node++) {
                parents[node] = new int[] {random.nextInt(node), random.nextInt(node)};
            }
            var groups = new ArrayList<List<Integer>>();
            for (int unit = 0; unit < 4; unit++) groups.add(new ArrayList<>());
            for (int node = 0; node < parents.length; node++) {
                groups.get(random.nextInt(groups.size())).add(node);
            }
            groups.removeIf(List::isEmpty);
            Collections.shuffle(groups, random);
            compare(graph(parents), groups);
        }
    }

    @Test void repeatedPortsMultipleOutputsAndIntraUnitEdgesPreserveOccurrences() {
        var producer = new CompiledNode(new NodeId(0),
                new Operation(DropoutKind.DROPOUT, new DropoutAttrs(.25)),
                List.of(new ValueId(0), new ValueId(1)),
                List.of(new ValueId(2), new ValueId(3), new ValueId(4)));
        var nodes = List.of(producer, add(1, 3, 3, 5), add(2, 2, 4, 6), add(3, 5, 6, 7));
        var dag = dag(nodes);
        var edges = List.copyOf(dag.edges());
        var split = compare(dag, List.of(List.of(3), List.of(2), List.of(1), List.of(0)));
        assertEquals(List.of(List.of(), List.of(0), List.of(0), List.of(1, 2)),
                IntStream.range(0, 4).mapToObj(split::dependencies).toList());
        var contracted = compare(dag, List.of(List.of(3), List.of(0, 1), List.of(2)));
        assertEquals(List.of(), contracted.dependencies(0));
        assertEquals(6, dag.edges().size());
        assertEquals(List.of(0, 1), dag.consumers(new ValueId(3)).stream()
                .map(PartitionDag.ConsumerOccurrence::inputPosition).toList());
        assertEquals(1, dag.producer(new ValueId(3)).orElseThrow().outputPosition());
        assertEquals(2, dag.producer(new ValueId(4)).orElseThrow().outputPosition());
        assertEquals(edges, dag.edges());
    }

    @Test void eachInvocationUsesCurrentMembershipAndRequiresExactNodeIdentity() {
        var dag = graph(new int[][] {{}, {0}, {1}});
        compare(dag, List.of(List.of(2), List.of(1), List.of(0)));
        compare(dag, List.of(List.of(0, 1), List.of(2)));
        compare(dag, List.of(List.of(0, 2), List.of(1)));
        compare(dag, List.of(List.of(2), List.of(1), List.of(0)));
        var node = dag.nodes().getFirst();
        var equalCopy = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs());
        assertEquals(node, equalCopy);
        assertNotSame(node, equalCopy);
        // Even a value-equal key must not substitute for the original graph occurrence.
        var wrongOrdinals = ordinals(dag);
        wrongOrdinals.remove(node);
        wrongOrdinals.put(equalCopy, 0);
        assertThrows(NullPointerException.class, () -> new CpuUnitTopology(
                dag.nodes().stream().map(List::of).toList(), wrongOrdinals, dag.edges()));
    }

    @Test void largeSparseTopologyVisitsEveryEdgeExactlyOnceIncludingDependencyEmission() {
        int count = 20_000;
        var nodes = new ArrayList<CompiledNode>(count);
        for (int node = 0; node < count; node++) nodes.add(add(node, node, node, node + 1));
        var dag = dag(nodes);
        var edges = new CountingEdges(dag.edges());
        var members = new ArrayList<>(nodes.stream().map(List::of).toList());
        Collections.reverse(members);
        var topology = new CpuUnitTopology(members, ordinals(dag), edges);
        assertEquals(1, edges.iterations);
        assertEquals(2 * (count - 1), edges.visits);
        for (int index = 0; index < count; index++) {
            assertEquals(count - index - 1, topology.order().get(index));
            assertEquals(List.of(index), topology.memberNodeOrdinals(index));
            assertEquals(index == 0 ? List.of() : List.of(index - 1),
                    topology.dependencies(index));
        }
        assertEquals(1, edges.iterations);
        assertEquals(39_998, edges.visits);
    }

    private static CpuUnitTopology compare(PartitionDag dag, List<List<Integer>> positions) {
        var members = positions.stream().map(group -> group.stream().map(dag.nodes()::get).toList())
                .toList();
        var ordinals = ordinals(dag);
        List<Integer> expected;
        try {
            expected = oldOrder(members, ordinals, dag);
        } catch (IllegalArgumentException cycle) {
            var actual = assertThrows(IllegalArgumentException.class,
                    () -> new CpuUnitTopology(members, ordinals, dag.edges()));
            assertEquals(cycle.getMessage(), actual.getMessage());
            return null;
        }
        var actual = new CpuUnitTopology(members, ordinals, dag.edges());
        assertEquals(expected, actual.order());
        for (int index = 0; index < expected.size(); index++) {
            int unit = expected.get(index);
            assertEquals(oldProducers(members, unit, dag).stream().map(expected::indexOf)
                    .sorted().toList(), actual.dependencies(index));
            assertEquals(positions.get(unit), actual.memberNodeOrdinals(index));
            for (int member = 0; member < members.get(unit).size(); member++) {
                assertSame(dag.nodes().get(actual.memberNodeOrdinals(index).get(member)),
                        members.get(actual.order().get(index)).get(member));
            }
        }
        return actual;
    }

    // Bounded test-only copy of the former repeated producer/edge scan, including its tie-break.
    private static List<Integer> oldOrder(List<List<CompiledNode>> members,
            Map<CompiledNode, Integer> ordinals, PartitionDag dag) {
        var remaining = new ArrayList<>(IntStream.range(0, members.size()).boxed().toList());
        var result = new ArrayList<Integer>();
        while (!remaining.isEmpty()) {
            int ready = remaining.stream().filter(unit ->
                            result.containsAll(oldProducers(members, unit, dag)))
                    .min(Comparator.comparingInt(unit -> members.get(unit).stream()
                            .mapToInt(ordinals::get).min().orElseThrow()))
                    .orElseThrow(() -> new IllegalArgumentException("CPU unit topology is cyclic"));
            result.add(ready);
            remaining.remove(Integer.valueOf(ready));
        }
        return result;
    }

    private static List<Integer> oldProducers(List<List<CompiledNode>> members, int target,
            PartitionDag dag) {
        return IntStream.range(0, members.size()).filter(unit -> unit != target
                && dag.edges().stream().anyMatch(edge -> members.get(unit).stream()
                        .anyMatch(node -> node == edge.producer().node())
                        && members.get(target).stream()
                        .anyMatch(node -> node == edge.consumer().node()))).boxed().toList();
    }

    private static IdentityHashMap<CompiledNode, Integer> ordinals(PartitionDag dag) {
        var result = new IdentityHashMap<CompiledNode, Integer>();
        for (int index = 0; index < dag.nodes().size(); index++) {
            result.put(dag.nodes().get(index), index);
        }
        return result;
    }

    private static PartitionDag graph(int[][] parents) {
        var nodes = new ArrayList<CompiledNode>();
        for (int node = 0; node < parents.length; node++) {
            int[] inputs = parents[node];
            nodes.add(add(node, inputs.length == 0 ? 0 : inputs[0] + 1,
                    inputs.length < 2 ? 0 : inputs[1] + 1, node + 1));
        }
        return dag(nodes);
    }

    private static CompiledNode add(int id, int left, int right, int output) {
        return new CompiledNode(new NodeId(id),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(left), new ValueId(right)), List.of(new ValueId(output)));
    }

    private static PartitionDag dag(List<CompiledNode> nodes) {
        return new PartitionDag(new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList()), nodes);
    }

    private static final class CountingEdges implements Iterable<PartitionDag.Edge> {
        private final List<PartitionDag.Edge> edges;
        private int iterations;
        private int visits;

        private CountingEdges(List<PartitionDag.Edge> edges) { this.edges = edges; }

        @Override public Iterator<PartitionDag.Edge> iterator() {
            iterations++;
            var iterator = edges.iterator();
            return new Iterator<>() {
                @Override public boolean hasNext() { return iterator.hasNext(); }
                @Override public PartitionDag.Edge next() { visits++; return iterator.next(); }
            };
        }
    }
}
