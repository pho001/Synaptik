package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeSet;

/**
 * Immutable structural presence facts for one exact generated Scan body, not derivative formulas.
 *
 * <p>Roots are declaration slots, ordered as {@code carries} followed by {@code seedFormals};
 * tied actual values never coalesce roots. A scope is the disjunction of its root memberships;
 * an empty scope is absent. Every local executable node has exactly one scope, including work
 * that reads only primal values. Optional ordered ADDs additionally retain each operand's scope,
 * and temporal accumulators remain separate from those optional list merges.</p>
 *
 * <p>Optional merges require identical operand and result data types and Shapes. With only one
 * present operand the result is its raw identity, not an ordinary ADD promotion or broadcast.</p>
 *
 * <p>Construction validates intrinsic local structure and input availability, defensively sorts
 * scope memberships and snapshots maps in graph-node order. Numeric temporal accumulator
 * formals and primal-only formals are unconditional; controlled carry/seed inputs retain their
 * distinct roots. Containing consumers independently check header positions, binding roles,
 * scope/dataflow applicability and ordinary operation guards.
 * The exact graph reference is retained; structural equality cannot substitute another body.</p>
 *
 * @param graph non-null exact immutable local compiled body
 * @param carries non-null ordered distinct numeric carry positions and next-presence scopes
 * @param seedFormals non-null ordered distinct local input positions, disjoint from numeric carries
 * @param scopes non-null scope table; each non-null membership has distinct valid root indices
 * @param nodeScopes non-null complete local-node to scope-index mapping
 * @param merges non-null optional-merge mapping naming local binary single-output ordinary ADDs
 * @param accumulators non-null distinct temporal shared-carry positions and contribution scopes
 */
public record ScanActivityPlan(
        CompiledGraphModel graph, List<Carry> carries, List<Integer> seedFormals,
        List<List<Integer>> scopes, Map<NodeId, Integer> nodeScopes,
        Map<NodeId, Merge> merges, List<Accumulator> accumulators) {
    /**
     * Snapshots a complete activity plan and rejects malformed local membership or merge shapes.
     *
     * @param graph non-null exact body graph
     * @param carries non-null ordered unique numeric carry declarations
     * @param seedFormals non-null ordered unique seed input positions
     * @param scopes non-null scope memberships, copied into ascending root order
     * @param nodeScopes non-null mapping covering every local node exactly
     * @param merges non-null mapping of optional ordinary ADD occurrences
     * @param accumulators non-null ordered unique temporal accumulation declarations
     * @throws NullPointerException if a component, element, key or value is null
     * @throws IllegalArgumentException if positions, scopes, membership or merge shapes are invalid
     * @throws ArithmeticException if the number of roots overflows an int
     */
    public ScanActivityPlan {
        Objects.requireNonNull(graph, "graph");
        carries = List.copyOf(carries);
        seedFormals = List.copyOf(seedFormals);
        Objects.requireNonNull(scopes, "scopes");
        Objects.requireNonNull(nodeScopes, "nodeScopes");
        Objects.requireNonNull(merges, "merges");
        accumulators = List.copyOf(accumulators);
        int rootCount = Math.addExact(carries.size(), seedFormals.size());
        List<List<Integer>> memberships = new ArrayList<>(scopes.size());
        for (List<Integer> membership : scopes) {
            TreeSet<Integer> sorted = new TreeSet<>();
            for (Integer root : Objects.requireNonNull(membership, "scope membership")) {
                Objects.requireNonNull(root, "scope root");
                if (root < 0 || root >= rootCount || !sorted.add(root)) {
                    throw new IllegalArgumentException("scope roots must be distinct valid root indices");
                }
            }
            memberships.add(List.copyOf(sorted));
        }
        scopes = List.copyOf(memberships);
        Set<Integer> numericPositions = new HashSet<>();
        for (Carry carry : carries) {
            requirePosition(carry.numericCarryPosition(), graph.inputs().size(), "carry");
            requireScope(carry.nextScope(), scopes.size());
            if (!numericPositions.add(carry.numericCarryPosition())) {
                throw new IllegalArgumentException("duplicate numeric carry position");
            }
        }
        for (Accumulator accumulator : accumulators) {
            requirePosition(accumulator.numericSharedCarryPosition(), graph.inputs().size(), "accumulator");
            requireScope(accumulator.contributionScope(), scopes.size());
            if (!numericPositions.add(accumulator.numericSharedCarryPosition())) {
                throw new IllegalArgumentException("duplicate numeric carry/accumulator position");
            }
        }
        Set<Integer> seeds = new HashSet<>();
        for (int formal : seedFormals) {
            requirePosition(formal, graph.inputs().size(), "seed formal");
            if (!seeds.add(formal) || numericPositions.contains(formal)) {
                throw new IllegalArgumentException("seed formals must be unique and disjoint from numeric carries");
            }
        }
        nodeScopes.forEach((id, scope) -> {
            Objects.requireNonNull(id, "nodeScopes key");
            Objects.requireNonNull(scope, "nodeScopes[" + id + "]");
        });
        merges.forEach((id, merge) -> {
            Objects.requireNonNull(id, "merges key");
            Objects.requireNonNull(merge, "merges[" + id + "]");
        });
        if (!nodeScopes.keySet().equals(graph.nodePhases().keySet())) {
            throw new IllegalArgumentException("nodeScopes must cover exactly every local node");
        }
        LinkedHashMap<NodeId, Integer> orderedScopes = new LinkedHashMap<>();
        LinkedHashMap<NodeId, Merge> orderedMerges = new LinkedHashMap<>();
        Map<ValueId, TensorDescriptor> descriptors = merges.isEmpty() ? Map.of() : new HashMap<>();
        if (!merges.isEmpty()) {
            for (var value : graph.values()) descriptors.put(value.id(), value.descriptor());
        }
        for (CompiledNode node : graph.nodes()) {
            Integer scope = Objects.requireNonNull(nodeScopes.get(node.id()), "nodeScopes[" + node.id() + "]");
            requireScope(scope, scopes.size());
            orderedScopes.put(node.id(), scope);
            if (merges.containsKey(node.id())) {
                Merge merge = Objects.requireNonNull(merges.get(node.id()), "merges[" + node.id() + "]");
                requireScope(merge.leftScope(), scopes.size());
                requireScope(merge.rightScope(), scopes.size());
                if (node.operation().kind() != BinaryArithmeticKind.ADD
                        || node.inputs().size() != 2 || node.outputs().size() != 1
                        || node.scanBody().isPresent()) {
                    throw new IllegalArgumentException("merge must name a local ordinary binary ADD");
                }
                TensorDescriptor result = descriptors.get(node.outputs().getFirst());
                for (ValueId input : node.inputs()) {
                    TensorDescriptor operand = descriptors.get(input);
                    if (operand.dataType() != result.dataType() || !operand.shape().equals(result.shape())) {
                        throw new IllegalArgumentException(
                                "merge operands and result must have identical data types and Shapes");
                    }
                }
                TreeSet<Integer> union = new TreeSet<>(scopes.get(merge.leftScope()));
                union.addAll(scopes.get(merge.rightScope()));
                if (!scopes.get(scope).equals(List.copyOf(union))) {
                    throw new IllegalArgumentException("merge node scope must equal its operand-scope union");
                }
                orderedMerges.put(node.id(), merge);
            }
        }
        if (!orderedScopes.keySet().equals(nodeScopes.keySet())
                || !orderedMerges.keySet().equals(merges.keySet())) {
            throw new IllegalArgumentException("activity maps contain nonlocal node keys");
        }
        nodeScopes = Collections.unmodifiableMap(orderedScopes);
        merges = Collections.unmodifiableMap(orderedMerges);
        validateDataflow(graph, carries, seedFormals, scopes, nodeScopes, merges);
    }

    private static void validateDataflow(
            CompiledGraphModel graph, List<Carry> carries, List<Integer> seeds,
            List<List<Integer>> scopes, Map<NodeId, Integer> nodeScopes, Map<NodeId, Merge> merges) {
        Map<ValueId, List<Integer>> availability = new HashMap<>();
        for (int root = 0; root < carries.size(); root++) {
            availability.put(graph.inputs().get(carries.get(root).numericCarryPosition()), List.of(root));
        }
        for (int root = 0; root < seeds.size(); root++) {
            availability.put(graph.inputs().get(seeds.get(root)), List.of(carries.size() + root));
        }
        for (CompiledNode node : graph.nodes()) {
            Merge merge = merges.get(node.id());
            List<Integer> owner = scopes.get(nodeScopes.get(node.id()));
            for (int input = 0; input < node.inputs().size(); input++) {
                List<Integer> required = merge == null ? owner : scopes.get(
                        input == 0 ? merge.leftScope() : merge.rightScope());
                List<Integer> available = availability.get(node.inputs().get(input));
                if (available != null && !available.containsAll(required)) {
                    throw new IllegalArgumentException("node scope reads an unavailable controlled input");
                }
            }
            for (ValueId output : node.outputs()) {
                availability.put(output, owner);
            }
        }
    }

    private static void requirePosition(int position, int count, String role) {
        if (position < 0 || position >= count) {
            throw new IllegalArgumentException(role + " position is outside local input declarations");
        }
    }

    private static void requireScope(int scope, int count) {
        if (scope < 0 || scope >= count) {
            throw new IllegalArgumentException("scope index is outside scope table");
        }
    }

    /**
     * One carried structural root and its next-step presence.
     * @param numericCarryPosition nonnegative numeric carry declaration position
     * @param initialPresence initial structural presence, independent of numeric bits
     * @param nextScope nonnegative scope-table index for the next carry
     */
    public record Carry(int numericCarryPosition, boolean initialPresence, int nextScope) {
        /**
         * Validates nonnegative intrinsic indices; the plan validates their upper bounds.
         * @param numericCarryPosition nonnegative numeric carry position
         * @param initialPresence initial structural presence
         * @param nextScope nonnegative next-presence scope index
         * @throws IllegalArgumentException if either index is negative
         */
        public Carry {
            if (numericCarryPosition < 0 || nextScope < 0) {
                throw new IllegalArgumentException("carry indices must be nonnegative");
            }
        }
    }

    /**
     * Ordered optional-list ADD operand scopes, not temporal accumulation.
     * @param leftScope nonnegative scope index for the left operand
     * @param rightScope nonnegative scope index for the right operand
     */
    public record Merge(int leftScope, int rightScope) {
        /**
         * Validates nonnegative operand indices; the plan checks upper bounds and ADD shape.
         * @param leftScope nonnegative left scope index
         * @param rightScope nonnegative right scope index
         * @throws IllegalArgumentException if either scope is negative
         */
        public Merge {
            if (leftScope < 0 || rightScope < 0) {
                throw new IllegalArgumentException("merge scopes must be nonnegative");
            }
        }
    }

    /**
     * Separate old-plus-contribution temporal accumulation for a numeric shared carry.
     * @param numericSharedCarryPosition nonnegative shared carry declaration position
     * @param contributionScope nonnegative scope index controlling only the new contribution
     */
    public record Accumulator(int numericSharedCarryPosition, int contributionScope) {
        /**
         * Validates nonnegative indices; the containing consumer checks shared-role applicability.
         * @param numericSharedCarryPosition nonnegative shared carry position
         * @param contributionScope nonnegative contribution scope index
         * @throws IllegalArgumentException if either index is negative
         */
        public Accumulator {
            if (numericSharedCarryPosition < 0 || contributionScope < 0) {
                throw new IllegalArgumentException("accumulator indices must be nonnegative");
            }
        }
    }
}
