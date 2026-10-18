package io.github.pho001.synaptik.checkpoint.graph;

import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Call-local exact identity association around Compiler's raw source, not another graph model.
 * Normalized Model IDs put declared inputs first, unique named state second and captured nodes
 * last. Only Compiler ever maps ValueIds back to Tensors or constructs stored operations.
 */
final class GraphSource {
    final GraphCheckpoint.Source source;
    final CompiledGraphModel graph;
    final List<Tensor> states;
    final List<GraphCheckpoint.Role> roles;
    final List<Integer> registrations;

    /**
     * Retains already validated call-local source associations without a cache or ownership.
     * @param source immutable containers of borrowed expressions
     * @param graph normalized existing Model graph
     * @param states unique named identities in registration encounter order
     * @param roles state-aligned explicit roles
     * @param registrations registration-aligned state indices
     */
    private GraphSource(GraphCheckpoint.Source source, CompiledGraphModel graph, List<Tensor> states,
                        List<GraphCheckpoint.Role> roles, List<Integer> registrations) {
        this.source = source;
        this.graph = graph;
        this.states = List.copyOf(states);
        this.roles = List.copyOf(roles);
        this.registrations = List.copyOf(registrations);
    }

    /**
     * Captures once and reconciles declared slots with first-encounter leaves by exact identity.
     * @param source non-null borrowed source, never mutated
     * @return normalized graph and bounded call-local association, never null
     * @throws IllegalArgumentException if names, roles, closure, source families or geometry fail
     * @throws ArithmeticException if checked canonical byte/stride geometry overflows
     */
    static GraphSource capture(GraphCheckpoint.Source source) {
        var positions = new IdentityHashMap<Tensor, Integer>();
        var values = new ArrayList<GraphValue>();
        var inputs = new ArrayList<ValueId>();
        for (Tensor input : source.inputs()) {
            if (input.provenance().isPresent() || positions.put(input, positions.size()) != null)
                throw new IllegalArgumentException("Inputs require unique provenance-free identities");
            addInput(values, inputs, input);
        }
        var states = new ArrayList<Tensor>();
        var roles = new ArrayList<GraphCheckpoint.Role>();
        var registrations = new ArrayList<Integer>();
        var names = new HashSet<String>();
        for (var leaf : source.leaves()) {
            nameLength(leaf.name());
            if (!names.add(leaf.name())) throw new IllegalArgumentException("Duplicate leaf name");
            Tensor tensor = leaf.tensor();
            if (tensor.provenance().isPresent()) throw new IllegalArgumentException("Named state has provenance");
            if (leaf.role() == GraphCheckpoint.Role.PARAMETER
                    && (!tensor.descriptor().dataType().isFloating() || !tensor.descriptor().requiresGrad()))
                throw new IllegalArgumentException("PARAMETER requires floating requiresGrad=true");
            Integer position = positions.get(tensor);
            if (position == null) {
                position = positions.size();
                positions.put(tensor, position);
                states.add(tensor);
                roles.add(leaf.role());
                addInput(values, inputs, tensor);
            } else if (position < source.inputs().size()) {
                throw new IllegalArgumentException("Named state overlaps input prototypes");
            } else if (roles.get(position - source.inputs().size()) != leaf.role()) {
                throw new IllegalArgumentException("Conflicting alias roles");
            }
            registrations.add(position - source.inputs().size());
        }
        for (String name : names) {
            for (int dot = name.indexOf('.'); dot >= 0; dot = name.indexOf('.', dot + 1))
                if (names.contains(name.substring(0, dot)))
                    throw new IllegalArgumentException("Leaf name prefix collision");
        }
        var captured = GraphCompilationPort.captureSource(source.outputs());
        var remap = new HashMap<ValueId, ValueId>();
        var reachable = new IdentityHashMap<Tensor, Boolean>();
        for (int i = 0; i < captured.leaves().size(); i++) {
            Tensor tensor = captured.leaves().get(i);
            Integer position = positions.get(tensor);
            if (position == null) throw new IllegalArgumentException("Reachable non-input leaf is unnamed");
            remap.put(captured.graph().inputs().get(i), new ValueId(position));
            reachable.put(tensor, Boolean.TRUE);
        }
        for (Tensor state : states)
            if (!reachable.containsKey(state)) throw new IllegalArgumentException("Unreachable named state");
        var descriptors = new HashMap<ValueId, GraphValue>();
        for (GraphValue value : captured.graph().values()) {
            GraphValueStorage.bytes(value.descriptor());
            descriptors.put(value.id(), value);
        }
        var nodes = new ArrayList<CompiledNode>();
        var phases = new HashMap<NodeId, GraphPhase>();
        for (CompiledNode node : captured.graph().nodes()) {
            ValueId id = new ValueId(values.size());
            remap.put(node.outputs().getFirst(), id);
            values.add(new GraphValue(id, descriptors.get(node.outputs().getFirst()).descriptor()));
            NodeId nodeId = new NodeId(nodes.size());
            nodes.add(new CompiledNode(nodeId, node.operation(),
                    node.inputs().stream().map(remap::get).toList(), List.of(id)));
            phases.put(nodeId, GraphPhase.FORWARD);
        }
        var graph = new CompiledGraphModel(values, nodes, inputs,
                captured.graph().outputs().stream().map(remap::get).toList(), phases);
        return new GraphSource(source, graph, states, roles, registrations);
    }

    /**
     * Adds a prototype or state descriptor after canonical leaf geometry validation.
     * @param values ordered normalized graph values to extend
     * @param inputs graph input IDs to extend
     * @param tensor provenance-free borrowed leaf, not read for storage
     */
    private static void addInput(List<GraphValue> values, List<ValueId> inputs, Tensor tensor) {
        GraphValueStorage.leafBytes(tensor.descriptor());
        ValueId id = new ValueId(values.size());
        values.add(new GraphValue(id, tensor.descriptor()));
        inputs.add(id);
    }

    /**
     * Validates exact Unicode and NN-compatible nonblank dot segments without normalization.
     * @param name non-null registration name
     * @return checked strict UTF-8 byte length, positive and within JVM byte-array bounds
     * @throws IllegalArgumentException if Unicode, segment grammar or size is invalid
     */
    static int nameLength(String name) {
        long bytes = 0;
        boolean nonblank = false;
        for (int i = 0; i < name.length(); i++) {
            char ch = name.charAt(i);
            if (ch == '.') {
                if (!nonblank) throw new IllegalArgumentException("Blank leaf name segment");
                nonblank = false;
                bytes++;
                continue;
            }
            int cp = ch;
            if (Character.isHighSurrogate(ch)) {
                if (++i >= name.length() || !Character.isLowSurrogate(name.charAt(i)))
                    throw new IllegalArgumentException("Malformed Unicode leaf name");
                cp = Character.toCodePoint(ch, name.charAt(i));
                bytes += 4;
            } else if (Character.isLowSurrogate(ch)) {
                throw new IllegalArgumentException("Malformed Unicode leaf name");
            } else bytes += ch < 0x80 ? 1 : ch < 0x800 ? 2 : 3;
            if (!Character.isWhitespace(cp)) nonblank = true;
        }
        if (!nonblank || bytes > Integer.MAX_VALUE - 8)
            throw new IllegalArgumentException("Blank or oversized leaf name");
        return (int) bytes;
    }

    /**
     * Substitutes only explicit signature slots, leaving state and opaque upstream DAGs intact.
     * @param replacements non-null full-descriptor-compatible ordered replacements
     * @return immutable ordinary roots, potentially repeated
     * @throws NullPointerException if replacements or an element is null
     * @throws IllegalArgumentException if count or descriptors differ
     */
    List<Tensor> rebind(List<Tensor> replacements) {
        List<Tensor> snapshot = List.copyOf(Objects.requireNonNull(replacements, "inputs"));
        if (snapshot.size() != source.inputs().size()) throw new IllegalArgumentException("Input count mismatch");
        for (int i = 0; i < snapshot.size(); i++)
            if (!snapshot.get(i).descriptor().equals(source.inputs().get(i).descriptor()))
                throw new IllegalArgumentException("Input descriptor mismatch at " + i);
        var boundary = new ArrayList<>(snapshot);
        boundary.addAll(states);
        return GraphCompilationPort.reconstructSource(graph, boundary);
    }

    /**
     * Admits every storage association before BOOL inspection or detached allocation/copy.
     * @return independent source, retaining no original prototypes, graph or state
     * @throws IllegalArgumentException if a logical payload exceeds JVM bounds or BOOL is invalid
     * @throws IllegalStateException if any state cannot be read
     */
    GraphCheckpoint.Source copy() {
        var segments = GraphValueStorage.admit(states);
        GraphValueStorage.validateBools(states, segments);
        var boundary = new ArrayList<Tensor>();
        for (Tensor input : source.inputs()) boundary.add(TensorFactory.create(input.descriptor()));
        for (int i = 0; i < states.size(); i++)
            boundary.add(GraphValueStorage.copy(states.get(i).descriptor(), segments.get(i)));
        return finish(boundary, GraphCompilationPort.reconstructSource(graph, boundary));
    }

    /**
     * Reassociates registrations with final leaves without retaining this call-local source.
     * @param boundary final prototypes followed by unique final state
     * @param outputs ordinary reconstructed roots
     * @return immutable boundary lists, borrowing only the final Tensors
     */
    private GraphCheckpoint.Source finish(List<Tensor> boundary, List<Tensor> outputs) {
        var leaves = new ArrayList<GraphCheckpoint.Leaf>();
        for (int i = 0; i < source.leaves().size(); i++) {
            var old = source.leaves().get(i);
            leaves.add(new GraphCheckpoint.Leaf(old.name(), old.role(),
                    boundary.get(source.inputs().size() + registrations.get(i))));
        }
        return new GraphCheckpoint.Source(boundary.subList(0, source.inputs().size()), outputs, leaves);
    }
}
