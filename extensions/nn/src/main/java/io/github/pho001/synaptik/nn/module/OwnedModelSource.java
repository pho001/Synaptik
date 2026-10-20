package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Leaf;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Role;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Source;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Cold acquisition and live identity guards for one privately owned ordinary Tensor source.
 * Constants belong only to the source; Module wrappers remain the sole parameter/buffer registry.
 * Scan presence is cached from immutable ordinary provenance for every acquisition route, not
 * inferred from input types or saved role metadata. It adds no per-run traversal or execution path.
 * Callers serialize all access through ModelExecutionState and exclude inherited Module mutation.
 */
final class OwnedModelSource {
    final Source source;
    final boolean containsScan;
    private final Module root;
    private final ModelForward<List<Tensor>, List<Tensor>> composition;
    private final StateDictionary dictionary;
    private final Map<String, Parameter> parameters;
    private final Map<String, Buffer> buffers;
    private final List<Module> modules;
    private final List<ForwardMode> modes;
    private final List<Map<String, Module>> children;
    private final IdentityHashMap<Tensor, HostTensorStorage> stores = new IdentityHashMap<>();

    /**
     * Captures guards after final state installation.
     * @param root non-null final Module owner
     * @param source acquired source borrowing only final owned leaves
     */
    private OwnedModelSource(Module root, Source source) {
        this(root, source, null);
    }

    /**
     * Captures the same guards for final private composition state, without source admission.
     * @param root final eval owner
     * @param source borrowed final ordinary Tensor boundaries and leaves
     * @param composition internal adapter retaining only instantiated owned steps, or null for reconstruction
     */
    private OwnedModelSource(Module root, Source source,
            ModelForward<List<Tensor>, List<Tensor>> composition) {
        this.root = root;
        this.source = source;
        this.composition = composition;
        containsScan = containsScan(source.outputs());
        dictionary = root.stateDictionary();
        parameters = root.parametersRecursively();
        buffers = root.buffersRecursively();
        modules = new ArrayList<>();
        modes = new ArrayList<>();
        children = new ArrayList<>();
        var pending = new ArrayDeque<Module>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Module module = pending.removeFirst();
            modules.add(module);
            modes.add(module.mode());
            children.add(module.children());
            pending.addAll(module.children().values());
        }
        refreshStorage();
    }

    /**
     * Inventories ordinary provenance, acquires external leaves once and installs final wrappers.
     * @param root eval-mode defined tree after its single prototype forward
     * @param input declared storage-free prototype
     * @param output callback's exact output expression
     * @return owned source with no retained callback or original state graph
     * @throws IllegalArgumentException if source/state cannot be represented by GraphCheckpoint
     */
    static OwnedModelSource acquireDefinition(Module root, Tensor input, Tensor output) {
        List<Leaf> leaves = inventory(root, List.of(input), List.of(output), false);
        Source owned = GraphCheckpoint.copy(new Source(List.of(input), List.of(output), leaves));
        root.loadStateDictionary(dictionary(owned));
        return new OwnedModelSource(root, owned);
    }

    /**
     * Retains a builder's final private state and output without capture, copy or reconstruction.
     * @param root final eval Model owning the chain
     * @param inputs ordered storage-free prototypes, including unused declared slots
     * @param outputs exact ordered results of the single prototype composition
     * @param composition internal adapter over instantiated owned steps used only by cold forward
     * @return guarded source borrowing only private final leaves
     * @throws IllegalArgumentException if any registered state is unreachable
     */
    static OwnedModelSource acquireComposition(Module root, List<Tensor> inputs, List<Tensor> outputs,
            ModelForward<List<Tensor>, List<Tensor>> composition) {
        List<Leaf> leaves = inventory(root, inputs, outputs, true);
        return new OwnedModelSource(root,
                new Source(inputs, outputs, leaves), composition);
    }

    /**
     * Inventories wrapper roles and fresh constants through ordinary Tensor provenance.
     * @param root final state owner
     * @param inputs every declared prototype, excluded from constants even when unused
     * @param outputs exact reachable ordered expression roots
     * @param verifyReachability whether to check state here rather than at existing Checkpoint admission
     * @return canonical state followed by deterministic collision-free constants
     * @throws IllegalArgumentException if a registered leaf is not reachable
     */
    private static List<Leaf> inventory(Module root, List<Tensor> inputs, List<Tensor> outputs,
            boolean verifyReachability) {
        List<Leaf> leaves = new ArrayList<>();
        IdentityHashMap<Tensor, Boolean> registered = new IdentityHashMap<>();
        for (StateEntry entry : root.stateDictionary().entries()) {
            leaves.add(new Leaf(entry.path(), entry.kind() == StateKind.PARAMETER
                    ? Role.PARAMETER : Role.BUFFER, entry.value()));
            registered.put(entry.value(), true);
        }
        var visited = new IdentityHashMap<Tensor, Boolean>();
        var pending = new ArrayDeque<Tensor>();
        for (int i = outputs.size() - 1; i >= 0; i--) pending.push(outputs.get(i));
        int constant = 0;
        while (!pending.isEmpty()) {
            Tensor tensor = pending.pop();
            if (visited.put(tensor, true) != null) continue;
            if (tensor.provenance().isPresent()) {
                var operands = tensor.provenance().orElseThrow().inputs();
                for (int i = operands.size() - 1; i >= 0; i--) pending.push(operands.get(i));
            } else if (!inputs.contains(tensor) && !registered.containsKey(tensor)) {
                String name;
                do {
                    name = "__constant" + constant++;
                } while (collides(name, leaves));
                leaves.add(new Leaf(name, Role.CONSTANT, tensor));
                registered.put(tensor, true);
            }
        }
        for (StateEntry entry : verifyReachability ? root.stateDictionary().entries() : List.<StateEntry>of()) {
            if (!visited.containsKey(entry.value())) {
                throw new IllegalArgumentException("registered state is unreachable: " + entry.path());
            }
        }
        return leaves;
    }

    /**
     * Detects fixed scans once from ordinary provenance, including final-state-only roots and heads.
     * @param outputs acquired ordered roots; their upstream provenance is immutable
     * @return whether any reachable producer has a fixed recurrent-scan kind
     */
    private static boolean containsScan(List<Tensor> outputs) {
        var visited = new IdentityHashMap<Tensor, Boolean>();
        var pending = new ArrayDeque<Tensor>(outputs);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.pop();
            if (visited.put(tensor, true) != null || tensor.provenance().isEmpty()) continue;
            var provenance = tensor.provenance().orElseThrow();
            if (provenance.operation().kind() instanceof RecurrentScanKind) return true;
            pending.addAll(provenance.inputs());
        }
        return false;
    }

    /**
     * Composes builder chains cold or delegates unchanged source paths to Checkpoint reconstruction.
     * Validates every declared descriptor before invoking an internal composition adapter; produced
     * substitutions keep their upstream provenance. Only explicitly requested cold forward composes
     * again: compute uses the fixed acquired outputs, not this adapter.
     * @param values non-null exact full-descriptor expressions in declared order, including unused
     *     slots; borrowed for composition without retaining caller input storage
     * @return immutable ordinary outputs retaining the same owned parameters
     * @throws NullPointerException if values or an element is null
     * @throws IllegalArgumentException if arity or full descriptors differ
     */
    List<Tensor> forward(List<Tensor> values) {
        if (composition == null) return GraphCheckpoint.rebind(source, values);
        List<Tensor> stable = List.copyOf(Objects.requireNonNull(values, "inputs"));
        if (stable.size() != source.inputs().size()) {
            throw new IllegalArgumentException("inputs must match the full builder signature");
        }
        for (int i = 0; i < stable.size(); i++) {
            if (!source.inputs().get(i).descriptor().equals(stable.get(i).descriptor())) {
                throw new IllegalArgumentException("input must match the full builder signature at " + i);
            }
        }
        return composition.forward(stable);
    }

    /**
     * Adopts already-acquired graph/load leaves without a second copy and installs namespaces.
     * @param root empty private model receiving the state
     * @param source privately copied or loaded source
     * @return guarded source retaining the exact supplied final leaves
     * @throws IllegalArgumentException if paths are not canonical Module traversal order
     */
    static OwnedModelSource adopt(Module root, Source source) {
        Map<String, StateNamespace> namespaces = new LinkedHashMap<>();
        for (Leaf leaf : source.leaves()) {
            if (leaf.role() == Role.CONSTANT) continue;
            String[] segments = leaf.name().split("\\.", -1);
            Module owner = root;
            String path = "";
            for (int i = 0; i < segments.length - 1; i++) {
                path = path.isEmpty() ? segments[i] : path + "." + segments[i];
                StateNamespace namespace = namespaces.get(path);
                if (namespace == null) {
                    namespace = new StateNamespace();
                    owner.child(segments[i], namespace);
                    namespaces.put(path, namespace);
                }
                owner = namespace;
            }
            String name = segments[segments.length - 1];
            if (leaf.role() == Role.PARAMETER) owner.parameter(name, leaf.tensor());
            else owner.buffer(name, leaf.tensor());
        }
        if (!root.stateDictionary().equals(dictionary(source))) {
            throw new IllegalArgumentException("state paths must follow canonical Module traversal order");
        }
        root.eval();
        return new OwnedModelSource(root, source);
    }

    /**
     * Checks all frozen modes, topology, wrappers, Tensor bindings and storage associations.
     * @throws IllegalStateException if the source was externally mutated
     */
    void validate() {
        for (int i = 0; i < modules.size(); i++) {
            Module module = modules.get(i);
            if (module.mode() != modes.get(i)
                    || !orderedEquals(module.children(), children.get(i))) changed();
        }
        if (!dictionary.equals(root.stateDictionary())
                || !orderedEquals(parameters, root.parametersRecursively())
                || !orderedEquals(buffers, root.buffersRecursively())) changed();
        for (var entry : stores.entrySet()) {
            if (entry.getKey().hostStorage().orElse(null) != entry.getValue()) changed();
        }
    }

    /** Refreshes associations only after the Model's explicit successful native transition. */
    void refreshStorage() {
        for (Leaf leaf : source.leaves()) {
            stores.put(leaf.tensor(), leaf.tensor().hostStorage().orElse(null));
        }
    }

    /**
     * Compares encounter order as well as identity (Map.equals alone ignores order).
     * @param expected captured ordered map
     * @param actual current ordered map
     * @return true only for identical ordered keys and reference values
     */
    private static boolean orderedEquals(Map<String, ?> expected, Map<String, ?> actual) {
        if (!new ArrayList<>(expected.keySet()).equals(new ArrayList<>(actual.keySet()))) return false;
        for (String key : expected.keySet()) if (expected.get(key) != actual.get(key)) return false;
        return true;
    }

    /**
     * Projects only registered state, preserving source order and aliases.
     * @param source final source
     * @return immutable shallow dictionary for strict Module installation/comparison
     */
    private static StateDictionary dictionary(Source source) {
        return new StateDictionary(source.leaves().stream()
                .filter(leaf -> leaf.role() != Role.CONSTANT)
                .map(leaf -> new StateEntry(leaf.name(), leaf.role() == Role.PARAMETER
                        ? StateKind.PARAMETER : StateKind.BUFFER, leaf.tensor())).toList());
    }

    /**
     * Avoids direct and prefix collisions with every registered name.
     * @param name candidate constant name
     * @param leaves prior registrations
     * @return whether the candidate conflicts with an existing path
     */
    private static boolean collides(String name, List<Leaf> leaves) {
        return leaves.stream().anyMatch(leaf -> leaf.name().equals(name)
                || leaf.name().startsWith(name + ".") || name.startsWith(leaf.name() + "."));
    }

    /** @throws IllegalStateException always: the capability cannot recapture mutable state */
    private static void changed() {
        throw new IllegalStateException("owned Model source changed; construct a replacement Model");
    }

    /** Structural path namespace only; it has no forward recipe or execution responsibility. */
    private static final class StateNamespace extends Module {
    }
}
