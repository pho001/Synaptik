package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Leaf;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Role;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint.Source;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
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
 * General-scan acquisition delegates its bounded cold body inventory to Checkpoint and Compiler,
 * independently of execution support. Training and backend closure remain with the existing
 * Compiler/Engine owners rather than a source-kind approximation.
 * Callers serialize all access through ModelExecutionState and exclude inherited Module mutation.
 */
final class OwnedModelSource {
    final Source source;
    private final Module root;
    private final ModelForward<List<Tensor>, List<Tensor>> composition;
    private final StateDictionary dictionary;
    private final Map<String, Parameter> parameters;
    private final Map<String, Buffer> buffers;
    private final List<ModuleGuard> modules = new ArrayList<>();
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
        dictionary = root.stateDictionary();
        parameters = root.parametersRecursively();
        buffers = root.buffersRecursively();
        modules.add(guard(root, -1, ""));
        for (int i = 0; i < modules.size(); i++) {
            for (var child : modules.get(i).children().entrySet())
                modules.add(guard(child.getValue(), i, child.getKey()));
        }
        refreshStorage();
    }

    /**
     * Inventories ordinary provenance, acquires external leaves once and installs final wrappers.
     * Checkpoint owns source/storage admission and delegates general bodies to Compiler's exact
     * cold constructor inventory. Body formals are not registered state, and no CPU predicate is
     * consulted here. The callback's prototype expression is not retained after private reconstruction.
     * @param root eval-mode defined tree after its single prototype forward
     * @param input declared storage-free prototype
     * @param output callback's exact output expression
     * @return owned source with no retained callback or original state graph
     * @throws IllegalArgumentException if source/state cannot be represented by GraphCheckpoint
     * @throws IllegalStateException if named state storage is absent, dead or inaccessible
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
     * Namespace keys retain only owner identity and local segments, not cumulative paths.
     * Adopts already-acquired graph/load leaves without a second copy and installs namespaces.
     * @param root empty private model receiving the state
     * @param source privately copied or loaded source
     * @return guarded source retaining the exact supplied final leaves
     * @throws IllegalArgumentException if paths are not canonical Module traversal order
     */
    static OwnedModelSource adopt(Module root, Source source) {
        var namespaces = new IdentityHashMap<Module, Map<String, StateNamespace>>();
        for (Leaf leaf : source.leaves()) {
            if (leaf.role() == Role.CONSTANT) continue;
            String[] segments = leaf.name().split("\\.", -1);
            Module owner = root;
            for (int i = 0; i < segments.length - 1; i++) {
                var local = namespaces.computeIfAbsent(owner, ignored -> new LinkedHashMap<>());
                StateNamespace namespace = local.get(segments[i]);
                if (namespace == null) {
                    namespace = new StateNamespace();
                    owner.child(segments[i], namespace);
                    local.put(segments[i], namespace);
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
     * The first failing guard identifies its category and captured module/state path; module
     * paths are assembled only on failure, keeping deep-tree guard metadata linear.
     * @throws IllegalStateException if the source was externally mutated, with category and path
     */
    void validate() {
        for (int i = 0; i < modules.size(); i++) {
            ModuleGuard guard = modules.get(i);
            if (guard.module().mode() != guard.mode()) changed("mode", modulePath(i));
            if (firstMismatch(guard.children(), guard.module().children()) != null)
                changed("topology", modulePath(i));
        }
        String path = firstMismatch(parameters, root.parametersRecursively());
        if (path != null) changed("parameter wrapper", path);
        path = firstMismatch(buffers, root.buffersRecursively());
        if (path != null) changed("buffer wrapper", path);
        List<StateEntry> current = root.stateDictionary().entries();
        for (int i = 0; i < dictionary.entries().size(); i++) {
            StateEntry expected = dictionary.entries().get(i);
            StateEntry actual = current.get(i);
            if (!expected.path().equals(actual.path()) || expected.kind() != actual.kind())
                changed("state topology", expected.path());
            if (expected.value() != actual.value()) changed("Tensor binding", expected.path());
        }
        for (Leaf leaf : source.leaves()) {
            if (leaf.tensor().hostStorage().orElse(null) != stores.get(leaf.tensor()))
                changed("storage association", leaf.name());
        }
    }

    /** Refreshes associations only after the Model's explicit successful native transition. */
    void refreshStorage() {
        for (Leaf leaf : source.leaves()) {
            stores.put(leaf.tensor(), leaf.tensor().hostStorage().orElse(null));
        }
    }

    /**
     * Finds the first changed ordered key or reference without allocating key snapshots.
     * @param expected captured ordered map
     * @param actual current ordered map
     * @return captured path of the first mismatch, new key for an addition, or null if unchanged
     */
    private static String firstMismatch(Map<String, ?> expected, Map<String, ?> actual) {
        var old = expected.entrySet().iterator();
        var live = actual.entrySet().iterator();
        while (old.hasNext()) {
            var entry = old.next();
            if (!live.hasNext()) return entry.getKey();
            var other = live.next();
            if (!entry.getKey().equals(other.getKey()) || entry.getValue() != other.getValue())
                return entry.getKey();
        }
        return live.hasNext() ? live.next().getKey() : null;
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

    /**
     * Formats a captured module location only after a failing guard.
     * @param index captured module ordinal
     * @return exact dot path, or {@code <root>} for the root
     */
    private String modulePath(int index) {
        var segments = new ArrayDeque<String>();
        for (int i = index; modules.get(i).parent() >= 0; i = modules.get(i).parent())
            segments.addFirst(modules.get(i).name());
        return segments.isEmpty() ? "<root>" : String.join(".", segments);
    }

    /**
     * Captures one module without retaining its full ancestor path.
     * @param module final owned module
     * @param parent parent guard ordinal, or -1 for root
     * @param name local child name, empty for root
     * @return immutable guard over exact module identity and ordered children
     */
    private static ModuleGuard guard(Module module, int parent, String name) {
        return new ModuleGuard(module, parent, name, module.mode(), module.children());
    }

    /** Linear parent/local-name location plus the existing live-source guards. */
    private record ModuleGuard(Module module, int parent, String name, ForwardMode mode,
            Map<String, Module> children) { }

    /**
     * Reports the existing producer's precise failure without recapturing mutable state.
     * @param category failing association category
     * @param path stable captured module or state path
     * @throws IllegalStateException always: the capability requires replacement
     */
    private static void changed(String category, String path) {
        throw new IllegalStateException("owned Model source changed: " + category + " at " + path
                + "; construct a replacement Model");
    }

    /** Structural path namespace only; it has no forward recipe or execution responsibility. */
    private static final class StateNamespace extends Module {
    }
}
