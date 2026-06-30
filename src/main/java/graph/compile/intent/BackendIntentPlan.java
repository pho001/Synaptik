package graph.compile.intent;

import backend.contract.ComputeBackend;
import graph.optimizer.state.GraphRewriteMap;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Compile-local backend intent ownership for semantic tensors.
 *
 * <p>Backend intent is not tensor state. The compiler carries this plan beside graph snapshots so optimizer rewrites
 * and compiled-node snapshots can preserve explicit accelerator anchors without mutating user-owned tensors.
 */
public final class BackendIntentPlan {
    private static final BackendIntentPlan EMPTY = new BackendIntentPlan(new IdentityHashMap<>());

    private final IdentityHashMap<Tensor, ComputeBackend> intents;

    private BackendIntentPlan(IdentityHashMap<Tensor, ComputeBackend> intents) {
        this.intents = copyAcceleratorIntents(intents);
    }

    /**
     * Returns an empty plan where all tensors default to CPU.
     *
     * @return empty backend intent plan
     */
    public static BackendIntentPlan empty() {
        return EMPTY;
    }

    /**
     * Creates a plan with one explicit backend intent.
     *
     * @param tensor tensor to anchor
     * @param backend backend intent
     * @return backend intent plan
     */
    public static BackendIntentPlan of(Tensor tensor, ComputeBackend backend) {
        return empty().withBackend(tensor, backend);
    }

    /**
     * Creates a plan with multiple tensors assigned to one explicit backend.
     *
     * @param backend backend intent
     * @param tensors tensors to anchor
     * @return backend intent plan
     */
    public static BackendIntentPlan of(ComputeBackend backend, Tensor... tensors) {
        return empty().withBackends(backend, tensors);
    }

    /**
     * Resolves backend intent for a tensor.
     *
     * @param tensor tensor to inspect
     * @return explicit backend or CPU when none is present
     */
    public ComputeBackend backend(Tensor tensor) {
        if (tensor == null) {
            return ComputeBackend.CPU;
        }
        return intents.getOrDefault(tensor, ComputeBackend.CPU);
    }

    /**
     * Returns a new plan with {@code tensor} assigned to {@code backend}. CPU clears explicit intent.
     *
     * @param tensor tensor to update
     * @param backend backend intent
     * @return updated plan
     */
    public BackendIntentPlan withBackend(Tensor tensor, ComputeBackend backend) {
        if (tensor == null) {
            return this;
        }
        IdentityHashMap<Tensor, ComputeBackend> copy = mutableCopy();
        if (isAcceleratorBackend(backend)) {
            copy.put(tensor, backend);
        } else {
            copy.remove(tensor);
        }
        return fromMutable(copy);
    }

    /**
     * Returns a new plan with all supplied tensors assigned to one explicit backend.
     *
     * @param backend backend intent
     * @param tensors tensors to anchor
     * @return updated plan
     */
    public BackendIntentPlan withBackends(ComputeBackend backend, Tensor... tensors) {
        if (tensors == null || tensors.length == 0) {
            return this;
        }
        BackendIntentPlan current = this;
        for (Tensor tensor : tensors) {
            current = current.withBackend(tensor, backend);
        }
        return current;
    }

    /**
     * Returns a new plan that preserves backend intent from {@code source} on {@code target}.
     *
     * @param target replacement tensor
     * @param source original tensor
     * @return updated plan
     */
    public BackendIntentPlan preserve(Tensor target, Tensor source) {
        if (target == null || source == null) {
            return this;
        }
        return withBackend(target, backend(source));
    }

    /**
     * Returns a new plan mapped from original graph tensors to snapshot tensors.
     *
     * @param originalBySnapshot snapshot-to-original tensor map
     * @return remapped backend intent plan
     */
    public BackendIntentPlan remapFromOriginalBySnapshot(Map<Tensor, Tensor> originalBySnapshot) {
        if (originalBySnapshot == null || originalBySnapshot.isEmpty() || intents.isEmpty()) {
            return empty();
        }
        IdentityHashMap<Tensor, ComputeBackend> remapped = new IdentityHashMap<>();
        for (Map.Entry<Tensor, Tensor> entry : originalBySnapshot.entrySet()) {
            Tensor snapshot = entry.getKey();
            Tensor original = entry.getValue();
            ComputeBackend backend = backend(original);
            if (isAcceleratorBackend(backend)) {
                remapped.put(snapshot, backend);
            }
        }
        return fromMutable(remapped);
    }

    /**
     * Returns a new plan remapped through an optimizer rewrite map.
     *
     * @param rewriteMap cumulative optimizer rewrite map
     * @return remapped backend intent plan
     */
    public BackendIntentPlan remapThrough(GraphRewriteMap rewriteMap) {
        if (rewriteMap == null || intents.isEmpty()) {
            return this;
        }
        IdentityHashMap<Tensor, ComputeBackend> remapped = new IdentityHashMap<>();
        for (Map.Entry<Tensor, ComputeBackend> entry : intents.entrySet()) {
            Tensor rewritten = rewriteMap.resolve(entry.getKey());
            if (rewritten != null && isAcceleratorBackend(entry.getValue())) {
                putIntent(remapped, rewritten, entry.getValue());
            }
        }
        return fromMutable(remapped);
    }

    IdentityHashMap<Tensor, ComputeBackend> mutableCopy() {
        return copyAcceleratorIntents(intents);
    }

    static BackendIntentPlan fromMutable(IdentityHashMap<Tensor, ComputeBackend> intents) {
        if (intents == null || intents.isEmpty()) {
            return empty();
        }
        return new BackendIntentPlan(intents);
    }

    static boolean isAcceleratorBackend(ComputeBackend backend) {
        return backend != null && backend != ComputeBackend.CPU;
    }

    private static void putIntent(
            IdentityHashMap<Tensor, ComputeBackend> intents,
            Tensor tensor,
            ComputeBackend backend
    ) {
        ComputeBackend existing = intents.get(tensor);
        if (existing != null && existing != backend) {
            throw new IllegalStateException(
                    "Conflicting backend intents after graph rewrite for tensor "
                            + tensor.getLabel()
                            + ": "
                            + existing
                            + " vs "
                            + backend
            );
        }
        intents.put(tensor, backend);
    }

    private static IdentityHashMap<Tensor, ComputeBackend> copyAcceleratorIntents(
            IdentityHashMap<Tensor, ComputeBackend> input
    ) {
        IdentityHashMap<Tensor, ComputeBackend> copy = new IdentityHashMap<>();
        if (input == null) {
            return copy;
        }
        for (Map.Entry<Tensor, ComputeBackend> entry : input.entrySet()) {
            Tensor tensor = Objects.requireNonNull(entry.getKey(), "backend intent tensor cannot be null");
            ComputeBackend backend = entry.getValue();
            if (isAcceleratorBackend(backend)) {
                copy.put(tensor, backend);
            }
        }
        return copy;
    }
}
