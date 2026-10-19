package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Optimizer;
import io.github.pho001.synaptik.training.TrainingParameterStorage;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingStep;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * Exclusive high-level admission and cached public-owner composition for one owned Model.
 * All ordinary operations hold one admission through synchronous result closure. Close waits for
 * admission and retains attempt-all cleanup failures. Inherited Module mutation is caller-excluded.
 */
final class ModelExecutionState {
    static final ArtifactLimits DEFAULT_LIMITS =
            new ArtifactLimits(256L << 20, 4 << 20, 16384, 64, 64 << 20, 256L << 20);
    private final Model<?, ?> model;
    private final OwnedModelSource owned;
    private Engine engine;
    private boolean ownsEngine;
    private boolean frozen;
    private boolean busy;
    private boolean closing;
    private boolean closed;
    private Throwable closeFailure;
    private Throwable invalid;
    private PreparedExecution inference;
    private int[] inferenceSlots;
    private Tensor[] inferenceFixed;
    private TrainingSession session;
    private int[] trainingSlots;
    private Tensor[] trainingFixed;
    private List<Tensor> targets;
    private Arena nativeArena;
    private boolean nativeAdmitted;

    /**
     * Retains the already-acquired source; creates no Engine or native resource.
     * @param model final private Model owner
     * @param owned final installed source and live guards
     */
    ModelExecutionState(Model<?, ?> model, OwnedModelSource owned) {
        this.model = model;
        this.owned = owned;
    }

    /**
     * Selects a borrowed Engine before any setup attempt; preserves selection on failure.
     * @param candidate non-null open caller-owned Engine
     * @throws IllegalStateException if closed, busy, frozen or the candidate is closed
     */
    void withEngine(Engine candidate) {
        begin();
        try {
            if (frozen) throw new IllegalStateException("Model Engine binding is frozen");
            Objects.requireNonNull(candidate, "engine");
            if (candidate.isClosed()) throw new IllegalStateException("candidate Engine is closed");
            engine = candidate;
        } finally { end(); }
    }

    /**
     * Performs an explicit cold symbolic substitution without affecting cached execution.
     * @param values non-null exact-descriptor symbolic values, possibly produced expressions
     * @return immutable ordinary output expressions retaining owned state
     */
    List<Tensor> forward(List<Tensor> values) {
        begin();
        try {
            validateSource();
            return GraphCheckpoint.rebind(owned.source, values);
        } finally { end(); }
    }

    /**
     * Applies the one-input/output cold boundary under ordinary admission.
     * @param value non-null descriptor-compatible expression
     * @return the sole reconstructed output expression
     */
    Tensor forward(Tensor value) {
        begin();
        try {
            singleBoundary();
            validateSource();
            return GraphCheckpoint.rebind(owned.source, List.of(Objects.requireNonNull(value, "input")))
                    .getFirst();
        } finally { end(); }
    }

    /**
     * Computes a single-boundary result, validating arity before input inspection.
     * @param value borrowed stored exact-signature input
     * @return detached sole result, readable after every owner closes
     */
    HostTensorValue compute(Tensor value) {
        begin();
        try {
            singleBoundary();
            return computeOpen(List.of(Objects.requireNonNull(value, "input")), Long.MAX_VALUE).getFirst();
        } finally { end(); }
    }

    /**
     * Computes every ordered output and bounds all publications before detaching any.
     * @param values borrowed stored inputs in declared signature order
     * @param maximum total canonical byte ceiling, nonnegative
     * @return immutable detached ordered results, never a partially materialized list
     */
    List<HostTensorValue> compute(List<Tensor> values, long maximum) {
        begin();
        try { return computeOpen(values, maximum); }
        finally { end(); }
    }

    /**
     * Runs within one admission; builds positional associations only when preparing.
     * @param values declared-order stored batch
     * @param maximum aggregate byte ceiling
     * @return detached values after RunResult closure
     * @throws IllegalArgumentException if input metadata/storage or byte bounds reject
     */
    private List<HostTensorValue> computeOpen(List<Tensor> values, long maximum) {
        if (maximum < 0) throw new IllegalArgumentException("maximumTotalBytes must be non-negative");
        List<Tensor> stable = validateValues(owned.source.inputs(), values, true);
        validateSource();
        frozen = true;
        Engine selected = engine();
        if (inference == null) {
            CompiledGraph graph = selected.compile(owned.source.outputs());
            int[] slots = new int[graph.inputs().size()];
            Tensor[] fixed = new Tensor[slots.length];
            bindings(graph.inputs(), List.of(), slots, fixed);
            PreparedExecution prepared = selected.prepare(graph);
            inferenceSlots = slots;
            inferenceFixed = fixed;
            inference = prepared;
        }
        try (RunResult result = selected.runValues(inference,
                values(stable, List.of(), inferenceSlots, inferenceFixed))) {
            if (result.publications().size() != owned.source.outputs().size()) {
                throw new IllegalStateException("Model publication count changed");
            }
            long total = 0;
            for (var publication : result.publications()) {
                var descriptor = publication.descriptor();
                if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty()) {
                    throw new IllegalArgumentException("publication requires resolved static geometry");
                }
                long bytes = Math.multiplyExact(descriptor.shape().knownElementCount().orElseThrow(),
                        descriptor.dataType().byteWidth());
                if (bytes > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("publication exceeds JVM byte[] limit");
                }
                total = Math.addExact(total, bytes);
            }
            if (total > maximum) throw new IllegalArgumentException("publications exceed maximumTotalBytes");
            List<HostTensorValue> detached = new ArrayList<>(result.resultCount());
            for (var publication : result.publications()) {
                detached.add(result.materialize(publication, maximum));
            }
            return List.copyOf(detached);
        }
    }

    /**
     * Configures a single-output objective without changing Module mode.
     * @param signature non-null fully static target descriptor
     * @param builder non-null pure one-attempt objective construction
     * @param optimizer non-null Training-owned optimizer configuration
     */
    void configure(TensorDescriptor signature, BiFunction<Tensor, Tensor, Tensor> builder,
            Optimizer optimizer) {
        begin();
        try {
            singleBoundary();
            Objects.requireNonNull(signature, "targetSignature");
            Objects.requireNonNull(builder, "objectiveBuilder");
            configureOpen(List.of(TensorFactory.create(signature)),
                    (outputs, target) -> builder.apply(outputs.getFirst(), target.getFirst()), optimizer);
        } finally { end(); }
    }

    /**
     * Configures one persistent TrainingSession using explicit target prototypes.
     * @param prototypes ordered provenance-free static target prototypes, disjoint from source
     * @param builder pure callback invoked once per admitted explicit setup attempt
     * @param optimizer non-null optimizer configuration
     */
    void configure(List<Tensor> prototypes,
            BiFunction<List<Tensor>, List<Tensor>, Tensor> builder, Optimizer optimizer) {
        begin();
        try { configureOpen(prototypes, builder, optimizer); }
        finally { end(); }
    }

    /**
     * Performs the explicit heap-to-native transition before constructing a session.
     * @param prototypes declared targets, which are never mutated or retained as callback state
     * @param builder pure one-shot objective builder
     * @param optimizer Training-owned immutable optimizer
     * @throws IllegalStateException if configured already or source guards reject
     */
    private void configureOpen(List<Tensor> prototypes,
            BiFunction<List<Tensor>, List<Tensor>, Tensor> builder, Optimizer optimizer) {
        if (session != null) throw new IllegalStateException("Model training is already configured");
        List<Tensor> stable = List.copyOf(Objects.requireNonNull(prototypes, "targetPrototypes"));
        Objects.requireNonNull(builder, "objectiveBuilder");
        Objects.requireNonNull(optimizer, "optimizer");
        List<Tensor> seen = new ArrayList<>(owned.source.inputs());
        for (var leaf : owned.source.leaves()) seen.add(leaf.tensor());
        for (Tensor target : stable) {
            if (target.provenance().isPresent() || !target.descriptor().shape().isFullyStatic()
                    || seen.contains(target)) {
                throw new IllegalArgumentException("targets require unique static provenance-free prototypes");
            }
            byteSize(target.descriptor());
            seen.add(target);
        }
        validateSource();
        frozen = true;
        // The callback is cold and not retained, even after a failed attempt.
        Tensor objective = Objects.requireNonNull(builder.apply(owned.source.outputs(), stable), "objective");
        var descriptor = objective.descriptor();
        if (descriptor.shape().rank() != 0 || !descriptor.requiresGrad()
                || (descriptor.dataType() != DataType.FLOAT32 && descriptor.dataType() != DataType.FLOAT64)) {
            throw new IllegalArgumentException("objective must be scalar gradient-eligible FLOAT32/FLOAT64");
        }
        validateObjectiveLeaves(objective, stable);
        validateSource();
        Engine selected = engine();
        if (!nativeAdmitted) {
            closeInference();
            Arena candidate = Arena.ofShared();
            try {
                TrainingParameterStorage.copyToSharedNative(model.trainingParameters(), candidate);
            } catch (RuntimeException | Error failure) {
                cleanup(candidate, failure);
                throw failure;
            }
            nativeArena = candidate;
            nativeAdmitted = true;
            owned.refreshStorage();
        }
        TrainingSession candidate = TrainingSession.open(selected, model.trainingParameters(), objective, optimizer);
        try {
            int[] slots = new int[candidate.inputs().size()];
            Tensor[] fixed = new Tensor[slots.length];
            bindings(candidate.inputs(), stable, slots, fixed);
            validateSource();
            trainingSlots = slots;
            trainingFixed = fixed;
            targets = stable;
            session = candidate;
        } catch (RuntimeException | Error failure) {
            cleanup(candidate, failure);
            throw failure;
        }
    }

    /**
     * Rejects arbitrary external objective leaves before session preparation.
     * @param objective callback result whose ordinary provenance is inspected
     * @param prototypes declared targets
     * @throws IllegalArgumentException if a leaf cannot be bound by this Model
     */
    private void validateObjectiveLeaves(Tensor objective, List<Tensor> prototypes) {
        var visited = new java.util.IdentityHashMap<Tensor, Boolean>();
        var pending = new java.util.ArrayDeque<Tensor>();
        pending.push(objective);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.pop();
            if (visited.put(tensor, true) != null) continue;
            if (tensor.provenance().isPresent()) {
                pending.addAll(tensor.provenance().orElseThrow().inputs());
            } else if (!owned.source.inputs().contains(tensor) && !prototypes.contains(tensor)
                    && owned.source.leaves().stream().noneMatch(leaf -> leaf.tensor() == tensor)) {
                throw new IllegalArgumentException("objective contains an unbindable external leaf");
            }
        }
    }

    /**
     * Delegates one configured RESET_AND_STEP transaction.
     * @param inputs fresh batch in declared order
     * @param targetValues fresh targets in declared order
     * @return detached TrainingStep, retaining no batch storage
     */
    TrainingStep train(List<Tensor> inputs, List<Tensor> targetValues) {
        begin();
        try { return trainOpen(inputs, targetValues); }
        finally { end(); }
    }

    /**
     * Checks the single boundary under admission before constructing the ordered value lists.
     * @param input borrowed stored input
     * @param target borrowed stored target
     * @return detached Training transaction result
     */
    TrainingStep train(Tensor input, Tensor target) {
        begin();
        try {
            singleBoundary();
            if (session == null) throw new IllegalStateException("configureTraining is required");
            if (targets.size() != 1) throw new IllegalArgumentException("single target boundary required");
            return trainOpen(List.of(Objects.requireNonNull(input, "input")),
                    List.of(Objects.requireNonNull(target, "target")));
        } finally { end(); }
    }

    /**
     * Executes within admission; Training retains its overlap, live precommit and atomicity checks.
     * @param inputs declared-order batch
     * @param targetValues declared-order targets
     * @return detached step after Training closes its internal result
     */
    private TrainingStep trainOpen(List<Tensor> inputs, List<Tensor> targetValues) {
        if (session == null) throw new IllegalStateException("configureTraining is required");
        validateSource();
        List<Tensor> batch = validateValues(owned.source.inputs(), inputs, true);
        List<Tensor> targetBatch = validateValues(targets, targetValues, true);
        return session.runValues(values(batch, targetBatch, trainingSlots, trainingFixed),
                GradientMode.RESET_AND_STEP);
    }

    /**
     * Persists current owned values without an Engine, callback or state recopy.
     * @param path non-null destination
     * @param limits non-null artifact ceilings
     * @throws IOException if artifact publication fails
     */
    void save(Path path, ArtifactLimits limits) throws IOException {
        begin();
        try {
            validateSource();
            GraphCheckpoint.save(path, owned.source, limits);
        } finally { end(); }
    }

    /**
     * Resolves each final Engine/Training input once, tolerating eliminated declared slots.
     * @param inputs final compiled external metadata
     * @param targetPrototypes declared target slots (empty for inference)
     * @param slots output positions into concatenated batch/targets, or -1 for owned state
     * @param fixed output owned-state references for -1 positions
     * @throws IllegalArgumentException if compilation exposes an unmapped leaf
     */
    private void bindings(List<CompiledGraph.Input> inputs, List<Tensor> targetPrototypes,
            int[] slots, Tensor[] fixed) {
        List<Tensor> external = new ArrayList<>(owned.source.inputs());
        external.addAll(targetPrototypes);
        for (int i = 0; i < inputs.size(); i++) {
            var input = inputs.get(i);
            slots[i] = -1;
            for (int j = 0; j < external.size(); j++) {
                if (input.tensorId().equals(external.get(j).id())) slots[i] = j;
            }
            if (slots[i] >= 0) continue;
            for (var leaf : owned.source.leaves()) {
                if (input.tensorId().equals(leaf.tensor().id())) fixed[i] = leaf.tensor();
            }
            if (fixed[i] == null) throw new IllegalArgumentException("unmapped compiled Model input");
        }
    }

    /**
     * Fills the cached positional plan without rebuilding or mutating any Tensor.
     * @param batch current batch
     * @param targetBatch current targets
     * @param slots precomputed declared positions
     * @param fixed precomputed owned leaves
     * @return call-local ordered bindings borrowed only until result closure
     */
    private static List<Tensor> values(List<Tensor> batch, List<Tensor> targetBatch,
            int[] slots, Tensor[] fixed) {
        List<Tensor> values = new ArrayList<>(slots.length);
        for (int i = 0; i < slots.length; i++) {
            int slot = slots[i];
            values.add(slot < 0 ? fixed[i] : slot < batch.size()
                    ? batch.get(slot) : targetBatch.get(slot - batch.size()));
        }
        return values;
    }

    /**
     * Validates even eliminated declared values before setup/run; storage stays caller-owned.
     * @param prototypes full declared signature
     * @param values supplied values
     * @param stored whether values require usable storage
     * @return immutable structural snapshot retaining exact supplied Tensors
     * @throws IllegalArgumentException if arity, full descriptors or capacity differ
     * @throws IllegalStateException if required storage is absent, dead or inaccessible
     */
    private static List<Tensor> validateValues(List<Tensor> prototypes, List<Tensor> values, boolean stored) {
        List<Tensor> stable = List.copyOf(Objects.requireNonNull(values, "values"));
        if (stable.size() != prototypes.size()) throw new IllegalArgumentException("Model input arity mismatch");
        for (int i = 0; i < stable.size(); i++) {
            Tensor value = stable.get(i);
            TensorDescriptor descriptor = prototypes.get(i).descriptor();
            if (!descriptor.equals(value.descriptor())) throw new IllegalArgumentException("Model signature mismatch at " + i);
            long bytes = byteSize(descriptor);
            if (stored) {
                var storage = value.hostStorage().orElseThrow(() -> new IllegalStateException("input has no host storage"));
                if (!storage.isAlive() || !storage.segment().isAccessibleBy(Thread.currentThread())) {
                    throw new IllegalStateException("input storage is not alive or accessible");
                }
                if (storage.dataType() != descriptor.dataType() || storage.byteSize() < bytes) {
                    throw new IllegalArgumentException("input storage capacity/type mismatch");
                }
            }
        }
        return stable;
    }

    /**
     * Computes checked static capacity, honoring present layout spans.
     * @param descriptor fully static full descriptor
     * @return required nonnegative bytes
     * @throws IllegalArgumentException if shape is not static
     * @throws ArithmeticException if geometry overflows
     */
    private static long byteSize(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic()) throw new IllegalArgumentException("static signature required");
        long count = descriptor.layout().isPresent() ? descriptor.layout().orElseThrow().referencedElementSpan()
                : descriptor.shape().knownElementCount().orElseThrow();
        return Math.multiplyExact(count, descriptor.dataType().byteWidth());
    }

    /** @throws IllegalArgumentException if a single boundary method is used on a list-only graph */
    private void singleBoundary() {
        if (owned.source.inputs().size() != 1 || owned.source.outputs().size() != 1) {
            throw new IllegalArgumentException("single input/output boundary required; use list methods");
        }
    }

    /** @return the selected Engine, creating the model-owned CPU choice once if needed */
    private Engine engine() {
        if (engine == null) {
            engine = Engine.standard();
            ownsEngine = true;
        }
        return engine;
    }

    /**
     * Invalidates instead of recapturing externally mutated state; cleanup failures stay suppressed.
     * @throws IllegalStateException if already invalid or current source guards reject
     */
    private void validateSource() {
        if (invalid != null) throw new IllegalStateException("owned Model capability is invalid", invalid);
        try { owned.validate(); }
        catch (RuntimeException | Error failure) {
            invalid = failure;
            closeCached(failure);
            throw failure;
        }
    }

    /** @throws IllegalStateException if selected Engine, Model lifecycle or busy admission rejects */
    private synchronized void begin() {
        if (engine != null && engine.isClosed()) throw new IllegalStateException("selected Engine is closed");
        if (closing || closed) throw new IllegalStateException("Model is closed or closing");
        if (busy) throw new IllegalStateException("Model is busy");
        busy = true;
    }

    /** Releases admission and wakes a waiting close after all synchronous work/result closure. */
    private synchronized void end() {
        busy = false;
        notifyAll();
    }

    /** Closes and forgets inference before any explicit native storage replacement. */
    private void closeInference() {
        PreparedExecution previous = inference;
        inference = null;
        inferenceSlots = null;
        inferenceFixed = null;
        if (previous != null) previous.close();
    }

    /**
     * Attempts all cached-owner cleanup, preserving an existing primary failure.
     * @param primary earlier failure, or null
     * @return first failure with distinct later failures suppressed, or null
     */
    private Throwable closeCached(Throwable primary) {
        TrainingSession oldSession = session;
        PreparedExecution oldInference = inference;
        session = null;
        inference = null;
        primary = cleanup(oldSession, primary);
        return cleanup(oldInference, primary);
    }

    /**
     * Rejects new work, waits uninterruptibly with interrupt restoration, then closes in owner order.
     * @throws RuntimeException if retained cleanup failed; repeated/concurrent close replays it
     * @throws Error if retained cleanup failed fatally
     */
    void close() {
        boolean interrupted = false;
        synchronized (this) {
            while (closing && !closed) {
                try { wait(); } catch (InterruptedException ignored) { interrupted = true; }
            }
            if (closed) {
                if (interrupted) Thread.currentThread().interrupt();
                rethrow(closeFailure);
                return;
            }
            closing = true;
            while (busy) {
                try { wait(); } catch (InterruptedException ignored) { interrupted = true; }
            }
        }
        Throwable failure = closeCached(null);
        failure = cleanup(nativeArena, failure);
        if (ownsEngine) failure = cleanup(engine, failure);
        synchronized (this) {
            closeFailure = failure;
            closed = true;
            notifyAll();
        }
        if (interrupted) Thread.currentThread().interrupt();
        rethrow(failure);
    }

    /**
     * Attempts one cleanup without skipping later owners or self-suppressing.
     * @param owner nullable owner to close
     * @param primary nullable earlier failure
     * @return retained primary or first cleanup failure
     */
    private static Throwable cleanup(AutoCloseable owner, Throwable primary) {
        if (owner == null) return primary;
        try { owner.close(); }
        catch (Throwable failure) {
            if (primary == null) return failure;
            if (failure != primary) primary.addSuppressed(failure);
        }
        return primary;
    }

    /**
     * Replays the retained unchecked cleanup result.
     * @param failure nullable retained cleanup failure
     */
    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
        if (failure != null) throw new IllegalStateException("Model cleanup failed", failure);
    }
}
