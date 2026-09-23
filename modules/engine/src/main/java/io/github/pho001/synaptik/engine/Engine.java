package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Owns one explicitly composed Synaptik Engine and its public lifecycle.
 *
 * <p>{@link #builder()} accepts concrete CPU and Metal integrations through entry-time ownership
 * transfer and freezes their provider and availability snapshots in registration order. Compile
 * considers every registered entry deterministically. Cold preparation validates the complete
 * non-empty owner set and required transfer directions before analysis, composes all finalized
 * partition contributions into one immutable schedule, and captures direct input/publication
 * adapters. {@link #standard()} constructs one fresh CPU integration through that same
 * composition path.</p>
 *
 * <p>The ordinary surface compiles Tensor expressions, prepares immutable reusable recipes, binds
 * logical input Tensors by identity in arbitrary order, and returns publication leases whose
 * exact occurrences can be materialized explicitly as detached immutable host values. A
 * one-shot compute convenience discovers reachable expression leaves, then performs fresh
 * compilation, preparation, execution, complete publication and aggregate-byte preflight,
 * ordered host materialization, and cleanup in one synchronous call. Forward publications retain
 * requested-output identity; gradient publications retain explicit target identity and position
 * even when roles alias inwardly. A one-shot scalar-objective backward convenience accepts an
 * explicit target list and returns detached objective and target-aligned gradient values without
 * mutating any Tensor. Implicit targets remain outside the current surface. Prepared handles are
 * explicit closeable outward owners; Engine closure is the final safety boundary for handles a
 * caller leaves open. Lifecycle observation and closure are thread-safe and inherit the owned
 * lifecycle's idempotent, failure-retaining semantics.</p>
 */
public final class Engine implements AutoCloseable {
    private final AdvancedEngine delegate;

    /**
     * Creates an empty single-use construction owner.
     *
     * @return a fresh non-null open builder owning no integrations
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Single-use owner for explicit built-in backend composition.
     *
     * <p>Each non-null integration transfers ownership at method entry, including calls made after
     * this builder is spent or closed. Registration snapshots identity, provider, and availability
     * once. A successful build atomically transfers every accepted entry to Engine. Failed build
     * is terminal and closes accepted entries in reverse registration order. Explicit close is
     * idempotent, attempt-all, and failure-retaining.</p>
     */
    public static final class Builder implements AutoCloseable {
        private enum State { OPEN, SPENT, CLOSED }

        private final ArrayList<EngineBackendRegistry.Registration> registrations =
                new ArrayList<>();
        private State state = State.OPEN;
        private Throwable closeFailure;

        private Builder() {
        }

        /**
         * Transfers ownership of one CPU integration at method entry.
         *
         * @param integration non-null integration that the caller must never use or close after
         *     this call begins, whether registration succeeds or fails
         * @return this same builder
         * @throws NullPointerException if {@code integration} is {@code null}; no ownership moves
         * @throws IllegalArgumentException if its captured identity is already registered or its
         *     provider and availability identities disagree
         * @throws IllegalStateException if this builder is spent or closed
         * @throws RuntimeException if interrogation or rejected-integration cleanup fails
         * @throws Error if interrogation or cleanup reports a fatal failure
         */
        public synchronized Builder takeOwnership(CpuBackendIntegration integration) {
            Objects.requireNonNull(integration, "integration");
            return registerTransferred(
                    integration, () -> new CpuEngineBackendComposition(integration));
        }

        /**
         * Transfers ownership of one Metal integration at method entry.
         *
         * @param integration non-null integration that the caller must never use or close after
         *     this call begins, whether registration succeeds or fails
         * @return this same builder
         * @throws NullPointerException if {@code integration} is {@code null}; no ownership moves
         * @throws IllegalArgumentException if its captured identity is already registered or its
         *     provider and availability identities disagree
         * @throws IllegalStateException if this builder is spent or closed
         * @throws RuntimeException if interrogation or rejected-integration cleanup fails
         * @throws Error if interrogation or cleanup reports a fatal failure
         */
        public synchronized Builder takeOwnership(MetalBackendIntegration integration) {
            Objects.requireNonNull(integration, "integration");
            return registerTransferred(
                    integration, () -> new MetalEngineBackendComposition(integration));
        }

        /**
         * Package-private deterministic lifecycle seam for Engine-owned adapter tests.
         *
         * @param entry non-null adapter whose ownership transfers at method entry
         * @return this same builder
         */
        synchronized Builder takeOwnership(EngineBackendComposition entry) {
            Objects.requireNonNull(entry, "entry");
            return registerTransferred(entry, () -> entry);
        }

        /**
         * Builds one Engine and atomically transfers all accepted integrations to it.
         *
         * @return a new non-null Engine owning the complete ordered registry
         * @throws IllegalArgumentException if no integration was accepted
         * @throws IllegalStateException if this builder is spent or closed
         * @throws RuntimeException if construction or reverse rollback fails
         * @throws Error if construction or rollback reports a fatal failure
         */
        public synchronized Engine build() {
            return build(registry -> new Engine(new AdvancedEngine(registry)));
        }

        /**
         * Package-private deterministic construction-failure seam.
         *
         * @param construction non-null action receiving the registry after ownership transfer
         * @return the non-null constructed Engine
         */
        synchronized Engine build(
                java.util.function.Function<EngineBackendRegistry, Engine> construction) {
            Objects.requireNonNull(construction, "construction");
            requireOpen();
            if (registrations.isEmpty()) {
                IllegalArgumentException failure =
                        new IllegalArgumentException("at least one backend integration is required");
                failBuild(failure);
                throw failure;
            }
            try {
                EngineBackendRegistry registry =
                        EngineBackendRegistry.fromCaptured(registrations);
                Engine engine = Objects.requireNonNull(
                        construction.apply(registry), "constructed engine");
                registrations.clear();
                state = State.SPENT;
                return engine;
            } catch (RuntimeException | Error failure) {
                failBuild(failure);
                throw failure;
            }
        }

        /**
         * Closes every still-owned integration in reverse registration order.
         * Repeated calls after an explicit close replay the exact first cleanup failure. Closing a
         * successfully spent builder is a no-op.
         *
         * @throws RuntimeException if cleanup first reports an unchecked failure
         * @throws Error if cleanup first reports a fatal failure
         */
        @Override
        public synchronized void close() {
            if (state == State.SPENT) return;
            if (state == State.OPEN) {
                state = State.CLOSED;
                closeFailure = closeEntries(null);
            }
            rethrow(closeFailure);
        }

        private Builder registerTransferred(
                AutoCloseable transferred,
                EntryFactory entryFactory) {
            if (state != State.OPEN) {
                IllegalStateException failure =
                        new IllegalStateException("engine builder is spent or closed");
                closeTransferred(transferred, failure);
                throw failure;
            }
            EngineBackendComposition entry = null;
            try {
                entry = Objects.requireNonNull(entryFactory.create(), "backend entry");
                EngineBackendRegistry.Registration captured =
                        EngineBackendRegistry.capture(entry);
                for (EngineBackendRegistry.Registration accepted : registrations) {
                    if (accepted.backendId().equals(captured.backendId())) {
                        throw new IllegalArgumentException(
                                "duplicate backend ID: " + captured.backendId().value());
                    }
                }
                registrations.add(captured);
                return this;
            } catch (RuntimeException | Error failure) {
                closeTransferred(entry == null ? transferred : entry, failure);
                throw failure;
            }
        }


        private void requireOpen() {
            if (state != State.OPEN) {
                throw new IllegalStateException("engine builder is spent or closed");
            }
        }

        private void failBuild(Throwable failure) {
            state = State.CLOSED;
            closeEntries(failure);
        }

        private Throwable closeEntries(Throwable primary) {
            Throwable first = primary;
            for (int index = registrations.size() - 1; index >= 0; index--) {
                try {
                    registrations.get(index).adapter().close();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (first == null) first = cleanupFailure;
                    else if (cleanupFailure != first) first.addSuppressed(cleanupFailure);
                }
            }
            registrations.clear();
            if (primary == null) closeFailure = first;
            return first;
        }

        private static void closeTransferred(AutoCloseable transferred, Throwable primary) {
            try {
                transferred.close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != primary) primary.addSuppressed(cleanupFailure);
            } catch (Exception checkedFailure) {
                primary.addSuppressed(new IllegalStateException(
                        "backend integration cleanup reported a checked failure", checkedFailure));
            }
        }

        private static void rethrow(Throwable failure) {
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
        }


        @FunctionalInterface
        private interface EntryFactory {
            EngineBackendComposition create();
        }
    }

    /**
     * Opens a new independent CPU Engine through the explicit builder path.
     *
     * <p>Every call opens one fresh default CPU integration without discovery or global reuse and
     * transfers it at entry to a fresh builder. Construction and rollback therefore use the same
     * ownership, identity, snapshot, and terminal-failure rules as explicit composition.</p>
     *
     * @return a new non-null open Engine owning an independent CPU-only composition
     * @throws RuntimeException if CPU opening or Engine construction fails
     * @throws Error if construction or rollback reports a fatal failure
     */
    public static Engine standard() {
        return builder().takeOwnership(CpuBackendIntegration.open()).build();
    }

    /**
     * Creates an ordinary Engine that takes cleanup ownership of one exact lifecycle owner.
     *
     * @param delegate non-null lifecycle owner whose cleanup ownership transfers on success;
     *     this constructor does not inspect its current lifecycle state
     * @throws NullPointerException if {@code delegate} is null, with message {@code delegate}
     */
    Engine(AdvancedEngine delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate");
    }

    AdvancedEngine lifecycleOwner() {
        return delegate;
    }

    /**
     * Compiles one non-empty identity-unique ordered forward boundary with standard settings.
     *
     * @param forwardOutputs non-null non-empty list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained
     * @return a fresh non-null owner-bound immutable compile handle
     * @throws NullPointerException if the list or an element is {@code null}
     * @throws IllegalArgumentException if the list is empty, repeats an exact Tensor, or Compiler
     *     rejects the graph
     * @throws IllegalStateException if closure has begun or no backend is eligible
     * @throws RuntimeException if inward compilation reports another unchecked failure
     * @throws Error if inward compilation reports a fatal failure
     */
    public CompiledGraph compile(List<Tensor> forwardOutputs) {
        return delegate.compileOrdinary(this, forwardOutputs);
    }

    /**
     * Compiles one explicitly seeded first-order reverse-mode request with standard settings.
     * Seeds align one-for-one with forward outputs and may repeat or be expressions; targets are
     * never inferred.
     *
     * @param forwardOutputs non-null non-empty identity-unique ordered forward outputs
     * @param cotangentSeeds non-null output-aligned explicit non-null cotangent seeds
     * @param targets non-null non-empty identity-unique ordered differentiation targets
     * @return a fresh non-null owner-bound immutable compile handle
     * @throws NullPointerException if a list or element is {@code null}
     * @throws IllegalArgumentException if local structure or Compiler semantics reject the request
     * @throws IllegalStateException if closure has begun or no backend is eligible
     * @throws RuntimeException if inward compilation reports another unchecked failure
     * @throws Error if inward compilation reports a fatal failure
     */
    public CompiledGraph compile(
            List<Tensor> forwardOutputs,
            List<Tensor> cotangentSeeds,
            List<Tensor> targets) {
        return delegate.compileOrdinary(this, forwardOutputs, cotangentSeeds, targets);
    }

    /**
     * Prepares one compile handle created by this exact Engine. Preparation requires a non-empty
     * plan whose owners exactly match registered adapters. It accepts single-owner plans and mixed
     * CPU/Metal plans in the bounded positive rank-1..16 static canonical contiguous FLOAT32
     * transfer domain; unsupported ownership, direction, layout, type, or geometry may therefore
     * reject an artifact that compiled successfully before backend analysis.
     *
     * <p>The returned handle owns the exact inward Runtime preparation and must be closed when
     * reuse ends, preferably with try-with-resources. Closing this Engine closes any retained
     * handle the caller leaves open, after all retained results and before backend composition.</p>
     *
     * @param compiledGraph non-null owner-bound compile handle from this Engine
     * @return a fresh non-null immutable reusable closeable prepared handle owned by this Engine
     * @throws NullPointerException if {@code compiledGraph} is {@code null}
     * @throws IllegalArgumentException if ownership or inward preparation is invalid
     * @throws IllegalStateException if closure has begun
     * @throws RuntimeException if inward preparation reports another unchecked failure
     * @throws Error if inward preparation reports a fatal failure
     */
    public PreparedExecution prepare(CompiledGraph compiledGraph) {
        return delegate.prepareOrdinary(this, compiledGraph);
    }

    /**
     * Performs one bounded two-phase CPU autotuning transaction and returns a fresh production
     * preparation, or the explicitly permitted safe heuristic fallback.
     *
     * <p>Phase 1 selects and authenticates the CPU local-workload decision. Phase 2 holds that
     * decision fixed while it checks every complete-plan candidate for exact canonical-byte
     * correctness before any warmup or timed execution. After authenticating the complete-plan
     * winner and cleaning representative resources, Engine freshly prepares only that winner;
     * no trial preparation becomes production state. When the request explicitly allows safe
     * fallback, a recoverable tuning failure instead produces one fresh ordinary heuristic
     * preparation and reports the fallback outcome without tuning evidence. Every trial result
     * closes before its temporary preparation; the returned preparation contains the sole
     * closeable production owner.</p>
     *
     * @param compiledGraph non-null compile handle created by this exact Engine
     * @param request non-null configuration, identity, and live representative inputs
     * @return a complete non-null preparation whose {@link
     *     ModelAutotuningPreparation#preparedExecution()} is the sole closeable production owner,
     *     plus immutable outcome metadata
     * @throws NullPointerException if an argument or required request value is null
     * @throws IllegalArgumentException if ownership, representative binding, or tuning evidence
     *     is invalid
     * @throws IllegalStateException if closure has begun or required tuning cannot complete
     * @throws RuntimeException if tuning, preparation, execution, or cleanup fails
     * @throws Error if inward work or cleanup reports a fatal failure
     */
    public ModelAutotuningPreparation prepareTuned(
            CompiledGraph compiledGraph, ModelAutotuningRequest request) {
        return delegate.prepareTunedOrdinary(this, compiledGraph, request);
    }

    /**
     * Runs one prepared recipe using current caller-owned host associations matched by Tensor ID.
     *
     * <p>The input list may use any order. Associations are snapshotted once after complete
     * logical validation; caller storage must remain usable until the returned result closes.
     * A handle whose close has begun rejects the run. A run already admitted by Runtime may
     * complete while handle close returns without waiting; its lease performs any deferred
     * persistent-resource cleanup.</p>
     *
     * @param preparedExecution non-null prepared handle created by this exact Engine
     * @param inputs non-null list supplying every required logical Tensor exactly once
     * @return a fresh non-null publication lease supporting metadata access and explicit
     *     per-occurrence host materialization while open
     * @throws NullPointerException if an argument or input element is {@code null}
     * @throws IllegalArgumentException if ownership, identity, descriptor, storage geometry, or
     *     inward binding is invalid
     * @throws IllegalStateException if closure has begun or storage is absent, dead, inaccessible,
     *     or inward execution state is invalid
     * @throws RuntimeException if inward execution reports another unchecked failure
     * @throws Error if inward execution reports a fatal failure
     */
    public RunResult run(PreparedExecution preparedExecution, List<Tensor> inputs) {
        return delegate.runOrdinary(this, preparedExecution, inputs);
    }

    /**
     * Computes one forward output without an aggregate caller byte limit.
     *
     * <p>This overload delegates to {@link #compute(Tensor, long)} with
     * {@link Long#MAX_VALUE}. Checked aggregate arithmetic, the per-value JVM array ceiling, and
     * all descriptor, storage, and inward execution constraints still apply.</p>
     *
     * @param output non-null forward output Tensor
     * @return a fresh non-null detached immutable host value
     * @throws NullPointerException if {@code output} is null
     * @throws IllegalArgumentException if output metadata is invalid, a payload exceeds the JVM
     *     array ceiling, or an inward compile, prepare, binding, or selected-backend validation
     *     rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical byte-count arithmetic overflows
     * @throws RuntimeException if inward execution, copying, or cleanup reports another unchecked
     *     failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public HostTensorValue compute(Tensor output) {
        return compute(output, Long.MAX_VALUE);
    }

    /**
     * Computes one forward output through a fresh complete lifecycle and returns its detached
     * canonical host value.
     *
     * <p>This is the singleton specialization of the ordered-output overload. One Engine
     * admission spans argument validation, transient iterative discovery of provenance-free
     * leaves, compilation, Compiler-authoritative input selection, preparation, logical input
     * binding, execution, complete publication and aggregate-size preflight, copying, and
     * temporary-result cleanup. Discovery follows immutable Tensor provenance by exact object
     * identity; final {@link CompiledGraph#inputs()} metadata alone determines which discovered
     * leaves are bound and in what order. No Tensor or provenance state is retained after the
     * synchronous call. The call does not cache or reuse a compiled or prepared recipe; callers
     * performing repeated runs or selective materialization should use {@link #compile(List)},
     * {@link #prepare(CompiledGraph)}, {@link #run(PreparedExecution, List)}, and
     * {@link RunResult#materialize}.</p>
     *
     * <p>The byte limit covers only the returned canonical payload, not inputs, intermediate or
     * peak memory, recipes, object overhead, or backend workspace. The caller retains ownership
     * of selected leaf host storage and must keep it live, accessible, and free from conflicting
     * mutation through synchronous completion. Storage of an unselected discovered leaf is not
     * inspected. The returned value is immutable, owns no closeable
     * resource, and remains readable after this Engine closes. On failure, no value is returned;
     * temporary-result cleanup still runs, and a distinct cleanup failure is suppressed on the
     * primary failure.</p>
     *
     * @param output non-null forward output Tensor
     * @param maximumTotalBytes non-negative aggregate upper bound, in bytes, for the returned
     *     canonical payload
     * @return a fresh non-null detached immutable host value
     * @throws NullPointerException if {@code output} is null
     * @throws IllegalArgumentException if the byte limit is negative, logical binding or output
     *     metadata is invalid, the payload exceeds the limit or JVM array ceiling, or an inward
     *     compile, prepare, binding, or selected-backend validation rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical byte-count arithmetic overflows
     * @throws RuntimeException if inward execution, copying, or cleanup reports another unchecked
     *     failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public HostTensorValue compute(Tensor output, long maximumTotalBytes) {
        return delegate.computeOrdinary(this, output, maximumTotalBytes);
    }

    /**
     * Computes an ordered non-empty forward boundary without an aggregate caller byte limit.
     *
     * <p>This overload delegates to {@link #compute(List, long)} with
     * {@link Long#MAX_VALUE}. Checked aggregate arithmetic, each value's JVM array ceiling, and all
     * descriptor, storage, and inward execution constraints still apply.</p>
     *
     * @param outputs non-null non-empty ordered list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained or mutated
     * @return a fresh non-null immutable ordered list of detached immutable host values
     * @throws NullPointerException if {@code outputs} or an output element is null
     * @throws IllegalArgumentException if outputs are empty or repeat an exact Tensor, output
     *     metadata is invalid, an individual payload exceeds the JVM array ceiling, or inward
     *     validation rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical or aggregate byte-count arithmetic overflows
     * @throws RuntimeException if inward compilation, preparation, execution, copying, or cleanup
     *     reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public List<HostTensorValue> compute(List<Tensor> outputs) {
        return compute(outputs, Long.MAX_VALUE);
    }

    /**
     * Computes an ordered non-empty forward boundary through one fresh complete lifecycle and
     * returns detached canonical host values in exact requested publication order.
     *
     * <p>One Engine admission spans argument validation, one transient identity-safe inventory of
     * reachable provenance-free leaves, one compilation, Compiler-authoritative ordered input
     * selection, one preparation, one run, complete
     * aggregate-size preflight before any physical copy, one copy per forward occurrence, and
     * temporary-result cleanup. Inventory traversal is iterative and carries no liveness or input
     * ordering meaning; {@link CompiledGraph#inputs()} supplies both final membership and order.
     * Unselected leaves are ignored without storage access, and no Tensor or provenance reference
     * escapes the call. The returned list and values are immutable and lifecycle-independent.
     * Distinct occurrences are copied independently even if they alias an inward representation.
     * No compilation, preparation, result, Tensor-inventory, or byte cache is created. A
     * failure returns no partial list; temporary-result cleanup still runs, and a distinct cleanup
     * failure is suppressed on the primary failure.</p>
     *
     * <p>The aggregate byte limit covers only the sum of canonical returned payload lengths, not
     * inputs, Runtime buffers or workspaces, recipes, object overhead, defensive copies, peak
     * memory, or other allocation. Selected leaf storage remains caller-owned and must stay live,
     * accessible, and free from conflicting mutation through synchronous completion. Execution
     * uses the complete captured mixed-owner schedule; each host copy uses the adapter captured
     * for that publication and requires a fully static resolved final layout.</p>
     *
     * @param outputs non-null non-empty ordered list of non-null identity-unique output Tensors;
     *     the container is snapshotted and not retained or mutated
     * @param maximumTotalBytes non-negative aggregate upper bound, in bytes, for the sum of all
     *     returned canonical payloads
     * @return a fresh non-null immutable ordered list of detached immutable host values
     * @throws NullPointerException if {@code outputs} or an output element is null
     * @throws IllegalArgumentException if outputs are empty or repeat an exact Tensor, the byte
     *     limit is negative, logical binding or output metadata is invalid, the aggregate exceeds
     *     the limit, an individual payload exceeds the JVM array ceiling, or inward validation
     *     rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected caller
     *     storage is absent, dead, or inaccessible, or result metadata is inconsistent
     * @throws ArithmeticException if checked logical or aggregate byte-count arithmetic overflows
     * @throws RuntimeException if inward compilation, preparation, execution, copying, or cleanup
     *     reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public List<HostTensorValue> compute(
            List<Tensor> outputs, long maximumTotalBytes) {
        return delegate.computeOrdinary(this, outputs, maximumTotalBytes);
    }

    /**
     * Computes one scalar objective and its first derivatives for explicit ordered targets.
     *
     * <p>One Engine admission spans structural validation, transient identity-safe discovery of
     * reachable provenance-free leaves, exactly one Compiler functional-gradient request,
     * Compiler-authoritative input selection, fresh preparation, one run, publication validation,
     * complete aggregate byte preflight, independent objective-then-gradient copies, and cleanup.
     * The Compiler receives one absent cotangent seed and
     * {@link io.github.pho001.synaptik.compiler.FunctionalGradientRequest.DisconnectedPolicy#ERROR};
     * it alone validates scalar, floating, gradient, connectivity, and derivative semantics.
     * Callers needing explicit seeds, multiple outputs, another policy, reuse, or the full request
     * surface should use the reusable ordinary compile lifecycle or {@link AdvancedEngine}.</p>
     *
     * <p>The limit covers the canonical bytes of the objective and every gradient, not inputs,
     * recipes, Runtime resources, object overhead, or peak memory. All descriptor sizes and the
     * complete sum are checked before any physical copy. Selected input storage remains
     * caller-owned and must remain live, accessible, and free from conflicting mutation through
     * synchronous cleanup. The returned carrier and values own no closeable resource and remain
     * readable after Engine and caller storage closure. Each call compiles and prepares afresh and
     * creates no cache or retained Tensor state.</p>
     *
     * @param objective non-null scalar floating gradient-eligible forward Tensor; semantic
     *     eligibility is validated by Compiler
     * @param targets non-null non-empty ordered list of non-null exact-object-identity-unique
     *     differentiation targets; membership is snapshotted and target order defines gradient
     *     order
     * @param maximumTotalBytes non-negative aggregate upper bound, in bytes, for the detached
     *     objective and all gradient payloads
     * @return a fresh non-null immutable detached objective and target-aligned gradient result
     * @throws NullPointerException if the objective, target list, or an indexed target is null
     * @throws IllegalArgumentException if targets are empty or repeat an exact Tensor, the limit
     *     is negative, Compiler rejects scalar/gradient/connectivity semantics, result metadata is
     *     unsuitable for host copying, an individual payload exceeds the JVM array ceiling, the
     *     aggregate exceeds the limit, or inward validation rejects the request
     * @throws IllegalStateException if Engine closure has begun, reachable Tensor identity is
     *     inconsistent, an authoritative compiled input has no reachable leaf, selected storage
     *     is absent/dead/inaccessible, or publication metadata is inconsistent
     * @throws ArithmeticException if checked logical or aggregate byte arithmetic overflows
     * @throws RuntimeException if inward compilation, preparation, execution, copying, or cleanup
     *     reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    public ScalarObjectiveBackwardResult backward(
            Tensor objective, List<Tensor> targets, long maximumTotalBytes) {
        return delegate.backwardOrdinary(this, objective, targets, maximumTotalBytes);
    }

    /** Package-private focused seam proving prepared ordinary work performs no registry lookup. */
    void poisonBackendLookupForTesting() {
        delegate.poisonBackendLookupForTesting();
    }

    /**
     * Reports whether closure of this Engine has begun.
     *
     * <p>This is a thread-safe point-in-time observation; another thread may begin closure
     * immediately after a {@code false} result.</p>
     *
     * @return {@code true} from the instant the first close call starts closure
     */
    public boolean isClosed() {
        return delegate.isClosed();
    }

    /**
     * Closes the owned composition through its concurrent, idempotent lifecycle protocol.
     * The first close rejects new work, waits for admitted Engine operations, then attempts all
     * retained results in reverse run-publication order, all retained prepared handles in reverse
     * prepare-publication order, and the backend composition. Repeated and concurrent callers
     * wait for that attempt and observe its exact retained first failure, if any.
     *
     * @throws RuntimeException if owned cleanup reports an unchecked failure
     * @throws Error if owned cleanup reports a fatal failure
     */
    @Override
    public void close() {
        delegate.close();
    }
}
