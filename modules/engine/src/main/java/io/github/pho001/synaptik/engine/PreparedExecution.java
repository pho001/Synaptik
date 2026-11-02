package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.compiler.BatchBindingPlan;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.List;
import java.util.Objects;

/**
 * Immutable reusable ordinary prepared handle owned by one exact {@link Engine}.
 *
 * <p>The handle is the outward owner of one private inward Runtime preparation. Explicit close or
 * owner-Engine close terminally rejects new runs and delegates physical prepared-resource
 * lifetime to Runtime's non-waiting lease protocol. Each admitted run still receives isolated
 * Runtime state. Closing the handle does not close a completed run result or detached host value;
 * those values retain their own documented lifetimes. Originating compile metadata remains
 * readable after closure. {@link Engine#run} binds exact IDs in arbitrary order, while
 * {@link Engine#runValues} binds fresh fully static values in compiled order: exact static
 * descriptors or the qualified ordinary/Scan CPU named-B binding, including homogeneous F32/F64
 * symbolic-length Scan. Fixed nonzero feature products are checked independently of empty axes.
 * B, numeric arrays and actual
 * publication descriptors belong to each run, never this prepared owner.
 * Both retrieve/check this exact inward delegate before inspecting inputs; Runtime alone admits
 * the later run lease. Positional geometry is precomputed without rejecting preparations that
 * remain usable by identity-based run.</p>
 */
public final class PreparedExecution implements AutoCloseable {
    private final Engine owner;
    private final CompiledGraph compiledGraph;
    private final io.github.pho001.synaptik.runtime.execution.PreparedExecution execution;
    private final List<EngineBackendComposition> inputAdapters;
    private final List<EngineBackendComposition> publicationAdapters;
    private final long[] valueElementSpans;
    private final BatchBindingPlan batchPlan;
    private final int[] publicationValues;
    private final List<java.util.function.LongConsumer> batchNumericChecks;
    private boolean closed;
    private boolean cleanupComplete;
    private Throwable closeFailure;

    /**
     * Retains the exact ordinary owner, compile handle, immutable Runtime recipe, and direct
     * adapters aligned with every caller-input and publication occurrence.
     *
     * @param owner non-null exact Engine owner
     * @param compiledGraph non-null exact originating compile handle
     * @param execution non-null immutable inward prepared recipe
     * @param inputAdapters non-null exact adapters in caller-input occurrence order
     * @param publicationAdapters non-null exact adapters in publication occurrence order
     * @throws NullPointerException if an argument or indexed adapter is {@code null}
     * @throws IllegalArgumentException if the compiled handle has another owner
     */
    PreparedExecution(
            Engine owner,
            CompiledGraph compiledGraph,
            io.github.pho001.synaptik.runtime.execution.PreparedExecution execution,
            List<EngineBackendComposition> inputAdapters,
            List<EngineBackendComposition> publicationAdapters) {
        this(owner, compiledGraph, execution, inputAdapters, publicationAdapters, batchPlan(compiledGraph));
    }

    /**
     * Retains an already indexed semantic plan and preindexes CPU logical int-limit checks for
     * outer values and every scoped Scan declaration, including unused formals and latent siblings.
     * Local IDs are never deduplicated against outer IDs or another occurrence. No actual B or
     * caller storage is retained here; an invocation performs no graph walk for these checks.
     * @param owner non-null exact ordinary Engine owner
     * @param compiledGraph non-null exact originating metadata
     * @param execution non-null inward recipe, with the same explicit batch mode
     * @param inputAdapters non-null ordered direct caller adapters; membership snapshotted
     * @param publicationAdapters non-null ordered direct publication adapters; snapshotted
     * @param batchPlan immutable semantic plan, or null exactly for static execution
     * @throws NullPointerException if a required reference or adapter is null
     * @throws IllegalArgumentException if ownership, modes or dimension vocabulary disagree
     * @throws ArithmeticException if fixed CPU numeric geometry exceeds int limits
     */
    PreparedExecution(Engine owner, CompiledGraph compiledGraph,
            io.github.pho001.synaptik.runtime.execution.PreparedExecution execution,
            List<EngineBackendComposition> inputAdapters, List<EngineBackendComposition> publicationAdapters,
            BatchBindingPlan batchPlan) {
        this.owner = Objects.requireNonNull(owner, "owner");
        this.compiledGraph = Objects.requireNonNull(compiledGraph, "compiledGraph");
        this.execution = Objects.requireNonNull(execution, "execution");
        this.inputAdapters = List.copyOf(inputAdapters);
        this.publicationAdapters = List.copyOf(publicationAdapters);
        this.batchPlan = batchPlan;
        var numericChecks = new java.util.ArrayList<java.util.function.LongConsumer>();
        var numericValues = new java.util.ArrayList<>(compiledGraph.artifacts().graph().values());
        // Keep scoped values as occurrences, not a map keyed by coincident local/outer IDs.
        // Unused formals and latent sibling ports still carry numeric representability duties.
        for (var node : compiledGraph.artifacts().graph().nodes()) node.scanBody()
                .ifPresent(body -> numericValues.addAll(body.graph().values()));
        if (batchPlan != null) for (var value : numericValues) {
            var descriptor = value.descriptor();
            long[] constants = new long[descriptor.shape().rank()];
            boolean[] variable = new boolean[constants.length];
            long fixedFeatures = 1;
            for (int axis = 0; axis < constants.length; axis++) {
                var dimension = descriptor.shape().dimension(axis);
                if (dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension) variable[axis] = true;
                else constants[axis] = Math.toIntExact(dimension.staticSize().orElseThrow(
                        () -> new IllegalArgumentException("ordinary CPU requires named or fixed dimensions")));
                if (!variable[axis] && constants[axis] != 0)
                    fixedFeatures = Math.toIntExact(Math.multiplyExact(fixedFeatures,constants[axis]));
            }
            // Fixed explicit view origins/strides are checked even for an empty domain. All
            // symbolic view strides in this bounded ordinary/Scan closure come from canonical B/fixed
            // axes via permutation, insertion, removal or zero-stride broadcast.
            descriptor.layout().ifPresent(layout -> {
                Math.toIntExact(layout.storageOffset());
                for (long stride : layout.strides()) Math.toIntExact(stride);
            });
            numericChecks.add(batch -> {
                long suffix = 1;
                for (int axis = constants.length - 1; axis >= 0; axis--)
                    suffix = Math.toIntExact(Math.multiplyExact(suffix, variable[axis] ? Math.toIntExact(batch) : constants[axis]));
            });
        }
        batchNumericChecks = List.copyOf(numericChecks);
        if (execution.memoryPlan().requiresBatchSize() != (batchPlan != null)) {
            throw new IllegalArgumentException("semantic and physical batch modes disagree");
        }
        var artifacts = compiledGraph.artifacts();
        var positions = new java.util.HashMap<io.github.pho001.synaptik.model.graph.ValueId, Integer>();
        for (int i = 0; i < artifacts.graph().values().size(); i++) positions.put(artifacts.graph().values().get(i).id(), i);
        var indices = new java.util.ArrayList<Integer>();
        artifacts.publication().forwardBindings().forEach(binding -> indices.add(positions.get(binding.valueId())));
        artifacts.publication().gradientBindings().forEach(binding -> indices.add(positions.get(binding.valueId())));
        publicationValues = indices.stream().mapToInt(Integer::intValue).toArray();
        valueElementSpans = new long[compiledGraph.inputs().size()];
        for (int index = 0; index < valueElementSpans.length; index++) {
            var descriptor = compiledGraph.inputs().get(index).descriptor();
            long span = -1;
            if (descriptor.shape().isFullyStatic()) {
                try {
                    // Canonical strides as well as element/byte spans must be representable.
                    var layout = descriptor.layout().orElseGet(
                            () -> LayoutDescriptor.contiguous(descriptor.shape()));
                    span = layout.referencedElementSpan();
                    Math.multiplyExact(span, descriptor.dataType().byteWidth());
                } catch (ArithmeticException invalidGeometry) {
                    span = -1;
                }
            }
            valueElementSpans[index] = span;
        }
        if (compiledGraph.owner() != owner) {
            throw new IllegalArgumentException("compiled graph belongs to another engine");
        }
    }

    /**
     * Indexes symbolic metadata once, before backend preparation acquires resources. A scoped-only
     * symbol without any outer binding rejects rather than entering the static execution path.
     * @param graph non-null final compile handle
     * @return non-null immutable binding plan for a symbolic outer graph, or null when all outer
     *     and scoped values are static
     * @throws NullPointerException if {@code graph} is null
     * @throws IllegalArgumentException if a scoped symbol lacks an outer binding or symbolic
     *     admission cannot be represented
     */
    static BatchBindingPlan batchPlan(CompiledGraph graph) {
        boolean outer = graph.artifacts().graph().values().stream()
                .anyMatch(value -> !value.descriptor().shape().isFullyStatic());
        boolean local = graph.artifacts().graph().nodes().stream().flatMap(node -> node.scanBody().stream())
                .flatMap(body -> body.graph().values().stream()).anyMatch(value -> !value.descriptor().shape().isFullyStatic());
        if (local && !outer) throw new IllegalArgumentException("scoped batch symbol has no outer binding");
        return outer ? BatchBindingPlan.prepare(graph.artifacts()) : null;
    }

    /** @return immutable semantic plan, or null exactly for static execution */
    BatchBindingPlan batchPlan() { return batchPlan; }

    /**
     * Checks every preindexed outer and scoped logical domain, including unused declarations,
     * latent siblings and intermediate scalar-backed broadcasts. Empty leading axes do not hide
     * an overflowing trailing stride or contributor domain.
     * No descriptor, graph traversal or backend lowerer is used during this pre-borrow pass.
     * @param batch actual non-negative admitted B
     * @throws ArithmeticException if a dimension, trailing stride or count exceeds CPU int limits
     */
    void validateBatchGeometry(long batch) {
        Math.toIntExact(batch);
        for (var check : batchNumericChecks) check.accept(batch);
    }

    /**
     * Resolves only preindexed public occurrences, never eager intermediate descriptors.
     * @param binding non-null current invocation's semantic binding
     * @return immutable actual generated-value descriptors in publication occurrence order
     */
    List<TensorDescriptor> publicationDescriptors(BatchBindingPlan.Binding binding) {
        var descriptors = new java.util.ArrayList<TensorDescriptor>(publicationValues.length);
        for (int index : publicationValues) descriptors.add(binding.descriptor(index));
        return List.copyOf(descriptors);
    }

    /**
     * Returns the exact ordinary compile handle from which this recipe was prepared.
     *
     * @return the retained non-null originating handle by reference identity
     */
    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    /**
     * Returns precomputed positional geometry without restricting identity-run preparation.
     *
     * @param index zero-based final compiled input position
     * @return non-negative minimum physical element capacity for a fresh value
     * @throws IllegalArgumentException if static canonical/referenced element or byte geometry
     *     could not be proved during preparation
     */
    long valueElementSpan(int index) {
        long span = valueElementSpans[index];
        if (span < 0) {
            throw new IllegalArgumentException(
                    "input requires fully static checked element/byte geometry at position " + index);
        }
        return span;
    }

    Engine owner() {
        return owner;
    }

    /**
     * Returns the direct adapters aligned with caller-input occurrences to cold run orchestration.
     *
     * @return non-null immutable ordered adapter snapshot retained at successful preparation
     */
    List<EngineBackendComposition> inputAdapters() {
        return inputAdapters;
    }

    /**
     * Returns direct adapters aligned with result publication occurrences to result ownership.
     *
     * @return non-null immutable ordered adapter snapshot retained at successful preparation
     */
    List<EngineBackendComposition> publicationAdapters() {
        return publicationAdapters;
    }

    /**
     * Returns the inward recipe only to package-private orchestration while this handle is open.
     *
     * @return the non-null immutable inward prepared execution
     * @throws IllegalStateException if outward close has begun
     */
    synchronized io.github.pho001.synaptik.runtime.execution.PreparedExecution execution() {
        if (closed) {
            throw new IllegalStateException("prepared execution is closed");
        }
        return execution;
    }

    /**
     * Reports whether this Engine-facing handle has terminally closed.
     *
     * <p>A {@code true} result means that the inward Runtime owner has already transitioned to
     * reject new leases. An earlier run may still hold a lease, so physical release of persistent
     * prepared resources may complete later when that run releases its lease.</p>
     *
     * @return {@code true} from the terminal outward close transition
     */
    public synchronized boolean isClosed() {
        return closed;
    }

    /**
     * Closes the exact inward Runtime preparation once and unregisters this handle from its
     * Engine owner even when inward cleanup fails.
     *
     * <p>Concurrent and repeated callers wait uninterruptibly for the first cleanup attempt,
     * restore interruption before returning, and rethrow the same retained immediate cleanup
     * failure by identity. Delegate retrieval and the inward close transition share this
     * wrapper's synchronization boundary, but a complete run does not hold that boundary. A run
     * that retrieved the delegate first therefore arbitrates with close only at Runtime's unique
     * lease authority. Runtime close does not wait for an already leased run; a failure raised
     * later by deferred cleanup belongs to the lease-releasing run path and is not retroactively
     * replayed by this wrapper.</p>
     *
     * @throws RuntimeException if inward cleanup reports an unchecked failure
     * @throws Error if inward cleanup reports a fatal failure
     */
    @Override
    public void close() {
        boolean interrupted = false;
        synchronized (this) {
            if (closed) {
                while (!cleanupComplete) {
                    try {
                        wait();
                    } catch (InterruptedException ignored) {
                        interrupted = true;
                    }
                }
                if (interrupted) Thread.currentThread().interrupt();
                rethrow(closeFailure);
                return;
            }
            closed = true;
            Throwable failure = null;
            try {
                execution.close();
            } catch (RuntimeException | Error cleanupFailure) {
                failure = cleanupFailure;
            } finally {
                owner.lifecycleOwner().unregister(this);
                closeFailure = failure;
                cleanupComplete = true;
                notifyAll();
            }
            rethrow(failure);
        }
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error error) throw error;
    }
}
