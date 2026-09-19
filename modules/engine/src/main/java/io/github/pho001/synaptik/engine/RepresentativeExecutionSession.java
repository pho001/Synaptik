package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Owns one synchronous representative-input borrowing and trial-execution session.
 *
 * <p>The session snapshots every required caller storage before borrowing any representation and
 * borrows each input exactly once in the final compiled-input order. Every action delegates to
 * the stateless Runtime runner, which creates a fresh isolated run state and completes every
 * publication. The completion-only action validates the result count and closes the result
 * without inspecting payloads. Correctness capture and comparison instead copy every ordered
 * publication occurrence to fresh canonical bytes while the result lease is open, close the
 * result, and only then publish a reference or exact match/mismatch outcome.</p>
 *
 * <p>The session is deliberately package-private and single-threaded. Its Engine admission begins
 * before owner or representative-input inspection and spans all trial executions, reverse-order
 * wrapper cleanup, and the final selected or fallback preparation. Engine closure therefore waits
 * for that entire admitted operation. Caller storage remains caller-owned and must stay live,
 * accessible, and unmodified until session cleanup completes. A correctness reference is opaque,
 * owned by this session, and usable only while the same session remains open and healthy. Any
 * execution, canonical-copy, or cleanup failure invalidates the session and prevents both
 * selected preparation and fallback; a successfully completed mismatch does not.</p>
 */
final class RepresentativeExecutionSession implements AutoCloseable {
    private enum State { OPEN, POISONED, CLOSED }

    private final List<BufferRepresentation> borrowedInputs;
    private final List<CompiledGraph.PublicationSpec> publicationSpecs;
    private final List<TensorDescriptor> publicationDescriptors;
    private final int expectedPublicationCount;
    private final PreparedExecutionRunner runner;
    private final EngineBackendComposition composition;
    private final io.github.pho001.synaptik.compiler.CompileArtifacts artifacts;
    private final AdvancedEngine lifecycleOwner;
    private State state = State.OPEN;
    private boolean poisoned;
    private boolean inputsCleanupAttempted;
    private boolean admissionReleased;
    private Throwable retainedCleanupFailure;
    private Throwable representativeExecutionFailure;
    private RepresentativePlanCorrectness.Reference correctnessReference;
    private long[] correctnessByteCounts;

    /**
     * Validates and borrows one input set inside an already-admitted Engine operation.
     * The complete binding and every current storage association are snapshotted and validated
     * before the first borrow. Successfully acquired non-owning wrappers belong to this session;
     * their underlying storage remains caller-owned.
     *
     * @param lifecycleOwner non-null lifecycle owner whose admission remains held until cleanup
     *     and any final preparation complete
     * @param compiledGraph non-null owner-validated compiled graph
     * @param inputs non-null representative Tensors supplied in any order; not retained
     * @param composition non-null exact admitted Engine composition
     * @param runner non-null stateless Runtime runner
     * @throws NullPointerException if an argument or input element is null
     * @throws IllegalArgumentException if identities, descriptors, storage type, or capacity do
     *     not match the compiled contract, or if borrowing rejects a storage
     * @throws IllegalStateException if required storage is absent, dead, or inaccessible
     * @throws RuntimeException if borrowing reports another unchecked failure; a distinct rollback
     *     cleanup failure is suppressed on that exact primary failure
     * @throws Error if borrowing reports a fatal failure; a distinct rollback cleanup failure is
     *     suppressed on that exact primary failure
     */
    RepresentativeExecutionSession(
            AdvancedEngine lifecycleOwner,
            CompiledGraph compiledGraph,
            List<Tensor> inputs,
            EngineBackendComposition composition,
            PreparedExecutionRunner runner) {
        Objects.requireNonNull(compiledGraph, "compiledGraph");
        Objects.requireNonNull(inputs, "inputs");
        this.composition = Objects.requireNonNull(composition, "composition");
        this.runner = Objects.requireNonNull(runner, "runner");
        this.lifecycleOwner = Objects.requireNonNull(lifecycleOwner, "lifecycleOwner");
        artifacts = compiledGraph.artifacts();
        publicationSpecs = List.copyOf(compiledGraph.publicationSpecs());
        var descriptors = new ArrayList<TensorDescriptor>(publicationSpecs.size());
        publicationSpecs.forEach(specification -> descriptors.add(specification.descriptor));
        publicationDescriptors = List.copyOf(descriptors);
        expectedPublicationCount = publicationSpecs.size();

        List<Tensor> supplied = snapshotElements(inputs);
        List<CompiledGraph.Input> required = compiledGraph.inputs();
        Map<TensorId, CompiledGraph.Input> expected = new HashMap<>();
        required.forEach(input -> expected.put(input.tensorId(), input));
        Map<TensorId, Tensor> suppliedById = new HashMap<>();
        for (Tensor tensor : supplied) {
            TensorId id = tensor.id();
            CompiledGraph.Input input = expected.get(id);
            if (input == null) {
                throw new IllegalArgumentException("unexpected input " + id);
            }
            if (suppliedById.putIfAbsent(id, tensor) != null) {
                throw new IllegalArgumentException("duplicate input " + id);
            }
            if (!tensor.descriptor().equals(input.descriptor())) {
                throw new IllegalArgumentException("input descriptor does not match " + id);
            }
        }
        if (suppliedById.size() != required.size()) {
            throw new IllegalArgumentException(
                    "input count must equal required input count: expected="
                            + required.size() + ", actual=" + suppliedById.size());
        }

        var storages = new ArrayList<HostTensorStorage>(required.size());
        for (CompiledGraph.Input input : required) {
            Tensor tensor = suppliedById.get(input.tensorId());
            if (tensor == null) {
                throw new IllegalArgumentException("missing input " + input.tensorId());
            }
            HostTensorStorage storage = tensor.hostStorage().orElseThrow(
                    () -> new IllegalStateException(
                            "input has no host storage: " + input.tensorId()));
            storages.add(storage);
        }
        for (int index = 0; index < required.size(); index++) {
            validateStorage(required.get(index), storages.get(index));
        }

        var borrowed = new ArrayList<BufferRepresentation>(storages.size());
        try {
            for (HostTensorStorage storage : storages) {
                borrowed.add(Objects.requireNonNull(
                        this.composition.borrow(storage), "composition.borrow(storage)"));
            }
        } catch (RuntimeException | Error failure) {
            closeReverse(borrowed, failure, null);
            throw failure;
        }
        borrowedInputs = List.copyOf(borrowed);
    }

    /**
     * Reports whether one throwable is the exact failure retained from representative execution.
     * Identity, rather than equality or causal association, determines the result.
     *
     * @param failure nullable throwable to compare by identity; not retained
     * @return {@code true} only when {@code failure} is the exact retained execution failure
     */
    synchronized boolean isRepresentativeExecutionFailure(Throwable failure) {
        return representativeExecutionFailure == failure;
    }

    /**
     * Reports whether a non-execution tuning failure may still enter cleanup and fallback.
     *
     * @return {@code true} only while admission is retained and this session is open, healthy,
     *     and has recorded no representative execution failure
     */
    synchronized boolean canResolveRecoverableTuningFailure() {
        return !admissionReleased && state == State.OPEN && !poisoned
                && representativeExecutionFailure == null;
    }

    /**
     * Executes one freshly prepared complete trial recipe and consumes completion only.
     * Runtime creates fresh run-owned state and fresh result resources for this call. The session
     * validates completed publication count and closes the result without inspecting payloads.
     * A failure remains primary, triggers one reverse cleanup attempt, and prevents every later
     * trial or production preparation through this session; distinct cleanup failures are
     * suppressed in encounter order.
     *
     * @param execution non-null immutable recipe freshly prepared for this trial action
     * @throws NullPointerException if {@code execution} is null
     * @throws IllegalStateException if cleanup has begun, a prior action poisoned the session, or
     *     Runtime returns a publication count different from the compiled boundary
     * @throws RuntimeException if Runtime work or cleanup reports another unchecked failure
     * @throws Error if Runtime work or cleanup reports a fatal failure
     */
    synchronized void execute(PreparedExecution execution) {
        requireOpen();
        Objects.requireNonNull(execution, "execution");
        io.github.pho001.synaptik.runtime.run.RunResult result = null;
        boolean resultCleanupAttempted = false;
        try {
            result = runner.run(execution, borrowedInputs);
            if (result.resultCount() != expectedPublicationCount) {
                throw new IllegalStateException(
                        "Runtime result count does not match publication count: expected="
                                + expectedPublicationCount + ", actual=" + result.resultCount());
            }
            resultCleanupAttempted = true;
            try {
                result.close();
            } catch (RuntimeException | Error cleanupFailure) {
                recordCleanupFailure(cleanupFailure);
                throw cleanupFailure;
            }
        } catch (RuntimeException | Error failure) {
            poisonAfterExecutionFailure(result, resultCleanupAttempted, failure);
            throw failure;
        }
    }

    /**
     * Executes the first complete recipe and captures every ordered publication as detached
     * canonical bytes after full descriptor and aggregate-limit preflight.
     *
     * <p>All compiled publication descriptors and the aggregate limit are preflighted before the
     * supplied recipe executes once with fresh Runtime state. Every occurrence is copied
     * independently while the result lease is open, including aliases and empty payloads, and the
     * result is closed before the returned reference becomes observable. The first successful
     * call owns the sole reference for this session. Preflight rejection runs nothing and leaves
     * the session open; execution, copying, validation, allocation, or cleanup failure poisons
     * and cleans the session under the ordinary representative failure protocol.</p>
     *
     * @param execution non-null immutable complete recipe freshly prepared for this capture
     * @param maximumTotalBytes non-negative upper bound for the aggregate canonical payload bytes
     * @return the new non-null opaque reference associated only with this live session
     * @throws NullPointerException if {@code execution} is null
     * @throws IllegalArgumentException if the limit or a publication descriptor is invalid, the
     *     aggregate exceeds the limit, or no compiled publication exists
     * @throws IllegalStateException if the session is not open, a reference was already captured,
     *     Runtime returns a different count, or a copied payload has an inconsistent length
     * @throws ArithmeticException if checked descriptor or aggregate arithmetic overflows
     * @throws RuntimeException if Runtime, copying, or cleanup reports another unchecked failure
     * @throws Error if allocation, Runtime, copying, or cleanup reports a fatal failure
     */
    synchronized RepresentativePlanCorrectness.Reference captureCorrectnessReference(
            PreparedExecution execution,
            long maximumTotalBytes) {
        requireOpen();
        Objects.requireNonNull(execution, "execution");
        if (correctnessReference != null) {
            throw new IllegalStateException("correctness reference is already captured");
        }
        long[] byteCounts = AdvancedEngine.preflightCanonicalDescriptorByteCounts(
                publicationDescriptors, maximumTotalBytes);
        byte[][] publications = executeAndCopy(execution, byteCounts);
        try {
            RepresentativePlanCorrectness.Reference reference =
                    RepresentativePlanCorrectness.capture(this, publications);
            correctnessByteCounts = byteCounts;
            correctnessReference = reference;
            return reference;
        } catch (RuntimeException | Error failure) {
            poisonAfterExecutionFailure(null, true, failure);
            throw failure;
        }
    }

    /**
     * Executes one later complete recipe and compares all detached canonical publication bytes
     * with this session's sole reference.
     *
     * <p>Comparison occurs only after every occurrence has copied successfully and the Runtime
     * result has closed. Occurrence order, aliases, zero-length values, canonical byte lengths,
     * and every represented bit are significant, so signed zero and distinct NaN payloads remain
     * different. The result carries no bytes. A clean mismatch does not poison or close the
     * session.</p>
     *
     * @param reference non-null exact opaque reference captured successfully by this session
     * @param execution non-null immutable complete recipe freshly prepared for this comparison
     * @return {@link RepresentativePlanCorrectness.Comparison#MATCH} for identical occurrence
     *     lengths and bytes, otherwise
     *     {@link RepresentativePlanCorrectness.Comparison#MISMATCH}
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if {@code reference} belongs to another session
     * @throws IllegalStateException if the session is not open, no reference has been captured,
     *     the supplied reference is not this session's captured value, Runtime returns a different
     *     count, or a copied payload has an inconsistent length
     * @throws RuntimeException if Runtime, copying, or cleanup reports another unchecked failure
     * @throws Error if allocation, Runtime, copying, or cleanup reports a fatal failure
     */
    synchronized RepresentativePlanCorrectness.Comparison compareCorrectness(
            RepresentativePlanCorrectness.Reference reference,
            PreparedExecution execution) {
        requireOpen();
        Objects.requireNonNull(reference, "reference");
        Objects.requireNonNull(execution, "execution");
        if (correctnessReference == null) {
            throw new IllegalStateException("correctness reference has not been captured");
        }
        RepresentativePlanCorrectness.requireAssociation(reference, this);
        if (reference != correctnessReference) {
            throw new IllegalStateException(
                    "correctness reference is not this session's captured reference");
        }
        byte[][] publications = executeAndCopy(execution, correctnessByteCounts);
        return RepresentativePlanCorrectness.compare(reference, this, publications);
    }

    /**
     * Runs one recipe, copies every publication under its result lease, and closes that lease.
     * Any failure enters the shared poisoning and cleanup protocol before it is rethrown.
     *
     * @param execution non-null immutable complete recipe to execute once in fresh Runtime state
     * @param byteCounts non-null exact preflighted per-occurrence canonical byte counts aligned
     *     with the compiled publication snapshot; retained by neither Runtime nor backend
     * @return new private detached canonical payload arrays after successful result cleanup
     * @throws RuntimeException if execution, count validation, copying, or cleanup fails
     * @throws Error if allocation or inward work reports a fatal failure
     */
    private byte[][] executeAndCopy(PreparedExecution execution, long[] byteCounts) {
        io.github.pho001.synaptik.runtime.run.RunResult result = null;
        boolean resultCleanupAttempted = false;
        try {
            result = runner.run(execution, borrowedInputs);
            if (result.resultCount() != expectedPublicationCount) {
                throw new IllegalStateException(
                        "Runtime result count does not match publication count: expected="
                                + expectedPublicationCount + ", actual=" + result.resultCount());
            }
            byte[][] publications = new byte[expectedPublicationCount][];
            for (int index = 0; index < expectedPublicationCount; index++) {
                byte[] bytes = Objects.requireNonNull(
                        composition.copyToCanonicalHostBytes(
                                result.publicationRepresentation(index),
                                publicationSpecs.get(index).descriptor,
                                byteCounts[index]),
                        "canonicalBytes");
                if (bytes.length != byteCounts[index]) {
                    throw new IllegalStateException(
                            "backend canonical byte count does not match descriptor: expected="
                                    + byteCounts[index] + ", actual=" + bytes.length);
                }
                publications[index] = bytes;
            }
            resultCleanupAttempted = true;
            try {
                result.close();
            } catch (RuntimeException | Error cleanupFailure) {
                recordCleanupFailure(cleanupFailure);
                throw cleanupFailure;
            }
            return publications;
        } catch (RuntimeException | Error failure) {
            poisonAfterExecutionFailure(result, resultCleanupAttempted, failure);
            throw failure;
        }
    }

    /**
     * Preserves one execution-path primary while completing result, input, and admission cleanup.
     * Repeated wrapper or admission cleanup is prevented by the session's existing once-only
     * guards; distinct cleanup failures are retained and suppressed in encounter order.
     *
     * @param result nullable result whose cleanup is still required when not already attempted
     * @param resultCleanupAttempted whether result cleanup has already begun
     * @param failure non-null exact execution-path primary to retain
     */
    private void poisonAfterExecutionFailure(
            io.github.pho001.synaptik.runtime.run.RunResult result,
            boolean resultCleanupAttempted,
            Throwable failure) {
        representativeExecutionFailure = failure;
        if (result != null && !resultCleanupAttempted) {
            try {
                result.close();
            } catch (RuntimeException | Error cleanupFailure) {
                recordCleanupFailure(cleanupFailure);
                suppressDistinct(failure, cleanupFailure);
            }
        }
        poisoned = true;
        state = State.POISONED;
        try {
            cleanupInputs(failure);
        } finally {
            releaseAdmission();
        }
    }

    /**
     * Completes reverse-order wrapper cleanup and proves this session safe for final preparation.
     * Successful return is the only state from which selected or ordinary fallback preparation is
     * permitted. Earlier trial failure, cleanup failure, or released admission rejects that path.
     *
     * @throws IllegalStateException if an earlier execution failed without a retained cleanup
     *     failure, or the admission was already released
     * @throws RuntimeException if wrapper cleanup first reported an unchecked failure
     * @throws Error if wrapper cleanup first reported an error
     */
    synchronized void cleanupForProductionPreparation() {
        if (poisoned) {
            rethrow(retainedCleanupFailure);
            throw new IllegalStateException("representative execution session is poisoned");
        }
        if (admissionReleased) {
            throw new IllegalStateException(
                    "representative execution session admission is released");
        }
        if (state == State.OPEN) {
            state = State.CLOSED;
            cleanupInputs(null);
        } else if (state == State.POISONED) {
            state = State.CLOSED;
            rethrow(retainedCleanupFailure);
        }
    }

    /**
     * Performs one ordinary safe-heuristic preparation for this session's exact artifacts.
     * This method is valid only after successful representative cleanup and invokes ordinary
     * preparation once; it does not interpret a tuning failure or decide fallback policy.
     *
     * @return a fresh non-null prepared execution
     * @throws IllegalStateException if cleanup has not completed successfully, a trial or cleanup
     *     failed, or the Engine admission was released
     * @throws RuntimeException if ordinary preparation reports an unchecked failure
     * @throws Error if ordinary preparation reports a fatal failure
     */
    synchronized PreparedExecution prepareOrdinaryFallback() {
        if (admissionReleased || state != State.CLOSED || poisoned
                || retainedCleanupFailure != null) {
            throw new IllegalStateException(
                    "representative execution session is not ready for fallback");
        }
        return composition.prepare(artifacts);
    }

    /**
     * Publishes one final recipe through the retained Engine admission.
     * The admission is consumed by this attempt. If Engine closure won the race after preparation,
     * publication rejects the recipe instead of returning it for production use.
     *
     * @param preparedExecution non-null freshly prepared production recipe
     * @return the exact recipe when Engine closure has not begun
     * @throws NullPointerException if {@code preparedExecution} is null
     * @throws IllegalStateException if Engine closure won before publication
     */
    synchronized PreparedExecution completeProductionPreparation(
            PreparedExecution preparedExecution) {
        Objects.requireNonNull(preparedExecution, "preparedExecution");
        if (admissionReleased) {
            throw new IllegalStateException(
                    "representative execution session admission is released");
        }
        admissionReleased = true;
        return lifecycleOwner.finishRepresentativePreparation(preparedExecution);
    }

    /**
     * Publishes one already-complete public autotuning result through the retained admission.
     *
     * @param result non-null complete result that must not escape if closure won the race
     * @return the exact result when the Engine remains open
     * @throws NullPointerException if {@code result} is null
     * @throws IllegalStateException if admission was already released or Engine closure wins
     */
    synchronized ModelAutotuningPreparation completeModelAutotuningPreparation(
            ModelAutotuningPreparation result) {
        Objects.requireNonNull(result, "result");
        if (admissionReleased) {
            throw new IllegalStateException(
                    "representative execution session admission is released");
        }
        admissionReleased = true;
        return lifecycleOwner.finishRepresentativePreparation(result);
    }

    /**
     * Releases the one Engine admission retained by the admitted construction path.
     * Repeated release is inert; releasing it allows a waiting Engine close to continue.
     */
    synchronized void releaseAdmission() {
        if (admissionReleased) return;
        admissionReleased = true;
        if (lifecycleOwner != null) lifecycleOwner.finishRepresentativeOperation();
    }

    /**
     * Closes every borrowed wrapper once in reverse acquisition order and releases admission.
     * Caller storage is never closed. Repeated close performs no repeated cleanup and rethrows the
     * same retained cleanup failure, including its deterministic suppressed failures.
     *
     * @throws RuntimeException if wrapper cleanup first reported an unchecked failure
     * @throws Error if wrapper cleanup first reported an error
     */
    @Override
    public synchronized void close() {
        try {
            if (state == State.OPEN) {
                state = State.CLOSED;
                cleanupInputs(null);
            } else if (state == State.POISONED) {
                state = State.CLOSED;
            }
            rethrow(retainedCleanupFailure);
        } finally {
            releaseAdmission();
        }
    }

    private void requireOpen() {
        if (state != State.OPEN) {
            throw new IllegalStateException("representative execution session is not open");
        }
    }

    private void cleanupInputs(Throwable primary) {
        if (inputsCleanupAttempted) return;
        inputsCleanupAttempted = true;
        closeReverse(borrowedInputs, primary, this);
        if (primary == null) rethrow(retainedCleanupFailure);
    }

    private static void closeReverse(
            List<BufferRepresentation> borrowed,
            Throwable primary,
            RepresentativeExecutionSession session) {
        for (int index = borrowed.size() - 1; index >= 0; index--) {
            try {
                borrowed.get(index).close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (session != null) session.recordCleanupFailure(cleanupFailure);
                if (primary != null
                        && (session == null || primary != session.retainedCleanupFailure)) {
                    suppressDistinct(primary, cleanupFailure);
                }
            }
        }
    }

    private void recordCleanupFailure(Throwable failure) {
        if (retainedCleanupFailure == null) {
            retainedCleanupFailure = failure;
        } else {
            suppressDistinct(retainedCleanupFailure, failure);
        }
    }

    private static List<Tensor> snapshotElements(List<Tensor> inputs) {
        var snapshot = new ArrayList<Tensor>(inputs.size());
        for (int index = 0; index < inputs.size(); index++) {
            snapshot.add(Objects.requireNonNull(inputs.get(index), "inputs[" + index + "]"));
        }
        return List.copyOf(snapshot);
    }

    private static void validateStorage(
            CompiledGraph.Input input, HostTensorStorage storage) {
        TensorId id = input.tensorId();
        if (storage.dataType() != input.descriptor().dataType()) {
            throw new IllegalArgumentException("input storage data type does not match " + id);
        }
        input.descriptor().layout().ifPresent(layout -> {
            if (storage.elementCapacity() < layout.referencedElementSpan()) {
                throw new IllegalArgumentException(
                        "input storage capacity is insufficient for " + id);
            }
        });
        if (!storage.isAlive()) {
            throw new IllegalStateException("input storage is not alive: " + id);
        }
        if (!storage.segment().isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException(
                    "input storage is not accessible to current thread: " + id);
        }
    }

    private static void suppressDistinct(Throwable primary, Throwable next) {
        if (next == primary) return;
        for (Throwable suppressed : primary.getSuppressed()) {
            if (suppressed == next) return;
        }
        primary.addSuppressed(next);
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) throw runtimeFailure;
        if (failure instanceof Error error) throw error;
    }
}
