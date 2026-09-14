package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.run.RunResult;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Owns one advanced CPU compile, prepare, and representation-level run lifecycle.
 *
 * <p>This low-level facade is thread-safe. It owns exactly one explicitly supplied CPU lifecycle
 * adapter and admits synchronous operations through a cold lifecycle gate. Compilation produces
 * an opaque owner-bound handle, preparation produces an opaque reusable owner-bound handle, and
 * each run creates isolated mutable Runtime state behind a lifecycle-only result. Engine closure
 * closes successful open results in reverse run order before closing the adapter. Caller storage
 * and borrowed input representations remain caller-owned.</p>
 *
 * <p>Closure rejects new work, waits uninterruptibly for admitted calls while restoring the
 * waiting thread's interrupt status, attempts all cleanup, and retains the first unchecked
 * cleanup failure. This API intentionally supplies no typed logical binding, publication access,
 * host materialization, discovery, tuning, one-shot execution, or backward convenience.</p>
 */
public final class AdvancedEngine implements AutoCloseable {
    private static final String CLOSED_MESSAGE = "advanced engine is closed";

    private enum Lifecycle { OPEN, CLOSING, CLOSED }

    private final Object lifecycleLock = new Object();
    private final EngineBackendComposition composition;
    private final PreparedExecutionRunner runner = new PreparedExecutionRunner();
    private final ArrayList<AdvancedRunResult> openResults = new ArrayList<>();
    private Lifecycle lifecycle = Lifecycle.OPEN;
    private int activeOperations;
    private Throwable closeFailure;

    /**
     * Transfers cleanup ownership of one exact CPU integration to a new Engine.
     *
     * <p>After this method succeeds, the caller must neither use nor close the supplied adapter
     * independently. Engine closure closes it exactly once after all Engine-owned open results.
     * A null argument transfers nothing.</p>
     *
     * @param cpuIntegration non-null open CPU integration whose ownership transfers on success
     * @return a new non-null open Engine owning the exact adapter
     * @throws NullPointerException if {@code cpuIntegration} is null, with message
     *     {@code cpuIntegration}
     */
    public static AdvancedEngine takeOwnership(CpuBackendIntegration cpuIntegration) {
        Objects.requireNonNull(cpuIntegration, "cpuIntegration");
        return new AdvancedEngine(new CpuEngineBackendComposition(cpuIntegration));
    }

    /**
     * Creates an Engine over one exact owned package-private composition.
     *
     * @param composition non-null composition whose ownership transfers to this Engine
     * @throws NullPointerException if {@code composition} is null
     */
    AdvancedEngine(EngineBackendComposition composition) {
        this.composition = Objects.requireNonNull(composition, "composition");
    }

    /**
     * Compiles one ordered Tensor-expression boundary with the owned CPU capability facts.
     *
     * @param mode non-null graph-scope compile mode
     * @param forwardOutputs non-null, non-empty ordered forward-output list; not mutated or
     *     retained by Engine, with nested constraints enforced by Compiler
     * @param functionalGradientRequest non-null optional functional gradient request; absent for
     *     forward-only compilation and present for a backward-capable mode
     * @param optimizationConfig non-null graph-optimization permission
     * @param backendIntent non-null backend intent
     * @param partitionScoringConfig non-null backend-neutral scoring preference
     * @return a new non-null opaque immutable handle consumable only by this exact Engine
     * @throws NullPointerException if a top-level argument is null or Compiler rejects a nested
     *     null according to its contract
     * @throws IllegalStateException if closure has begun or Compiler reports no eligible backend
     * @throws IllegalArgumentException if Compiler rejects the request
     * @throws RuntimeException if a capability provider reports another unchecked failure
     * @throws Error if inward compilation reports a fatal failure
     */
    public AdvancedCompiledGraph compile(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            Optional<FunctionalGradientRequest> functionalGradientRequest,
            GraphOptimizationConfig optimizationConfig,
            BackendIntent backendIntent,
            PartitionScoringConfig partitionScoringConfig) {
        beginOperation();
        AdvancedCompiledGraph result;
        try {
            Objects.requireNonNull(mode, "mode");
            Objects.requireNonNull(forwardOutputs, "forwardOutputs");
            Objects.requireNonNull(functionalGradientRequest, "functionalGradientRequest");
            Objects.requireNonNull(optimizationConfig, "optimizationConfig");
            Objects.requireNonNull(backendIntent, "backendIntent");
            Objects.requireNonNull(partitionScoringConfig, "partitionScoringConfig");
            CompileArtifacts artifacts = GraphCompilationPort.compile(
                    mode,
                    forwardOutputs,
                    functionalGradientRequest,
                    optimizationConfig,
                    backendIntent,
                    partitionScoringConfig,
                    composition.capabilityProviders(),
                    composition.availabilitySnapshots());
            result = new AdvancedCompiledGraph(this, artifacts);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        return finishHandle(result);
    }

    /**
     * Prepares one handle created by this Engine through the same owned CPU composition.
     * A call begun after closure first fails the lifecycle gate, before null, owner, or inward
     * preparation validation.
     *
     * @param compiledGraph non-null opaque handle created by this exact Engine
     * @return a new non-null opaque immutable reusable handle consumable only by this exact
     *     Engine and safe to share across concurrent runs
     * @throws NullPointerException if {@code compiledGraph} is null
     * @throws IllegalArgumentException if the handle belongs to another Engine or CPU/Prepare
     *     rejects zero, empty, mixed, multiple, or otherwise invalid partitions
     * @throws IllegalStateException if closure has begun or the CPU adapter is closed
     * @throws RuntimeException if preparation reports another unchecked failure
     * @throws Error if preparation reports a fatal failure
     */
    public AdvancedPreparedExecution prepare(AdvancedCompiledGraph compiledGraph) {
        beginOperation();
        AdvancedPreparedExecution result;
        try {
            Objects.requireNonNull(compiledGraph, "compiledGraph");
            requireOwner(compiledGraph.owner());
            result = new AdvancedPreparedExecution(
                    this, composition.prepare(compiledGraph.artifacts()));
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        return finishHandle(result);
    }

    /**
     * Borrows caller-owned host storage as a representation accepted by this Engine's CPU runs.
     *
     * <p>The Engine does not track or close the returned non-owning wrapper or the storage. The
     * caller must keep the wrapper, storage, and underlying memory scope usable for every run that
     * borrows them, and must close the wrapper independently when appropriate.</p>
     *
     * @param storage non-null live CPU-compatible host storage; ownership remains with caller
     * @return a new non-null CPU-compatible borrowed representation retaining the storage without
     *     taking ownership
     * @throws NullPointerException if {@code storage} is null
     * @throws IllegalArgumentException if intrinsic storage facts are unsupported
     * @throws IllegalStateException if closure has begun or storage is closed or inaccessible
     * @throws ArithmeticException if storage byte-size calculation overflows
     * @throws Error if backend work reports a fatal failure
     */
    public BufferRepresentation borrow(HostTensorStorage storage) {
        beginOperation();
        BufferRepresentation result;
        try {
            Objects.requireNonNull(storage, "storage");
            result = composition.borrow(storage);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        synchronized (lifecycleLock) {
            if (lifecycle == Lifecycle.OPEN) {
                activeOperations--;
                lifecycleLock.notifyAll();
                return result;
            }
        }
        IllegalStateException closed = closedFailure();
        try {
            result.close();
        } catch (RuntimeException | Error rollbackFailure) {
            if (rollbackFailure != closed) {
                closed.addSuppressed(rollbackFailure);
            }
        }
        finishFailure();
        throw closed;
    }

    /**
     * Runs one prepared handle synchronously with dense ordered borrowed representations.
     * The caller-input list is validated but not retained or mutated by Engine; each successful
     * call creates isolated mutable Runtime state behind the returned result.
     * A call begun after closure first fails the lifecycle gate, before argument, owner, or inward
     * Runtime validation.
     *
     * @param preparedExecution non-null prepared handle created by this exact Engine
     * @param callerInputs non-null dense ordered caller-owned representations in the prepared
     *     caller-input order; elements are non-null and remain caller-owned
     * @return a new non-null lifecycle-only result registered with this Engine; it exposes the
     *     publication count but no publication value or representation
     * @throws NullPointerException if an argument or caller-input element is null
     * @throws IllegalArgumentException if the handle belongs to another Engine or Runtime rejects
     *     binding
     * @throws IllegalStateException if closure has begun or Runtime state/action validation fails
     * @throws RuntimeException if Runtime work reports another unchecked failure
     * @throws Error if Runtime work reports a fatal failure
     */
    public AdvancedRunResult run(
            AdvancedPreparedExecution preparedExecution,
            List<BufferRepresentation> callerInputs) {
        beginOperation();
        RunResult delegate;
        try {
            Objects.requireNonNull(preparedExecution, "preparedExecution");
            Objects.requireNonNull(callerInputs, "callerInputs");
            requireOwner(preparedExecution.owner());
            for (int index = 0; index < callerInputs.size(); index++) {
                Objects.requireNonNull(callerInputs.get(index), "callerInputs[" + index + "]");
            }
            delegate = runner.run(preparedExecution.execution(), callerInputs);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        AdvancedRunResult result = new AdvancedRunResult(this, delegate);
        synchronized (lifecycleLock) {
            if (lifecycle == Lifecycle.OPEN) {
                openResults.add(result);
                activeOperations--;
                lifecycleLock.notifyAll();
                return result;
            }
        }
        IllegalStateException closed = closedFailure();
        try {
            result.close();
        } catch (RuntimeException | Error rollbackFailure) {
            if (rollbackFailure != closed) {
                closed.addSuppressed(rollbackFailure);
            }
        }
        finishFailure();
        throw closed;
    }

    /**
     * Reports whether Engine closure has begun. This is a point-in-time lifecycle observation;
     * another thread may begin closure immediately after a {@code false} result.
     *
     * @return {@code true} from the instant the first close call starts closure
     */
    public boolean isClosed() {
        synchronized (lifecycleLock) {
            return lifecycle != Lifecycle.OPEN;
        }
    }

    /**
     * Quiesces admitted calls, closes registered results in reverse successful-run order, and
     * then closes the owned composition. Concurrent/repeated callers wait for the first attempt
     * and rethrow its exact retained first cleanup failure, if any.
     *
     * @throws RuntimeException if result or composition cleanup first reports one
     * @throws Error if result or composition cleanup first reports one
     */
    @Override
    public void close() {
        boolean interrupted = false;
        synchronized (lifecycleLock) {
            if (lifecycle != Lifecycle.OPEN) {
                while (lifecycle == Lifecycle.CLOSING) {
                    try {
                        lifecycleLock.wait();
                    } catch (InterruptedException ignored) {
                        interrupted = true;
                    }
                }
                if (interrupted) {
                    Thread.currentThread().interrupt();
                }
                rethrow(closeFailure);
                return;
            }
            lifecycle = Lifecycle.CLOSING;
            while (activeOperations != 0) {
                try {
                    lifecycleLock.wait();
                } catch (InterruptedException ignored) {
                    interrupted = true;
                }
            }
        }

        Throwable failure = null;
        try {
            List<AdvancedRunResult> results;
            synchronized (lifecycleLock) {
                results = List.copyOf(openResults);
            }
            for (int index = results.size() - 1; index >= 0; index--) {
                failure = closeAndAccumulate(results.get(index), failure);
            }
            failure = closeAndAccumulate(composition, failure);
        } finally {
            synchronized (lifecycleLock) {
                closeFailure = failure;
                lifecycle = Lifecycle.CLOSED;
                lifecycleLock.notifyAll();
            }
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
        rethrow(failure);
    }

    /**
     * Removes a closed result from Engine ownership without affecting cleanup order of others.
     *
     * @param result non-null exact result whose cleanup has been claimed
     */
    void unregister(AdvancedRunResult result) {
        synchronized (lifecycleLock) {
            openResults.remove(result);
        }
    }

    private void beginOperation() {
        synchronized (lifecycleLock) {
            if (lifecycle != Lifecycle.OPEN) {
                throw closedFailure();
            }
            activeOperations++;
        }
    }

    private <T> T finishHandle(T handle) {
        synchronized (lifecycleLock) {
            activeOperations--;
            lifecycleLock.notifyAll();
            if (lifecycle != Lifecycle.OPEN) {
                throw closedFailure();
            }
            return handle;
        }
    }

    private void finishFailure() {
        synchronized (lifecycleLock) {
            activeOperations--;
            lifecycleLock.notifyAll();
        }
    }

    private void requireOwner(AdvancedEngine owner) {
        if (owner != this) {
            throw new IllegalArgumentException("handle belongs to another advanced engine");
        }
    }

    private static IllegalStateException closedFailure() {
        return new IllegalStateException(CLOSED_MESSAGE);
    }

    private static Throwable closeAndAccumulate(AutoCloseable closeable, Throwable first) {
        try {
            closeable.close();
        } catch (RuntimeException | Error next) {
            if (first == null) {
                return next;
            }
            if (next != first) {
                first.addSuppressed(next);
            }
        } catch (Exception impossible) {
            throw new AssertionError("close contract declared an unexpected checked failure", impossible);
        }
        return first;
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }
}
