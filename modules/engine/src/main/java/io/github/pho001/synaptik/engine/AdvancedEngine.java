package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuLocalWorkloadTuning;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningCandidateBatch;
import io.github.pho001.synaptik.prepare.analysis.BackendTuningDecision;
import io.github.pho001.synaptik.tools.tuning.BackendWorkloadTuning;
import io.github.pho001.synaptik.tools.tuning.WorkloadTuning;
import io.github.pho001.synaptik.tools.tuning.WorkloadTuningRequest;
import io.github.pho001.synaptik.tools.tuning.WorkloadTuningResult;

/**
 * Owns one advanced CPU compile, prepare, and representation-level run lifecycle.
 *
 * <p>This low-level facade is thread-safe. It owns exactly one explicitly supplied CPU lifecycle
 * adapter and admits synchronous operations through a cold lifecycle gate. Compilation produces
 * an opaque owner-bound handle, preparation produces an opaque reusable owner-bound handle, and
 * each run creates isolated mutable Runtime state behind a lifecycle-only result. Engine closure
 * closes successful open results in reverse run order before closing the adapter. Caller storage
 * and advanced caller-created borrowed input representations remain caller-owned. Ordinary runs
 * reuse this lifecycle owner privately but transfer cleanup of their Engine-created non-owning
 * wrappers to the registered result; the wrapped storage never transfers.</p>
 *
 * <p>Closure rejects new work, waits uninterruptibly for admitted calls while restoring the
 * waiting thread's interrupt status, attempts all cleanup, and retains the first unchecked
 * cleanup failure. This advanced API intentionally supplies no typed logical binding, publication
 * access, host materialization, discovery, tuning, or backward convenience. The ordinary owner
 * privately reuses this lifecycle gate for transient Tensor-expression leaf discovery, fresh
 * compile, authoritative input selection, prepare, run, complete publication and aggregate-byte
 * preflight, ordered materialization, and cleanup without widening this advanced public
 * surface.</p>
 */
public final class AdvancedEngine implements AutoCloseable {
    private static final String CLOSED_MESSAGE = "advanced engine is closed";

    private enum Lifecycle { OPEN, CLOSING, CLOSED }

    private final Object lifecycleLock = new Object();
    private final EngineBackendComposition composition;
    private final ModelAutotuningTuning<?, ?, ?> tuningOverride;
    private final PreparedExecutionRunner runner = new PreparedExecutionRunner();
    private final ArrayList<AdvancedRunResult> openResults = new ArrayList<>();
    private Lifecycle lifecycle = Lifecycle.OPEN;
    private int activeOperations;
    private Throwable closeFailure;

    /** Narrow typed cold-tuning seam implemented by CPU production composition and focused tests. */
    interface ModelAutotuningTuning<C extends BackendTuningCandidateBatch,
            D extends BackendTuningDecision, K> extends BackendWorkloadTuning<C, D, K> {
        /**
         * Obtains the optional CPU-local candidate handoff for the exact compile artifacts.
         * @param artifacts non-null immutable artifacts for the admitted compiled graph
         * @return a non-null optional containing the sole eligible handoff, or empty when tuning
         *     is unavailable
         */
        Optional<BackendPartitionTuningHandoff<C, D>> candidateHandoff(CompileArtifacts artifacts);
        /**
         * Freshly prepares one complete trial recipe without starting Runtime execution.
         * @param batch non-null backend-owned compatible candidate batch
         * @param candidate non-null opaque candidate from that batch
         * @return a fresh non-null complete trial recipe
         * @throws Exception if cold trial preparation cannot complete
         */
        io.github.pho001.synaptik.runtime.execution.PreparedExecution prepareTrial(C batch, K candidate)
                throws Exception;
        /**
         * Freshly prepares the final production recipe for an authenticated decision.
         * @param batch non-null original backend-owned candidate batch
         * @param decision non-null authenticated selected decision
         * @return a fresh non-null production recipe
         * @throws RuntimeException if selected preparation fails
         * @throws Error if selected preparation reports a fatal failure
         */
        io.github.pho001.synaptik.runtime.execution.PreparedExecution prepareSelected(C batch, D decision);
    }

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
        this(composition, null);
    }

    /**
     * Creates an Engine with a package-private focused tuning collaboration override.
     * The override is retained without access and is first used only after representative-session
     * construction under an existing admission.
     *
     * @param composition non-null owned Engine composition
     * @param tuningOverride optional focused typed collaboration used instead of CPU production
     *     adaptation; retained without invoking it
     */
    AdvancedEngine(
            EngineBackendComposition composition,
            ModelAutotuningTuning<?, ?, ?> tuningOverride) {
        this.composition = Objects.requireNonNull(composition, "composition");
        this.tuningOverride = tuningOverride;
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
        io.github.pho001.synaptik.runtime.run.RunResult delegate;
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

    CompiledGraph compileOrdinary(Engine owner, List<Tensor> forwardOutputs) {
        beginOperation();
        CompiledGraph result;
        try {
            Objects.requireNonNull(owner, "owner");
            List<Tensor> outputs = snapshotIdentityUnique(
                    forwardOutputs, "forwardOutputs", true);
            result = compileForwardOrdinaryOpen(owner, outputs);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        return finishHandle(result);
    }

    CompiledGraph compileOrdinary(
            Engine owner,
            List<Tensor> forwardOutputs,
            List<Tensor> cotangentSeeds,
            List<Tensor> targets) {
        beginOperation();
        CompiledGraph result;
        try {
            Objects.requireNonNull(owner, "owner");
            List<Tensor> outputs = snapshotIdentityUnique(
                    forwardOutputs, "forwardOutputs", true);
            List<Tensor> seeds = snapshotElements(cotangentSeeds, "cotangentSeeds");
            List<Tensor> targetSnapshot = snapshotIdentityUnique(targets, "targets", true);
            if (seeds.size() != outputs.size()) {
                throw new IllegalArgumentException(
                        "cotangentSeeds size must equal forwardOutputs size");
            }
            var outputReferences = outputs.stream()
                    .map(FunctionalGradientRequest.ForwardTensorReference::new)
                    .map(reference -> (FunctionalGradientRequest.OutputReference) reference)
                    .toList();
            var seedOptionals = seeds.stream().map(Optional::of).toList();
            var stage = new FunctionalGradientRequest.Stage(
                    outputReferences,
                    seedOptionals,
                    targetSnapshot,
                    false,
                    FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            CompileArtifacts artifacts = GraphCompilationPort.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    outputs,
                    Optional.of(new FunctionalGradientRequest(List.of(stage))),
                    GraphOptimizationConfig.standard(),
                    BackendIntent.unconstrained(),
                    PartitionScoringConfig.neutral(),
                    composition.capabilityProviders(),
                    composition.availabilitySnapshots());
            result = new CompiledGraph(owner, artifacts);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        return finishHandle(result);
    }

    io.github.pho001.synaptik.engine.PreparedExecution prepareOrdinary(
            Engine owner, CompiledGraph compiledGraph) {
        beginOperation();
        io.github.pho001.synaptik.engine.PreparedExecution result;
        try {
            Objects.requireNonNull(owner, "owner");
            Objects.requireNonNull(compiledGraph, "compiledGraph");
            requireOrdinaryOwner(owner, compiledGraph.owner());
            result = prepareOrdinaryOpen(owner, compiledGraph);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        return finishHandle(result);
    }

    io.github.pho001.synaptik.engine.RunResult runOrdinary(
            Engine owner,
            io.github.pho001.synaptik.engine.PreparedExecution preparedExecution,
            List<Tensor> inputs) {
        beginOperation();
        OrdinaryRun ordinaryRun;
        try {
            ordinaryRun = runOrdinaryOpen(owner, preparedExecution, inputs);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
        synchronized (lifecycleLock) {
            if (lifecycle == Lifecycle.OPEN) {
                openResults.add(ordinaryRun.owner());
                activeOperations--;
                lifecycleLock.notifyAll();
                return ordinaryRun.result();
            }
        }
        IllegalStateException closed = closedFailure();
        try {
            ordinaryRun.owner().close();
        } catch (RuntimeException | Error rollbackFailure) {
            suppressDistinct(closed, rollbackFailure);
        }
        finishFailure();
        throw closed;
    }

    /**
     * Admits one synchronous representative-execution session after checking exact ownership.
     * Engine lifecycle admission precedes owner and representative-input inspection; owner
     * validation then precedes representative-input inspection. The returned package-private
     * session retains that one admission across every trial, reverse cleanup, and the final
     * selected or fallback preparation, so Engine closure waits for the complete operation.
     *
     * @param owner non-null ordinary Engine facade owning {@code compiledGraph}
     * @param compiledGraph non-null compiled graph created by {@code owner}
     * @param representativeInputs representative Tensors validated only after admission and owner
     *     checks
     * @return a non-null admitted session that must be closed if not completed by a final
     *     preparation boundary
     * @throws NullPointerException if an argument or representative element is null
     * @throws IllegalArgumentException if the compiled graph belongs to another ordinary Engine
     * @throws IllegalStateException if Engine closure has begun or representative storage is
     *     absent, dead, or inaccessible
     * @throws RuntimeException if binding, borrowing, or rollback reports another failure
     * @throws Error if binding, borrowing, or rollback reports a fatal failure
     */
    RepresentativeExecutionSession openRepresentativeExecutionSession(
            Engine owner,
            CompiledGraph compiledGraph,
            List<Tensor> representativeInputs) {
        beginOperation();
        try {
            Objects.requireNonNull(owner, "owner");
            Objects.requireNonNull(compiledGraph, "compiledGraph");
            requireOrdinaryOwner(owner, compiledGraph.owner());
            return new RepresentativeExecutionSession(
                    this, compiledGraph, representativeInputs, composition, runner);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }
    }

    /**
     * Completes the exact admission retained by one representative session.
     * This may release an Engine close that is waiting for the synchronous session.
     */
    void finishRepresentativeOperation() {
        finishFailure();
    }

    /**
     * Publishes one final representative-session recipe under the retained admission.
     *
     * @param execution non-null fresh selected or fallback production recipe
     * @return the exact recipe if Engine closure has not begun
     * @throws NullPointerException if {@code execution} is null
     * @throws IllegalStateException if Engine closure won the final publication race
     */
    io.github.pho001.synaptik.runtime.execution.PreparedExecution
            finishRepresentativePreparation(
                    io.github.pho001.synaptik.runtime.execution.PreparedExecution execution) {
        return finishHandle(execution);
    }

    /** Publishes a complete public autotuning result through one retained admission. */
    ModelAutotuningPreparation finishRepresentativePreparation(
            ModelAutotuningPreparation preparation) {
        return finishHandle(preparation);
    }

    /**
     * Admits one ordinary public tuning request before any argument, adapter, or composition
     * inspection. Owner/request validation and already-admitted representative-session
     * construction precede CPU adapter construction and acquisition. Adapter-setup failure keeps
     * its exact throwable primary, closes representative wrappers, suppresses a distinct cleanup
     * failure once, and releases the sole admission without entering fallback.
     *
     * @param owner raw ordinary facade reference forwarded by {@link Engine}
     * @param compiledGraph raw owner-bound compile handle reference
     * @param request raw public tuning request reference
     * @return one completely constructed selected or safe-fallback preparation
     * @throws RuntimeException if lifecycle, validation, tuning, preparation, publication, or
     *     cleanup reports an unchecked failure
     * @throws Error if inward work or cleanup reports a fatal failure
     */
    ModelAutotuningPreparation prepareTunedOrdinary(
            Engine owner, CompiledGraph compiledGraph, ModelAutotuningRequest request) {
        beginOperation();
        return prepareTunedOrdinaryAlreadyAdmitted(
                owner, compiledGraph, request, tuningOverride);
    }

    /**
     * Focused typed test entry with the same first-statement admission and exact orchestration
     * body as production. The supplied collaboration is not accessed until validation and
     * representative-session construction succeed.
     *
     * @param owner raw ordinary facade reference
     * @param compiledGraph raw compile handle reference
     * @param request raw public request reference
     * @param tuning non-null typed synthetic collaboration, retained until post-session use
     * @param <C> opaque candidate-batch type
     * @param <D> opaque selected-decision type
     * @param <K> opaque candidate type
     * @return one completely constructed selected or safe-fallback preparation
     * @throws RuntimeException if lifecycle, validation, tuning, preparation, publication, or
     *     cleanup reports an unchecked failure
     * @throws Error if inward work or cleanup reports a fatal failure
     */
    <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision, K>
            ModelAutotuningPreparation prepareTunedOrdinary(
                    Engine owner,
                    CompiledGraph compiledGraph,
                    ModelAutotuningRequest request,
                    ModelAutotuningTuning<C, D, K> tuning) {
        beginOperation();
        return prepareTunedOrdinaryAlreadyAdmitted(owner, compiledGraph, request, tuning);
    }

    private ModelAutotuningPreparation prepareTunedOrdinaryAlreadyAdmitted(
            Engine owner,
            CompiledGraph compiledGraph,
            ModelAutotuningRequest request,
            ModelAutotuningTuning<?, ?, ?> suppliedTuning) {
        RepresentativeExecutionSession session;
        ModelAutotuningConfig config;
        ModelAutotuningRequest.ModelIdentity modelIdentity;
        try {
            Objects.requireNonNull(compiledGraph, "compiledGraph");
            requireOrdinaryOwner(owner, compiledGraph.owner());
            Objects.requireNonNull(request, "request");
            config = request.config();
            modelIdentity = request.modelIdentity();
            session = new RepresentativeExecutionSession(
                    this, compiledGraph, request.representativeInputs(), composition, runner);
        } catch (RuntimeException | Error failure) {
            finishFailure();
            throw failure;
        }

        ModelAutotuningTuning<?, ?, ?> tuning;
        try {
            tuning = suppliedTuning;
            if (tuning == null) {
                tuning = new CpuTuningAdapter((CpuEngineBackendComposition) composition);
            }
        } catch (RuntimeException | Error failure) {
            closeWithSuppression(session, failure);
            throw failure;
        }
        return prepareTunedOrdinaryWithSession(
                owner, compiledGraph, config, modelIdentity, session, tuning);
    }

    private <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision, K>
            ModelAutotuningPreparation prepareTunedOrdinaryWithSession(
                    Engine owner,
                    CompiledGraph compiledGraph,
                    ModelAutotuningConfig config,
                    ModelAutotuningRequest.ModelIdentity modelIdentity,
                    RepresentativeExecutionSession session,
                    ModelAutotuningTuning<C, D, K> tuning) {
        RuntimeException recoverable = null;
        boolean selectionAuthenticated = false;
        try {
            var optionalHandoff = tuning.candidateHandoff(compiledGraph.artifacts());
            if (optionalHandoff.isEmpty()) {
                throw new IllegalStateException("no eligible CPU local-workload tuning handoff");
            }
            var handoff = optionalHandoff.orElseThrow();
            byte[] context = ByteBuffer.allocate(8).putInt(0).putInt(0).array();
            WorkloadTuningRequest<C, D> tuningRequest = tuningRequest(
                            config, modelIdentity, handoff, context);
            WorkloadTuningResult<C, D> result;
            try {
                result = WorkloadTuning.tune(tuningRequest, tuning,
                        (batch, candidate) -> {
                            var trial = tuning.prepareTrial(batch, candidate);
                            session.execute(trial);
                        });
            } catch (RuntimeException failure) {
                if (session.isRepresentativeExecutionFailure(failure)) throw failure;
                throw failure;
            } catch (IOException failure) {
                throw new UncheckedIOException(failure);
            } catch (Exception failure) {
                throw new IllegalStateException("unexpected checked tuning failure", failure);
            }
            BackendPartitionTuningHandoff<C, D> selected = authenticate(
                            result, handoff, tuningRequest, context);
            selectionAuthenticated = true;
            session.cleanupForProductionPreparation();
            var inward = tuning.prepareSelected(
                    selected.candidateBatch(), selected.selectedDecision().orElseThrow());
            ModelAutotuningPreparation.Evidence evidence = translateEvidence(
                    result.evidence(), config, modelIdentity, context);
            var prepared = new io.github.pho001.synaptik.engine.PreparedExecution(
                    owner, compiledGraph, inward);
            var publicResult = new ModelAutotuningPreparation(
                    prepared, ModelAutotuningPreparation.Outcome.TUNED, Optional.of(evidence));
            return session.completeModelAutotuningPreparation(publicResult);
        } catch (Error failure) {
            if (!session.isRepresentativeExecutionFailure(failure)) closeWithSuppression(session, failure);
            throw failure;
        } catch (RuntimeException failure) {
            if (session.isRepresentativeExecutionFailure(failure)) throw failure;
            if (!canResolveRecoverableTuningFailure(session, selectionAuthenticated)) {
                session.releaseAdmission();
                throw failure;
            }
            recoverable = failure;
        }

        boolean allowed = config.fallbackPolicy()
                == ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC;
        try {
            session.cleanupForProductionPreparation();
        } catch (RuntimeException | Error cleanupFailure) {
            suppressDistinctOnce(recoverable, cleanupFailure);
            session.releaseAdmission();
            throw recoverable;
        }
        if (!allowed) {
            session.releaseAdmission();
            throw recoverable;
        }
        try {
            var inward = session.prepareOrdinaryFallback();
            var prepared = new io.github.pho001.synaptik.engine.PreparedExecution(
                    owner, compiledGraph, inward);
            var result = new ModelAutotuningPreparation(prepared,
                    ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK,
                    Optional.empty());
            return session.completeModelAutotuningPreparation(result);
        } catch (RuntimeException | Error fallbackFailure) {
            suppressDistinctOnce(fallbackFailure, recoverable);
            throw fallbackFailure;
        } finally {
            session.releaseAdmission();
        }
    }

    HostTensorValue computeOrdinary(
            Engine owner, Tensor output, long maximumTotalBytes) {
        beginOperation();
        try {
            Objects.requireNonNull(owner, "owner");
            Tensor outputSnapshot = Objects.requireNonNull(output, "output");
            requireNonNegativeTotalLimit(maximumTotalBytes);
            return computeOrdinaryOpen(
                    owner, List.of(outputSnapshot), maximumTotalBytes).getFirst();
        } finally {
            finishFailure();
        }
    }

    List<HostTensorValue> computeOrdinary(
            Engine owner,
            List<Tensor> outputs,
            long maximumTotalBytes) {
        beginOperation();
        try {
            Objects.requireNonNull(owner, "owner");
            List<Tensor> outputSnapshot = snapshotIdentityUnique(outputs, "outputs", true);
            requireNonNegativeTotalLimit(maximumTotalBytes);
            return computeOrdinaryOpen(owner, outputSnapshot, maximumTotalBytes);
        } finally {
            finishFailure();
        }
    }

    /**
     * Admits one ordinary scalar-objective backward call and keeps the admission through cleanup.
     * Structural validation occurs after admission and before any Compiler work.
     *
     * @param owner non-null ordinary Engine facade using this exact lifecycle owner
     * @param objective non-null scalar objective whose semantic eligibility Compiler validates
     * @param targets non-null non-empty exact-object-identity-unique ordered target list
     * @param maximumTotalBytes non-negative aggregate returned-payload limit in bytes
     * @return a fresh detached objective and target-aligned gradient carrier
     * @throws NullPointerException if a reference or indexed target is null
     * @throws IllegalArgumentException if local structure, the byte limit, Compiler semantics, or
     *     an inward lifecycle boundary rejects the request
     * @throws IllegalStateException if closure has begun or discovered input/publication/runtime
     *     state is inconsistent
     * @throws RuntimeException if inward work or cleanup reports another unchecked failure
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    ScalarObjectiveBackwardResult backwardOrdinary(
            Engine owner,
            Tensor objective,
            List<Tensor> targets,
            long maximumTotalBytes) {
        beginOperation();
        try {
            Objects.requireNonNull(owner, "owner");
            Tensor objectiveSnapshot = Objects.requireNonNull(objective, "objective");
            List<Tensor> targetSnapshot = snapshotIdentityUnique(targets, "targets", true);
            requireNonNegativeTotalLimit(maximumTotalBytes);
            return backwardOrdinaryOpen(
                    owner, objectiveSnapshot, targetSnapshot, maximumTotalBytes);
        } finally {
            finishFailure();
        }
    }

    private List<HostTensorValue> computeOrdinaryOpen(
            Engine owner,
            List<Tensor> outputs,
            long maximumTotalBytes) {
        Map<TensorId, Tensor> leaves = inventoryReachableLeaves(outputs);
        CompiledGraph compiled = compileForwardOrdinaryOpen(owner, outputs);
        List<Tensor> selectedInputs = selectCompiledInputs(compiled, leaves);
        io.github.pho001.synaptik.engine.PreparedExecution prepared =
                prepareOrdinaryOpen(owner, compiled);
        OrdinaryRun ordinaryRun = runOrdinaryOpen(owner, prepared, selectedInputs);
        boolean cleanupAttempted = false;
        try {
            List<io.github.pho001.synaptik.engine.RunResult.Publication> publications =
                    validateForwardPublications(ordinaryRun.result(), outputs);
            long[] byteCounts = preflightCanonicalByteCounts(
                    publications, maximumTotalBytes);
            var values = new ArrayList<HostTensorValue>(publications.size());
            for (int index = 0; index < publications.size(); index++) {
                values.add(ordinaryRun.owner().materializeUnderAdmission(
                        ordinaryRun.result(), publications.get(index), byteCounts[index], composition));
            }
            List<HostTensorValue> result = List.copyOf(values);
            cleanupAttempted = true;
            ordinaryRun.owner().close();
            return result;
        } catch (RuntimeException | Error failure) {
            if (!cleanupAttempted) {
                cleanupAttempted = true;
                try {
                    ordinaryRun.owner().close();
                } catch (RuntimeException | Error cleanupFailure) {
                    suppressDistinct(failure, cleanupFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Performs one already-admitted backward lifecycle without nesting a public gate.
     *
     * @param owner non-null ordinary Engine facade
     * @param objective non-null prevalidated objective reference
     * @param targets non-null immutable non-empty prevalidated target snapshot
     * @param maximumTotalBytes non-negative aggregate payload bound in bytes
     * @return a fresh detached result after temporary run cleanup succeeds
     * @throws RuntimeException if compilation, preparation, binding, execution, publication
     *     validation, preflight, copying, construction, or cleanup fails
     * @throws Error if inward work, allocation, copying, or cleanup reports a fatal failure
     */
    private ScalarObjectiveBackwardResult backwardOrdinaryOpen(
            Engine owner,
            Tensor objective,
            List<Tensor> targets,
            long maximumTotalBytes) {
        Map<TensorId, Tensor> leaves = inventoryReachableLeaves(List.of(objective));
        CompiledGraph compiled = compileScalarObjectiveBackwardOrdinaryOpen(
                owner, objective, targets);
        List<Tensor> selectedInputs = selectCompiledInputs(compiled, leaves);
        io.github.pho001.synaptik.engine.PreparedExecution prepared =
                prepareOrdinaryOpen(owner, compiled);
        OrdinaryRun ordinaryRun = runOrdinaryOpen(owner, prepared, selectedInputs);
        boolean cleanupAttempted = false;
        try {
            List<io.github.pho001.synaptik.engine.RunResult.Publication> publications =
                    validateScalarObjectiveBackwardPublications(
                            ordinaryRun.result(), objective, targets);
            long[] byteCounts = preflightCanonicalByteCounts(
                    publications, maximumTotalBytes);
            HostTensorValue objectiveValue = ordinaryRun.owner().materializeUnderAdmission(
                    ordinaryRun.result(), publications.getFirst(), byteCounts[0], composition);
            var gradients = new ArrayList<HostTensorValue>(targets.size());
            for (int index = 0; index < targets.size(); index++) {
                gradients.add(ordinaryRun.owner().materializeUnderAdmission(
                        ordinaryRun.result(), publications.get(index + 1),
                        byteCounts[index + 1], composition));
            }
            ScalarObjectiveBackwardResult result =
                    new ScalarObjectiveBackwardResult(objectiveValue, gradients);
            cleanupAttempted = true;
            ordinaryRun.owner().close();
            return result;
        } catch (RuntimeException | Error failure) {
            if (!cleanupAttempted) {
                cleanupAttempted = true;
                try {
                    ordinaryRun.owner().close();
                } catch (RuntimeException | Error cleanupFailure) {
                    suppressDistinct(failure, cleanupFailure);
                }
            }
            throw failure;
        }
    }

    private CompiledGraph compileForwardOrdinaryOpen(Engine owner, List<Tensor> outputs) {
        CompileArtifacts artifacts = GraphCompilationPort.compile(
                CompileMode.FORWARD_ONLY,
                outputs,
                Optional.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                composition.capabilityProviders(),
                composition.availabilitySnapshots());
        return new CompiledGraph(owner, artifacts);
    }

    /**
     * Lowers one exact absent-seed, ERROR-policy functional-gradient stage to Compiler.
     *
     * @param owner non-null ordinary owner retained by the returned compile handle
     * @param objective non-null exact singleton forward output and stage output
     * @param targets non-null immutable ordered exact differentiation targets
     * @return a fresh non-null ordinary compile handle over the complete artifacts
     * @throws RuntimeException if Compiler or capability planning rejects the request
     */
    private CompiledGraph compileScalarObjectiveBackwardOrdinaryOpen(
            Engine owner, Tensor objective, List<Tensor> targets) {
        var stage = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
                List.of(Optional.empty()),
                targets,
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        CompileArtifacts artifacts = GraphCompilationPort.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new FunctionalGradientRequest(List.of(stage))),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                composition.capabilityProviders(),
                composition.availabilitySnapshots());
        return new CompiledGraph(owner, artifacts);
    }

    /**
     * Iteratively inventories the exact provenance-free Tensor leaves reachable from outputs.
     * Visitation uses object identity, while the returned transient lookup uses Tensor identity
     * for later joining to compiled input metadata. A repeated Tensor ID on a different exact
     * leaf is rejected rather than resolved arbitrarily.
     *
     * @param outputs prevalidated non-null output snapshot containing non-null Tensors
     * @return a new mutable transient mapping from each reachable leaf ID to its exact Tensor
     * @throws IllegalStateException if different exact leaf objects carry the same Tensor ID
     */
    static Map<TensorId, Tensor> inventoryReachableLeaves(List<Tensor> outputs) {
        IdentityHashMap<Tensor, Boolean> visited = new IdentityHashMap<>();
        Map<TensorId, Tensor> leaves = new HashMap<>();
        var pending = new ArrayList<Tensor>(outputs);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (visited.put(tensor, Boolean.TRUE) != null) {
                continue;
            }
            Optional<io.github.pho001.synaptik.model.tensor.TensorProvenance> provenance =
                    tensor.provenance();
            if (provenance.isEmpty()) {
                Tensor previous = leaves.putIfAbsent(tensor.id(), tensor);
                if (previous != null && previous != tensor) {
                    throw new IllegalStateException(
                            "different Tensor objects share Tensor ID " + tensor.id());
                }
                continue;
            }
            List<Tensor> inputs = provenance.orElseThrow().inputs();
            for (int index = inputs.size() - 1; index >= 0; index--) {
                pending.add(inputs.get(index));
            }
        }
        return leaves;
    }

    /**
     * Selects reachable leaves in the exact authoritative compiled-input order.
     * Inventory entries absent from the compiled metadata are ignored without storage access.
     *
     * @param compiled non-null compiled graph whose immutable inputs determine membership and order
     * @param leaves non-null transient reachable-leaf inventory keyed by Tensor ID
     * @return an immutable list containing the exact selected Tensor objects in compiled order
     * @throws IllegalStateException if an authoritative compiled Tensor ID has no reachable leaf
     */
    static List<Tensor> selectCompiledInputs(
            CompiledGraph compiled, Map<TensorId, Tensor> leaves) {
        var selected = new ArrayList<Tensor>(compiled.inputs().size());
        for (CompiledGraph.Input input : compiled.inputs()) {
            Tensor tensor = leaves.get(input.tensorId());
            if (tensor == null) {
                throw new IllegalStateException(
                        "compiled input has no reachable provenance-free Tensor: "
                                + input.tensorId());
            }
            selected.add(tensor);
        }
        return List.copyOf(selected);
    }

    private io.github.pho001.synaptik.engine.PreparedExecution prepareOrdinaryOpen(
            Engine owner, CompiledGraph compiledGraph) {
        return new io.github.pho001.synaptik.engine.PreparedExecution(
                owner, compiledGraph, composition.prepare(compiledGraph.artifacts()));
    }

    private OrdinaryRun runOrdinaryOpen(
            Engine owner,
            io.github.pho001.synaptik.engine.PreparedExecution preparedExecution,
            List<Tensor> inputs) {
        var borrowed = new ArrayList<BufferRepresentation>();
        io.github.pho001.synaptik.runtime.run.RunResult inwardResult = null;
        boolean ownershipTransferred = false;
        try {
            Objects.requireNonNull(owner, "owner");
            Objects.requireNonNull(preparedExecution, "preparedExecution");
            Objects.requireNonNull(inputs, "inputs");
            requireOrdinaryOwner(owner, preparedExecution.owner());
            List<Tensor> supplied = snapshotElements(inputs, "inputs");
            List<CompiledGraph.Input> required = preparedExecution.compiledGraph().inputs();

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
            for (HostTensorStorage storage : storages) {
                borrowed.add(composition.borrow(storage));
            }

            inwardResult = runner.run(preparedExecution.execution(), borrowed);
            List<CompiledGraph.PublicationSpec> specifications =
                    preparedExecution.compiledGraph().publicationSpecs();
            if (inwardResult.resultCount() != specifications.size()) {
                throw new IllegalStateException(
                        "Runtime result count does not match publication count: expected="
                                + specifications.size() + ", actual=" + inwardResult.resultCount());
            }
            AdvancedRunResult resultOwner = new AdvancedRunResult(this, inwardResult, borrowed);
            io.github.pho001.synaptik.engine.RunResult result =
                    new io.github.pho001.synaptik.engine.RunResult(resultOwner, specifications);
            ownershipTransferred = true;
            return new OrdinaryRun(resultOwner, result);
        } catch (RuntimeException | Error failure) {
            if (!ownershipTransferred) {
                if (inwardResult != null) {
                    try {
                        inwardResult.close();
                    } catch (RuntimeException | Error cleanupFailure) {
                        suppressDistinct(failure, cleanupFailure);
                    }
                }
                closeBorrowedReverse(borrowed, failure);
            }
            throw failure;
        }
    }

    private static List<io.github.pho001.synaptik.engine.RunResult.Publication>
            validateForwardPublications(
                    io.github.pho001.synaptik.engine.RunResult result, List<Tensor> outputs) {
        List<io.github.pho001.synaptik.engine.RunResult.Publication> publications =
                result.publications();
        if (result.resultCount() != outputs.size()) {
            throw new IllegalStateException(
                    "forward result count does not match output count: expected="
                            + outputs.size() + ", actual=" + result.resultCount());
        }
        if (publications.size() != outputs.size()) {
            throw new IllegalStateException(
                    "forward publication count does not match output count: expected="
                            + outputs.size() + ", actual=" + publications.size());
        }
        for (int index = 0; index < publications.size(); index++) {
            var publication = publications.get(index);
            if (publication.index() != index) {
                throw new IllegalStateException(
                        "forward publication index mismatch at " + index + ": actual="
                                + publication.index());
            }
            if (publication.role() != io.github.pho001.synaptik.engine.RunResult.Role.FORWARD) {
                throw new IllegalStateException(
                        "forward publication role mismatch at " + index + ": actual="
                                + publication.role());
            }
            if (!publication.tensorId().equals(outputs.get(index).id())) {
                throw new IllegalStateException(
                        "forward publication Tensor ID mismatch at " + index);
            }
            if (publication.derivativeOrder().isPresent()) {
                throw new IllegalStateException(
                        "forward publication derivative order is present at " + index);
            }
            if (publication.targetIndex().isPresent()) {
                throw new IllegalStateException(
                        "forward publication target index is present at " + index);
            }
            Objects.requireNonNull(publication.descriptor(),
                    "forward publication descriptor[" + index + "]");
        }
        return publications;
    }

    /**
     * Validates the complete objective-first, target-ordered backward publication boundary.
     *
     * @param result non-null temporary ordinary result
     * @param objective non-null exact requested objective
     * @param targets non-null ordered requested targets
     * @return the result's exact publication list after complete validation
     * @throws IllegalStateException if count, role, identity, order, or metadata differs
     */
    private static List<io.github.pho001.synaptik.engine.RunResult.Publication>
            validateScalarObjectiveBackwardPublications(
                    io.github.pho001.synaptik.engine.RunResult result,
                    Tensor objective,
                    List<Tensor> targets) {
        List<io.github.pho001.synaptik.engine.RunResult.Publication> publications =
                result.publications();
        int expectedCount = Math.addExact(1, targets.size());
        if (result.resultCount() != expectedCount) {
            throw new IllegalStateException(
                    "backward result count mismatch: expected=" + expectedCount
                            + ", actual=" + result.resultCount());
        }
        if (publications.size() != expectedCount) {
            throw new IllegalStateException(
                    "backward publication count mismatch: expected=" + expectedCount
                            + ", actual=" + publications.size());
        }
        validateBackwardForwardPublication(publications.getFirst(), objective);
        for (int targetIndex = 0; targetIndex < targets.size(); targetIndex++) {
            validateBackwardGradientPublication(
                    publications.get(targetIndex + 1), targets.get(targetIndex), targetIndex);
        }
        return publications;
    }

    /**
     * Validates the exact forward role at backward result index zero.
     *
     * @param publication non-null index-zero publication occurrence
     * @param objective non-null exact requested objective
     * @throws IllegalStateException if role, index, identity, or derivative metadata differs
     * @throws NullPointerException if the publication descriptor is unexpectedly null
     */
    private static void validateBackwardForwardPublication(
            io.github.pho001.synaptik.engine.RunResult.Publication publication,
            Tensor objective) {
        if (publication.index() != 0) {
            throw new IllegalStateException(
                    "backward objective publication index mismatch: actual="
                            + publication.index());
        }
        if (publication.role() != io.github.pho001.synaptik.engine.RunResult.Role.FORWARD) {
            throw new IllegalStateException(
                    "backward objective publication role mismatch: actual="
                            + publication.role());
        }
        if (!publication.tensorId().equals(objective.id())) {
            throw new IllegalStateException(
                    "backward objective publication Tensor ID mismatch at 0");
        }
        if (publication.derivativeOrder().isPresent()) {
            throw new IllegalStateException(
                    "backward objective publication derivative order is present at 0");
        }
        if (publication.targetIndex().isPresent()) {
            throw new IllegalStateException(
                    "backward objective publication target index is present at 0");
        }
        Objects.requireNonNull(publication.descriptor(),
                "backward publication descriptor[0]");
    }

    /**
     * Validates one exact first-gradient role against its requested target position.
     *
     * @param publication non-null gradient publication occurrence
     * @param target non-null exact requested target at {@code targetIndex}
     * @param targetIndex non-negative requested target-list position
     * @throws IllegalStateException if role, index, identity, derivative order, or target position
     *     differs
     * @throws NullPointerException if the publication descriptor is unexpectedly null
     */
    private static void validateBackwardGradientPublication(
            io.github.pho001.synaptik.engine.RunResult.Publication publication,
            Tensor target,
            int targetIndex) {
        int publicationIndex = targetIndex + 1;
        if (publication.index() != publicationIndex) {
            throw new IllegalStateException(
                    "backward gradient publication index mismatch at " + publicationIndex
                            + ": actual=" + publication.index());
        }
        if (publication.role() != io.github.pho001.synaptik.engine.RunResult.Role.GRADIENT) {
            throw new IllegalStateException(
                    "backward gradient publication role mismatch at " + publicationIndex
                            + ": actual=" + publication.role());
        }
        if (!publication.tensorId().equals(target.id())) {
            throw new IllegalStateException(
                    "backward gradient publication Tensor ID mismatch at " + publicationIndex);
        }
        if (publication.derivativeOrder().orElse(-1) != 1) {
            throw new IllegalStateException(
                    "backward gradient publication derivative order mismatch at "
                            + publicationIndex);
        }
        if (publication.targetIndex().orElse(-1) != targetIndex) {
            throw new IllegalStateException(
                    "backward gradient publication target index mismatch at "
                            + publicationIndex);
        }
        Objects.requireNonNull(publication.descriptor(),
                "backward publication descriptor[" + publicationIndex + "]");
    }

    private static long[] preflightCanonicalByteCounts(
            List<io.github.pho001.synaptik.engine.RunResult.Publication> publications,
            long maximumTotalBytes) {
        long[] byteCounts = new long[publications.size()];
        long total = 0;
        for (int index = 0; index < publications.size(); index++) {
            var descriptor = publications.get(index).descriptor();
            if (!descriptor.shape().isFullyStatic()) {
                throw new IllegalArgumentException(
                        "host snapshot requires a fully static shape: " + descriptor.shape());
            }
            if (descriptor.layout().isEmpty()) {
                throw new IllegalArgumentException("host snapshot requires a resolved layout");
            }
            long elementCount = descriptor.shape().knownElementCount().orElseThrow();
            long byteCount = Math.multiplyExact(elementCount, descriptor.dataType().byteWidth());
            if (byteCount > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "canonical byte count exceeds JVM byte[] limit: " + byteCount);
            }
            byteCounts[index] = byteCount;
            total = Math.addExact(total, byteCount);
        }
        if (total > maximumTotalBytes) {
            throw new IllegalArgumentException(
                    "total canonical byte count exceeds maximumTotalBytes: required=" + total
                            + ", maximum=" + maximumTotalBytes);
        }
        return byteCounts;
    }

    private static void requireNonNegativeTotalLimit(long maximumTotalBytes) {
        if (maximumTotalBytes < 0) {
            throw new IllegalArgumentException(
                    "maximumTotalBytes must be non-negative: " + maximumTotalBytes);
        }
    }

    private record OrdinaryRun(
            AdvancedRunResult owner, io.github.pho001.synaptik.engine.RunResult result) {}

    /**
     * Admits and synchronously materializes one occurrence under its registered result monitor.
     * Engine admission precedes all argument inspection. The result monitor then rejects result
     * closure and remains held through occurrence validation, Runtime lookup, CPU copy, and
     * detached-value construction, so Engine close waits for an admitted call.
     *
     * @param resultOwner non-null registered result owner
     * @param result non-null ordinary result backed by {@code resultOwner}
     * @param publication possibly null occurrence selector validated after Engine and result
     *     lifecycle admission
     * @param maximumBytes caller byte limit validated after occurrence authentication
     * @return a fresh detached host value ready before admission is released
     * @throws IllegalStateException if Engine or result closure has begun; Engine closure wins
     *     over argument validation
     * @throws RuntimeException if validation or copying fails
     * @throws Error if copying reports a fatal failure
     */
    HostTensorValue materializeOrdinary(
            AdvancedRunResult resultOwner,
            io.github.pho001.synaptik.engine.RunResult result,
            io.github.pho001.synaptik.engine.RunResult.Publication publication,
            long maximumBytes) {
        beginOperation();
        try {
            return resultOwner.materializeUnderAdmission(
                    result, publication, maximumBytes, composition);
        } finally {
            finishFailure();
        }
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

    private static void requireOrdinaryOwner(Engine expected, Engine actual) {
        if (actual != expected) {
            throw new IllegalArgumentException("handle belongs to another engine");
        }
    }

    private static List<Tensor> snapshotElements(List<Tensor> values, String name) {
        Objects.requireNonNull(values, name);
        var snapshot = new ArrayList<Tensor>(values.size());
        for (int index = 0; index < values.size(); index++) {
            snapshot.add(Objects.requireNonNull(values.get(index), name + "[" + index + "]"));
        }
        return List.copyOf(snapshot);
    }

    private static List<Tensor> snapshotIdentityUnique(
            List<Tensor> values, String name, boolean requireNonEmpty) {
        List<Tensor> snapshot = snapshotElements(values, name);
        if (requireNonEmpty && snapshot.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
        for (int index = 0; index < snapshot.size(); index++) {
            Integer first = positions.putIfAbsent(snapshot.get(index), index);
            if (first != null) {
                throw new IllegalArgumentException(
                        name + "[" + index + "] duplicates " + name + "[" + first + "]");
            }
        }
        return snapshot;
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

    private static void closeBorrowedReverse(
            List<BufferRepresentation> borrowed, Throwable primary) {
        for (int index = borrowed.size() - 1; index >= 0; index--) {
            try {
                borrowed.get(index).close();
            } catch (RuntimeException | Error cleanupFailure) {
                suppressDistinct(primary, cleanupFailure);
            }
        }
    }

    private static void suppressDistinct(Throwable primary, Throwable next) {
        if (next != primary) {
            primary.addSuppressed(next);
        }
    }

    /**
     * Closes representative borrowing before freshly preparing the selected production recipe.
     * Cleanup failure prevents preparation. The retained Engine admission spans cleanup,
     * preparation, and final publication; a close race rejects the fresh recipe. This
     * package-private boundary performs no candidate interpretation or fallback.
     *
     * @param session non-null representative session whose cleanup must succeed first
     * @param tuning non-null CPU-owned typed tuning collaboration
     * @param batch non-null opaque batch that owns {@code decision}
     * @param decision non-null exact selected decision
     * @return a fresh non-null production recipe prepared after representative cleanup
     * @throws NullPointerException if an argument or selected preparation result is null
     * @throws IllegalStateException if the session is poisoned or not ready, or Engine closure
     *     wins the final publication race
     * @throws RuntimeException if cleanup or selected preparation reports another unchecked
     *     failure
     * @throws Error if cleanup or selected preparation reports a fatal failure
     */
    static io.github.pho001.synaptik.runtime.execution.PreparedExecution
            prepareSelectedAfterRepresentativeCleanup(
                    RepresentativeExecutionSession session,
                    CpuLocalWorkloadTuning tuning,
                    CpuLocalWorkloadTuning.CandidateBatch batch,
                    CpuLocalWorkloadTuning.SelectedDecision decision) {
        RepresentativeExecutionSession admittedSession =
                Objects.requireNonNull(session, "session");
        try {
            admittedSession.cleanupForProductionPreparation();
            var prepared = Objects.requireNonNull(tuning, "tuning").prepareSelected(
                    Objects.requireNonNull(batch, "batch"),
                    Objects.requireNonNull(decision, "decision"));
            return admittedSession.completeProductionPreparation(prepared);
        } finally {
            admittedSession.releaseAdmission();
        }
    }

    /**
     * Resolves a recoverable tuning failure after representative cleanup.
     * Required tuning rethrows the exact failure after successful cleanup. Allowed fallback
     * invokes the existing ordinary preparation exactly once and only after cleanup proves the
     * session unpoisoned. Cleanup failure forbids fallback: the exact tuning failure remains
     * primary and receives the cleanup failure as a distinct suppression. If fallback preparation
     * or final publication fails, that exact failure becomes primary and the tuning failure is
     * suppressed when distinct. Callers must perform Engine admission and ownership checks before
     * entering this boundary, so lifecycle or ownership rejection never becomes fallback input.
     *
     * @param session non-null representative session whose cleanup must succeed first
     * @param tuningFailure non-null recoverable tuning failure; an {@link Error} cannot enter
     * @param fallbackAllowed whether one ordinary safe-heuristic preparation is permitted
     * @return a fresh non-null ordinary production recipe when fallback is allowed and succeeds
     * @throws NullPointerException if {@code session} or {@code tuningFailure} is null
     * @throws RuntimeException the exact tuning failure when fallback is forbidden or cleanup
     *     fails; otherwise the exact fallback-preparation or final-publication unchecked failure
     * @throws Error if ordinary preparation reports a fatal failure; the tuning failure is
     *     suppressed on that exact primary error
     */
    static io.github.pho001.synaptik.runtime.execution.PreparedExecution
            prepareAfterRecoverableTuningFailure(
                    RepresentativeExecutionSession session,
                    RuntimeException tuningFailure,
                    boolean fallbackAllowed) {
        Objects.requireNonNull(tuningFailure, "tuningFailure");
        RepresentativeExecutionSession admittedSession =
                Objects.requireNonNull(session, "session");
        try {
            admittedSession.cleanupForProductionPreparation();
        } catch (RuntimeException | Error cleanupFailure) {
            suppressDistinctOnce(tuningFailure, cleanupFailure);
            admittedSession.releaseAdmission();
            throw tuningFailure;
        }
        try {
            if (!fallbackAllowed) throw tuningFailure;
            return admittedSession.completeProductionPreparation(
                    admittedSession.prepareOrdinaryFallback());
        } catch (RuntimeException | Error preparationFailure) {
            suppressDistinct(preparationFailure, tuningFailure);
            throw preparationFailure;
        } finally {
            admittedSession.releaseAdmission();
        }
    }

    private static void suppressDistinctOnce(Throwable primary, Throwable next) {
        if (next == primary) return;
        for (Throwable suppressed : primary.getSuppressed()) {
            if (suppressed == next) return;
        }
        primary.addSuppressed(next);
    }

    /**
     * Observes the exact fallback boundary without inspecting a tuning result. Once selection is
     * authenticated, cleanup, selected preparation, public translation/construction, and final
     * publication failures are never recoverable tuning failures.
     *
     * @param session non-null admitted representative session
     * @param selectionAuthenticated whether result authentication completed successfully
     * @return whether a runtime failure may enter the recoverable cleanup/fallback state machine
     */
    static boolean canResolveRecoverableTuningFailure(
            RepresentativeExecutionSession session, boolean selectionAuthenticated) {
        return !selectionAuthenticated
                && Objects.requireNonNull(session, "session")
                        .canResolveRecoverableTuningFailure();
    }

    private static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision>
            WorkloadTuningRequest<C, D> tuningRequest(
                    ModelAutotuningConfig config,
                    ModelAutotuningRequest.ModelIdentity modelIdentity,
                    BackendPartitionTuningHandoff<C, D> handoff,
                    byte[] context) {
        Objects.requireNonNull(config, "config");
        var profile = config.representativeProfile();
        var budget = config.budget();
        return new WorkloadTuningRequest<>(
                new WorkloadTuningRequest.ModelFingerprint(
                        modelIdentity.schemaVersion(), modelIdentity.bytes()),
                new WorkloadTuningRequest.ProfileFingerprint(
                        profile.schemaVersion(), profile.bytes()),
                List.of(new WorkloadTuningRequest.Occurrence<>(handoff, 1, context)),
                switch (config.objective()) {
                    case MIN_MEDIAN_ELAPSED_NANOS ->
                            WorkloadTuningRequest.Objective.MIN_MEDIAN_ELAPSED_NANOS;
                },
                new WorkloadTuningRequest.Budget(
                        budget.maximumDistinctCacheMisses(),
                        budget.maximumCandidatesPerMiss(),
                        budget.warmupCount(), budget.timedSampleCount()),
                config.workloadCache());
    }

    private static <C extends BackendTuningCandidateBatch, D extends BackendTuningDecision>
            BackendPartitionTuningHandoff<C, D> authenticate(
                    WorkloadTuningResult<C, D> result,
                    BackendPartitionTuningHandoff<C, D> original,
                    WorkloadTuningRequest<C, D> request,
                    byte[] context) {
        Objects.requireNonNull(result, "tuning result");
        if (result.selectedHandoffs().size() != 1) {
            throw new IllegalArgumentException("tuning result must contain exactly one handoff");
        }
        var selected = result.selectedHandoffs().getFirst();
        if (selected.partition() != original.partition()
                || selected.candidateBatch() != original.candidateBatch()
                || selected.selectedDecision().isEmpty()) {
            throw new IllegalArgumentException("selected handoff does not match the original handoff");
        }
        var evidence = result.evidence();
        if (!evidence.modelFingerprint().equals(request.modelFingerprint())
                || !evidence.profileFingerprint().equals(request.profileFingerprint())
                || evidence.objective() != request.objective()
                || !evidence.budget().equals(request.budget())
                || evidence.workloads().size() != 1) {
            throw new IllegalArgumentException("tuning result evidence does not match the request");
        }
        var workload = evidence.workloads().getFirst();
        if (workload.totalWeight() != 1 || workload.occurrences().size() != 1
                || workload.occurrences().getFirst().weight() != 1
                || !java.util.Arrays.equals(
                        workload.occurrences().getFirst().contextFingerprint(), context)) {
            throw new IllegalArgumentException("tuning occurrence evidence does not match the request");
        }
        return selected;
    }

    private static ModelAutotuningPreparation.Evidence translateEvidence(
            WorkloadTuningResult.Evidence inward,
            ModelAutotuningConfig config,
            ModelAutotuningRequest.ModelIdentity modelIdentity,
            byte[] context) {
        var workloads = new ArrayList<ModelAutotuningPreparation.WorkloadEvidence>();
        for (Object value : inward.workloads()) {
            var workload = (WorkloadTuningResult.WorkloadEvidence) value;
            var compatibility = workload.compatibility();
            var occurrences = new ArrayList<ModelAutotuningPreparation.OccurrenceEvidence>();
            for (Object occurrenceValue : workload.occurrences()) {
                var occurrence = (WorkloadTuningResult.OccurrenceEvidence) occurrenceValue;
                if (!java.util.Arrays.equals(occurrence.contextFingerprint(), context)) {
                    throw new IllegalArgumentException("unexpected occurrence context");
                }
                occurrences.add(new ModelAutotuningPreparation.OccurrenceEvidence(
                        0, 0, occurrence.weight(),
                        new ModelAutotuningPreparation.ContextIdentity(1, context)));
            }
            var candidates = new ArrayList<ModelAutotuningPreparation.CandidateEvidence>();
            for (Object candidateValue : workload.candidates()) {
                var candidate = (WorkloadTuningResult.CandidateEvidence) candidateValue;
                candidates.add(new ModelAutotuningPreparation.CandidateEvidence(
                        new ModelAutotuningPreparation.CandidateIdentity(
                                candidate.identity().bytes()),
                        candidate.elapsedSamplesNanos(), summary(candidate.summary())));
            }
            workloads.add(new ModelAutotuningPreparation.WorkloadEvidence(
                    new ModelAutotuningPreparation.CompatibilityIdentity(
                            compatibility.schemaVersion(), compatibility.bytes(),
                            switch (compatibility.reuseScope()) {
                                case SESSION -> ModelAutotuningPreparation.ReuseScope.SESSION;
                                case PERSISTENT -> ModelAutotuningPreparation.ReuseScope.PERSISTENT;
                            }),
                    workload.totalWeight(), occurrences,
                    switch (workload.source()) {
                        case CACHE_HIT -> ModelAutotuningPreparation.Source.CACHE_HIT;
                        case MEASURED -> ModelAutotuningPreparation.Source.MEASURED;
                    },
                    candidates,
                    new ModelAutotuningPreparation.CandidateIdentity(
                            workload.winnerIdentity().bytes()),
                    summary(workload.winnerSummary())));
        }
        return new ModelAutotuningPreparation.Evidence(
                modelIdentity, config.representativeProfile(), config.objective(), config.budget(),
                workloads);
    }

    private static ModelAutotuningPreparation.SampleSummary summary(
            WorkloadTuningResult.SampleSummary summary) {
        return new ModelAutotuningPreparation.SampleSummary(
                summary.minimumNanos(), summary.medianNanos(), summary.maximumNanos(),
                summary.sampleCount());
    }

    private static void closeWithSuppression(
            RepresentativeExecutionSession session, Throwable primary) {
        try {
            session.close();
        } catch (RuntimeException | Error cleanupFailure) {
            suppressDistinctOnce(primary, cleanupFailure);
        }
    }

    private record CpuTuningAdapter(CpuEngineBackendComposition composition)
            implements ModelAutotuningTuning<CpuLocalWorkloadTuning.CandidateBatch,
                    CpuLocalWorkloadTuning.SelectedDecision,
                    CpuLocalWorkloadTuning.Candidate> {
        private CpuTuningAdapter {
            Objects.requireNonNull(composition, "composition");
        }

        private CpuLocalWorkloadTuning tuning() { return composition.localWorkloadTuning(); }

        @Override public Optional<BackendPartitionTuningHandoff<
                CpuLocalWorkloadTuning.CandidateBatch, CpuLocalWorkloadTuning.SelectedDecision>>
                candidateHandoff(CompileArtifacts artifacts) {
            return tuning().candidateHandoff(artifacts);
        }

        @Override public io.github.pho001.synaptik.runtime.execution.PreparedExecution prepareTrial(
                CpuLocalWorkloadTuning.CandidateBatch batch,
                CpuLocalWorkloadTuning.Candidate candidate) {
            return tuning().prepareTrial(batch, candidate);
        }

        @Override public io.github.pho001.synaptik.runtime.execution.PreparedExecution prepareSelected(
                CpuLocalWorkloadTuning.CandidateBatch batch,
                CpuLocalWorkloadTuning.SelectedDecision decision) {
            return tuning().prepareSelected(batch, decision);
        }

        @Override public List<CpuLocalWorkloadTuning.Candidate> candidates(
                CpuLocalWorkloadTuning.CandidateBatch batch) {
            return tuning().candidates(batch);
        }

        @Override public WorkloadTuningRequest.WorkloadCompatibility compatibility(
                CpuLocalWorkloadTuning.CandidateBatch batch) {
            var value = tuning().compatibility(batch);
            return new WorkloadTuningRequest.WorkloadCompatibility(
                    value.schemaVersion(), value.bytes(), switch (value.reuseScope()) {
                        case SESSION -> WorkloadTuningRequest.ReuseScope.SESSION;
                        case PERSISTENT -> WorkloadTuningRequest.ReuseScope.PERSISTENT;
                    });
        }

        @Override public WorkloadTuningRequest.CandidateIdentity candidateIdentity(
                CpuLocalWorkloadTuning.Candidate candidate) {
            return new WorkloadTuningRequest.CandidateIdentity(
                    tuning().candidateIdentity(candidate).bytes());
        }

        @Override public CpuLocalWorkloadTuning.SelectedDecision selectedDecision(
                CpuLocalWorkloadTuning.CandidateBatch batch,
                CpuLocalWorkloadTuning.Candidate candidate) {
            return tuning().selectedDecision(batch, candidate);
        }

        @Override public byte[] encodeDecision(
                CpuLocalWorkloadTuning.SelectedDecision decision) {
            return tuning().encodeDecision(decision);
        }

        @Override public Optional<CpuLocalWorkloadTuning.SelectedDecision>
                decodeCompatibleDecision(
                        CpuLocalWorkloadTuning.CandidateBatch batch, byte[] encodedDecision) {
            return tuning().decodeCompatibleDecision(batch, encodedDecision);
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
