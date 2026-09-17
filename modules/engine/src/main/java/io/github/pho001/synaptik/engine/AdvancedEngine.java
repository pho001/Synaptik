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
