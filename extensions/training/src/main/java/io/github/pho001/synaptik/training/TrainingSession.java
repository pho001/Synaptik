package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.InferenceSession;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Reusable public owner for one compiled/prepared training graph and its optimizer state.
 *
 * <p>{@link #open} borrows one caller-owned {@link Engine} and {@link Module}, snapshots every
 * recursive parameter path, wrapper, Tensor, and host-storage identity, compiles one scalar
 * objective with those exact parameter targets, and opens one {@link InferenceSession}. Each
 * {@link #run} supplies only the remaining bindable inputs. The session inserts its stable
 * parameters and private scalar cotangent seed in {@link CompiledGraph#inputs()} occurrence order,
 * reuses the one prepared execution, authenticates the objective-then-gradient publications by
 * target index, and applies backend-neutral optimizer arithmetic after detaching all values.</p>
 *
 * <p>The initial update domain is non-empty, connected {@code FLOAT32}/{@code FLOAT64} parameters
 * with finite initial values, fully static dense-contiguous offset-zero non-view descriptors, and
 * exact-capacity writable, shareable native host storage. Unsupported parameter capture fails
 * before private seed allocation or compilation. Values are updated in place so the compiled
 * Tensor identities remain stable. The caller grants this session exclusive use of the Module's
 * parameter wrappers, captured parameter storage associations, and parameter bytes until close.
 * Parameter replacement, storage replacement, overlapping external input storage, or concurrent
 * caller mutation violates that contract and is rejected when observable before mutation.</p>
 *
 * <p>All input/storage validation, Engine execution, result closure, gradient decoding,
 * candidate calculation, and a second complete parameter-view validation precede mutation.
 * Current parameters, decoded gradients, accumulation, momentum, optimizer intermediates, and
 * candidates must remain finite; NaN, infinity, and represented overflow fail before commit,
 * while signed zero is finite. Candidate, rollback, momentum, and accumulation primitive arrays
 * are allocated once at open and reused; the optimizer element loop allocates no object per
 * element. Validated native writes are non-throwing after validation. Ordinary failure before or
 * during the documented precommit work leaves parameter bytes, optimizer slots, counters, and
 * pending gradients unchanged. Fatal JVM failure or caller violation of exclusive storage access
 * is outside this atomicity guarantee.</p>
 *
 * <p>One operation may be in flight. Admission precedes operation-argument validation and
 * observes Engine closure first, then session closing/closure, then an already-admitted busy
 * operation. Close atomically rejects later admission, waits uninterruptibly for an earlier
 * admitted operation while restoring interrupt status, then attempts the owned inference session
 * and seed arena exactly once. Repeated close replays the retained cleanup result. Close never
 * closes the borrowed Engine, Module, Tensor, or caller storage. Engine close remains the final
 * composition boundary. Detached {@link TrainingStep} and {@link TrainingState} values outlive
 * both owners.</p>
 */
public final class TrainingSession implements AutoCloseable {
    private static final String CLOSED_MESSAGE = "training session is closed";
    private static final String ENGINE_CLOSED_MESSAGE = "training engine is closed";
    private static final String BUSY_MESSAGE = "training session already has an operation in flight";

    private enum Lifecycle { OPEN, CLOSING, CLOSED }

    enum Operation { RUN, ZERO_GRAD, STATE, RESTORE }

    static class Hooks {
        private static final Hooks NONE = new Hooks();

        void afterAdmission(Operation operation) { }

        void afterObjectiveMaterialization() { }

        void afterExecution() { }

        void beforeParameterWrite(int parameterIndex) { }

        void beforeOwnedResourceClose() { }
    }

    private final Object lifecycleLock = new Object();
    private final Engine engine;
    private final Module module;
    private final Tensor objective;
    private final Sgd optimizer;
    private final Tensor cotangentSeed;
    private final Arena seedArena;
    private final List<TrainingParameter> parameters;
    private final List<CompiledGraph.Input> externalInputs;
    private final CompiledGraph compiledGraph;
    private final InferenceSession inferenceSession;
    private final Tensor[] fixedCompiledInputs;
    private final int[] externalIndexByCompiledInput;
    private final Hooks hooks;

    private Lifecycle lifecycle = Lifecycle.OPEN;
    private boolean operationInFlight;
    private long executions;
    private long optimizerSteps;
    private long accumulatedGradientRuns;
    private Throwable closeFailure;

    private TrainingSession(
            Engine engine,
            Module module,
            Tensor objective,
            Sgd optimizer,
            Tensor cotangentSeed,
            Arena seedArena,
            List<TrainingParameter> parameters,
            List<CompiledGraph.Input> externalInputs,
            CompiledGraph compiledGraph,
            InferenceSession inferenceSession,
            Tensor[] fixedCompiledInputs,
            int[] externalIndexByCompiledInput,
            Hooks hooks) {
        this.engine = engine;
        this.module = module;
        this.objective = objective;
        this.optimizer = optimizer;
        this.cotangentSeed = cotangentSeed;
        this.parameters = parameters;
        this.seedArena = seedArena;
        this.externalInputs = externalInputs;
        this.compiledGraph = compiledGraph;
        this.inferenceSession = inferenceSession;
        this.fixedCompiledInputs = fixedCompiledInputs;
        this.externalIndexByCompiledInput = externalIndexByCompiledInput;
        this.hooks = hooks;
    }

    /**
     * Compiles and prepares one scalar-objective training session.
     *
     * <p>Arguments are null-checked in declaration order. Parameter discovery uses the Module's
     * deterministic recursive order and requires at least one parameter. Every parameter is a
     * differentiation target; Compiler disconnected-policy validation rejects an unused target.
     * Exact Tensor and non-empty storage overlap checks complete before compilation. The objective
     * must be a scalar {@code FLOAT32} or {@code FLOAT64} Tensor requesting gradients. The supplied
     * optimizer is borrowed as immutable configuration; this session initializes independent
     * zero momentum and gradient-accumulation slots.</p>
     *
     * <p>Construction performs exactly one ordinary Engine gradient compile and exactly one
     * ordinary session preparation. A preparation failure publishes no TrainingSession and closes
     * the private seed arena. The Engine and Module remain caller-owned and must outlive this
     * session; successful construction gives the result sole ownership of the new inward
     * inference session and private seed arena.</p>
     *
     * @param engine non-null open Engine borrowed for the complete session lifetime
     * @param module non-null caller-owned Module whose complete recursive parameter set is targeted
     * @param objective non-null scalar FLOAT32/FLOAT64 gradient-eligible expression built from the
     *     Module's exact current parameter Tensors
     * @param optimizer non-null immutable built-in optimizer configuration; currently {@link Sgd}
     * @return a fresh non-null open training owner with one compiled and prepared graph
     * @throws NullPointerException if an argument, discovered path/wrapper/value/storage, or
     *     inward required value is {@code null}
     * @throws IllegalArgumentException if objective, optimizer narrowing, parameter schema,
     *     parameter identity/storage uniqueness, Compiler semantics, or preparation is unsupported
     * @throws IllegalStateException if the Module has no parameters, parameter storage is absent,
     *     dead, or inaccessible, Engine closure has begun, or inward lifecycle state is invalid
     * @throws ArithmeticException if checked shape, byte, or counter-independent size arithmetic
     *     overflows
     * @throws RuntimeException if compilation or preparation reports another unchecked failure
     * @throws Error if allocation, compilation, or preparation reports a fatal failure
     */
    public static TrainingSession open(
            Engine engine,
            Module module,
            Tensor objective,
            Optimizer optimizer) {
        return openInternal(engine, module, objective, optimizer, Hooks.NONE);
    }

    static TrainingSession openForTesting(
            Engine engine,
            Module module,
            Tensor objective,
            Optimizer optimizer,
            Hooks hooks) {
        return openInternal(engine, module, objective, optimizer, hooks);
    }

    private static TrainingSession openInternal(
            Engine engine,
            Module module,
            Tensor objective,
            Optimizer optimizer,
            Hooks hooks) {
        Hooks lifecycleHooks = Objects.requireNonNull(hooks, "hooks");
        Engine borrowedEngine = Objects.requireNonNull(engine, "engine");
        Module borrowedModule = Objects.requireNonNull(module, "module");
        Tensor scalarObjective = Objects.requireNonNull(objective, "objective");
        Optimizer suppliedOptimizer = Objects.requireNonNull(optimizer, "optimizer");
        Sgd sgd = switch (suppliedOptimizer) {
            case Sgd configuration -> configuration;
        };

        validateObjective(scalarObjective);
        Map<String, Parameter> discovered = borrowedModule.parametersRecursively();
        if (discovered.isEmpty()) {
            throw new IllegalStateException("training module has no parameters");
        }
        var parameterSnapshot = new ArrayList<TrainingParameter>(discovered.size());
        IdentityHashMap<Tensor, String> tensors = new IdentityHashMap<>();
        for (Map.Entry<String, Parameter> entry : discovered.entrySet()) {
            TrainingParameter captured = TrainingParameter.capture(entry.getKey(), entry.getValue());
            String firstPath = tensors.putIfAbsent(captured.tensor(), captured.path());
            if (firstPath != null) {
                throw new IllegalArgumentException(
                        "parameter Tensor identity is repeated at paths "
                                + firstPath + " and " + captured.path());
            }
            for (TrainingParameter previous : parameterSnapshot) {
                if (previous.overlaps(captured.storage())) {
                    throw new IllegalArgumentException(
                            "parameter storages overlap at paths "
                                    + previous.path() + " and " + captured.path());
                }
            }
            parameterSnapshot.add(captured);
        }
        validateOptimizerNarrowing(sgd, parameterSnapshot);
        List<TrainingParameter> stableParameters = List.copyOf(parameterSnapshot);

        Arena seedArena = Arena.ofShared();
        try {
            Tensor seedInput = nativeScalarOne(
                    scalarObjective.descriptor().dataType(), seedArena);
            Tensor seed = seedInput.contiguous();
            List<Tensor> targets =
                    stableParameters.stream().map(TrainingParameter::tensor).toList();
            CompiledGraph graph = borrowedEngine.compile(
                    List.of(scalarObjective), List.of(seed), targets);
            BindingPlan bindings = bindingPlan(graph, seedInput, stableParameters);
            InferenceSession session = borrowedEngine.session(graph);
            return new TrainingSession(
                    borrowedEngine,
                    borrowedModule,
                    scalarObjective,
                    sgd,
                    seedInput,
                    seedArena,
                    stableParameters,
                    bindings.externalInputs,
                    graph,
                    session,
                    bindings.fixedInputs,
                    bindings.externalIndices,
                    lifecycleHooks);
        } catch (RuntimeException | Error failure) {
            try {
                seedArena.close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Returns the exact immutable compiled graph reused by every run.
     *
     * <p>Its metadata remains readable after session or Engine close. The handle is still owned by
     * the borrowed Engine, and this accessor does not grant a second prepared-execution owner.</p>
     *
     * @return the non-null exact graph created once during {@link #open}
     */
    public CompiledGraph compiledGraph() {
        return compiledGraph;
    }

    /**
     * Returns caller-supplied input occurrences after removing captured parameters and the private
     * cotangent seed from Compiler input order.
     *
     * <p>Each {@link #run} must supply one matching Tensor for every listed identity, in any list
     * order. The immutable result preserves relative Compiler occurrence order and carries no
     * storage or execution authority.</p>
     *
     * @return the same non-null immutable ordered metadata list on every call; possibly empty
     */
    public List<CompiledGraph.Input> inputs() {
        return externalInputs;
    }

    /**
     * Returns the exact immutable optimizer configuration borrowed at construction.
     *
     * @return the non-null configuration retained by exact reference
     */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Runs the fixed prepared graph and applies the selected gradient lifecycle atomically.
     *
     * <p>The input list contains only occurrences reported by {@link #inputs()} and may use any
     * order. Every required Tensor identity must occur exactly once with its exact compiled
     * descriptor and present, live, accessible storage of sufficient capacity. Foreign or
     * duplicate identities, missing inputs, and any non-empty storage overlap with another
     * compiled input are rejected before Engine execution. Captured parameter wrapper, Tensor,
     * storage, schema, liveness, accessibility, and non-overlap are checked before the run and
     * again immediately before update commit.</p>
     *
     * <p>The returned objective describes values before an optional parameter update. A reset or
     * accumulate-and-step success advances the optimizer step once; accumulate-only success does
     * not. Lifecycle admission happens before either argument is inspected. Any ordinary failure,
     * including a non-finite decoded gradient or arithmetic result, closes the temporary Engine
     * result and leaves all session-owned mutable state and counters unchanged.</p>
     *
     * @param inputs non-null list supplying every non-parameter bindable Tensor exactly once
     * @param gradientMode non-null reset, accumulate-only, or accumulate-and-step behavior
     * @return a fresh non-null detached result that outlives this session and its Engine
     * @throws NullPointerException if an argument or input element/storage is {@code null} after
     *     successful lifecycle admission
     * @throws IllegalArgumentException if input identity, descriptor, capacity, overlap, or
     *     publication metadata is invalid
     * @throws IllegalStateException if the Engine is closed, close has begun, another operation is
     *     active, the inward session is closed, a required storage is absent/dead/inaccessible, a
     *     captured parameter binding changed, or publication state is inconsistent
     * @throws ArithmeticException if a counter/byte calculation overflows or any required
     *     optimizer value or intermediate is non-finite before mutation
     * @throws RuntimeException if Engine work, materialization, or update reports another unchecked
     *     failure
     * @throws Error if inward work or allocation reports a fatal failure
     */
    public TrainingStep run(List<Tensor> inputs, GradientMode gradientMode) {
        beginOperation();
        try {
            hooks.afterAdmission(Operation.RUN);
            Objects.requireNonNull(inputs, "inputs");
            GradientMode mode = Objects.requireNonNull(gradientMode, "gradientMode");
            long nextExecution = Math.incrementExact(executions);
            boolean updates = mode != GradientMode.ACCUMULATE;
            long nextOptimizerStep = updates
                    ? Math.incrementExact(optimizerSteps) : optimizerSteps;
            long nextAccumulation = mode == GradientMode.RESET_AND_STEP
                    ? 0L : Math.incrementExact(accumulatedGradientRuns);

            validateModuleAndParameters();
            List<Tensor> compiledInputs = bindInputs(inputs);
            HostTensorValue objectiveValue = executeAndDecode(compiledInputs);
            hooks.afterExecution();

            if (mode != GradientMode.RESET_AND_STEP) {
                for (TrainingParameter parameter : parameters) {
                    parameter.stageAccumulation();
                }
            }
            if (!updates) {
                validateModuleAndParameters();
                for (TrainingParameter parameter : parameters) {
                    parameter.commitAccumulation();
                }
                publishCounters(nextExecution, optimizerSteps, nextAccumulation);
                return new TrainingStep(
                        nextExecution,
                        OptionalLong.empty(),
                        nextAccumulation,
                        objectiveValue);
            }

            boolean useAccumulated = mode == GradientMode.ACCUMULATE_AND_STEP;
            for (TrainingParameter parameter : parameters) {
                parameter.stageUpdate(optimizer, optimizerSteps, useAccumulated);
            }
            validateModuleAndParameters();
            commitParameterCandidates();
            for (TrainingParameter parameter : parameters) {
                parameter.commitMomentum();
                parameter.clearAccumulation();
            }
            publishCounters(nextExecution, nextOptimizerStep, 0L);
            return new TrainingStep(
                    nextExecution,
                    OptionalLong.of(nextOptimizerStep),
                    0L,
                    objectiveValue);
        } finally {
            finishOperation();
        }
    }

    /**
     * Clears pending gradient accumulation without changing parameter values, momentum slots, or
     * either progress counter.
     *
     * @throws IllegalStateException if close has begun, the Engine-owned session is closed, or
     *     another operation is active
     */
    public void zeroGrad() {
        beginOperation();
        try {
            hooks.afterAdmission(Operation.ZERO_GRAD);
            for (TrainingParameter parameter : parameters) {
                parameter.clearAccumulation();
            }
            publishCounters(executions, optimizerSteps, 0L);
        } finally {
            finishOperation();
        }
    }

    /**
     * Captures complete detached parameter, optimizer, progress, and pending-gradient state.
     *
     * <p>Lifecycle admission precedes capture. The operation validates the complete Module,
     * captured storage view, and finite parameter/momentum/accumulation values before copying.
     * The immutable result contains canonical bytes and remains readable after this session or its
     * Engine closes. It is not a durable file format.</p>
     *
     * @return a fresh non-null immutable in-memory state snapshot
     * @throws IllegalStateException if the Engine is closed, close has begun, another operation is
     *     active, the inward session is closed, captured Module/storage state changed, or current
     *     parameter/optimizer state is non-finite
     * @throws RuntimeException if current host memory cannot be read
     * @throws Error if snapshot allocation reports a fatal failure
     */
    public TrainingState state() {
        beginOperation();
        try {
            hooks.afterAdmission(Operation.STATE);
            validateModuleAndParameters();
            var snapshots = new ArrayList<TrainingState.ParameterState>(parameters.size());
            for (TrainingParameter parameter : parameters) {
                snapshots.add(parameter.snapshot());
            }
            return new TrainingState(
                    optimizer,
                    executions,
                    optimizerSteps,
                    accumulatedGradientRuns,
                    snapshots);
        } finally {
            finishOperation();
        }
    }

    /**
     * Restores one complete compatible in-memory state after validate-before-install preflight.
     *
     * <p>Lifecycle admission precedes inspection of {@code state}. The exact SGD configuration,
     * counters, parameter count, ordered paths, data types, Shapes, payload lengths, finite payload
     * values, momentum-disabled slot values, and current Module/storage view all validate before
     * any parameter write. Success installs parameter bytes in path order, then publishes
     * momentum, pending gradients, and counters. Failure before commit leaves all current state
     * unchanged. The supplied immutable state is not retained.</p>
     *
     * @param state non-null detached state from a schema-compatible session
     * @throws NullPointerException if {@code state} is {@code null} after successful admission
     * @throws IllegalArgumentException if optimizer, counters, paths, schemas, or finite payloads
     *     differ
     * @throws IllegalStateException if the Engine is closed, close has begun, another operation is
     *     active, the inward session is closed, or current Module/storage state changed
     * @throws RuntimeException if validated host parameter installation unexpectedly fails
     * @throws Error if fatal host access fails
     */
    public void restore(TrainingState state) {
        beginOperation();
        try {
            hooks.afterAdmission(Operation.RESTORE);
            TrainingState candidate = Objects.requireNonNull(state, "state");
            if (!optimizer.equals(candidate.optimizer())) {
                throw new IllegalArgumentException("training state optimizer configuration differs");
            }
            if (candidate.optimizerSteps() > candidate.executions()
                    || candidate.accumulatedGradientRuns() > candidate.executions()) {
                throw new IllegalArgumentException("training state counters are inconsistent");
            }
            if (candidate.parameters().size() != parameters.size()) {
                throw new IllegalArgumentException(
                        "training state parameter count mismatch: expected=" + parameters.size()
                                + ", actual=" + candidate.parameters().size());
            }
            validateModuleAndParameters();
            for (int index = 0; index < parameters.size(); index++) {
                parameters.get(index).validateState(candidate.parameters().get(index), optimizer);
            }
            for (int index = 0; index < parameters.size(); index++) {
                parameters.get(index).stageRestore(candidate.parameters().get(index));
            }
            validateModuleAndParameters();
            commitParameterCandidates();
            for (TrainingParameter parameter : parameters) {
                parameter.commitRestore();
            }
            publishCounters(
                    candidate.executions(),
                    candidate.optimizerSteps(),
                    candidate.accumulatedGradientRuns());
        } finally {
            finishOperation();
        }
    }

    /**
     * Returns the number of successful forward/backward executions.
     *
     * <p>The point-in-time value remains readable after close. An admitted operation publishes its
     * new value only after all of its state commits.</p>
     *
     * @return the current non-negative successful execution count
     */
    public long executions() {
        synchronized (lifecycleLock) {
            return executions;
        }
    }

    /**
     * Returns the number of successful optimizer updates.
     *
     * @return the current non-negative update count, no greater than {@link #executions()}
     */
    public long optimizerSteps() {
        synchronized (lifecycleLock) {
            return optimizerSteps;
        }
    }

    /**
     * Returns how many successful runs contribute to the pending gradient sum.
     *
     * @return the current non-negative pending-run count
     */
    public long accumulatedGradientRuns() {
        synchronized (lifecycleLock) {
            return accumulatedGradientRuns;
        }
    }

    /**
     * Reports whether this session or its owned Engine inference session has begun closure.
     *
     * @return {@code true} after explicit close admission or when Engine cleanup has closed the
     *     inward session; otherwise {@code false}
     */
    public boolean isClosed() {
        synchronized (lifecycleLock) {
            return lifecycle != Lifecycle.OPEN || inferenceSession.isClosed();
        }
    }

    /**
     * Rejects new operations, waits for an admitted operation, and closes owned resources once.
     *
     * <p>The first closer becomes the cleanup owner. Concurrent closers wait for the same result.
     * Waiting is uninterruptible and restores the thread's interrupt status. Cleanup attempts the
     * inference session and private seed arena in that order. Repeated calls replay the exact
     * retained unchecked cleanup failure, if any. The borrowed Engine, Module, parameter Tensors,
     * and storage are never closed.</p>
     *
     * @throws RuntimeException if owned-resource cleanup first reports an unchecked failure
     * @throws Error if cleanup reports a fatal failure
     */
    @Override
    public void close() {
        boolean interrupted = false;
        synchronized (lifecycleLock) {
            if (lifecycle == Lifecycle.CLOSED) {
                rethrow(closeFailure);
                return;
            }
            if (lifecycle == Lifecycle.CLOSING) {
                while (lifecycle != Lifecycle.CLOSED) {
                    try {
                        lifecycleLock.wait();
                    } catch (InterruptedException interruption) {
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
            while (operationInFlight) {
                try {
                    lifecycleLock.wait();
                } catch (InterruptedException interruption) {
                    interrupted = true;
                }
            }
        }

        Throwable failure = null;
        try {
            inferenceSession.close();
        } catch (RuntimeException | Error cleanupFailure) {
            failure = cleanupFailure;
        }
        try {
            hooks.beforeOwnedResourceClose();
        } catch (RuntimeException | Error cleanupFailure) {
            if (failure == null) {
                failure = cleanupFailure;
            } else if (cleanupFailure != failure) {
                failure.addSuppressed(cleanupFailure);
            }
        }
        try {
            seedArena.close();
        } catch (RuntimeException | Error cleanupFailure) {
            if (failure == null) {
                failure = cleanupFailure;
            } else if (cleanupFailure != failure) {
                failure.addSuppressed(cleanupFailure);
            }
        }
        synchronized (lifecycleLock) {
            closeFailure = failure;
            lifecycle = Lifecycle.CLOSED;
            lifecycleLock.notifyAll();
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        rethrow(failure);
    }

    private HostTensorValue executeAndDecode(List<Tensor> inputs) {
        RunResult result = null;
        Throwable primary = null;
        try {
            result = inferenceSession.run(inputs);
            validatePublications(result);
            List<RunResult.Publication> publications = result.publications();
            HostTensorValue objectiveValue = result.materialize(
                    publications.getFirst(), canonicalByteSize(publications.getFirst()));
            hooks.afterObjectiveMaterialization();
            for (int index = 0; index < parameters.size(); index++) {
                RunResult.Publication publication = publications.get(index + 1);
                HostTensorValue gradient = result.materialize(
                        publication, canonicalByteSize(publication));
                parameters.get(index).decodeGradient(gradient);
            }
            return objectiveValue;
        } catch (RuntimeException | Error failure) {
            primary = failure;
            throw failure;
        } finally {
            if (result != null) {
                try {
                    result.close();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (primary == null) {
                        throw cleanupFailure;
                    }
                    if (cleanupFailure != primary) {
                        primary.addSuppressed(cleanupFailure);
                    }
                }
            }
        }
    }

    private void validatePublications(RunResult result) {
        if (result.resultCount() != parameters.size() + 1) {
            throw new IllegalStateException(
                    "training publication count mismatch: expected=" + (parameters.size() + 1)
                            + ", actual=" + result.resultCount());
        }
        List<RunResult.Publication> publications = result.publications();
        RunResult.Publication forward = publications.getFirst();
        if (forward.index() != 0
                || forward.role() != RunResult.Role.FORWARD
                || !forward.tensorId().equals(objective.id())
                || forward.derivativeOrder().isPresent()
                || forward.targetIndex().isPresent()) {
            throw new IllegalStateException("training objective publication metadata is invalid");
        }
        if (forward.descriptor().dataType() != objective.descriptor().dataType()
                || !forward.descriptor().shape().equals(objective.descriptor().shape())) {
            throw new IllegalStateException("training objective publication schema is invalid");
        }
        for (int index = 0; index < parameters.size(); index++) {
            TrainingParameter parameter = parameters.get(index);
            RunResult.Publication gradient = publications.get(index + 1);
            if (gradient.index() != index + 1
                    || gradient.role() != RunResult.Role.GRADIENT
                    || !gradient.tensorId().equals(parameter.tensor().id())
                    || gradient.derivativeOrder().orElse(-1) != 1
                    || gradient.targetIndex().orElse(-1) != index
                    || gradient.descriptor().dataType() != parameter.dataType()
                    || !gradient.descriptor().shape().equals(parameter.shape())) {
                throw new IllegalStateException(
                        "training gradient publication metadata is invalid at path "
                                + parameter.path());
            }
        }
    }

    private List<Tensor> bindInputs(List<Tensor> suppliedInputs) {
        if (suppliedInputs.size() != externalInputs.size()) {
            throw new IllegalArgumentException(
                    "training input count mismatch: expected=" + externalInputs.size()
                            + ", actual=" + suppliedInputs.size());
        }
        Tensor[] external = new Tensor[externalInputs.size()];
        for (int suppliedIndex = 0; suppliedIndex < suppliedInputs.size(); suppliedIndex++) {
            Tensor supplied = Objects.requireNonNull(
                    suppliedInputs.get(suppliedIndex), "inputs[" + suppliedIndex + "]");
            int expectedIndex = findExternalInput(supplied.id());
            if (expectedIndex < 0) {
                throw new IllegalArgumentException(
                        "foreign training input Tensor identity: " + supplied.id());
            }
            if (external[expectedIndex] != null) {
                throw new IllegalArgumentException(
                        "duplicate training input Tensor identity: " + supplied.id());
            }
            CompiledGraph.Input expected = externalInputs.get(expectedIndex);
            if (!supplied.descriptor().equals(expected.descriptor())) {
                throw new IllegalArgumentException(
                        "training input descriptor does not match " + expected.tensorId());
            }
            external[expectedIndex] = supplied;
        }
        for (int index = 0; index < external.length; index++) {
            if (external[index] == null) {
                throw new IllegalArgumentException(
                        "missing training input Tensor identity: "
                                + externalInputs.get(index).tensorId());
            }
        }

        var compiled = new ArrayList<Tensor>(fixedCompiledInputs.length);
        HostTensorStorage[] storages = new HostTensorStorage[fixedCompiledInputs.length];
        String[] roles = new String[fixedCompiledInputs.length];
        for (int compiledIndex = 0; compiledIndex < fixedCompiledInputs.length; compiledIndex++) {
            Tensor tensor = fixedCompiledInputs[compiledIndex];
            if (tensor == null) {
                tensor = external[externalIndexByCompiledInput[compiledIndex]];
            }
            HostTensorStorage storage = tensor.hostStorage().orElse(null);
            if (storage == null) {
                throw new IllegalStateException(
                        "training input has no host storage: " + tensor.id());
            }
            validateInputStorage(
                    compiledGraph.inputs().get(compiledIndex), storage, tensor == cotangentSeed);
            compiled.add(tensor);
            storages[compiledIndex] = storage;
            roles[compiledIndex] = tensor == cotangentSeed
                    ? "cotangent seed"
                    : parameterPath(tensor.id()).orElse("input " + tensor.id());
        }
        for (int left = 0; left < storages.length; left++) {
            if (storages[left].byteSize() == 0) {
                continue;
            }
            for (int right = left + 1; right < storages.length; right++) {
                if (storages[right].byteSize() != 0
                        && storages[left].segment()
                                .asOverlappingSlice(storages[right].segment()).isPresent()) {
                    throw new IllegalArgumentException(
                            "training input storages overlap: "
                                    + roles[left] + " and " + roles[right]);
                }
            }
        }
        return List.copyOf(compiled);
    }

    private int findExternalInput(TensorId id) {
        for (int index = 0; index < externalInputs.size(); index++) {
            if (externalInputs.get(index).tensorId().equals(id)) {
                return index;
            }
        }
        return -1;
    }

    private Optional<String> parameterPath(TensorId id) {
        for (TrainingParameter parameter : parameters) {
            if (parameter.tensor().id().equals(id)) {
                return Optional.of("parameter " + parameter.path());
            }
        }
        return Optional.empty();
    }

    private void validateModuleAndParameters() {
        Map<String, Parameter> current = module.parametersRecursively();
        if (current.size() != parameters.size()) {
            throw new IllegalStateException(
                    "module parameter structure changed while training session is open");
        }
        Iterator<Map.Entry<String, Parameter>> currentEntries = current.entrySet().iterator();
        for (TrainingParameter expected : parameters) {
            Map.Entry<String, Parameter> actual = currentEntries.next();
            if (!expected.path().equals(actual.getKey())
                    || expected.parameter() != actual.getValue()) {
                throw new IllegalStateException(
                        "module parameter structure changed while training session is open");
            }
            expected.validateBinding();
            expected.validateFiniteState();
        }
        for (int left = 0; left < parameters.size(); left++) {
            for (int right = left + 1; right < parameters.size(); right++) {
                if (parameters.get(left).overlaps(parameters.get(right).storage())) {
                    throw new IllegalStateException(
                            "parameter storages overlap while training session is open: "
                                    + parameters.get(left).path() + " and "
                                    + parameters.get(right).path());
                }
            }
        }
    }

    private void commitParameterCandidates() {
        int committing = -1;
        try {
            for (int index = 0; index < parameters.size(); index++) {
                committing = index;
                hooks.beforeParameterWrite(index);
                parameters.get(index).writeParameterCandidate();
            }
        } catch (RuntimeException | Error failure) {
            for (int index = committing; index >= 0; index--) {
                try {
                    parameters.get(index).rollbackParameter();
                } catch (RuntimeException | Error rollbackFailure) {
                    if (rollbackFailure != failure) {
                        failure.addSuppressed(rollbackFailure);
                    }
                }
            }
            throw failure;
        }
    }

    private void beginOperation() {
        synchronized (lifecycleLock) {
            if (engine.isClosed()) {
                throw new IllegalStateException(ENGINE_CLOSED_MESSAGE);
            }
            if (lifecycle != Lifecycle.OPEN || inferenceSession.isClosed()) {
                throw new IllegalStateException(CLOSED_MESSAGE);
            }
            if (operationInFlight) {
                throw new IllegalStateException(BUSY_MESSAGE);
            }
            operationInFlight = true;
        }
    }

    private void publishCounters(
            long executionCount, long optimizerStepCount, long accumulationCount) {
        synchronized (lifecycleLock) {
            executions = executionCount;
            optimizerSteps = optimizerStepCount;
            accumulatedGradientRuns = accumulationCount;
        }
    }

    private void finishOperation() {
        synchronized (lifecycleLock) {
            operationInFlight = false;
            lifecycleLock.notifyAll();
        }
    }

    private static void validateObjective(Tensor objective) {
        var descriptor = objective.descriptor();
        if (descriptor.dataType() != DataType.FLOAT32
                && descriptor.dataType() != DataType.FLOAT64) {
            throw new IllegalArgumentException(
                    "training objective must use FLOAT32 or FLOAT64: "
                            + descriptor.dataType());
        }
        if (descriptor.shape().rank() != 0) {
            throw new IllegalArgumentException(
                    "training objective must be scalar: " + descriptor.shape());
        }
        if (!descriptor.requiresGrad()) {
            throw new IllegalArgumentException("training objective must require gradients");
        }
    }

    private static void validateOptimizerNarrowing(
            Sgd optimizer, List<TrainingParameter> parameters) {
        boolean hasFloat32 = parameters.stream()
                .anyMatch(parameter -> parameter.dataType() == DataType.FLOAT32);
        if (!hasFloat32) {
            return;
        }
        float learningRate = (float) optimizer.learningRate();
        float momentum = (float) optimizer.momentum();
        float dampening = (float) optimizer.dampening();
        float weightDecay = (float) optimizer.weightDecay();
        if (!Float.isFinite(learningRate) || learningRate <= 0.0f) {
            throw new IllegalArgumentException(
                    "learningRate is not finite and positive after FLOAT32 narrowing");
        }
        if (!Float.isFinite(momentum) || momentum < 0.0f || momentum >= 1.0f) {
            throw new IllegalArgumentException(
                    "momentum is not in [0, 1) after FLOAT32 narrowing");
        }
        if (!Float.isFinite(dampening) || dampening < 0.0f || dampening > 1.0f) {
            throw new IllegalArgumentException(
                    "dampening is not in [0, 1] after FLOAT32 narrowing");
        }
        if (!Float.isFinite(weightDecay) || weightDecay < 0.0f) {
            throw new IllegalArgumentException(
                    "weightDecay is not finite and non-negative after FLOAT32 narrowing");
        }
        if (optimizer.nesterov() && momentum == 0.0f) {
            throw new IllegalArgumentException(
                    "Nesterov momentum is zero after FLOAT32 narrowing");
        }
        if (optimizer.nesterov() && dampening != 0.0f) {
            throw new IllegalArgumentException(
                    "Nesterov dampening is non-zero after FLOAT32 narrowing");
        }

    }

    private static BindingPlan bindingPlan(
            CompiledGraph graph,
            Tensor seed,
            List<TrainingParameter> parameters) {
        List<CompiledGraph.Input> compiledInputs = graph.inputs();
        Tensor[] fixed = new Tensor[compiledInputs.size()];
        int[] externalIndices = new int[compiledInputs.size()];
        Arrays.fill(externalIndices, -1);
        boolean[] parameterSeen = new boolean[parameters.size()];
        boolean seedSeen = false;
        var external = new ArrayList<CompiledGraph.Input>();
        for (int compiledIndex = 0; compiledIndex < compiledInputs.size(); compiledIndex++) {
            CompiledGraph.Input input = compiledInputs.get(compiledIndex);
            if (input.tensorId().equals(seed.id())) {
                if (seedSeen) {
                    throw new IllegalStateException("cotangent seed input occurrence is duplicated");
                }
                seedSeen = true;
                fixed[compiledIndex] = seed;
                continue;
            }
            int parameterIndex = findParameter(parameters, input.tensorId());
            if (parameterIndex >= 0) {
                if (parameterSeen[parameterIndex]) {
                    throw new IllegalStateException(
                            "parameter input occurrence is duplicated: "
                                    + parameters.get(parameterIndex).path());
                }
                parameterSeen[parameterIndex] = true;
                fixed[compiledIndex] = parameters.get(parameterIndex).tensor();
                continue;
            }
            externalIndices[compiledIndex] = external.size();
            external.add(input);
        }
        if (!seedSeen) {
            throw new IllegalStateException("compiled training graph has no cotangent seed input");
        }
        for (int index = 0; index < parameterSeen.length; index++) {
            if (!parameterSeen[index]) {
                throw new IllegalStateException(
                        "compiled training graph has no input for parameter path "
                                + parameters.get(index).path());
            }
        }
        return new BindingPlan(
                List.copyOf(external), fixed, externalIndices);
    }

    private static int findParameter(
            List<TrainingParameter> parameters, TensorId tensorId) {
        for (int index = 0; index < parameters.size(); index++) {
            if (parameters.get(index).tensor().id().equals(tensorId)) {
                return index;
            }
        }
        return -1;
    }

    private static void validateInputStorage(
            CompiledGraph.Input input, HostTensorStorage storage, boolean requireExact) {
        Objects.requireNonNull(storage, "storage");
        if (storage.dataType() != input.descriptor().dataType()) {
            throw new IllegalArgumentException(
                    "training input storage data type does not match " + input.tensorId());
        }
        long required = input.descriptor().layout()
                .map(layout -> layout.referencedElementSpan()).orElse(0L);
        if (requireExact ? storage.elementCapacity() != required
                : storage.elementCapacity() < required) {
            throw new IllegalArgumentException(
                    "training input storage capacity is incompatible with " + input.tensorId());
        }
        if (!storage.isAlive()) {
            throw new IllegalStateException(
                    "training input storage is not alive: " + input.tensorId());
        }
        if (!storage.segment().isAccessibleBy(Thread.currentThread())) {
            throw new IllegalStateException(
                    "training input storage is not accessible to current thread: "
                            + input.tensorId());
        }
    }

    private static long canonicalByteSize(RunResult.Publication publication) {
        if (!publication.descriptor().shape().isFullyStatic()) {
            throw new IllegalStateException("training publication shape is not fully static");
        }
        return Math.multiplyExact(
                publication.descriptor().shape().knownElementCount().orElseThrow(),
                publication.descriptor().dataType().byteWidth());
    }
    private static Tensor nativeScalarOne(DataType dataType, Arena arena) {
        Shape shape = Shape.scalar();
        TensorDescriptor descriptor = new TensorDescriptor(
                dataType,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        MemorySegmentStorage storage = new MemorySegmentStorage(
                dataType,
                1,
                arena.allocate(dataType.byteWidth(), dataType.byteWidth()));
        if (dataType == DataType.FLOAT32) {
            storage.segment().set(ValueLayout.JAVA_FLOAT, 0, 1.0f);
        } else {
            storage.segment().set(ValueLayout.JAVA_DOUBLE, 0, 1.0d);
        }
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
    }


    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
    }

    private record BindingPlan(
            List<CompiledGraph.Input> externalInputs,
            Tensor[] fixedInputs,
            int[] externalIndices) {
        private BindingPlan {
            Objects.requireNonNull(externalInputs, "externalInputs");
            Objects.requireNonNull(fixedInputs, "fixedInputs");
            Objects.requireNonNull(externalIndices, "externalIndices");
        }
    }
}
