package io.github.pho001.synaptik.training;

import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
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
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

/**
 * Reusable public owner for one compiled/prepared training graph and its optimizer state.
 *
 * <p>{@link #open} borrows one caller-owned {@link Engine} and live {@link TrainingParameters}, snapshots every
 * ordered parameter path, Tensor, and host-storage identity, compiles one scalar
 * objective with those exact parameter targets, and opens one {@link PreparedExecution}. Each
 * {@link #run} or {@link #runValues} supplies only the remaining bindable inputs. The former uses
 * exact compiled identities in arbitrary order; the latter accepts fresh stored values in
 * {@link #inputs()} order with exact full static descriptors. The session inserts its stable
 * parameters and private scalar cotangent seed in {@link CompiledGraph#inputs()} occurrence order,
 * reuses the one prepared execution, authenticates the objective-then-gradient publications by
 * target index, and applies backend-neutral optimizer arithmetic after detaching all values.</p>
 *
 * <p>The initial update domain is non-empty, connected floating parameters with finite initial
 * values, fully static dense-contiguous offset-zero non-view descriptors, and exact-capacity
 * writable, shareable native host storage. Each parameter's declared-width and optimizer-width
 * snapshot payload must independently fit a Java byte array. {@code BFLOAT16} and
 * {@code FLOAT16} parameters retain private {@code FLOAT32} master values and publish one rounded
 * logical value per successful update. Unsupported parameter capture fails before private seed
 * allocation or compilation.
 * Tensor identities remain stable. The caller grants this session exclusive use of the live source's
 * membership, captured parameter storage associations, and parameter bytes until close.
 * Parameter-pair non-overlap is established at open: Model's sealed host-storage implementation
 * retains each segment, and run/state/restore binding checks reject any changed association.
 * Parameter replacement, storage replacement, overlapping external input storage, or concurrent
 * caller mutation violates that contract and is rejected when observable before mutation.</p>
 *
 * <p>All input/storage validation, Engine execution, result closure, gradient decoding,
 * candidate calculation, and a second complete parameter-view validation precede mutation.
 * Current parameters, decoded gradients, accumulation, optimizer slots, intermediates, and
 * candidates must remain finite; NaN, infinity, and represented overflow fail before commit,
 * while signed zero is finite. Active first slots and accumulation, parameter candidates, and
 * rollback arrays are allocated at open. Inactive accumulation staging is allocated on accumulation
 * or restore and released after successful clearing or restore; momentum staging is allocated
 * for nonzero assigned SGD momentum, adaptive updates, or restore and is not retained for exactly
 * zero assigned SGD momentum. An active second-moment array exists only for Adam/AdamW paths;
 * its candidate is allocated only when an update or restore needs it. Explicit
 * {@link ParameterGroup} paths choose one algorithm per parameter at open; unlisted paths use the
 * default. All parameters still share one run and atomic commit.
 * Staging allocation precedes parameter writes, and the optimizer element loop allocates no
 * object per element. Validated native writes and subsequent slot commits are non-allocating and
 * non-throwing after validation. Ordinary failure before or
 * during the documented precommit work leaves parameter bytes, optimizer slots, counters, and
 * pending gradients unchanged. Fatal JVM failure or caller violation of exclusive storage access
 * is outside this atomicity guarantee.</p>
 *
 * <p>One operation may be in flight. Admission precedes operation-argument validation and
 * observes Engine closure first, then session closing/closure, then an already-admitted busy
 * operation. Close atomically rejects later admission, waits uninterruptibly for an earlier
 * admitted operation while restoring interrupt status, then attempts the owned prepared execution
 * and seed arena exactly once. Repeated close replays the retained cleanup result. Close never
 * closes the borrowed Engine, parameter source, Tensor, or caller storage. Engine close remains the final
 * composition boundary. Detached {@link TrainingStep} and {@link TrainingState} values outlive
 * both owners.</p>
 *
 * <p>External storage is borrowed only through each synchronous run and its internal result
 * closure, never retained for the next call. Keep that storage alive, accessible and free of
 * conflicting mutation until the call returns. No run copies or rebinds prototype storage or
 * executes a supplied expression in place of its stored value. Training rejects nonempty input
 * overlap even where generic Engine positional execution permits repeated values.</p>
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
    private final TrainingParameters parameterSource;
    private final Tensor objective;
    private final Optimizer optimizer;
    private final List<ParameterGroup> parameterGroups;
    private final Tensor cotangentSeed;
    private final Arena seedArena;
    private final List<TrainingParameter> parameters;
    private final List<CompiledGraph.Input> externalInputs;
    private final CompiledGraph compiledGraph;
    private final PreparedExecution preparedExecution;
    private final Tensor[] fixedCompiledInputs;
    private final int[] externalIndexByCompiledInput;
    private final String[] compiledInputRoles;
    private final int[] nonParameterCompiledIndices;
    private final Hooks hooks;

    private Lifecycle lifecycle = Lifecycle.OPEN;
    private boolean operationInFlight;
    private long executions;
    private long optimizerSteps;
    private long accumulatedGradientRuns;
    private Throwable closeFailure;

    /**
     * Adopts a completely captured training plan after successful Engine preparation.
     * The caller retains rollback responsibility until construction returns; cleanup closes
     * prepared execution before seed memory. Parameter stores and the Engine remain borrowed.
     *
     * @param engine exact borrowed Engine used for compilation and every run
     * @param parameterSource borrowed live source requiring exclusive stable membership until close
     * @param objective exact scalar expression compiled for this session
     * @param optimizer immutable default optimizer configuration
     * @param parameterGroups immutable ordered parameter-group snapshot
     * @param cotangentSeed stable private scalar seed input with caller-invisible storage
     * @param seedArena owned shared arena backing the seed, closed after preparation
     * @param parameters stable ordered parameter captures and private optimizer state
     * @param externalInputs immutable metadata for inputs supplied by Training callers
     * @param compiledGraph exact owner-bound compiled forward/backward handle
     * @param preparedExecution exact owned Engine preparation, reused without recompilation
     * @param fixedCompiledInputs parameter/seed entries in final compiled order
     * @param externalIndexByCompiledInput external-input positions aligned with compiled inputs
     * @param compiledInputRoles diagnostic roles aligned with compiled input positions
     * @param nonParameterCompiledIndices compiled positions validated as non-parameter storage
     * @param hooks non-null test lifecycle hooks; production uses inert hooks
     */
    private TrainingSession(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer optimizer,
            List<ParameterGroup> parameterGroups,
            Tensor cotangentSeed,
            Arena seedArena,
            List<TrainingParameter> parameters,
            List<CompiledGraph.Input> externalInputs,
            CompiledGraph compiledGraph,
            PreparedExecution preparedExecution,
            Tensor[] fixedCompiledInputs,
            int[] externalIndexByCompiledInput,
            String[] compiledInputRoles,
            int[] nonParameterCompiledIndices,
            Hooks hooks) {
        this.engine = engine;
        this.parameterSource = parameterSource;
        this.objective = objective;
        this.optimizer = optimizer;
        this.parameterGroups = parameterGroups;
        this.cotangentSeed = cotangentSeed;
        this.parameters = parameters;
        this.seedArena = seedArena;
        this.externalInputs = externalInputs;
        this.compiledGraph = compiledGraph;
        this.preparedExecution = preparedExecution;
        this.fixedCompiledInputs = fixedCompiledInputs;
        this.externalIndexByCompiledInput = externalIndexByCompiledInput;
        this.compiledInputRoles = compiledInputRoles;
        this.nonParameterCompiledIndices = nonParameterCompiledIndices;
        this.hooks = hooks;
    }

    /**
     * Compiles and prepares one scalar-objective training session using one default optimizer.
     *
     * <p>Arguments are null-checked in declaration order. Parameter discovery uses the source's
     * immutable encounter order, validates its live owner, and requires at least one parameter.
     * Every parameter is a differentiation target; Compiler disconnected-policy validation
     * rejects an unused target. Null entries and duplicate paths reject before parameter capture.
     * Each parameter's logical and optimizer-width snapshot byte sizes must be at most
     * {@link Integer#MAX_VALUE}; this is checked during capture before that parameter's update
     * buffers or the private seed are allocated. Earlier parameters may already have been
     * captured. Exact Tensor and non-empty storage overlap checks complete
     * before compilation. The objective must be a scalar {@code FLOAT32} or {@code FLOAT64}
     * Tensor requesting gradients. The supplied optimizer is borrowed as immutable configuration;
     * this session initializes independent zero first-slot and gradient-accumulation arrays, plus
     * second-moment arrays only for adaptive parameters, but
     * allocates their inactive staging arrays only when an operation needs them.</p>
     *
     * <p>Construction performs exactly one ordinary Engine gradient compile and exactly one
     * ordinary graph preparation. A preparation failure publishes no TrainingSession and closes
     * the private seed arena. The Engine and source remain caller-owned and must outlive this
     * session; successful construction gives the result sole ownership of the new inward
     * prepared execution and private seed arena. Failed opening closes an acquired prepared owner
     * before seed memory, and every run closes its result before either owner.</p>
     *
     * <p>Admission accepts a generic live source, not an NN Module. Both {@link #run} and
     * {@link #runValues} reuse this preparation and the same optimizer transaction, with fixed
     * parameters and seed and call-local borrowed external storage.</p>
     *
     * @param engine non-null open Engine borrowed for the complete session lifetime
     * @param parameterSource non-null caller-owned live source whose complete ordered membership is targeted
     * @param objective non-null scalar FLOAT32/FLOAT64 gradient-eligible expression built from the
     *     source's exact current parameter Tensors
     * @param optimizer non-null immutable {@link Sgd}, {@link Adam}, or {@link AdamW} default
     * @return a fresh non-null open training owner with one compiled and prepared graph
     * @throws NullPointerException if an argument, discovered list/entry/path/Tensor/storage, or
     *     inward required value is {@code null}
     * @throws IllegalArgumentException if objective, optimizer narrowing, parameter schema or
     *     either snapshot byte size, parameter identity/storage uniqueness, Compiler semantics,
     *     or preparation is unsupported
     * @throws IllegalStateException if the source has no parameters or live membership changed,
     *     parameter storage is absent, dead or inaccessible, Engine closure has begun, or inward
     *     lifecycle state is invalid
     * @throws ArithmeticException if checked shape, byte, or counter-independent size arithmetic
     *     overflows
     * @throws RuntimeException if compilation or preparation reports another unchecked failure
     * @throws Error if allocation, compilation, or preparation reports a fatal failure
     */
    public static TrainingSession open(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer optimizer) {
        return openInternal(engine, parameterSource, objective, optimizer, List.of(), Hooks.NONE);
    }

    /**
     * Opens one training session with optimizer configurations assigned to selected parameters.
     *
     * <p>The ordered groups and each group's ordered paths are copied and form part of exact
     * {@link TrainingState} restore compatibility. Paths must match keys in the captured
     * {@link TrainingParameters#entries()} snapshot; every path may occur in at most one group.
     * Unlisted parameters use {@code defaultOptimizer}. All discovered parameters remain gradient
     * targets, and all groups share one graph, execution counter, optimizer-step counter,
     * accumulation lifecycle, and atomic commit. An empty list behaves as the four-argument
     * overload. Group membership and assigned coefficient narrowing validate before seed
     * allocation, compilation, or preparation.</p>
     *
     * @param engine non-null open Engine borrowed for the complete session lifetime
     * @param parameterSource non-null caller-owned live source with a nonempty ordered parameter set
     * @param objective non-null scalar FLOAT32/FLOAT64 gradient-eligible expression using the
     *     source's current parameter Tensors
     * @param defaultOptimizer non-null immutable {@link Sgd}, {@link Adam}, or {@link AdamW}
     * @param groups non-null ordered list of non-null immutable groups, copied at open
     * @return a fresh non-null session with one compiled and prepared graph
     * @throws NullPointerException if an argument, group, or required discovered value is null
     * @throws IllegalArgumentException if a group path is absent or claimed more than once, a
     *     coefficient cannot narrow for its assigned parameter, or capture/compile rejects input
     * @throws IllegalStateException if the source is empty or live membership changed, storage is
     *     unusable, or Engine is closed
     * @throws ArithmeticException if checked size arithmetic overflows
     * @throws RuntimeException if compilation or preparation reports another unchecked failure
     * @throws Error if allocation, compilation, or preparation reports a fatal failure
     */
    public static TrainingSession open(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer defaultOptimizer,
            List<ParameterGroup> groups) {
        return openInternal(engine, parameterSource, objective, defaultOptimizer, groups, Hooks.NONE);
    }

    /**
     * Opens the ordinary ungrouped path with package-private deterministic lifecycle hooks.
     *
     * @param engine non-null borrowed open Engine
     * @param parameterSource non-null live ordered caller-owned source
     * @param objective non-null scalar FLOAT32/FLOAT64 gradient-eligible objective
     * @param optimizer non-null immutable default configuration
     * @param hooks non-null deterministic test hooks
     * @return a fresh owned session using ordinary admission and cleanup
     * @throws RuntimeException if ordinary open admission, compile or prepare fails
     */
    static TrainingSession openForTesting(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer optimizer,
            Hooks hooks) {
        return openInternal(engine, parameterSource, objective, optimizer, List.of(), hooks);
    }

    /**
     * Opens the ordinary grouped path with package-private deterministic lifecycle hooks.
     *
     * @param engine non-null borrowed open Engine
     * @param parameterSource non-null live ordered caller-owned source
     * @param objective non-null scalar FLOAT32/FLOAT64 gradient-eligible objective
     * @param optimizer non-null immutable default configuration
     * @param groups non-null ordered groups with distinct admitted paths
     * @param hooks non-null deterministic test hooks
     * @return a fresh owned session using ordinary grouped admission and cleanup
     * @throws RuntimeException if ordinary open admission, compile or prepare fails
     */
    static TrainingSession openForTesting(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer optimizer,
            List<ParameterGroup> groups,
            Hooks hooks) {
        return openInternal(engine, parameterSource, objective, optimizer, groups, hooks);
    }

    /**
     * Snapshots complete live membership and native bindings before acquiring execution owners.
     *
     * @param engine borrowed open Engine
     * @param parameterSource live caller-owned source, validated independently of frozen entries
     * @param objective scalar gradient-eligible objective using exact source Tensors
     * @param optimizer immutable default configuration
     * @param groups ordered immutable group configurations, copied before capture
     * @param hooks non-null lifecycle test hooks, inert in production
     * @return a new session owning preparation and seed, never the source or caller storage
     * @throws RuntimeException if admission, capture, compile or prepare fails
     * @throws Error if allocation or inward execution-owner construction fails fatally
     */
    private static TrainingSession openInternal(
            Engine engine,
            TrainingParameters parameterSource,
            Tensor objective,
            Optimizer optimizer,
            List<ParameterGroup> groups,
            Hooks hooks) {
        Hooks lifecycleHooks = Objects.requireNonNull(hooks, "hooks");
        Engine borrowedEngine = Objects.requireNonNull(engine, "engine");
        TrainingParameters borrowedSource = Objects.requireNonNull(parameterSource, "parameters");
        Tensor scalarObjective = Objects.requireNonNull(objective, "objective");
        Optimizer suppliedOptimizer = Objects.requireNonNull(optimizer, "optimizer");
        List<ParameterGroup> stableGroups = List.copyOf(Objects.requireNonNull(groups, "groups"));

        validateObjective(scalarObjective);
        List<TrainingParameters.Entry> discovered = TrainingParameter.captureEntries(borrowedSource);
        Map<String, Optimizer> groupedOptimizers = new HashMap<>();
        for (ParameterGroup group : stableGroups) {
            for (String path : group.paths()) {
                if (discovered.stream().noneMatch(entry -> entry.path().equals(path))) {
                    throw new IllegalArgumentException("unknown parameter group path: " + path);
                }
                if (groupedOptimizers.putIfAbsent(path, group.optimizer()) != null) {
                    throw new IllegalArgumentException("parameter path belongs to multiple groups: "
                            + path);
                }
            }
        }
        var parameterSnapshot = new ArrayList<TrainingParameter>(discovered.size());
        IdentityHashMap<Tensor, String> tensors = new IdentityHashMap<>();
        for (TrainingParameters.Entry entry : discovered) {
            TrainingParameter captured = TrainingParameter.capture(entry.path(), entry.tensor());
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
            Optimizer assigned = groupedOptimizers.getOrDefault(captured.path(), suppliedOptimizer);
            captured.configureOptimizer(assigned);
            parameterSnapshot.add(captured);
        }
        List<TrainingParameter> stableParameters = List.copyOf(parameterSnapshot);

        Arena seedArena = Arena.ofShared();
        PreparedExecution prepared = null;
        try {
            Tensor seedInput = nativeScalarOne(
                    scalarObjective.descriptor().dataType(), seedArena);
            Tensor seed = seedInput.contiguous();
            List<Tensor> targets =
                    stableParameters.stream().map(TrainingParameter::tensor).toList();
            CompiledGraph graph = borrowedEngine.compile(
                    List.of(scalarObjective), List.of(seed), targets);
            BindingPlan bindings = bindingPlan(graph, seedInput, stableParameters);
            prepared = borrowedEngine.prepare(graph);
            return new TrainingSession(
                    borrowedEngine,
                    borrowedSource,
                    scalarObjective,
                    suppliedOptimizer,
                    stableGroups,
                    seedInput,
                    seedArena,
                    stableParameters,
                    bindings.externalInputs,
                    graph,
                    prepared,
                    bindings.fixedInputs,
                    bindings.externalIndices,
                    bindings.inputRoles,
                    bindings.nonParameterIndices,
                    lifecycleHooks);
        } catch (RuntimeException | Error failure) {
            if (prepared != null) {
                try {
                    prepared.close();
                } catch (RuntimeException | Error cleanupFailure) {
                    if (cleanupFailure != failure) failure.addSuppressed(cleanupFailure);
                }
            }
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
     * order. Each {@link #runValues} instead supplies one stored value in each listed position,
     * with an exact full static descriptor but no identity constraint. Targets and every other
     * external bindable leaf are included. The immutable result preserves relative final Compiler
     * occurrence order and carries no storage or execution authority.</p>
     *
     * @return the same non-null immutable ordered metadata list on every call; possibly empty
     */
    public List<CompiledGraph.Input> inputs() {
        return externalInputs;
    }

    /**
     * Returns the exact immutable default optimizer configuration borrowed at construction.
     *
     * @return the non-null configuration retained by exact reference
     */
    public Optimizer optimizer() {
        return optimizer;
    }

    /**
     * Returns the exact immutable ordered groups captured when this session opened.
     *
     * @return non-null immutable list, possibly empty for an ungrouped session
     */
    public List<ParameterGroup> parameterGroups() {
        return parameterGroups;
    }

    /**
     * Runs the fixed prepared graph and applies the selected gradient lifecycle atomically.
     *
     * <p>The input list contains only occurrences reported by {@link #inputs()} and may use any
     * order. Every required Tensor identity must occur exactly once with its exact compiled
     * descriptor and present, live, accessible storage of sufficient capacity. Foreign or
     * duplicate identities, missing inputs, and any non-empty storage overlap with another
     * compiled input are rejected before Engine execution. Parameter-pair non-overlap is
     * established at open for fixed storage occurrences; this run rechecks live source membership,
     * Tensor storage identity, schema, liveness, and accessibility. Overlap with external inputs
     * or the cotangent seed is checked on every run. The complete parameter view is validated
     * again immediately before update commit.</p>
     *
     * <p>External storage is borrowed only through this synchronous call and internal result
     * closure; callers keep it usable and exclude conflicting mutation until return. Fresh IDs
     * are not accepted here; use {@link #runValues} for positional stored values.</p>
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
     *     active, the inward preparation is closed, a required storage is absent/dead/inaccessible, a
     *     captured parameter binding changed, or publication state is inconsistent
     * @throws ArithmeticException if a counter/byte calculation overflows or any required
     *     optimizer value or intermediate is non-finite before mutation
     * @throws RuntimeException if Engine work, materialization, or update reports another unchecked
     *     failure
     * @throws Error if inward work or allocation reports a fatal failure
     */
    public TrainingStep run(List<Tensor> inputs, GradientMode gradientMode) {
        return runInternal(inputs, gradientMode, false);
    }

    /**
     * Runs fresh stored external values through the existing preparation and optimizer transaction.
     *
     * <p>Positions are exactly {@link #inputs()} order, excluding captured parameters and the
     * private cotangent seed. TensorIds may be fresh; dtype, fully static shape, layout presence
     * and geometry, and requiresGrad must match each position exactly. No identity remapping,
     * expression execution, prototype mutation, copy, recompilation or preparation occurs.
     * The fixed parameters and seed are inserted into final compiled order for
     * {@link Engine#runValues}. Engine validates checked referenced spans for resolved layouts
     * and checked canonical static geometry for absent layouts. This does not strengthen
     * {@link #run}'s existing unresolved-layout checks.</p>
     *
     * <p>Each stored value needs present, correctly typed, live, current-thread-accessible host
     * storage of sufficient capacity. Nonempty external/external, external/parameter and seed
     * overlap rejects, including one value repeated across positions; zero-byte overlap remains
     * exempt. Caller storage is borrowed through this synchronous call and internal result
     * closure only, never retained for a later run. The caller keeps it usable and excludes
     * conflicting mutation until return. The returned objective precedes any parameter update.</p>
     *
     * <p>Admission checks Engine closure, session closure and busy state before arguments. All
     * decoding, result closure, candidate staging and complete precommit validation precede
     * mutation. Ordinary failure preserves parameter/master bytes, slots, pending gradients and
     * counters, with the same rollback and cleanup behavior as {@link #run}.</p>
     *
     * @param values non-null stored values, one per external input position; elements non-null
     * @param gradientMode non-null reset, accumulate-only, or accumulate-and-step behavior
     * @return a fresh non-null detached step that outlives caller storage, this session and Engine
     * @throws NullPointerException if an argument or element is null after lifecycle admission
     * @throws IllegalArgumentException if count, descriptor, static geometry, storage dtype or
     *     capacity, nonempty overlap, or publication metadata is invalid
     * @throws IllegalStateException if Engine is closed, session close has begun, another operation
     *     is active, preparation is closed, storage is absent/dead/inaccessible, a captured live
     *     binding changed, or publication state is inconsistent
     * @throws ArithmeticException if checked arithmetic overflows or required optimizer values
     *     or intermediates are non-finite
     * @throws RuntimeException if Engine work, materialization or update otherwise fails
     * @throws Error if inward work or allocation fails fatally
     */
    public TrainingStep runValues(List<Tensor> values, GradientMode gradientMode) {
        return runInternal(values, gradientMode, true);
    }

    /**
     * Owns one admitted run and the shared decode, stage, precommit and rollback transaction.
     *
     * @param inputs non-null external inputs after admission, interpreted by the selected mode
     * @param gradientMode non-null gradient lifecycle after admission
     * @param positional whether to bind stored values by position rather than exact TensorId
     * @return detached pre-update objective and successfully committed progress
     * @throws RuntimeException if admission, validation, execution or optimizer work fails
     * @throws Error if allocation or inward work fails fatally
     */
    private TrainingStep runInternal(
            List<Tensor> inputs, GradientMode gradientMode, boolean positional) {
        beginOperation();
        try {
            hooks.afterAdmission(Operation.RUN);
            Objects.requireNonNull(inputs, positional ? "values" : "inputs");
            GradientMode mode = Objects.requireNonNull(gradientMode, "gradientMode");
            long nextExecution = Math.incrementExact(executions);
            boolean updates = mode != GradientMode.ACCUMULATE;
            long nextOptimizerStep = updates
                    ? Math.incrementExact(optimizerSteps) : optimizerSteps;
            long nextAccumulation = mode == GradientMode.RESET_AND_STEP
                    ? 0L : Math.incrementExact(accumulatedGradientRuns);

            validateModuleAndParameters();
            List<Tensor> compiledInputs = positional ? bindValues(inputs) : bindInputs(inputs);
            HostTensorValue objectiveValue = executeAndDecode(compiledInputs, positional);
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
            for (int index = 0; index < parameters.size(); index++) {
                parameters.get(index).stageUpdate(optimizerSteps, useAccumulated);
            }
            for (TrainingParameter parameter : parameters) {
                parameter.prepareParameterCandidate();
            }
            validateModuleAndParameters();
            commitParameterCandidates();
            for (int index = 0; index < parameters.size(); index++) {
                TrainingParameter parameter = parameters.get(index);
                parameter.commitOptimizerSlots();
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
     * Clears pending gradient accumulation without changing parameter values, optimizer slots, or
     * either progress counter.
     *
     * @throws IllegalStateException if close has begun, the Engine-owned preparation is closed, or
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
     * <p>Lifecycle admission precedes capture. The operation validates the complete live source,
     * captured storage view, and finite logical parameter, master parameter, optimizer slots, and
     * accumulation values before copying. The immutable result contains canonical logical and
     * optimizer-width bytes and remains readable after this session or its Engine closes. It is
     * not a durable file format.</p>
     *
     * @return a fresh non-null immutable in-memory state snapshot
     * @throws IllegalStateException if the Engine is closed, close has begun, another operation is
     *     active, the inward preparation is closed, captured source/storage state changed, or current
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
                    parameterGroups,
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
     * <p>Lifecycle admission precedes inspection of {@code state}. The exact default algorithm and
     * ordered parameter-group configurations,
     * counters, parameter count, ordered paths, data types, Shapes, logical and master payload
     * lengths, slot presence, finite payload values, non-negative adaptive second moments,
     * momentum-disabled SGD slot values, and current source/storage
     * view all validate before any parameter write. Success atomically installs logical and master
     * parameter values in path order, then publishes optimizer slots, pending gradients, and counters.
     * Failure before commit leaves all current state unchanged and discards newly staged inactive
     * accumulation or zero-momentum buffers. The supplied immutable state is not retained.</p>
     *
     * <p>The value may come from {@link #state()} or public {@link TrainingState} constructors.
     * Successful construction establishes structure only, not compatibility or payload validity
     * for this session; both sources receive the same complete restore validation.</p>
     *
     * @param state non-null detached captured or reconstructed state compatible with this session's
     *     exact optimizer configuration, ordered groups, and parameter schema
     * @throws NullPointerException if {@code state} is {@code null} after successful admission
     * @throws IllegalArgumentException if optimizer, counters, paths, schemas, or finite payloads
     *     differ
     * @throws IllegalStateException if the Engine is closed, close has begun, another operation is
     *     active, the inward preparation is closed, or current source/storage state changed
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
            if (!parameterGroups.equals(candidate.parameterGroups())) {
                throw new IllegalArgumentException("training state parameter groups differ");
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
                parameters.get(index).validateState(candidate.parameters().get(index));
            }
            try {
                for (int index = 0; index < parameters.size(); index++) {
                    parameters.get(index).stageRestore(candidate.parameters().get(index));
                }
                validateModuleAndParameters();
                commitParameterCandidates();
                for (int index = 0; index < parameters.size(); index++) {
                    parameters.get(index).commitRestore();
                }
                publishCounters(
                        candidate.executions(),
                        candidate.optimizerSteps(),
                        candidate.accumulatedGradientRuns());
            } catch (RuntimeException | Error failure) {
                for (int index = 0; index < parameters.size(); index++) {
                    parameters.get(index).discardFailedRestoreStaging();
                }
                throw failure;
            }
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
     * Reports whether this session or its owned Engine prepared execution has begun closure.
     *
     * @return {@code true} after explicit close admission or when Engine cleanup has closed the
     *     inward preparation; otherwise {@code false}
     */
    public boolean isClosed() {
        synchronized (lifecycleLock) {
            return lifecycle != Lifecycle.OPEN || preparedExecution.isClosed();
        }
    }

    /**
     * Rejects new operations, waits for an admitted operation, and closes owned resources once.
     *
     * <p>The first closer becomes the cleanup owner. Concurrent closers wait for the same result.
     * Waiting is uninterruptible and restores the thread's interrupt status. Cleanup attempts the
     * prepared execution and private seed arena in that order. Repeated calls replay the exact
     * retained unchecked cleanup failure, if any. The borrowed Engine, source, parameter Tensors,
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
            preparedExecution.close();
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

    /**
     * Executes one complete compiled-order binding, detaches publications and closes the result.
     * Cleanup failure suppresses onto a primary failure; no caller input is retained here.
     *
     * @param inputs non-null complete compiled-order values including fixed parameters and seed
     * @param positional whether to use public Engine fresh-value admission
     * @return detached pre-update scalar objective, never null
     * @throws RuntimeException if execution, publication validation, decoding or cleanup fails
     * @throws Error if inward execution or allocation fails fatally
     */
    private HostTensorValue executeAndDecode(List<Tensor> inputs, boolean positional) {
        RunResult result = null;
        Throwable primary = null;
        try {
            result = positional ? engine.runValues(preparedExecution, inputs)
                    : engine.run(preparedExecution, inputs);
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

    /**
     * Preserves exact-ID arbitrary-order admission and its diagnostics for the original run API.
     *
     * @param suppliedInputs non-null external inputs with exact compiled identities
     * @return complete immutable binding with validated storage and nonempty non-overlap
     * @throws RuntimeException if count, identity, descriptor or storage validation fails
     */
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

        return assembleInputs(Arrays.asList(external));
    }

    /**
     * Snapshots positional stored values with exact descriptors, leaving static capacity geometry
     * to Engine's checked positional admission rather than changing the original identity path.
     *
     * @param values non-null values in external compiled encounter order
     * @return immutable complete compiled-order binding with Training overlap checks applied
     * @throws RuntimeException if count, null element, descriptor or storage admission fails
     */
    private List<Tensor> bindValues(List<Tensor> values) {
        if (values.size() != externalInputs.size()) {
            throw new IllegalArgumentException(
                    "training input count mismatch: expected=" + externalInputs.size()
                            + ", actual=" + values.size());
        }
        List<Tensor> external = List.copyOf(values);
        for (int index = 0; index < external.size(); index++) {
            if (!external.get(index).descriptor().equals(externalInputs.get(index).descriptor())) {
                throw new IllegalArgumentException(
                        "training input descriptor does not match "
                                + externalInputs.get(index).tensorId());
            }
        }
        return assembleInputs(external);
    }

    /**
     * Inserts fixed parameters/seed and validates borrowed storage in final compiled order.
     * Fixed parameter pairs need no dynamic rescan; every pair involving external storage or
     * the seed keeps the existing ordered nonempty overlap diagnostic.
     *
     * @param external non-null fully admitted external values in relative compiled order
     * @return non-null immutable complete binding retained only through the synchronous run
     * @throws RuntimeException if storage presence, schema, capacity, access or overlap is invalid
     */
    private List<Tensor> assembleInputs(List<Tensor> external) {
        var compiled = new ArrayList<Tensor>(fixedCompiledInputs.length);
        HostTensorStorage[] storages = new HostTensorStorage[fixedCompiledInputs.length];
        for (int compiledIndex = 0; compiledIndex < fixedCompiledInputs.length; compiledIndex++) {
            Tensor tensor = fixedCompiledInputs[compiledIndex];
            if (tensor == null) {
                tensor = external.get(externalIndexByCompiledInput[compiledIndex]);
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
        }
        for (int left = 0; left < storages.length; left++) {
            if (storages[left].byteSize() == 0) {
                continue;
            }
            if (fixedCompiledInputs[left] != null
                    && fixedCompiledInputs[left] != cotangentSeed) {
                for (int right : nonParameterCompiledIndices) {
                    if (right > left) {
                        validateInputNonOverlap(storages, left, right);
                    }
                }
            } else {
                for (int right = left + 1; right < storages.length; right++) {
                    validateInputNonOverlap(storages, left, right);
                }
            }
        }
        return List.copyOf(compiled);
    }

    /**
     * Rejects one nonempty compiled-input pair involving an external input or cotangent seed.
     *
     * @param storages non-null validated storage array in compiled-input order
     * @param left index of the nonempty left storage in that array
     * @param right index greater than {@code left} of the candidate right storage
     * @throws IllegalArgumentException if both storages are nonempty and overlap
     */
    private void validateInputNonOverlap(HostTensorStorage[] storages, int left, int right) {
        if (storages[right].byteSize() != 0
                && storages[left].segment()
                        .asOverlappingSlice(storages[right].segment()).isPresent()) {
            throw new IllegalArgumentException(
                    "training input storages overlap: "
                            + compiledInputRoles[left] + " and " + compiledInputRoles[right]);
        }
    }

    private int findExternalInput(TensorId id) {
        for (int index = 0; index < externalInputs.size(); index++) {
            if (externalInputs.get(index).tensorId().equals(id)) {
                return index;
            }
        }
        return -1;
    }

    /**
     * Validates the current complete parameter view against the open-time fixed bindings.
     *
     * <p>Each accepted storage retains its exact segment, so fixed parameter pairs cannot acquire
     * overlap without a changed binding. This check still rejects changed bindings, dead or
     * inaccessible storage, and non-finite state before committed session-state mutation. Staging
     * arrays may already have been modified when this check runs at precommit.</p>
     *
     * @throws IllegalStateException if the live membership, binding, storage, or finite state
     *     differs from the admitted session view
     */
    private void validateModuleAndParameters() {
        parameterSource.validateUnchanged();
        for (TrainingParameter expected : parameters) {
            expected.validateBinding();
            expected.validateFiniteState();
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
            if (lifecycle != Lifecycle.OPEN || preparedExecution.isClosed()) {
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

    static void validateSgdNarrowing(Sgd optimizer, DataType parameterType) {
        if (parameterType == DataType.FLOAT64) {
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
        String[] inputRoles = new String[compiledInputs.size()];
        int[] nonParameterIndices = new int[compiledInputs.size()];
        int nonParameterCount = 0;
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
                inputRoles[compiledIndex] = "cotangent seed";
                nonParameterIndices[nonParameterCount++] = compiledIndex;
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
                inputRoles[compiledIndex] = "parameter " + parameters.get(parameterIndex).path();
                continue;
            }
            externalIndices[compiledIndex] = external.size();
            external.add(input);
            inputRoles[compiledIndex] = "input " + input.tensorId();
            nonParameterIndices[nonParameterCount++] = compiledIndex;
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
                List.copyOf(external), fixed, externalIndices, inputRoles,
                Arrays.copyOf(nonParameterIndices, nonParameterCount));
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
            int[] externalIndices,
            String[] inputRoles,
            int[] nonParameterIndices) {
        private BindingPlan {
            Objects.requireNonNull(externalInputs, "externalInputs");
            Objects.requireNonNull(fixedInputs, "fixedInputs");
            Objects.requireNonNull(externalIndices, "externalIndices");
            Objects.requireNonNull(inputRoles, "inputRoles");
            Objects.requireNonNull(nonParameterIndices, "nonParameterIndices");
        }
    }
}
