package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.training.Optimizer;
import io.github.pho001.synaptik.training.TrainingStep;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * Typed root module whose forward boundary may use arbitrary caller-selected Java types.
 *
 * <p>The generic input and output types describe Java composition only. A model may accept and
 * return Tensors directly or use caller-owned records for structured values; the functional
 * factory enforces non-null values at that boundary. These types add no Tensor semantics, tuple
 * type, graph representation or backward method. The separate owned tensor capability returns
 * detached values rather than pretending that materialized results have generic type O.</p>
 *
 * <p>{@link #define(ModelDefinition)} creates an immutable named child structure through a
 * short-lived sealed {@link Topology}. The resulting Model inherits recursive mode propagation,
 * parameter and buffer discovery, stable state-dictionary paths, strict state loading, and
 * compatible parameter replacement from {@link Module}. Structure is permanent after definition;
 * existing bindings and mode retain Module's caller-coordinated mutable lifecycle.</p>
 *
 * <p>Generic Models are not thread-safe. Callers must coordinate forward construction with mode changes,
 * binding replacement, state loading, and other mutable Module operations when one consistent
 * view matters. {@link #builder()}, {@link #defineTensor} and {@link #graph} establish a private
 * static source for compute and configured training; save additionally requires checkpoint admission.
 * Builder composition reuses existing layers independently of that eight-operation format limit.
 * Those Models serialize high-level operations
 * (competing calls fail busy), guard inherited mutable state, and close cached public Engine and
 * Training owners before their native storage. Inherited Module mutators still require caller
 * exclusion. Generic/custom/declarative Models retain their old forward contract; unsupported
 * tensor operations throw UnsupportedOperationException before file/Engine work and close is a no-op.
 * Owned construction and load select eval; configureTraining does not change mode. Old flexible
 * model.v1 and whole-model.v1 readers remain separate.</p>
 *
 * @param <I> non-null Java input type
 * @param <O> non-null Java output type
 */
public abstract class Model<I, O> extends Module implements AutoCloseable {
    private ModelExecutionState execution;
    /**
     * Creates an ordinary empty training-mode Model base for an advanced subclass.
     *
     * <p>A subclass may use inherited protected declaration and child-registration operations.
     * Unlike the functional factory implementation, the abstract base cannot enforce the
     * documented non-null input and output contract around an arbitrary override.</p>
     */
    protected Model() {
        super();
    }

    /**
     * Applies this model's typed forward composition.
     *
     * @param input non-null caller input accepted by this model; ownership and mutation semantics
     *     are defined by the concrete model and input type
     * @return non-null caller-defined output; concrete implementations define exact identity and
     *     side effects
     */
    public abstract O forward(I input);

    /**
     * Starts closed fluent construction of a privately owned ordinary Tensor Model.
     * No Engine is created; execution capability is independent of checkpoint operation admission.
     * @return a fresh caller-coordinated reusable builder with default seed zero
     */
    public static ModelBuilder builder() { return new ModelBuilder(); }

    /**
     * Installs an already-private frozen chain and constructs its prototype expression once.
     * @param signature canonical static input descriptor, never null
     * @param layers fresh exclusively owned sequence, never null
     * @return an eval Model retaining exact final state and ordinary expressions, without copying
     * @throws NullPointerException if signature or layers is null
     * @throws IllegalArgumentException if composition or state reachability fails
     */
    static Model<Tensor, Tensor> fromComposition(TensorDescriptor signature, Sequential layers) {
        Tensor input = TensorFactory.create(signature);
        SingleTensorModel model = new SingleTensorModel();
        model.registerNamedChildren(Map.of("layers", layers));
        model.eval();
        Tensor output = layers.forward(input);
        Model<Tensor, Tensor> result = model;
        result.execution = new ModelExecutionState(model,
                OwnedModelSource.acquireComposition(model, input, output, layers));
        return result;
    }

    /**
     * Defines one typed functional Model with an immutable named child topology.
     *
     * <p>The definition callback is invoked exactly once. Its topology collects modules without
     * ownership mutation and is sealed in a {@code finally} boundary. On success, a non-null
     * forward body and the complete encounter-ordered topology are retained only after Module has
     * preflighted every name, identity, cycle, and parent-ownership condition. The factory does
     * not inspect which local variables the body captures; an unused registered module is still
     * owned. Any callback, null-result, or validation failure leaves every previously unowned
     * candidate unattached and is propagated without publishing a partial Model.</p>
     *
     * <p>The functional implementation rejects null input before calling the retained body,
     * invokes that body exactly once, rejects a null result, and otherwise returns its exact
     * result reference. Body exceptions are propagated unchanged. Expressions and module-local
     * effects created by a failing or null-returning body prefix remain; forward performs no
     * rollback, lookup by module name, reflection, capture, compilation, or execution.</p>
     *
     * @param definition non-null one-shot definition callback
     * @param <I> inferred non-null Java input type of the resulting model
     * @param <O> inferred non-null Java output type of the resulting model
     * @return a non-null Model that permanently owns the declared modules and retains the exact
     *     forward body
     * @throws NullPointerException if {@code definition} or its returned forward body is null, or
     *     if the callback supplies a null topology name or module
     * @throws IllegalArgumentException if final topology validation rejects a name, repeated
     *     identity, or self/ancestor cycle
     * @throws IllegalStateException if final topology validation finds an already-owned module
     * @throws RuntimeException if the definition callback throws a runtime exception; the same
     *     exception is propagated after sealing and without attaching candidates
     */
    public static <I, O> Model<I, O> define(ModelDefinition<I, O> definition) {
        ModelDefinition<I, O> suppliedDefinition = Objects.requireNonNull(definition, "definition");
        Topology topology = new Topology();
        ModelForward<I, O> forward;
        try {
            forward = suppliedDefinition.define(topology);
        } finally {
            topology.seal();
        }
        ModelForward<I, O> suppliedForward = Objects.requireNonNull(forward, "model forward");
        return new FunctionalModel<>(suppliedForward, topology.snapshot());
    }

    /**
     * Acquires a static one-input/output source through one definition and one eval prototype forward.
     * Registered wrappers determine roles; other reachable stored leaves become constants.
     * Callbacks are discarded after acquisition. Caller Tensors/stores are never rebound or mutated;
     * callback effects themselves are not rollbackable. No Engine is created.
     * The static source admits ADD/MUL/NEG/MATMUL, rank-two PERMUTE[1,0] and floating
     * RELU/TANH/SIGMOID through GraphCheckpoint; execution support remains occurrence-specific.
     * @param inputSignature non-null fully static exact input descriptor
     * @param definition non-null one-shot definition; its adopted modules must not be mutated later
     * @return owned eval Model with private state and cold symbolic forward
     * @throws NullPointerException if a required argument/callback result is null
     * @throws IllegalArgumentException if signature, topology, source operations or state admission reject
     * @throws IllegalStateException if a child is owned already or state storage is unavailable
     */
    public static Model<Tensor, Tensor> defineTensor(
            TensorDescriptor inputSignature, ModelDefinition<Tensor, Tensor> definition) {
        Tensor input = TensorFactory.create(Objects.requireNonNull(inputSignature, "inputSignature"));
        if (!inputSignature.shape().isFullyStatic()) throw new IllegalArgumentException("static signature required");
        Objects.requireNonNull(definition, "definition");
        Topology topology = new Topology();
        ModelForward<Tensor, Tensor> body;
        try { body = definition.define(topology); }
        finally { topology.seal(); }
        Objects.requireNonNull(body, "model forward");
        SingleTensorModel model = new SingleTensorModel();
        model.registerNamedChildren(topology.snapshot());
        model.eval();
        Tensor output = Objects.requireNonNull(body.forward(input), "model output");
        Model<Tensor, Tensor> result = model;
        result.execution = new ModelExecutionState(model,
                OwnedModelSource.acquireDefinition(model, input, output));
        return model;
    }

    /**
     * Defines an owned source and selects a borrowed Engine without preparing it.
     * @param inputSignature non-null exact static descriptor
     * @param definition non-null one-shot definition
     * @param borrowedEngine non-null open Engine that remains caller-owned
     * @return owned eval Model borrowing the selected Engine
     * @throws NullPointerException if an argument or callback result is null
     * @throws IllegalArgumentException if source or signature admission rejects
     * @throws IllegalStateException if state is unavailable or Engine is closed
     */
    public static Model<Tensor, Tensor> defineTensor(TensorDescriptor inputSignature,
            ModelDefinition<Tensor, Tensor> definition, Engine borrowedEngine) {
        Objects.requireNonNull(borrowedEngine, "borrowedEngine");
        return select(defineTensor(inputSignature, definition), borrowedEngine);
    }

    /**
     * Acquires external state exactly once and reconstructs a private ordinary source.
     * @param inputs ordered unique static provenance-free prototypes, possibly unused
     * @param outputs nonempty distinct expressions using ADD/MUL/NEG/MATMUL, rank-two PERMUTE[1,0]
     *     or floating RELU/TANH/SIGMOID; ordinary family guards remain mandatory
     * @param namedLeaves complete reachable named state with explicit roles and canonical Module traversal order
     * @return owned eval list Model; constants remain outside stateDictionary
     * @throws NullPointerException if an argument/element is null
     * @throws IllegalArgumentException if source, roles, paths, geometry or values reject
     * @throws IllegalStateException if required host storage is unavailable
     */
    public static Model<List<Tensor>, List<Tensor>> graph(
            List<Tensor> inputs, List<Tensor> outputs, List<GraphCheckpoint.Leaf> namedLeaves) {
        return new TensorListModel(GraphCheckpoint.copy(new GraphCheckpoint.Source(inputs, outputs, namedLeaves)));
    }

    /**
     * Acquires a list source and selects a borrowed Engine without preparation.
     * @param inputs ordered static input prototypes
     * @param outputs ordered source outputs
     * @param namedLeaves complete canonical named state
     * @param borrowedEngine non-null open caller-owned Engine
     * @return owned eval Model borrowing that Engine
     * @throws NullPointerException if an argument/element is null
     * @throws IllegalArgumentException if source admission rejects
     * @throws IllegalStateException if storage is unavailable or Engine is closed
     */
    public static Model<List<Tensor>, List<Tensor>> graph(List<Tensor> inputs,
            List<Tensor> outputs, List<GraphCheckpoint.Leaf> namedLeaves, Engine borrowedEngine) {
        Objects.requireNonNull(borrowedEngine, "borrowedEngine");
        return select(graph(inputs, outputs, namedLeaves), borrowedEngine);
    }

    /**
     * Loads a single-boundary graph using the six NN artifact ceilings, without creating an Engine.
     * @param path non-null graph-model.v1/v2 file, not an old flexible model artifact
     * @return owned eval Model adopting final loaded heap state without another copy
     * @throws IOException if reading or artifact validation fails
     * @throws IllegalArgumentException if boundaries require loadGraph or state order is noncanonical
     */
    public static Model<Tensor, Tensor> load(Path path) throws IOException {
        return load(path, ModelExecutionState.DEFAULT_LIMITS);
    }

    /**
     * Loads a single-boundary graph under explicit artifact bounds.
     * @param path non-null graph artifact file
     * @param limits non-null file/manifest/entry/key/payload/aggregate ceilings
     * @return owned eval Model with adopted loaded state and no Engine
     * @throws IOException if reading or artifact validation fails
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if boundaries require loadGraph or state order is noncanonical
     */
    public static Model<Tensor, Tensor> load(Path path, ArtifactLimits limits) throws IOException {
        GraphCheckpoint.Source source = GraphCheckpoint.load(path, limits);
        if (source.inputs().size() != 1 || source.outputs().size() != 1) {
            throw new IllegalArgumentException("multiple/empty boundaries require Model.loadGraph");
        }
        return new SingleTensorModel(source);
    }

    /**
     * Loads and selects a caller-owned Engine without inspecting backend availability.
     * @param path non-null graph artifact
     * @param limits non-null artifact ceilings
     * @param borrowedEngine non-null open Engine, never closed by the Model
     * @return owned eval single Model borrowing that Engine
     * @throws IOException if artifact reading/validation fails
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if source/boundaries reject
     * @throws IllegalStateException if the Engine is closed
     */
    public static Model<Tensor, Tensor> load(Path path, ArtifactLimits limits, Engine borrowedEngine)
            throws IOException {
        Objects.requireNonNull(borrowedEngine, "borrowedEngine");
        return select(load(path, limits), borrowedEngine);
    }

    /**
     * Loads arbitrary static list boundaries with the six NN artifact ceilings, cold.
     * @param path non-null graph artifact
     * @return owned eval list Model adopting final loaded state
     * @throws IOException if reading/validation fails
     * @throws IllegalArgumentException if Module state order is noncanonical
     */
    public static Model<List<Tensor>, List<Tensor>> loadGraph(Path path) throws IOException {
        return loadGraph(path, ModelExecutionState.DEFAULT_LIMITS);
    }

    /**
     * Loads list boundaries under explicit artifact ceilings without creating an Engine.
     * Both closed graph-model.v1 and graph-model.v2 sources are accepted without numerical conversion.
     * @param path non-null graph artifact
     * @param limits non-null artifact bounds
     * @return owned eval Model adopting the final loaded leaves without another copy
     * @throws IOException if reading/validation fails
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if Module traversal order is noncanonical
     */
    public static Model<List<Tensor>, List<Tensor>> loadGraph(Path path, ArtifactLimits limits)
            throws IOException {
        return new TensorListModel(GraphCheckpoint.load(path, limits));
    }

    /**
     * Loads list boundaries and selects a borrowed Engine without preparation.
     * @param path non-null graph artifact
     * @param limits non-null artifact bounds
     * @param borrowedEngine non-null open caller-owned Engine
     * @return owned eval Model borrowing the selected Engine
     * @throws IOException if reading/validation fails
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if source state order rejects
     * @throws IllegalStateException if the Engine is closed
     */
    public static Model<List<Tensor>, List<Tensor>> loadGraph(
            Path path, ArtifactLimits limits, Engine borrowedEngine) throws IOException {
        Objects.requireNonNull(borrowedEngine, "borrowedEngine");
        return select(loadGraph(path, limits), borrowedEngine);
    }

    /**
     * Selects an Engine on the same receiver before the first valid setup attempt.
     * Selected-Engine closure precedes Model closure/busy admission; freeze precedes candidate checks.
     * Invalid compute/configure arguments do not freeze; any setup attempt does, including failure.
     * @param engine non-null open borrowed Engine; neither previous nor new selection is closed
     * @return this exact Model, with unchanged generic type, wrappers and working state
     * @throws UnsupportedOperationException if this Model has no owned tensor capability
     * @throws IllegalStateException if selected Engine/Model is closed, busy, frozen, or candidate closed
     * @throws NullPointerException if an admitted unfrozen candidate is null
     */
    public Model<I, O> withEngine(Engine engine) {
        capability().withEngine(engine);
        return this;
    }

    /**
     * Computes one static batch using a cached public preparation and fresh positional values.
     * @param input non-null stored input with the exact full signature; borrowed through result closure
     * @return detached sole output, valid after Model/input/Engine closure
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if closed, busy, invalidated or storage is unusable
     * @throws IllegalArgumentException if boundary arity or signature rejects
     * @throws NullPointerException if admitted input is null
     */
    public HostTensorValue compute(Tensor input) { return capability().compute(input); }

    /**
     * Computes all outputs using an unbounded aggregate ceiling (Long.MAX_VALUE).
     * @param inputs non-null declared-order stored values, including unused declared slots
     * @return immutable ordered detached values
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if lifecycle, source or storage admission rejects
     * @throws IllegalArgumentException if arity/signatures or publication geometry reject
     * @throws NullPointerException if inputs or an element is null
     */
    public List<HostTensorValue> compute(List<Tensor> inputs) {
        return capability().compute(inputs, Long.MAX_VALUE);
    }

    /**
     * Bounds every publication and their aggregate before materializing any output occurrence.
     * RunResult closes on success/failure and no partial list escapes.
     * @param inputs non-null declared-order stored values, borrowed for the complete call
     * @param maximumTotalBytes nonnegative aggregate canonical byte ceiling
     * @return immutable detached outputs in exact source output occurrence order
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if lifecycle, source or storage admission rejects
     * @throws IllegalArgumentException if inputs or any publication/aggregate bound reject
     * @throws ArithmeticException if checked publication sizing overflows
     * @throws NullPointerException if inputs or an element is null
     */
    public List<HostTensorValue> compute(List<Tensor> inputs, long maximumTotalBytes) {
        return capability().compute(inputs, maximumTotalBytes);
    }

    /**
     * Configures one scalar objective and TrainingSession on the owned working weights.
     * Does not toggle Module mode. The pure builder runs once per explicit attempt and is discarded.
     * A successful configuration is final; failed setup may retry on the frozen Engine choice.
     * @param targetSignature non-null fully static target descriptor
     * @param objectiveBuilder non-null pure builder returning a scalar gradient-eligible FLOAT32/FLOAT64 expression
     * @param optimizer non-null immutable Training optimizer configuration
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if closed, busy, invalid, already configured or storage unusable
     * @throws IllegalArgumentException if boundaries, objective, native parameter or source admission rejects
     * @throws NullPointerException if an argument or objective result is null
     */
    public void configureTraining(TensorDescriptor targetSignature,
            BiFunction<Tensor, Tensor, Tensor> objectiveBuilder, Optimizer optimizer) {
        capability().configure(targetSignature, objectiveBuilder, optimizer);
    }

    /**
     * Configures a list-boundary objective against owned outputs and explicit target prototypes.
     * Heap parameters are admitted to a Model-owned shared Arena once, preserving Tensor identity;
     * inference preparations close before replacement. Training owns optimizer mathematics/atomicity.
     * @param targetPrototypes non-null unique static provenance-free targets disjoint from source leaves
     * @param objectiveBuilder non-null pure one-attempt builder; arbitrary external leaves reject
     * @param optimizer non-null Training-owned immutable configuration
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if lifecycle, source, storage or repeated configuration rejects
     * @throws IllegalArgumentException if targets, objective or Training parameter admission rejects
     * @throws NullPointerException if an argument/element/objective is null
     */
    public void configureTraining(List<Tensor> targetPrototypes,
            BiFunction<List<Tensor>, List<Tensor>, Tensor> objectiveBuilder, Optimizer optimizer) {
        capability().configure(targetPrototypes, objectiveBuilder, optimizer);
    }

    /**
     * Performs RESET_AND_STEP on one configured batch and target without changing Module mode.
     * @param input non-null fresh stored exact-signature input borrowed through synchronous execution
     * @param target non-null fresh stored exact-signature target with no forbidden storage overlap
     * @return detached objective and committed step metadata, readable after all owners close
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if unconfigured, closed, busy, invalid or storage unusable
     * @throws IllegalArgumentException if boundary, descriptor or Training overlap admission rejects
     * @throws ArithmeticException if Training rejects nonfinite gradients, staged updates or optimizer arithmetic
     * @throws NullPointerException if an admitted argument is null
     */
    public TrainingStep train(Tensor input, Tensor target) { return capability().train(input, target); }

    /**
     * Performs the same Training transaction for explicit ordered batch and target slots.
     * @param inputs non-null fresh stored inputs in declared order, including eliminated slots
     * @param targets non-null fresh stored targets in declared order
     * @return detached TrainingStep after internal result closure
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if lifecycle/configuration/source/storage rejects
     * @throws IllegalArgumentException if arity, descriptors or overlap reject
     * @throws ArithmeticException if Training rejects nonfinite transaction values
     * @throws NullPointerException if a list or element is null
     */
    public TrainingStep train(List<Tensor> inputs, List<Tensor> targets) {
        return capability().train(inputs, targets);
    }

    /**
     * Saves the fixed source and current live owned leaves, even before first compute.
     * Builder compositions outside the eight-kind graph format reject before file work, without
     * freezing Engine selection or disabling subsequent compute/training.
     * Uses NN ceilings: 256 MiB file/aggregate, 4 MiB manifest, 16384 entries, 64-byte keys,
     * 64 MiB payload. Does not persist objective, optimizer, RNG, cursor or Java callback state.
     * @param path non-null destination published atomically by GraphCheckpoint
     * @throws IOException if encoding/publication fails
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if closed, busy, source changed or storage unusable
     * @throws IllegalArgumentException if source/values/limits reject
     * @throws NullPointerException if admitted path is null
     */
    public void save(Path path) throws IOException { capability().save(path, ModelExecutionState.DEFAULT_LIMITS); }

    /**
     * Saves under explicit artifact ceilings without execution or source reconstruction.
     * @param path non-null destination
     * @param limits non-null six artifact ceilings
     * @throws IOException if encoding/publication fails
     * @throws UnsupportedOperationException if tensor capability is absent
     * @throws IllegalStateException if lifecycle, live source or storage rejects
     * @throws IllegalArgumentException if source/values/bounds reject
     * @throws NullPointerException if an admitted argument is null
     */
    public void save(Path path, ArtifactLimits limits) throws IOException { capability().save(path, limits); }

    /**
     * Closes owned tensor resources, or does nothing for generic/custom/declarative Models.
     * Bypasses Engine/capability/busy admission, rejects new work and waits for admitted work.
     * Closes session/preparation before native owners before a lazily owned Engine; borrowed Engines
     * are never closed. Cleanup is attempt-all with suppression and retained repeated/concurrent replay.
     * Detached compute values and TrainingStep remain usable. Inherited mutation stays caller-coordinated.
     * @throws RuntimeException if retained cleanup failed
     * @throws Error if retained cleanup failed fatally
     */
    @Override
    public void close() { if (execution != null) execution.close(); }

    /**
     * Checks capability before any Engine/file/argument work.
     * @return owned tensor state, never null
     * @throws UnsupportedOperationException if this Model retains generic forward semantics only
     */
    private ModelExecutionState capability() {
        if (execution == null) throw new UnsupportedOperationException("Model has no owned tensor capability");
        return execution;
    }

    /**
     * Selects on an unpublished candidate and cleans it if selection fails.
     * @param model fresh private candidate
     * @param engine borrowed Engine
     * @param <I> exact input specialization
     * @param <O> exact output specialization
     * @return the same candidate
     */
    private static <I, O> Model<I, O> select(Model<I, O> model, Engine engine) {
        try { return model.withEngine(engine); }
        catch (RuntimeException | Error failure) {
            try { model.close(); } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    /** Concrete specialization keeps generic forward type-correct without casting O. */
    private static final class SingleTensorModel extends Model<Tensor, Tensor> {
        /** Creates an unpublished root for the definition path. */
        private SingleTensorModel() { }

        /**
         * Adopts already-loaded private state.
         * @param source validated single-boundary loaded source
         */
        private SingleTensorModel(GraphCheckpoint.Source source) {
            super.execution = new ModelExecutionState(this, OwnedModelSource.adopt(this, source));
        }

        /**
         * Composes builder layers or reconstructs a definition/load source without a callback.
         * @param input exact-signature expression
         * @return cold reconstructed owned expression
         * @throws IllegalStateException if closed, busy or invalidated
         * @throws IllegalArgumentException if the descriptor differs
         */
        @Override public Tensor forward(Tensor input) { return super.execution.forward(input); }
    }

    /** Concrete list specialization for advanced graph boundaries and loadGraph. */
    private static final class TensorListModel extends Model<List<Tensor>, List<Tensor>> {
        /**
         * Installs final acquired/load state without copying again.
         * @param source private validated source with canonical Module state order
         */
        private TensorListModel(GraphCheckpoint.Source source) {
            super.execution = new ModelExecutionState(this, OwnedModelSource.adopt(this, source));
        }

        /**
         * Reconstructs ordinary ordered symbolic outputs on an explicit cold call.
         * @param inputs exact-descriptor expressions in declared order
         * @return cold ordered reconstructed expressions
         * @throws IllegalStateException if closed, busy or invalidated
         * @throws IllegalArgumentException if boundary metadata differs
         */
        @Override public List<Tensor> forward(List<Tensor> inputs) { return super.execution.forward(inputs); }
    }

    private static final class FunctionalModel<I, O> extends Model<I, O> {
        private final ModelForward<I, O> forward;

        private FunctionalModel(ModelForward<I, O> forward, Map<String, Module> children) {
            this.forward = forward;
            registerNamedChildren(children);
        }

        @Override
        public O forward(I input) {
            I suppliedInput = Objects.requireNonNull(input, "input");
            return Objects.requireNonNull(forward.forward(suppliedInput), "model output");
        }
    }
}
