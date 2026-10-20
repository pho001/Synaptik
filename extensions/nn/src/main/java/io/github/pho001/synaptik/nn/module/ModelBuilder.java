package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Conv1d;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.layers.Conv3d;
import io.github.pho001.synaptik.nn.layers.GruSequence;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.layers.LstmSequence;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Closed fluent recipes for an owned, fixed-signature Tensor Model, obtained from {@link Model#builder()}.
 *
 * <p>Declare input once before appending steps. Each build snapshots the recipes and owns fresh
 * layers/state, unaffected by later builder edits. Supplied Linear state is borrowed until build,
 * then copied once per Tensor identity before expressions are constructed; callers exclude storage
 * mutation and closure during build. Built Models retain no recipes or caller parameter leaves.
 * Builder mutation/build are caller-coordinated, not thread-safe.</p>
 *
 * <p>Automatic weights use existing Glorot-uniform algorithms and zero bias. Default seed is zero;
 * the last selected seed applies to the whole build. Appended step i, including transforms,
 * activations and finalHidden selection, uses {@code seed + 0x9E3779B97F4A7C15L * i} with long
 * wraparound and a fresh L64X128MixRandom generator per initialized layer. Input is canonical
 * contiguous, non-view, offset zero and not gradient-eligible. Forward/compute require its exact
 * full descriptor.</p>
 *
 * <p>Construction and cold forward create no Engine. Compute/train reuse existing Model execution
 * owners and fixed expressions, not recipes. Execution support is occurrence-specific and independent
 * of checkpoint admission: the selected default CNN and static recurrent compositions persist
 * as their ordinary Tensor DAG, not recipes. Unsupported save leaves the Model usable.</p>
 *
 * <p>Explicit rnnScan/gruScan/lstmScan recipes share a canonical runtime lengths input and produce
 * dense time-major outputs while skipping only each scan's padded transitions. Finish with
 * {@link #buildWithLengths()}; {@link #build()} remains single-input and static. Scan Models are
 * inference-only; current scan execution qualification is canonical FLOAT32 on CPU, not every
 * valid floating expression or ordinary-step composition. {@link #finalHidden()} selects the
 * immediately preceding scan's existing final state, whereas {@link #lastTimeStep()} selects a
 * dense time index. Ordinary bias and activations remain dense and do not acquire automatic
 * padding masks.</p>
 */
public final class ModelBuilder {
    private static final long STEP_SEED = 0x9E3779B97F4A7C15L;
    private static final RandomGeneratorFactory<RandomGenerator> RANDOM =
            RandomGeneratorFactory.of("L64X128MixRandom");
    private final List<LayerRecipe> recipes = new ArrayList<>();
    private final List<Tensor> suppliedLeaves = new ArrayList<>();
    private final List<Integer> scanSteps = new ArrayList<>();
    private TensorDescriptor input;
    private long seed;

    /** Creates an empty builder; public construction is through Model.builder(). */
    ModelBuilder() { }

    /**
     * Declares the exact values-input signature before any steps, snapshotting the shape array.
     * A scan terminal adds its separate shared lengths input; it does not change this descriptor.
     * @param type non-null floating element type
     * @param shape non-null rank-at-least-one array of strictly positive static extents
     * @return this same mutable builder
     * @throws NullPointerException if type or shape is null
     * @throws IllegalArgumentException if type/rank/extents are invalid
     * @throws IllegalStateException if input was already declared
     * @throws ArithmeticException if canonical layout sizing overflows
     */
    public ModelBuilder input(DataType type, long... shape) {
        if (input != null) throw new IllegalStateException("input is already declared");
        Objects.requireNonNull(type, "type");
        long[] snapshot = Objects.requireNonNull(shape, "shape").clone();
        if (!type.isFloating()) throw new IllegalArgumentException("input type must be floating");
        if (snapshot.length == 0) throw new IllegalArgumentException("input rank must be at least one");
        for (long extent : snapshot) positive(extent, "input extent");
        Shape dimensions = Shape.of(snapshot);
        input = new TensorDescriptor(type, dimensions,
                Optional.of(LayoutDescriptor.contiguous(dimensions)), false);
        return this;
    }

    /**
     * Selects the base seed for all steps of subsequent builds, including already appended recipes.
     * @param seed any Java long; wraparound in step derivation is intentional
     * @return this same mutable builder
     */
    public ModelBuilder seed(long seed) { this.seed = seed; return this; }

    /**
     * Appends a projection with inferred last-axis input width and zero bias enabled.
     * @param outFeatures strictly positive output width
     * @return this same builder
     * @throws IllegalArgumentException if outFeatures is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder linear(long outFeatures) { return linear(outFeatures, true); }

    /**
     * Appends a projection with inferred last-axis input width and Glorot-uniform weight.
     * @param outFeatures strictly positive output width
     * @param bias whether to create an exact zero trainable bias
     * @return this same builder
     * @throws IllegalArgumentException if outFeatures is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder linear(long outFeatures, boolean bias) {
        positive(outFeatures, "outFeatures");
        return append((type, stepSeed, copies) -> new Linear(outFeatures, bias, type,
                ParameterInitialization.glorotUniform(), RANDOM, stepSeed));
    }

    /**
     * Appends a no-bias projection borrowing weight until each build's private acquisition.
     * @param weight non-null stored provenance-free positive rank-two floating gradient-eligible
     *     weight [outFeatures,inFeatures], matching input dtype; never mutated
     * @return this same builder
     * @throws NullPointerException if weight is null
     * @throws IllegalArgumentException if Linear schema, provenance or dtype rejects
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder linear(Tensor weight) {
        requireInput();
        new Linear(weight); // Reuse the supported layer's complete supplied-state guards.
        supplied(weight);
        append((type, stepSeed, copies) -> new Linear(copies.get(weight)));
        suppliedLeaves.add(weight);
        return this;
    }

    /**
     * Appends a projection borrowing weight/bias until each build copies their identities once.
     * @param weight non-null stored provenance-free floating gradient-eligible weight with positive
     *     static rank-two shape [outFeatures,inFeatures], matching input dtype; never mutated
     * @param bias non-null stored provenance-free gradient-eligible [outFeatures] bias matching weight dtype
     * @return this same builder
     * @throws NullPointerException if weight or bias is null
     * @throws IllegalArgumentException if Linear schema, provenance or dtype rejects
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder linear(Tensor weight, Tensor bias) {
        requireInput();
        new Linear(weight, bias);
        supplied(weight);
        supplied(bias);
        append((type, stepSeed, copies) -> new Linear(copies.get(weight), copies.get(bias)));
        suppliedLeaves.add(weight);
        suppliedLeaves.add(bias);
        return this;
    }

    /**
     * Appends ordinary floating RELU without changing shape.
     * @return this same builder
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder relu() {
        return append((type, stepSeed, copies) -> new Activation(ModelSpecification.ActivationKind.RELU));
    }

    /**
     * Appends ordinary floating TANH without changing shape.
     * @return this same builder
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder tanh() {
        return append((type, stepSeed, copies) -> new Activation(ModelSpecification.ActivationKind.TANH));
    }

    /**
     * Appends ordinary floating SIGMOID without changing shape.
     * @return this same builder
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder sigmoid() {
        return append((type, stepSeed, copies) -> new Activation(ModelSpecification.ActivationKind.SIGMOID));
    }

    /**
     * Appends NCW convolution: stride/dilation/groups one, padding zero, zero bias enabled.
     * @param outChannels positive output-channel count
     * @param kernelWidth positive spatial kernel width
     * @return this same builder
     * @throws IllegalArgumentException if a count is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder conv1d(long outChannels, long kernelWidth) {
        positive(outChannels, "outChannels"); positive(kernelWidth, "kernelWidth");
        return append((type, stepSeed, copies) -> new Conv1d(outChannels, kernelWidth,
                1, 0, 1, 1, true, type, ParameterInitialization.glorotUniform(), stepSeed));
    }

    /**
     * Appends NCHW convolution: unit strides/dilations/groups, zero padding and zero bias enabled.
     * @param outChannels positive output-channel count
     * @param kernelHeight positive vertical kernel extent
     * @param kernelWidth positive horizontal kernel extent
     * @return this same builder
     * @throws IllegalArgumentException if a count is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder conv2d(long outChannels, long kernelHeight, long kernelWidth) {
        positive(outChannels, "outChannels");
        positive(kernelHeight, "kernelHeight"); positive(kernelWidth, "kernelWidth");
        return append((type, stepSeed, copies) -> new Conv2d(outChannels, kernelHeight, kernelWidth,
                1, 1, 0, 0, 1, 1, 1, true, type, ParameterInitialization.glorotUniform(), stepSeed));
    }

    /**
     * Appends NCDHW convolution with unit strides/dilations/groups, zero padding and zero bias.
     * Existing Conv3d backward rejection is unchanged.
     * @param outChannels positive output-channel count
     * @param kernelDepth positive depth kernel extent
     * @param kernelHeight positive vertical kernel extent
     * @param kernelWidth positive horizontal kernel extent
     * @return this same builder
     * @throws IllegalArgumentException if a count is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder conv3d(long outChannels, long kernelDepth, long kernelHeight, long kernelWidth) {
        positive(outChannels, "outChannels"); positive(kernelDepth, "kernelDepth");
        positive(kernelHeight, "kernelHeight"); positive(kernelWidth, "kernelWidth");
        return append((type, stepSeed, copies) -> new Conv3d(outChannels, kernelDepth, kernelHeight,
                kernelWidth, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, true, type,
                ParameterInitialization.glorotUniform(), stepSeed));
    }

    /**
     * Appends floor NCHW maximum pooling, stride equal to kernel, zero padding and unit dilation.
     * @param kernelHeight positive vertical window extent
     * @param kernelWidth positive horizontal window extent
     * @return this same builder
     * @throws IllegalArgumentException if a window extent is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder maxPool2d(long kernelHeight, long kernelWidth) {
        MaxPool2dAttrs attrs = new MaxPool2dAttrs(kernelHeight, kernelWidth,
                kernelHeight, kernelWidth, 0, 0, 1, 1, false);
        return append((type, stepSeed, copies) -> new MaxPoolStep(attrs));
    }

    /**
     * Appends rank-at-least-two flatten preserving axis zero and multiplying remaining extents.
     * CNN batch stays separate; TBF [T,B,H] becomes [T,B*H]. Select lastTimeStep, or a scan's
     * finalHidden, first when batch rows must remain separate. No padding mask is propagated.
     * @return this same builder
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder flatten() { return append((type, stepSeed, copies) -> new FlattenStep()); }

    /**
     * Appends an all-valid zero-initial-state static tanh RNN returning all hidden steps [T,B,H].
     * @param hiddenSize positive hidden width; input must be positive static TBF at build
     * @return this same builder
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder rnn(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        return append((type, stepSeed, copies) -> new RnnStep(new RnnSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed)));
    }

    /**
     * Appends an all-valid zero-state reset-after GRU returning every hidden step [T,B,H].
     * @param hiddenSize positive hidden width; input must be positive static TBF at build
     * @return this same builder
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder gru(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        return append((type, stepSeed, copies) -> new GruStep(new GruSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed)));
    }

    /**
     * Appends all-valid zero-hidden/zero-cell LSTM returning every hidden step [T,B,H], not cell state.
     * @param hiddenSize positive hidden width; input must be positive static TBF at build
     * @return this same builder
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder lstm(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        return append((type, stepSeed, copies) -> new LstmStep(new LstmSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed)));
    }

    /**
     * Appends a biased FORWARD tanh scan with independent zero hidden state and dense [T,B,H] output.
     * The first scan establishes positive static time/batch geometry and one shared INT64[B]
     * runtime lengths input; later scans must preserve T/B. Ordinary steps do not inherit masking.
     * Each row consumes its valid prefix, zero-pads the dense suffix, and returns zero final hidden
     * state for length zero. Geometry is checked at build; length values in [0,T] are checked at run.
     * @param hiddenSize positive hidden width; input at this step must be positive static TBF
     * @return this same builder; use buildWithLengths, not build, to finish a scan chain
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder rnnScan(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        append((type, stepSeed, copies) -> new RnnStep(new RnnSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed), hiddenSize));
        scanSteps.add(recipes.size() - 1);
        return this;
    }

    /**
     * Appends a biased FORWARD reset-after GRU scan, starting each batch from independent zero state.
     * Shares runtime lengths and exact T/B with every scan, returning dense [T,B,H]; padded
     * transitions are skipped only within each scan. Scan-containing Models are inference-only.
     * Length zero leaves that row's dense output and final hidden state zero. Geometry is checked
     * at build; runtime lengths must lie in [0,T]. Existing initialization and step seeds apply.
     * @param hiddenSize positive hidden width; input at this step must be positive static TBF
     * @return this same builder; finish with buildWithLengths
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder gruScan(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        append((type, stepSeed, copies) -> new GruStep(new GruSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed), hiddenSize));
        scanSteps.add(recipes.size() - 1);
        return this;
    }

    /**
     * Appends a biased FORWARD LSTM scan with independent zero hidden/cell state and dense [T,B,H].
     * All scans share runtime lengths with exact T/B. No state carries between batches; explicit
     * states or finalCell require the advanced graph/list boundary. The Model is inference-only.
     * Length zero leaves the row's output and final states zero. Geometry is checked at build;
     * runtime lengths must lie in [0,T]. Existing initialization and step seeds apply.
     * @param hiddenSize positive hidden width; input at this step must be positive static TBF
     * @return this same builder; finish with buildWithLengths
     * @throws IllegalArgumentException if hiddenSize is not positive
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder lstmScan(long hiddenSize) {
        positive(hiddenSize, "hiddenSize");
        append((type, stepSeed, copies) -> new LstmStep(new LstmSequence(hiddenSize, true,
                type, ParameterInitialization.glorotUniform(), stepSeed), hiddenSize));
        scanSteps.add(recipes.size() - 1);
        return this;
    }

    /**
     * Selects the immediately preceding scan's existing finalHidden [B,H] port, not a time index.
     * A zero-length row returns its initial zero hidden state. Append classifier layers afterwards;
     * their ordinary bias/activation may make zero-length rows nonzero. Counts as one seed step.
     * @return this same builder
     * @throws IllegalStateException if input is absent or the immediately preceding recipe is not
     *     a new scan, including repeated selection or selection after an ordinary transform
     */
    public ModelBuilder finalHidden() {
        requireInput();
        if (!scanSteps.contains(recipes.size() - 1)) {
            throw new IllegalStateException("finalHidden must immediately follow a scan recipe");
        }
        return append((type, stepSeed, copies) -> new LastTimeStepStep(true));
    }

    /**
     * Explicitly selects the last time step of rank-three TBF to [B,H] using select(0,T-1).
     * Ignores runtime lengths, even after a scan; this is not {@link #finalHidden()}. Ordinary
     * intervening steps may change a scan's padded zeros before this dense-time selection.
     * @return this same builder
     * @throws IllegalStateException if input has not been declared
     */
    public ModelBuilder lastTimeStep() {
        return append((type, stepSeed, copies) -> new LastTimeStepStep());
    }

    /**
     * Builds a fresh private eval Model and its prototype expression without creating an Engine.
     * All shape compatibility checks occur through existing layers/Tensor operations. A failed
     * build publishes nothing and changes no caller descriptors, storage associations or bytes.
     * Scan recipes reject before supplied-state copying or layer construction; use buildWithLengths.
     * @return a new owned single-Tensor Model; caller must close it
     * @throws IllegalStateException if input/steps are absent, a scan requires buildWithLengths,
     *     or supplied host storage is unavailable
     * @throws IllegalArgumentException if composition, supplied raw state or layer schema rejects
     * @throws ArithmeticException if checked shape, fan, layout or allocation sizing overflows
     */
    public Model<Tensor, Tensor> build() {
        requireInput();
        if (!scanSteps.isEmpty()) throw new IllegalStateException("scan recipes require buildWithLengths");
        List<LayerRecipe> snapshot = List.copyOf(recipes);
        if (snapshot.isEmpty()) throw new IllegalStateException("at least one step is required");
        return Model.fromComposition(input, instantiate(snapshot));
    }

    /**
     * Builds a fresh private inference-only list Model declaring [values, validLengths] and one output.
     * Lengths have canonical non-gradient INT64[B] metadata, with runtime values in [0,T], including
     * zero. T/B are established at the first scan after any ordinary prefix; later scans must match.
     * No Engine, length-value read, dummy transition, caller callback retention or implicit mask is involved.
     * Cold forward accepts exact-descriptor symbolic lists; compute(values,lengths) needs no forward.
     * Without a scan, rejection precedes supplied-state copying or layer construction. Each build
     * owns fresh private state. Exact descriptors include layout and gradient eligibility; the
     * lengths prototype is canonical contiguous, offset-zero and non-view. Saved Models use
     * Model.loadGraph and the same compute path, without new input-role metadata or a new format.
     * @return a non-null fresh owned eval Model with two ordered inputs and one chain result;
     *     caller closes it; compute returns detached values, while forward remains list-typed
     * @throws IllegalStateException if input or scan recipes are absent, or supplied storage is unavailable
     * @throws IllegalArgumentException if step geometry, T/B alignment or supplied state admission rejects
     * @throws ArithmeticException if checked geometry, initialization or allocation sizing overflows
     */
    public Model<List<Tensor>, List<Tensor>> buildWithLengths() {
        requireInput();
        if (scanSteps.isEmpty()) throw new IllegalStateException("buildWithLengths requires a scan recipe");
        Sequential layers = instantiate(List.copyOf(recipes));
        layers.eval();
        Tensor values = TensorFactory.create(input);
        List<Tensor> boundary = composeWithLengths(layers, List.of(values));
        return Model.fromComposition(List.of(values, boundary.getLast()), List.of(boundary.getFirst()),
                layers, inputs -> List.of(composeWithLengths(layers, inputs).getFirst()));
    }

    /**
     * Instantiates each snapshotted recipe once, copying supplied identities before any expression.
     * @param snapshot nonempty stable recipe list, retained only during this build
     * @return fresh exclusive chain; automatic parameters remain deferred until prototype composition
     */
    private Sequential instantiate(List<LayerRecipe> snapshot) {
        IdentityHashMap<Tensor, Tensor> copies = copySupplied();
        List<UnaryTensorModule> layers = new ArrayList<>(snapshot.size());
        long baseSeed = seed;
        for (int i = 0; i < snapshot.size(); i++) {
            layers.add(snapshot.get(i).create(input.dataType(), baseSeed + STEP_SEED * i, copies));
        }
        return new Sequential(layers);
    }

    /**
     * Applies instantiated steps with a call-local shared lengths value, never rebuilding recipes.
     * @param layers frozen owned chain in eval mode
     * @param inputs [values,lengths], or [values] only for the single prototype composition
     * @return [chain result,lengths]; a missing prototype length input is created at the first scan
     * @throws IllegalArgumentException if a scan lacks positive TBF geometry or changes established T/B
     */
    private static List<Tensor> composeWithLengths(Sequential layers, List<Tensor> inputs) {
        Tensor current = inputs.getFirst();
        Tensor lengths = inputs.size() == 2 ? inputs.get(1) : null;
        long time = -1, batch = -1;
        for (Module module : layers.children().values()) {
            boolean scan = module instanceof RnnStep r && r.hiddenSize > 0
                    || module instanceof GruStep g && g.hiddenSize > 0
                    || module instanceof LstmStep l && l.hiddenSize > 0;
            if (scan) {
                if (current.descriptor().shape().rank() != 3) {
                    throw new IllegalArgumentException("scan requires rank-three TBF");
                }
                long nextTime = extent(current, 0), nextBatch = extent(current, 1);
                positive(nextTime, "scan time"); positive(nextBatch, "scan batch");
                positive(extent(current, 2), "scan features");
                if (time < 0) {
                    time = nextTime; batch = nextBatch;
                    if (lengths == null) {
                        Shape shape = Shape.of(batch);
                        lengths = TensorFactory.create(new TensorDescriptor(DataType.INT64, shape,
                                Optional.of(LayoutDescriptor.contiguous(shape)), false));
                    }
                } else if (time != nextTime || batch != nextBatch) {
                    throw new IllegalArgumentException("later scans must preserve exact time and batch extents");
                }
                if (module instanceof RnnStep r) current = r.forwardScan(current, lengths);
                else if (module instanceof GruStep g) current = g.forwardScan(current, lengths);
                else current = ((LstmStep) module).forwardScan(current, lengths);
            } else {
                current = ((UnaryTensorModule) module).forward(current);
            }
        }
        return List.of(current, Objects.requireNonNull(lengths, "scan lengths"));
    }

    /**
     * Copies external identities in a zero-node source before constructing any expression.
     * @return temporary identity association of supplied leaves to private copies, never retained by a Model
     */
    private IdentityHashMap<Tensor, Tensor> copySupplied() {
        IdentityHashMap<Tensor, Tensor> copies = new IdentityHashMap<>();
        List<Tensor> unique = new ArrayList<>();
        List<GraphCheckpoint.Leaf> leaves = new ArrayList<>();
        for (Tensor leaf : suppliedLeaves) {
            if (copies.containsKey(leaf)) continue;
            copies.put(leaf, leaf);
            leaves.add(new GraphCheckpoint.Leaf("leaf" + unique.size(), GraphCheckpoint.Role.PARAMETER, leaf));
            unique.add(leaf);
        }
        if (!unique.isEmpty()) {
            var acquired = GraphCheckpoint.copy(new GraphCheckpoint.Source(List.of(), unique, leaves));
            for (int i = 0; i < unique.size(); i++) copies.put(unique.get(i), acquired.outputs().get(i));
        }
        return copies;
    }

    /**
     * Appends only an internal closed factory after input declaration.
     * @param recipe internal factory, never supplied by callers
     * @return this same builder
     */
    private ModelBuilder append(LayerRecipe recipe) { requireInput(); recipes.add(recipe); return this; }

    /** @throws IllegalStateException if no input signature has been declared */
    private void requireInput() {
        if (input == null) throw new IllegalStateException("declare input before steps/build");
    }

    /**
     * Rejects supplied expressions or implicit dtype conversion; storage admission remains the raw copier's.
     * @param leaf schema-validated Linear leaf, never null
     * @throws IllegalArgumentException if provenance or dtype rejects
     */
    private void supplied(Tensor leaf) {
        if (leaf.provenance().isPresent() || leaf.descriptor().dataType() != input.dataType()) {
            throw new IllegalArgumentException("supplied state must be a leaf matching input dtype");
        }
    }

    /**
     * Checks architectural extents before retaining a recipe.
     * @param value extent to check
     * @param name diagnostic role
     * @throws IllegalArgumentException if value is not strictly positive
     */
    private static void positive(long value, String name) {
        if (value <= 0) throw new IllegalArgumentException(name + " must be positive: " + value);
    }

    /**
     * Reads one proved static extent at the cold composition boundary.
     * @param input positive static Tensor
     * @param axis zero-based existing axis
     * @return exact extent in elements
     */
    private static long extent(Tensor input, int axis) {
        return ((StaticDimension) input.descriptor().shape().dimension(axis)).size();
    }

    /**
     * Packs every static sequence output in time order; it never chooses a final hidden state.
     * @param outputs nonempty ordinary [B,H] hidden outputs from an all-valid sequence
     * @return ordinary [T,B,H] concat of reshaped hidden outputs
     */
    private static Tensor allTimeSteps(List<Tensor> outputs) {
        Tensor[] steps = new Tensor[outputs.size()];
        for (int i = 0; i < steps.length; i++) {
            Tensor output = outputs.get(i);
            steps[i] = output.reshape(1, extent(output, 0), extent(output, 1));
        }
        return Tensor.concat(0, steps);
    }

    /** Closed construction factories retained only by the builder, never by a built Model. */
    @FunctionalInterface
    private interface LayerRecipe {
        /**
         * Creates a fresh child, without constructing its input expressions.
         * @param type exact floating input and parameter dtype
         * @param seed derived step seed
         * @param copies temporary supplied-leaf association
         * @return fresh unowned child, never null
         */
        UnaryTensorModule create(DataType type, long seed, IdentityHashMap<Tensor, Tensor> copies);
    }

    /** Stateless maximum-pool adapter; geometry is immutable and mode-independent. */
    private static final class MaxPoolStep extends UnaryTensorModule {
        private final MaxPool2dAttrs attrs;
        /**
         * Retains fixed pooling geometry without storing inputs or execution state.
         * @param attrs immutable complete pooling geometry
         */
        private MaxPoolStep(MaxPool2dAttrs attrs) { this.attrs = attrs; }
        /**
         * Constructs one ordinary pooling occurrence without retaining input or result state.
         * @param input NCHW expression accepted by Model pooling, never mutated
         * @return ordinary maximum-pool expression
         */
        @Override public Tensor forward(Tensor input) { return input.maxPool2d(attrs); }
    }

    /** Reshape adapter preserving the first axis rather than guessing batch/time semantics. */
    private static final class FlattenStep extends UnaryTensorModule {
        /**
         * Constructs a checked reshape without inferring whether axis zero means batch or time.
         * @param input static rank-at-least-two expression
         * @return [axis0, product of remaining extents] ordinary reshape
         * @throws IllegalArgumentException if rank is less than two
         * @throws ArithmeticException if the checked product overflows
         */
        @Override public Tensor forward(Tensor input) {
            int rank = input.descriptor().shape().rank();
            if (rank < 2) throw new IllegalArgumentException("flatten requires rank at least two");
            long features = 1;
            for (int axis = 1; axis < rank; axis++) features = Math.multiplyExact(features, extent(input, axis));
            return input.reshape(extent(input, 0), features);
        }
    }

    /** Explicit dense-time or scan-port selection; recurrent recipes never silently reduce outputs. */
    private static final class LastTimeStepStep extends UnaryTensorModule {
        private final boolean finalHidden;
        /** Creates the unchanged dense last-time selector. */
        private LastTimeStepStep() { this(false); }
        /**
         * Selects the existing scan port instead of creating a time-selection operation.
         * @param finalHidden true only for the recipe immediately following a scan
         */
        private LastTimeStepStep(boolean finalHidden) { this.finalHidden = finalHidden; }
        /**
         * Returns an existing scan sibling or constructs a dense-time selection with ordinary view metadata.
         * @param input positive static rank-three TBF expression; the exact scan outputs port when
         *     configured for final hidden selection
         * @return existing finalHidden port or ordinary last-time selection [B,H], as configured
         * @throws IllegalArgumentException if rank is not three
         */
        @Override public Tensor forward(Tensor input) {
            if (finalHidden) return input.provenance().orElseThrow().producer().output(1);
            if (input.descriptor().shape().rank() != 3) {
                throw new IllegalArgumentException("lastTimeStep requires rank-three TBF");
            }
            return input.select(0, extent(input, 0) - 1);
        }
    }

    /** Typed all-output RNN adapter owning its existing sequence under a stable state path. */
    private static final class RnnStep extends UnaryTensorModule {
        private final RnnSequence sequence;
        private final long hiddenSize;
        /**
         * Takes exclusive child ownership without constructing a recurrent expression yet.
         * @param sequence fresh unowned sequence, initialized by prototype forward
         */
        private RnnStep(RnnSequence sequence) { this(sequence, 0); }
        /**
         * Owns an existing sequence with a fixed scan width, without running a dummy transition.
         * @param sequence fresh sequence exclusively owned under sequence.cell
         * @param hiddenSize positive scan width, or zero for unchanged static unrolling
         */
        private RnnStep(RnnSequence sequence, long hiddenSize) {
            this.sequence = child("sequence", sequence); this.hiddenSize = hiddenSize;
        }
        /**
         * Composes one fixed scan with explicit zero initial state; reads no length values.
         * @param input positive TBF expression
         * @param lengths shared canonical INT64[B] expression
         * @return dense canonical scan outputs, retaining the finalHidden sibling
         */
        private Tensor forwardScan(Tensor input, Tensor lengths) {
            Tensor zero = TensorFactory.zeros(Shape.of(extent(input, 1), hiddenSize),
                    input.descriptor().dataType(), Optional.empty(), false);
            return sequence.forwardScan(input, lengths, zero, RecurrentDirection.FORWARD).outputs();
        }
        /**
         * Composes the existing all-valid sequence with fresh zero state and packs every output.
         * @param input positive static TBF expression
         * @return all hidden steps [T,B,H] from a fresh zero initial state
         */
        @Override public Tensor forward(Tensor input) { return allTimeSteps(sequence.forward(input).packedOutputs()); }
    }

    /** Typed all-output reset-after GRU adapter, with no streaming state. */
    private static final class GruStep extends UnaryTensorModule {
        private final GruSequence sequence;
        private final long hiddenSize;
        /**
         * Takes exclusive child ownership; state paths remain under sequence.cell.
         * @param sequence fresh unowned sequence, initialized by prototype forward
         */
        private GruStep(GruSequence sequence) { this(sequence, 0); }
        /**
         * Owns a fresh sequence without initialization expressions or caller input retention.
         * @param sequence exclusively owned sequence under sequence.cell
         * @param hiddenSize positive scan width, or zero for static unrolling
         */
        private GruStep(GruSequence sequence, long hiddenSize) {
            this.sequence = child("sequence", sequence); this.hiddenSize = hiddenSize;
        }
        /**
         * Composes a FORWARD reset-after scan, never carrying state from an earlier call.
         * @param input positive TBF expression
         * @param lengths shared non-gradient INT64[B] expression, never read
         * @return canonical dense output with finalHidden on the same producer
         */
        private Tensor forwardScan(Tensor input, Tensor lengths) {
            Tensor zero = TensorFactory.zeros(Shape.of(extent(input, 1), hiddenSize),
                    input.descriptor().dataType(), Optional.empty(), false);
            return sequence.forwardScan(input, lengths, zero, RecurrentDirection.FORWARD).outputs();
        }
        /**
         * Composes reset-after gates through the existing sequence, without cross-call state.
         * @param input positive static TBF expression
         * @return every hidden step [T,B,H] starting from zero
         */
        @Override public Tensor forward(Tensor input) { return allTimeSteps(sequence.forward(input).packedOutputs()); }
    }

    /** Typed all-output LSTM adapter; final cell/hidden states are never retained as future input. */
    private static final class LstmStep extends UnaryTensorModule {
        private final LstmSequence sequence;
        private final long hiddenSize;
        /**
         * Takes exclusive child ownership without retaining hidden/cell values between calls.
         * @param sequence fresh unowned sequence, initialized by prototype forward
         */
        private LstmStep(LstmSequence sequence) { this(sequence, 0); }
        /**
         * Owns one independent LSTM without dummy transitions or carried state.
         * @param sequence fresh sequence exclusively owned under sequence.cell
         * @param hiddenSize positive scan width, or zero for static unrolling
         */
        private LstmStep(LstmSequence sequence, long hiddenSize) {
            this.sequence = child("sequence", sequence); this.hiddenSize = hiddenSize;
        }
        /**
         * Composes a FORWARD scan with distinct zero hidden and cell leaves.
         * @param input positive TBF expression
         * @param lengths shared non-gradient INT64[B] expression, never read
         * @return dense canonical hidden outputs, with finalHidden/finalCell sibling ports
         */
        private Tensor forwardScan(Tensor input, Tensor lengths) {
            Shape shape = Shape.of(extent(input, 1), hiddenSize);
            Tensor hidden = TensorFactory.zeros(shape, input.descriptor().dataType(), Optional.empty(), false);
            Tensor cell = TensorFactory.zeros(shape, input.descriptor().dataType(), Optional.empty(), false);
            return sequence.forwardScan(input, lengths, hidden, cell, RecurrentDirection.FORWARD).outputs();
        }
        /**
         * Composes the existing LSTM sequence and publishes hidden outputs, not final cell state.
         * @param input positive static TBF expression
         * @return every hidden step [T,B,H] starting from zero hidden and cell states
         */
        @Override public Tensor forward(Tensor input) { return allTimeSteps(sequence.forward(input).packedOutputs()); }
    }
}
