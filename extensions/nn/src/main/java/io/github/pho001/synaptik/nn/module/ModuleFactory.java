package io.github.pho001.synaptik.nn.module;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Conv1d;
import io.github.pho001.synaptik.nn.layers.Conv2d;
import io.github.pho001.synaptik.nn.layers.Conv3d;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.nn.layers.GruSequence;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.layers.LstmSequence;
import io.github.pho001.synaptik.nn.layers.RnnSequence;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Stateless construction recipes for the standard initialized neural-network module families.
 *
 * <p>{@link #standard()} returns one immutable, instance-field-free namespace. Every recipe
 * creates and returns one fresh exact concrete module by invoking its existing public constructor
 * once. The caller chooses every architectural size, bias option, data type, initialization
 * policy, and seed explicitly; this factory retains none of them and caches no module, parameter,
 * Tensor, random source, or construction result.</p>
 *
 * <p>The factory neither registers nor owns a returned module. Functional models establish named
 * ownership only when caller code passes the result to {@link Topology#addModule(String, Module)}.
 * Recurrent recipes return a matching Sequence that already owns its fresh matching Cell under
 * local child name {@code cell}, exactly as the direct Sequence constructor specifies.</p>
 *
 * <p>This closed facade has no default or global layer configuration, provider or plugin
 * extension point, registry, lookup by name or type, service locator, reflection, or generic
 * Module lifecycle. Constructor validation, initialization effects, state paths, and forward
 * behavior remain contracts of the returned concrete type.</p>
 *
 * <p>Embedding construction is eager. Linear and recurrent recipes retain their current automatic
 * input-width binding behavior. Convolution recipes similarly infer only a positive static input-
 * channel extent and otherwise retain their complete rank-specific configuration. The Linear
 * recipe selects the exact deterministic JDK
 * {@code L64X128MixRandom} factory; recurrent constructors already select that standard algorithm
 * for random policies. Direct constructors remain the advanced path for caller-supplied state,
 * cells, recurrent states and lengths, random generators, or deterministic generator factories.
 * Recipe calls on this singleton may occur concurrently because the factory owns no mutable or
 * per-call state; each returned module retains its own documented threading contract.</p>
 */
public final class ModuleFactory {
    private static final ModuleFactory STANDARD = new ModuleFactory();

    /** Prevents caller construction because the standard namespace carries no identity state. */
    private ModuleFactory() {
    }

    /**
     * Returns the shared stateless standard recipe namespace.
     *
     * <p>This call creates no module, Tensor, random generator, parameter, topology registration,
     * or caller-specific configuration. Repeated calls return the same exact reference.</p>
     *
     * @return the exact non-null shared immutable standard factory
     */
    public static ModuleFactory standard() {
        return STANDARD;
    }

    /**
     * Creates one fresh eagerly initialized embedding table.
     *
     * <p>This recipe delegates once to the existing complete-table {@link Embedding} constructor.
     * Every row remains ordinary trainable state, and the result is not registered with a
     * topology.</p>
     *
     * @param vocabularySize positive number of ordinary trainable rows
     * @param embeddingSize positive width of each row
     * @param dataType non-null floating table type
     * @param weightInitialization non-null policy applied once to the complete table
     * @param seed any Java {@code long} seed used by the standard random source for a random
     *     policy; accepted but not used by zero or one
     * @return a non-null fresh unowned {@link Embedding} with its complete weight initialized
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated Embedding schema validation fails
     * @throws ArithmeticException if delegated checked Shape or element-count arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if delegated eager allocation fails
     */
    public Embedding embedding(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        return new Embedding(
                vocabularySize, embeddingSize, dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh automatic linear projection.
     *
     * <p>This recipe selects the exact deterministic JDK {@code L64X128MixRandom} factory and
     * delegates once to the current automatic {@link Linear} constructor. Construction retains
     * only that layer's immutable configuration and reserves its parameter names; it creates no
     * generator, Tensor, identifier, or Parameter. The layer infers only its input width on first
     * compatible forward use. Zero and one policies do not create a generator during binding.</p>
     *
     * @param outFeatures positive architectural output width
     * @param bias whether the layer will own a typed-zero bias
     * @param dataType non-null exact floating parameter and accepted input type
     * @param weightInitialization non-null policy applied to the complete bound weight
     * @param seed any Java {@code long} seed retained by the layer for random-policy
     *     initialization attempts; accepted but not used by zero or one
     * @return a non-null fresh unowned automatic {@link Linear}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if the named JDK generator factory is unavailable or
     *     delegated Linear validation fails
     */
    public Linear linear(
            long outFeatures,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        return new Linear(
                outFeatures,
                bias,
                dataType,
                weightInitialization,
                RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom"),
                seed);
    }

    /**
     * Creates one fresh automatic channels-first one-dimensional convolution layer.
     *
     * @param outChannels positive output-channel count divisible by {@code groups}
     * @param kernelWidth positive kernel width
     * @param stride positive output stride
     * @param padding non-negative symmetric padding per side
     * @param dilation positive kernel dilation
     * @param groups positive group count dividing output and eventual input channels
     * @param bias whether to own a typed-zero bias
     * @param dataType non-null exact floating input and parameter type
     * @param weightInitialization non-null weight policy
     * @param seed any seed retained by the layer for random-policy attempts; accepted but unused
     *     by zero and one policies
     * @return a fresh non-null unowned automatic {@link Conv1d}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated convolution schema validation fails
     * @throws ArithmeticException if delegated checked geometry or fan arithmetic overflows
     */
    public Conv1d conv1d(
            long outChannels, long kernelWidth, long stride, long padding, long dilation,
            long groups, boolean bias, DataType dataType,
            ParameterInitialization weightInitialization, long seed) {
        return new Conv1d(outChannels, kernelWidth, stride, padding, dilation, groups, bias,
                dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh automatic channels-first two-dimensional convolution layer.
     *
     * @param outChannels positive output-channel count divisible by {@code groups}
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param strideHeight positive output-height stride
     * @param strideWidth positive output-width stride
     * @param paddingHeight non-negative symmetric height padding per side
     * @param paddingWidth non-negative symmetric width padding per side
     * @param dilationHeight positive height dilation
     * @param dilationWidth positive width dilation
     * @param groups positive group count dividing output and eventual input channels
     * @param bias whether to own a typed-zero bias
     * @param dataType non-null exact floating input and parameter type
     * @param weightInitialization non-null weight policy
     * @param seed any seed retained by the layer for random-policy attempts; accepted but unused
     *     by zero and one policies
     * @return a fresh non-null unowned automatic {@link Conv2d}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated convolution schema validation fails
     * @throws ArithmeticException if delegated checked geometry or fan arithmetic overflows
     */
    public Conv2d conv2d(
            long outChannels, long kernelHeight, long kernelWidth,
            long strideHeight, long strideWidth, long paddingHeight, long paddingWidth,
            long dilationHeight, long dilationWidth, long groups, boolean bias,
            DataType dataType, ParameterInitialization weightInitialization, long seed) {
        return new Conv2d(outChannels, kernelHeight, kernelWidth, strideHeight, strideWidth,
                paddingHeight, paddingWidth, dilationHeight, dilationWidth, groups, bias,
                dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh automatic channels-first three-dimensional convolution layer.
     *
     * @param outChannels positive output-channel count divisible by {@code groups}
     * @param kernelDepth positive kernel depth
     * @param kernelHeight positive kernel height
     * @param kernelWidth positive kernel width
     * @param strideDepth positive output-depth stride
     * @param strideHeight positive output-height stride
     * @param strideWidth positive output-width stride
     * @param paddingDepth non-negative symmetric depth padding per side
     * @param paddingHeight non-negative symmetric height padding per side
     * @param paddingWidth non-negative symmetric width padding per side
     * @param dilationDepth positive depth dilation
     * @param dilationHeight positive height dilation
     * @param dilationWidth positive width dilation
     * @param groups positive group count dividing output and eventual input channels
     * @param bias whether to own a typed-zero bias
     * @param dataType non-null exact floating input and parameter type
     * @param weightInitialization non-null weight policy
     * @param seed any seed retained by the layer for random-policy attempts; accepted but unused
     *     by zero and one policies
     * @return a fresh non-null unowned automatic {@link Conv3d}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated convolution schema validation fails
     * @throws ArithmeticException if delegated checked geometry or fan arithmetic overflows
     */
    public Conv3d conv3d(
            long outChannels, long kernelDepth, long kernelHeight, long kernelWidth,
            long strideDepth, long strideHeight, long strideWidth,
            long paddingDepth, long paddingHeight, long paddingWidth,
            long dilationDepth, long dilationHeight, long dilationWidth, long groups,
            boolean bias, DataType dataType,
            ParameterInitialization weightInitialization, long seed) {
        return new Conv3d(outChannels, kernelDepth, kernelHeight, kernelWidth,
                strideDepth, strideHeight, strideWidth, paddingDepth, paddingHeight, paddingWidth,
                dilationDepth, dilationHeight, dilationWidth, groups, bias,
                dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh vanilla-RNN Sequence with one owned automatic Cell.
     *
     * @param hiddenSize positive recurrent hidden width
     * @param bias whether the cell will own a typed-zero bias
     * @param dataType non-null floating parameter and default-state type
     * @param weightInitialization non-null policy applied independently to both cell matrices
     * @param seed any Java {@code long} seed retained by the cell for random-policy
     *     initialization attempts; accepted but not used by zero or one
     * @return a non-null fresh unowned {@link RnnSequence} owning one fresh automatic Cell under
     *     {@code cell}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated RNN schema validation fails
     * @throws ArithmeticException if delegated checked Shape or count arithmetic overflows
     */
    public RnnSequence rnn(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        return new RnnSequence(hiddenSize, bias, dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh GRU Sequence with one owned automatic Cell.
     *
     * @param hiddenSize positive recurrent hidden width
     * @param bias whether the cell will own a complete packed typed-zero bias
     * @param dataType non-null floating parameter and default-state type
     * @param weightInitialization non-null policy applied independently to both packed matrices
     * @param seed any Java {@code long} seed retained by the cell for random-policy
     *     initialization attempts; accepted but not used by zero or one
     * @return a non-null fresh unowned {@link GruSequence} owning one fresh automatic Cell under
     *     {@code cell}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated GRU schema validation fails
     * @throws ArithmeticException if delegated packed-size, Shape, or count arithmetic overflows
     */
    public GruSequence gru(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        return new GruSequence(hiddenSize, bias, dataType, weightInitialization, seed);
    }

    /**
     * Creates one fresh LSTM Sequence with one owned automatic Cell.
     *
     * @param hiddenSize positive recurrent hidden and cell-state width
     * @param bias whether the cell will own a complete packed typed-zero input-side bias
     * @param dataType non-null floating parameter and default-state type
     * @param weightInitialization non-null policy applied independently to both packed matrices
     * @param seed any Java {@code long} seed retained by the cell for random-policy
     *     initialization attempts; accepted but not used by zero or one
     * @return a non-null fresh unowned {@link LstmSequence} owning one fresh automatic Cell under
     *     {@code cell}
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null
     * @throws IllegalArgumentException if delegated LSTM schema validation fails
     * @throws ArithmeticException if delegated packed-size, Shape, or count arithmetic overflows
     */
    public LstmSequence lstm(
            long hiddenSize,
            boolean bias,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        return new LstmSequence(hiddenSize, bias, dataType, weightInitialization, seed);
    }
}
