package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.util.Objects;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * A stateful embedding lookup with one rank-two floating weight table.
 *
 * <p>The table has the exact positive fully static Shape
 * {@code [vocabularySize, embeddingSize]}. Axis zero contains the rows selected by index values,
 * and axis one is appended as the final result Dimension. The table is declared as the sole
 * parameter under local name {@code weight}. Callers may either supply that table exactly or
 * request eager whole-table initialization from explicit sizes, floating type, policy, and seed.
 * The initialized form never infers vocabulary size from indices or a batch, and it does not wait
 * for a forward call. Every row is ordinary trainable state; this layer defines no padding index,
 * special row, invariant zero row, or frozen row.</p>
 *
 * <p>{@link #forward(Tensor)} reads the current weight binding once and delegates directly to
 * {@link Tensor#embedding(Tensor)}. Model therefore owns accepted index types, the ordinary
 * axis-zero Gather operation, result metadata, provenance, and inherited failures. This layer
 * adds no numerical lookup, value-bound enforcement, gradient rule, compiler behavior, storage,
 * backend behavior, or execution. Forward construction is identical in training and evaluation
 * mode.</p>
 *
 * <p>A successful {@link Parameter#replace(Tensor)} becomes visible to the next forward call.
 * Earlier Tensor references and already constructed expressions retain their prior exact table.
 * Replacement and forward construction are not thread-safe as a combined operation; callers must
 * coordinate them when one stable table snapshot matters.</p>
 */
public final class Embedding extends UnaryTensorModule {
    private final Parameter weight;

    /**
     * Creates an embedding layer from one exact caller-supplied weight table.
     *
     * <p>The supplied Tensor must be floating and gradient-eligible, with a fully static rank-two
     * Shape whose vocabulary and embedding extents are both positive. All validation completes
     * before the parameter is declared. Construction retains the exact Tensor reference without
     * creating or evaluating another Tensor, allocating storage, consuming randomness, or
     * mutating the supplied value.</p>
     *
     * @param weight non-null floating Tensor with {@code requiresGrad == true} and positive fully
     *     static rank-two Shape {@code [vocabularySize, embeddingSize]}; retained exactly
     * @throws NullPointerException if {@code weight} is {@code null}
     * @throws IllegalArgumentException if the weight type is not floating, gradient eligibility
     *     is false, rank is not two, Shape is not fully static, vocabulary extent is not positive,
     *     or embedding extent is not positive, checked in that order
     */
    public Embedding(Tensor weight) {
        Tensor suppliedWeight = Objects.requireNonNull(weight, "weight");
        validateWeight(suppliedWeight);
        this.weight = parameter("weight", suppliedWeight);
    }

    /**
     * Creates an embedding layer with one eagerly initialized complete weight table.
     *
     * <p>Construction validates the explicit schema before initialization, creates exact Shape
     * {@code [vocabularySize, embeddingSize]}, and dispatches the selected policy once. A random
     * policy uses one fresh standard {@code L64X128MixRandom} source created from {@code seed};
     * zero and one create no random source and ignore the seed. Fan policies interpret axis zero
     * as fan-out and axis one as fan-in, so this table uses {@code vocabularySize} as fan-out and
     * {@code embeddingSize} as fan-in.</p>
     *
     * <p>The returned fresh floating Tensor has {@code requiresGrad == true} and is declared under
     * local name {@code weight} only after initialization succeeds. Construction is entirely eager
     * and exposes no reservation, first-forward binding, padding-row rewrite, or frozen row.
     * Success therefore consumes the initializer's one Tensor identifier and immediately exposes
     * one complete permanent parameter. If sampling, allocation, or identifier acquisition fails,
     * completed draws or allocations are not rolled back, but no parameter wrapper is declared and
     * no layer is returned.</p>
     *
     * @param vocabularySize positive number of ordinary trainable table rows
     * @param embeddingSize positive width of each table row
     * @param dataType non-null floating table type; exactly FLOAT64, FLOAT32, or BFLOAT16
     * @param weightInitialization non-null existing policy applied once to the complete table
     * @param seed seed for the exact standard random source used by a random policy; accepted but
     *     otherwise ignored by zero and one
     * @throws NullPointerException if {@code dataType} or {@code weightInitialization} is null,
     *     checked in that order after both sizes
     * @throws IllegalArgumentException if a size is not positive, the type is not floating, or
     *     the complete table exceeds the Java array element limit, checked in that order
     * @throws ArithmeticException if checked table element-count or delegated layout arithmetic
     *     overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if delegated eager source or destination allocation fails
     */
    public Embedding(
            long vocabularySize,
            long embeddingSize,
            DataType dataType,
            ParameterInitialization weightInitialization,
            long seed) {
        if (vocabularySize <= 0) {
            throw new IllegalArgumentException(
                    "vocabularySize must be positive: " + vocabularySize);
        }
        if (embeddingSize <= 0) {
            throw new IllegalArgumentException(
                    "embeddingSize must be positive: " + embeddingSize);
        }
        DataType requestedType = Objects.requireNonNull(dataType, "dataType");
        ParameterInitialization initialization =
                Objects.requireNonNull(weightInitialization, "weightInitialization");
        if (!requestedType.isFloating()) {
            throw new IllegalArgumentException(
                    "embedding data type must be floating: " + requestedType);
        }
        Shape weightShape = Shape.of(vocabularySize, embeddingSize);
        long elementCount = weightShape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "embedding weight element count exceeds Java array limit: count="
                            + elementCount + ", maximum=" + Integer.MAX_VALUE);
        }

        Tensor initializedWeight;
        if (initialization.requiresRandomGenerator()) {
            RandomGenerator generator =
                    RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom").create(seed);
            initializedWeight = ParameterInitializers.initialize(
                    weightShape, requestedType, initialization, generator);
        } else {
            initializedWeight = ParameterInitializers.initialize(
                    weightShape, requestedType, initialization);
        }
        this.weight = parameter("weight", initializedWeight);
    }

    /**
     * Returns the stable weight parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code weight}; its
     *     {@link Parameter#value()} is the current table binding
     */
    public Parameter weight() {
        return weight;
    }

    /**
     * Builds one embedding Tensor expression from the current table and supplied indices.
     *
     * <p>The indices null check occurs before the current binding is read. That binding is read
     * exactly once and used as the receiver of {@link Tensor#embedding(Tensor)}. The inherited
     * Model contract accepts exact INT32 or INT64 indices of any rank, appends the table's exact
     * axis-one Dimension to their complete Shape, and creates one ordinary axis-zero GATHER with
     * ordered {@code [weight, indices]} provenance. Result type and gradient eligibility come
     * from the table. This method is mode-insensitive and reads no values, performs no bounds
     * check, and does not evaluate or execute the lookup.</p>
     *
     * @param indices non-null INT32 or INT64 Tensor of any rank accepted by the Model embedding
     *     convenience; retained as exact provenance input one and never mutated
     * @return a non-null fresh, unlabeled, storage-free, unresolved-layout Model GATHER expression
     *     using the exact current weight binding observed by this call
     * @throws NullPointerException if {@code indices} is {@code null}, with message {@code indices}
     * @throws IllegalArgumentException if inherited Model embedding index-type validation fails
     * @throws ArithmeticException if inherited checked Gather result-Shape construction overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    @Override
    public Tensor forward(Tensor indices) {
        Objects.requireNonNull(indices, "indices");
        Tensor currentWeight = weight.value();
        return currentWeight.embedding(indices);
    }

    private static void validateWeight(Tensor weight) {
        DataType weightType = weight.descriptor().dataType();
        if (!weightType.isFloating()) {
            throw new IllegalArgumentException(
                    "embedding weight must have a floating data type: " + weightType);
        }
        if (!weight.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "embedding weight must have requiresGrad == true");
        }
        Shape weightShape = weight.descriptor().shape();
        if (weightShape.rank() != 2) {
            throw new IllegalArgumentException(
                    "embedding weight must have rank two: " + weightShape.rank());
        }
        if (!weightShape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "embedding weight must have a fully static shape: " + weightShape);
        }
        long vocabularySize = ((StaticDimension) weightShape.dimension(0)).size();
        if (vocabularySize == 0) {
            throw new IllegalArgumentException(
                    "embedding weight must have positive vocabularySize: " + vocabularySize);
        }
        long embeddingSize = ((StaticDimension) weightShape.dimension(1)).size();
        if (embeddingSize == 0) {
            throw new IllegalArgumentException(
                    "embedding weight must have positive embeddingSize: " + embeddingSize);
        }
    }
}
