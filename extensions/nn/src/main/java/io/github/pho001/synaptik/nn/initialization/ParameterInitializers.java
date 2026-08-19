package io.github.pho001.synaptik.nn.initialization;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorRandoms;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.Objects;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * Creates eager floating Tensor leaves intended for neural-network parameter bindings.
 *
 * <p>Every method requires a fully static {@link Shape} and one of {@link DataType#FLOAT64},
 * {@link DataType#FLOAT32}, or {@link DataType#BFLOAT16}. Every successful call returns a fresh
 * dense-contiguous, host-backed, provenance-free Tensor that retains the exact supplied Shape and
 * data type, has no label, and has {@code requiresGrad == true}. Callers may bind that Tensor to a
 * module-owned {@link Parameter}; this class does not create, name, own, retain, or update a
 * parameter, and {@code Parameter} owns no initialization policy.</p>
 *
 * <p>Random values are sampled eagerly through the Model random factories from the exact
 * caller-supplied {@link RandomGenerator}. Each logical row-major element consumes one matching
 * {@link RandomGenerator#nextGaussian()} or
 * {@link RandomGenerator#nextDouble(double, double)} call; an empty Tensor consumes no draw. The
 * caller selects, configures, seeds, owns, advances, and coordinates access to the source. This
 * stateless class never selects a default source and never retains, substitutes, synchronizes,
 * resets, splits, serializes, or closes the supplied object.</p>
 *
 * <p>Glorot policies use fixed unit gain, while Kaiming policies use fixed rectified-linear-unit
 * (ReLU) gain and fan-in mode. Fan-based methods accept any fully static positive rank-two
 * weight shape in {@code [fanOut, fanIn]} orientation, including Linear
 * {@code [outFeatures, inFeatures]} and Embedding
 * {@code [vocabularySize, embeddingSize]} tables. They provide no
 * convolution fan inference, configurable gain, activation, fan mode, alias, or default source.
 * This eager host-data boundary neither accepts nor creates {@link GraphRngState}, which represents
 * deferred random state in a Tensor expression graph.</p>
 *
 * <p>The three-argument {@code initialize} overload accepts exactly zero and one policies and
 * cannot create or consume a random generator. The four-argument overload accepts exactly the
 * other six policies and forwards the exact supplied generator. Both are exhaustive dispatch
 * conveniences over the named initializer methods below; they add no fallback or hidden state.</p>
 */
public final class ParameterInitializers {
    /** Prevents instances because initialization policy is scoped to one static call. */
    private ParameterInitializers() {
    }

    /**
     * Applies an exact constant initialization policy without a random generator.
     *
     * @param shape non-null Shape passed unchanged to the selected initializer
     * @param dataType non-null data type passed unchanged to the selected initializer
     * @param initialization non-null {@link ParameterInitialization#zeros()} or
     *     {@link ParameterInitialization#ones()} policy
     * @return the fresh parameter Tensor returned by the selected existing initializer
     * @throws NullPointerException if an argument is null, checked in parameter order
     * @throws IllegalArgumentException if the policy requires a random generator or delegated
     *     initializer validation fails
     * @throws ArithmeticException if delegated checked element-count or layout arithmetic
     *     overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor initialize(
            Shape shape, DataType dataType, ParameterInitialization initialization) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        ParameterInitialization policy =
                Objects.requireNonNull(initialization, "initialization");
        return switch (policy.kind()) {
            case ZEROS -> zeros(shape, dataType);
            case ONES -> ones(shape, dataType);
            default -> throw new IllegalArgumentException(
                    "initialization policy requires a random generator: " + policy);
        };
    }

    /**
     * Applies a random initialization policy with the exact caller-owned generator.
     *
     * @param shape non-null Shape passed unchanged to the selected initializer
     * @param dataType non-null data type passed unchanged to the selected initializer
     * @param initialization non-null random policy
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return the fresh parameter Tensor returned by the selected existing initializer
     * @throws NullPointerException if an argument is null, checked in parameter order
     * @throws IllegalArgumentException if the policy is zero/one or delegated validation fails
     * @throws ArithmeticException if delegated checked element-count or layout arithmetic
     *     overflows
     * @throws RuntimeException if the random source throws while sampling; completed calls remain
     *     consumed and no Tensor is returned
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor initialize(
            Shape shape,
            DataType dataType,
            ParameterInitialization initialization,
            RandomGenerator randomGenerator) {
        Objects.requireNonNull(shape, "shape");
        Objects.requireNonNull(dataType, "dataType");
        ParameterInitialization policy =
                Objects.requireNonNull(initialization, "initialization");
        RandomGenerator source = Objects.requireNonNull(randomGenerator, "randomGenerator");
        return switch (policy.kind()) {
            case GLOROT_NORMAL -> glorotNormal(shape, dataType, source);
            case GLOROT_UNIFORM -> glorotUniform(shape, dataType, source);
            case KAIMING_RELU_NORMAL -> kaimingReluNormal(shape, dataType, source);
            case KAIMING_RELU_UNIFORM -> kaimingReluUniform(shape, dataType, source);
            case NORMAL -> normal(shape, dataType, policy.first(), policy.second(), source);
            case UNIFORM -> uniform(shape, dataType, policy.first(), policy.second(), source);
            case ZEROS, ONES -> throw new IllegalArgumentException(
                    "initialization policy does not accept a random generator: " + policy);
        };
    }

    /**
     * Creates a fresh floating parameter Tensor filled with exact typed zero.
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code shape} or {@code dataType} is null, checked in that
     *     order
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Model
     *     Java-array limit, or the data type is not floating
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model destination allocation fails
     */
    public static Tensor zeros(Shape shape, DataType dataType) {
        return TensorFactory.zeros(shape, dataType, Optional.empty(), true);
    }

    /**
     * Creates a fresh floating parameter Tensor filled with exact typed one.
     *
     * @param shape non-null fully static result shape; scalar and zero-element shapes are valid
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code shape} or {@code dataType} is null, checked in that
     *     order
     * @throws IllegalArgumentException if the shape is dynamic, its count exceeds the Model
     *     Java-array limit, or the data type is not floating
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor ones(Shape shape, DataType dataType) {
        return TensorFactory.ones(shape, dataType, Optional.empty(), true);
    }

    /**
     * Creates a fresh floating parameter Tensor from normally distributed samples.
     *
     * <p>Each row-major element consumes one {@link RandomGenerator#nextGaussian()} call and the
     * Model applies binary64 {@code mean + gaussian * standardDeviation} before its documented
     * FLOAT64, FLOAT32, or BFLOAT16 conversion. The caller owns and advances the exact supplied
     * source; this method neither retains nor manages it. An empty Tensor consumes no call. Model
     * validation, sampling, conversion, allocation, identifier, source-failure, and non-rollback
     * semantics are preserved.</p>
     *
     * @param shape non-null fully static result shape whose count fits a Java array
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param mean finite binary64 distribution mean
     * @param standardDeviation finite non-negative binary64 standard deviation
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code shape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if Model shape, type, distribution, or gradient validation
     *     fails
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor normal(
            Shape shape,
            DataType dataType,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator) {
        return TensorRandoms.randomNormal(
                shape,
                dataType,
                mean,
                standardDeviation,
                randomGenerator,
                Optional.empty(),
                true);
    }

    /**
     * Creates a fresh floating parameter Tensor from bounded continuous-uniform samples.
     *
     * <p>Each row-major element consumes one
     * {@link RandomGenerator#nextDouble(double, double)} call with the exact supplied bounds. The
     * caller owns and advances the exact supplied source; this method neither retains nor manages
     * it. An empty Tensor consumes no call. Model validation, sampling, conversion, allocation,
     * identifier, source-failure, and non-rollback semantics are preserved.</p>
     *
     * @param shape non-null fully static result shape whose count fits a Java array
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound, strictly greater than the
     *     lower bound
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code shape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if Model shape, type, bounds, or gradient validation fails
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor uniform(
            Shape shape,
            DataType dataType,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator) {
        return TensorRandoms.randomUniform(
                shape,
                dataType,
                lowerBoundInclusive,
                upperBoundExclusive,
                randomGenerator,
                Optional.empty(),
                true);
    }

    /**
     * Creates a Glorot unit-gain normal rank-two weight.
     *
     * <p>For {@code weightShape == [fanOut, fanIn]}, the distribution has mean zero and standard
     * deviation {@code sqrt(2.0 / (fanIn + fanOut))}. Fan addition and division use binary64,
     * avoiding signed-long addition overflow before exactly one Model normal-sampling call.</p>
     *
     * @param weightShape non-null fully static rank-two shape in {@code [fanOut, fanIn]}
     *     orientation with both extents positive
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied weight shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code weightShape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if the type is not floating, the Shape is not fully static
     *     rank two, {@code fanOut} or {@code fanIn} is zero, or the positive Shape's
     *     element count exceeds the Model Java-array limit, checked in that order
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked Model count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor glorotNormal(
            Shape weightShape, DataType dataType, RandomGenerator randomGenerator) {
        validateFanInputs(weightShape, dataType, randomGenerator);
        double fanOut = staticSize(weightShape, 0);
        double fanIn = staticSize(weightShape, 1);
        double standardDeviation = Math.sqrt(2.0d / (fanIn + fanOut));
        return normal(weightShape, dataType, 0.0d, standardDeviation, randomGenerator);
    }

    /**
     * Creates a Glorot unit-gain uniform rank-two weight.
     *
     * <p>For {@code weightShape == [fanOut, fanIn]}, the distribution interval is
     * {@code [-sqrt(6.0 / (fanIn + fanOut)), +sqrt(6.0 / (fanIn + fanOut)))}. Fan addition and
     * division use binary64, avoiding signed-long addition overflow before exactly one Model
     * bounded-uniform sampling call.</p>
     *
     * @param weightShape non-null fully static rank-two shape in {@code [fanOut, fanIn]}
     *     orientation with both extents positive
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied weight shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code weightShape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if the type is not floating, the Shape is not fully static
     *     rank two, {@code fanOut} or {@code fanIn} is zero, or the positive Shape's
     *     element count exceeds the Model Java-array limit, checked in that order
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked Model count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor glorotUniform(
            Shape weightShape, DataType dataType, RandomGenerator randomGenerator) {
        validateFanInputs(weightShape, dataType, randomGenerator);
        double fanOut = staticSize(weightShape, 0);
        double fanIn = staticSize(weightShape, 1);
        double bound = Math.sqrt(6.0d / (fanIn + fanOut));
        return uniform(weightShape, dataType, -bound, bound, randomGenerator);
    }

    /**
     * Creates a fan-in Kaiming normal rank-two weight with fixed ReLU gain.
     *
     * <p>For {@code weightShape == [fanOut, fanIn]}, the distribution has mean zero and standard
     * deviation {@code sqrt(2.0 / fanIn)}. The fan is converted to binary64 before division and
     * exactly one Model normal-sampling call follows validation.</p>
     *
     * @param weightShape non-null fully static rank-two shape in {@code [fanOut, fanIn]}
     *     orientation with both extents positive
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied weight shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code weightShape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if the type is not floating, the Shape is not fully static
     *     rank two, {@code fanOut} or {@code fanIn} is zero, or the positive Shape's
     *     element count exceeds the Model Java-array limit, checked in that order
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked Model count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor kaimingReluNormal(
            Shape weightShape, DataType dataType, RandomGenerator randomGenerator) {
        validateFanInputs(weightShape, dataType, randomGenerator);
        double fanIn = staticSize(weightShape, 1);
        double standardDeviation = Math.sqrt(2.0d / fanIn);
        return normal(weightShape, dataType, 0.0d, standardDeviation, randomGenerator);
    }

    /**
     * Creates a fan-in Kaiming uniform rank-two weight with fixed ReLU gain.
     *
     * <p>For {@code weightShape == [fanOut, fanIn]}, the distribution interval is
     * {@code [-sqrt(6.0 / fanIn), +sqrt(6.0 / fanIn))}. The fan is converted to binary64 before
     * division and exactly one Model bounded-uniform sampling call follows validation.</p>
     *
     * @param weightShape non-null fully static rank-two shape in {@code [fanOut, fanIn]}
     *     orientation with both extents positive
     * @param dataType non-null floating result type: FLOAT64, FLOAT32, or BFLOAT16
     * @param randomGenerator non-null transient caller-owned source, never retained
     * @return a non-null fresh dense provenance-free and unlabeled Tensor retaining the exact
     *     supplied weight shape and type, with {@code requiresGrad == true}
     * @throws NullPointerException if {@code weightShape}, {@code dataType}, or
     *     {@code randomGenerator} is null, checked in that order
     * @throws IllegalArgumentException if the type is not floating, the Shape is not fully static
     *     rank two, {@code fanOut} or {@code fanIn} is zero, or the positive Shape's
     *     element count exceeds the Model Java-array limit, checked in that order
     * @throws RuntimeException if the random source throws while sampling; completed source calls
     *     remain consumed, but no Tensor or identifier is created
     * @throws ArithmeticException if checked Model count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted after sampling
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public static Tensor kaimingReluUniform(
            Shape weightShape, DataType dataType, RandomGenerator randomGenerator) {
        validateFanInputs(weightShape, dataType, randomGenerator);
        double fanIn = staticSize(weightShape, 1);
        double bound = Math.sqrt(6.0d / fanIn);
        return uniform(weightShape, dataType, -bound, bound, randomGenerator);
    }

    private static void validateFanInputs(
            Shape weightShape, DataType dataType, RandomGenerator randomGenerator) {
        Objects.requireNonNull(weightShape, "weightShape");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(randomGenerator, "randomGenerator");
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "fan-based parameter initialization requires floating data type: " + dataType);
        }
        if (!weightShape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "fan-based parameter initialization requires a fully static shape: "
                            + weightShape);
        }
        if (weightShape.rank() != 2) {
            throw new IllegalArgumentException(
                    "fan-based parameter initialization requires rank-two weight shape: "
                            + weightShape);
        }
        long fanOut = ((StaticDimension) weightShape.dimension(0)).size();
        if (fanOut == 0) {
            throw new IllegalArgumentException(
                    "fan-based parameter initialization requires positive outFeatures: "
                            + fanOut);
        }
        long fanIn = ((StaticDimension) weightShape.dimension(1)).size();
        if (fanIn == 0) {
            throw new IllegalArgumentException(
                    "fan-based parameter initialization requires positive inFeatures: " + fanIn);
        }
    }

    private static double staticSize(Shape shape, int axis) {
        return (double) ((StaticDimension) shape.dimension(axis)).size();
    }
}
