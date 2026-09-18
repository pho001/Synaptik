package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Owns checked convolution kernel/fan arithmetic and dispatches the closed initialization policy.
 *
 * <p>The helper is deliberately package-private: fan values are derived implementation facts,
 * not layer configuration or public state. Random policies create one fresh standard
 * {@code L64X128MixRandom} source for each call; constant policies create no source. It retains
 * no Tensor, policy, seed, or random source.</p>
 */
final class ConvolutionParameterInitialization {
    private ConvolutionParameterInitialization() {
    }

    /**
     * Returns the one-dimensional kernel volume.
     *
     * @param first positive kernel extent
     * @return {@code first}
     */
    static long kernelVolume(long first) {
        return first;
    }

    /**
     * Returns the checked two-dimensional kernel volume.
     *
     * @param first positive first kernel extent
     * @param second positive second kernel extent
     * @return the product of both extents
     * @throws ArithmeticException if the product overflows {@code long}
     */
    static long kernelVolume(long first, long second) {
        return Math.multiplyExact(first, second);
    }

    /**
     * Returns the checked three-dimensional kernel volume.
     *
     * @param first positive first kernel extent
     * @param second positive second kernel extent
     * @param third positive third kernel extent
     * @return the product of all extents
     * @throws ArithmeticException if either product overflows {@code long}
     */
    static long kernelVolume(long first, long second, long third) {
        return Math.multiplyExact(Math.multiplyExact(first, second), third);
    }

    /**
     * Returns the checked per-output-channel fan-in for grouped convolution.
     *
     * @param inChannels positive input-channel count divisible by {@code groups}
     * @param groups positive group count
     * @param kernelVolume positive checked spatial kernel volume
     * @return {@code (inChannels / groups) * kernelVolume}
     * @throws ArithmeticException if the product overflows {@code long}
     */
    static long fanIn(long inChannels, long groups, long kernelVolume) {
        return Math.multiplyExact(inChannels / groups, kernelVolume);
    }

    /**
     * Returns the checked per-input-channel fan-out for grouped convolution.
     *
     * @param outChannels positive output-channel count divisible by {@code groups}
     * @param groups positive group count
     * @param kernelVolume positive checked spatial kernel volume
     * @return {@code (outChannels / groups) * kernelVolume}
     * @throws ArithmeticException if the product overflows {@code long}
     */
    static long fanOut(long outChannels, long groups, long kernelVolume) {
        return Math.multiplyExact(outChannels / groups, kernelVolume);
    }

    /**
     * Creates one gradient-eligible convolution weight using the supplied checked fan values.
     *
     * @param weightShape non-null fully static convolution weight Shape
     * @param dataType non-null floating element type
     * @param initialization non-null closed initialization policy
     * @param seed seed for the fresh standard source used by a random policy
     * @param fanIn positive checked grouped fan-in
     * @param fanOut positive checked grouped fan-out
     * @return a fresh non-null Tensor with the exact supplied Shape and type
     * @throws RuntimeException if initialization or a random source fails
     * @throws NullPointerException if {@code weightShape}, {@code dataType}, or
     *     {@code initialization} is null
     * @throws IllegalArgumentException if delegated policy, Shape, type, or distribution
     *     validation fails
     * @throws ArithmeticException if delegated checked storage arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if eager storage allocation fails
     */
    static Tensor initialize(
            Shape weightShape,
            DataType dataType,
            ParameterInitialization initialization,
            long seed,
            long fanIn,
            long fanOut) {
        if (!initialization.requiresRandomGenerator()) {
            return ParameterInitializers.initialize(weightShape, dataType, initialization);
        }
        RandomGenerator source = RandomGeneratorFactory.<RandomGenerator>of("L64X128MixRandom")
                .create(seed);
        if (initialization.equals(ParameterInitialization.glorotNormal())) {
            double standardDeviation = Math.sqrt(2.0d / ((double) fanIn + (double) fanOut));
            return ParameterInitializers.normal(
                    weightShape, dataType, 0.0d, standardDeviation, source);
        }
        if (initialization.equals(ParameterInitialization.glorotUniform())) {
            double bound = Math.sqrt(6.0d / ((double) fanIn + (double) fanOut));
            return ParameterInitializers.uniform(weightShape, dataType, -bound, bound, source);
        }
        if (initialization.equals(ParameterInitialization.kaimingReluNormal())) {
            double standardDeviation = Math.sqrt(2.0d / (double) fanIn);
            return ParameterInitializers.normal(
                    weightShape, dataType, 0.0d, standardDeviation, source);
        }
        if (initialization.equals(ParameterInitialization.kaimingReluUniform())) {
            double bound = Math.sqrt(6.0d / (double) fanIn);
            return ParameterInitializers.uniform(weightShape, dataType, -bound, bound, source);
        }
        return ParameterInitializers.initialize(weightShape, dataType, initialization, source);
    }

    /**
     * Returns the checked effective kernel extent for one spatial axis.
     *
     * @param kernel positive kernel extent
     * @param dilation positive dilation
     * @return {@code dilation * (kernel - 1) + 1}
     * @throws ArithmeticException if checked multiplication or addition overflows
     */
    static long effectiveKernel(long kernel, long dilation) {
        return Math.addExact(Math.multiplyExact(dilation, kernel - 1L), 1L);
    }

    /**
     * Returns twice the symmetric per-side padding using checked arithmetic.
     *
     * @param padding non-negative per-side padding
     * @return twice {@code padding}
     * @throws ArithmeticException if the product overflows {@code long}
     */
    static long doubledPadding(long padding) {
        return Math.multiplyExact(2L, padding);
    }

    /**
     * Validates every statically knowable output-geometry fact for one spatial axis.
     *
     * @param input non-null static or symbolic input extent
     * @param kernel positive kernel extent
     * @param stride positive stride
     * @param padding non-negative symmetric padding per side
     * @param dilation positive dilation
     * @param axis diagnostic spatial-axis name
     * @throws NullPointerException if {@code input} is null
     * @throws IllegalArgumentException if a static extent cannot fit the effective kernel
     * @throws ArithmeticException if checked geometry arithmetic overflows
     */
    static void validateSpatialGeometry(
            Dimension input, long kernel, long stride, long padding, long dilation, String axis) {
        long effectiveKernel = effectiveKernel(kernel, dilation);
        long doubledPadding = doubledPadding(padding);
        if (input instanceof StaticDimension staticDimension) {
            long padded = Math.addExact(staticDimension.size(), doubledPadding);
            long numerator = Math.subtractExact(padded, effectiveKernel);
            if (numerator < 0L) {
                throw new IllegalArgumentException(
                        "convolution " + axis + " geometry has negative output numerator: "
                                + numerator);
            }
            Math.addExact(numerator / stride, 1L);
        }
    }

    /**
     * Rejects a fully static parameter Shape that cannot use Model's Java-array storage.
     *
     * @param shape non-null fully static parameter Shape
     * @param role diagnostic parameter role
     * @throws NullPointerException if {@code shape} is null
     * @throws IllegalArgumentException if the known count exceeds {@link Integer#MAX_VALUE}
     * @throws ArithmeticException if the Shape count overflows
     */
    static void validateJavaArrayCount(Shape shape, String role) {
        long count = shape.knownElementCount().orElseThrow();
        if (count > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    role + " element count exceeds Java array limit: count=" + count
                            + ", maximum=" + Integer.MAX_VALUE);
        }
    }
}
