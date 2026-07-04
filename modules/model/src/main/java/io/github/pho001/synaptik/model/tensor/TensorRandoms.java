package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * Implements validated eager normal sampling for {@link TensorFactory} without retaining random
 * source or result state.
 *
 * <p>The helper accepts only fully static Java-array-sized shapes and the three floating data
 * types. It constructs canonical dense descriptors, samples directly into one matching primitive
 * carrier, and delegates exactly once to flat import. It never creates a Tensor or storage
 * directly, allocates an identifier, looks up or replaces a source, or synchronizes, seeds,
 * resets, splits, or closes the caller-owned generator.</p>
 */
final class TensorRandoms {
    /** Prevents instantiation because sampling is stateless and scoped to one factory call. */
    private TensorRandoms() {
    }

    /**
     * Validates a requested normal tensor, samples one typed carrier, and imports it once.
     *
     * @param shape non-null shape already checked by the public factory; must be fully static
     * @param dataType non-null requested type already checked by the public factory
     * @param mean requested finite binary64 mean
     * @param standardDeviation requested finite numerically non-negative binary64 deviation
     * @param randomGenerator exact non-null transient caller-owned source
     * @param label non-null optional label delegated unchanged to flat import
     * @param requiresGrad explicit result gradient intent
     * @return the exact tensor returned by one matching flat-import call; never {@code null}
     * @throws IllegalArgumentException if shape, count, type, mean, or deviation validation fails,
     *     or delegated label validation rejects blank text
     * @throws ArithmeticException if checked logical-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if identifier space is exhausted after all samples are drawn
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    static Tensor randomNormal(
            Shape shape,
            DataType dataType,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator,
            Optional<String> label,
            boolean requiresGrad) {
        TensorDescriptor descriptor = descriptor(
                shape, dataType, mean, standardDeviation, requiresGrad);
        int length = Math.toIntExact(shape.knownElementCount().orElseThrow());
        return switch (dataType) {
            case FLOAT64 -> sampleFloat64(
                    descriptor, mean, standardDeviation, randomGenerator, label, length);
            case FLOAT32 -> sampleFloat32(
                    descriptor, mean, standardDeviation, randomGenerator, label, length);
            case BFLOAT16 -> sampleBFloat16(
                    descriptor, mean, standardDeviation, randomGenerator, label, length);
            case INT32, INT64, BOOL -> throw new AssertionError("validated floating data type");
        };
    }

    /**
     * Validates metadata and creates the canonical dense descriptor before sampling or carrier
     * allocation.
     *
     * @param shape requested logical shape
     * @param dataType requested result type
     * @param mean requested binary64 mean
     * @param standardDeviation requested binary64 standard deviation
     * @param requiresGrad explicit gradient intent
     * @return a non-null validated dense-contiguous descriptor retaining {@code shape}
     * @throws IllegalArgumentException if the shape is dynamic, count exceeds the Java-array
     *     limit, type is non-floating, mean is non-finite, or deviation is non-finite or negative
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     */
    private static TensorDescriptor descriptor(
            Shape shape,
            DataType dataType,
            double mean,
            double standardDeviation,
            boolean requiresGrad) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "random tensor creation requires a fully static shape: " + shape);
        }
        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "random tensor element count exceeds Java array limit: required="
                            + elementCount
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "random normal creation requires floating data type: " + dataType);
        }
        if (!Double.isFinite(mean)) {
            throw new IllegalArgumentException("random normal mean must be finite: " + mean);
        }
        if (!Double.isFinite(standardDeviation) || standardDeviation < 0.0d) {
            throw new IllegalArgumentException(
                    "random normal standard deviation must be finite and non-negative: "
                            + standardDeviation);
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(
                dataType, shape, Optional.of(layout), requiresGrad);
    }

    /**
     * Samples transformed binary64 values into one source carrier and delegates once to matching
     * flat import.
     *
     * @param descriptor non-null validated dense FLOAT64 result descriptor
     * @param mean finite binary64 mean
     * @param standardDeviation finite numerically non-negative binary64 standard deviation
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one FLOAT64 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleFloat64(
            TensorDescriptor descriptor,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        double[] source = new double[length];
        for (int index = 0; index < length; index++) {
            double gaussian = randomGenerator.nextGaussian();
            source[index] = mean + gaussian * standardDeviation;
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples transformed binary64 values, narrows each once to binary32, and delegates one source
     * carrier to matching flat import.
     *
     * @param descriptor non-null validated dense FLOAT32 result descriptor
     * @param mean finite binary64 mean
     * @param standardDeviation finite numerically non-negative binary64 standard deviation
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one FLOAT32 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleFloat32(
            TensorDescriptor descriptor,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        float[] source = new float[length];
        for (int index = 0; index < length; index++) {
            double gaussian = randomGenerator.nextGaussian();
            double sample = mean + gaussian * standardDeviation;
            source[index] = (float) sample;
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples in binary64, narrows to binary32, converts to BFLOAT16 bits, and delegates once to
     * raw-BFLOAT16 flat import.
     *
     * @param descriptor non-null validated dense BFLOAT16 result descriptor
     * @param mean finite binary64 mean
     * @param standardDeviation finite numerically non-negative binary64 standard deviation
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one raw-BFLOAT16 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleBFloat16(
            TensorDescriptor descriptor,
            double mean,
            double standardDeviation,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        short[] source = new short[length];
        for (int index = 0; index < length; index++) {
            double gaussian = randomGenerator.nextGaussian();
            double sample = mean + gaussian * standardDeviation;
            source[index] = BFloat16Bits.fromFloat((float) sample);
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }
}
