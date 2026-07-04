package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.Optional;
import java.util.random.RandomGenerator;

/**
 * Implements validated eager normal, bounded continuous uniform, bounded integral, and Bernoulli
 * sampling for {@link TensorFactory} without retaining random source or result state.
 *
 * <p>The helper accepts only fully static Java-array-sized shapes and the three floating data
 * types for floating distributions, plus exact INT32 and INT64 output selected by primitive
 * integral bounds, plus canonical BOOL output selected by Bernoulli probability. It constructs
 * canonical dense descriptors, samples directly into one matching primitive carrier, and
 * delegates exactly once to flat import. It never creates a Tensor or storage directly, allocates
 * an identifier, looks up or replaces a source, or synchronizes, seeds, resets, splits, or closes
 * the caller-owned generator. The sampling paths remain cohesive
 * package-private factory mechanics rather than a public random package, source abstraction, seed
 * API, or distribution enum: callers already supply the exact generator, and no independent
 * random-domain model type is needed.</p>
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
     * Validates a requested uniform tensor, samples one typed carrier, and imports it once.
     *
     * <p>Validation requires a fully static Java-array-sized shape, one of FLOAT64, FLOAT32, or
     * BFLOAT16, finite bounds, and {@code lowerBoundInclusive < upperBoundExclusive}. After
     * constructing a canonical dense descriptor, the selected typed loop invokes
     * {@link RandomGenerator#nextDouble(double, double)} exactly once per row-major element with
     * the unchanged bounds. Empty output makes no source call and scalar output makes one.</p>
     *
     * <p>A conforming source returns a binary64 sample in
     * {@code [lowerBoundInclusive, upperBoundExclusive)}. FLOAT64 stores that result directly;
     * FLOAT32 narrows once; BFLOAT16 narrows once to binary32 and then uses
     * {@link BFloat16Bits#fromFloat(float)}. Narrowing may produce a stored value equal to the
     * narrowed upper bound or a lower-rounded representable value. A non-conforming custom source
     * result is converted without post-validation.</p>
     *
     * <p>The generator and carrier are transient and never retained. The caller owns generator
     * configuration, seeding, advancement, and thread-safe access. Equivalent output requires an
     * equivalent implementation and initial state, identical arguments, and no interfering use;
     * no cross-algorithm, provider, Java-version, seed-expansion, or concurrent-use promise is
     * made. Pre-sampling failures consume no source call or identifier. A source exception leaves
     * preceding calls consumed but creates no destination or identifier. Delegated blank-label
     * failure and exhaustion occur only after all source calls and destination allocation, with
     * the flat-import identifier effects documented by {@link TensorFactory#randomUniform}.</p>
     *
     * @param shape non-null shape already checked by the public factory; must be fully static
     * @param dataType non-null requested type already checked by the public factory
     * @param lowerBoundInclusive requested finite inclusive binary64 lower bound
     * @param upperBoundExclusive requested finite exclusive binary64 upper bound
     * @param randomGenerator exact non-null transient caller-owned source
     * @param label non-null optional label delegated unchanged to flat import
     * @param requiresGrad explicit result gradient intent
     * @return the exact tensor returned by one matching flat-import call; never {@code null}
     * @throws IllegalArgumentException if shape, count, type, or bound validation fails, or
     *     delegated label validation rejects blank text
     * @throws ArithmeticException if checked logical-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if identifier space is exhausted after all samples are drawn
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    static Tensor randomUniform(
            Shape shape,
            DataType dataType,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            boolean requiresGrad) {
        TensorDescriptor descriptor = uniformDescriptor(
                shape,
                dataType,
                lowerBoundInclusive,
                upperBoundExclusive,
                requiresGrad);
        int length = Math.toIntExact(shape.knownElementCount().orElseThrow());
        return switch (dataType) {
            case FLOAT64 -> sampleUniformFloat64(
                    descriptor,
                    lowerBoundInclusive,
                    upperBoundExclusive,
                    randomGenerator,
                    label,
                    length);
            case FLOAT32 -> sampleUniformFloat32(
                    descriptor,
                    lowerBoundInclusive,
                    upperBoundExclusive,
                    randomGenerator,
                    label,
                    length);
            case BFLOAT16 -> sampleUniformBFloat16(
                    descriptor,
                    lowerBoundInclusive,
                    upperBoundExclusive,
                    randomGenerator,
                    label,
                    length);
            case INT32, INT64, BOOL -> throw new AssertionError("validated floating data type");
        };
    }

    /**
     * Validates a requested INT32 tensor, samples one exact carrier, and imports it once.
     *
     * <p>Validation requires a fully static Java-array-sized shape and strictly ordered bounds.
     * A canonical dense non-differentiable INT32 descriptor is completed before one {@code int[]}
     * is allocated. Each row-major element is the direct result of one bounded
     * {@link RandomGenerator#nextInt(int, int)} invocation with the unchanged bounds. No modulo,
     * unbounded draw, floating arithmetic, conversion, stream, batching, or post-validation is
     * used. Empty output makes no source call and scalar output makes one.</p>
     *
     * <p>The source and carrier are transient. Pre-sampling failures consume no source call or
     * identifier; source exceptions leave earlier calls consumed but create no destination or
     * identifier. Blank-label and exhaustion effects occur after all calls and are inherited from
     * the one matching flat import. No state is rolled back.</p>
     *
     * <p>Primitive bounds select INT32 directly, gradients are disabled, and no data-type,
     * gradient, default-bound, or full-domain option is inferred. Because the exclusive bound has
     * the same carrier as the result, the interval cannot include {@link Integer#MAX_VALUE} as an
     * emitted value. The caller owns source configuration, state, and safe access; reproducibility
     * is limited to equivalent source implementation/state and identical arguments without
     * interfering use.</p>
     *
     * @param shape non-null shape already checked by the public factory; must be fully static
     * @param originInclusive inclusive signed 32-bit lower bound
     * @param boundExclusive exclusive signed 32-bit upper bound, greater than the origin
     * @param randomGenerator exact non-null transient caller-owned source
     * @param label non-null optional label delegated unchanged to flat import
     * @return the exact tensor returned by one INT32 flat-import call; never {@code null}
     * @throws IllegalArgumentException if shape, count, or bound validation fails, or delegated
     *     label validation rejects blank text
     * @throws ArithmeticException if checked logical-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if identifier space is exhausted after all samples are drawn
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    static Tensor randomInt(
            Shape shape,
            int originInclusive,
            int boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label) {
        TensorDescriptor descriptor = integralDescriptor(shape, originInclusive, boundExclusive);
        int length = Math.toIntExact(shape.knownElementCount().orElseThrow());
        return sampleInt32(
                descriptor, originInclusive, boundExclusive, randomGenerator, label, length);
    }

    /**
     * Validates a requested INT64 tensor, samples one exact carrier, and imports it once.
     *
     * <p>This overload applies the INT32 helper contract to primitive {@code long} bounds. It
     * creates a canonical dense non-differentiable INT64 descriptor, calls
     * {@link RandomGenerator#nextLong(long, long)} exactly once per row-major element, stores each
     * direct result in one {@code long[]} carrier, and delegates once to matching flat import.
     * Neither source nor carrier is retained, and no unbounded draw, modulo, floating arithmetic,
     * conversion, alternate carrier, or post-validation is used.</p>
     *
     * <p>Primitive bounds select INT64 directly and gradients are disabled. The same-carrier
     * exclusive bound cannot express a value above {@link Long#MAX_VALUE}, so that maximum cannot
     * be emitted by this bounded API and no full-domain alternative is supplied. Source ownership,
     * safe access, bounded reproducibility, and failure effects match the INT32 entry.</p>
     *
     * @param shape non-null shape already checked by the public factory; must be fully static
     * @param originInclusive inclusive signed 64-bit lower bound
     * @param boundExclusive exclusive signed 64-bit upper bound, greater than the origin
     * @param randomGenerator exact non-null transient caller-owned source
     * @param label non-null optional label delegated unchanged to flat import
     * @return the exact tensor returned by one INT64 flat-import call; never {@code null}
     * @throws IllegalArgumentException if shape, count, or bound validation fails, or delegated
     *     label validation rejects blank text
     * @throws ArithmeticException if checked logical-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if identifier space is exhausted after all samples are drawn
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    static Tensor randomInt(
            Shape shape,
            long originInclusive,
            long boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label) {
        TensorDescriptor descriptor = integralDescriptor(shape, originInclusive, boundExclusive);
        int length = Math.toIntExact(shape.knownElementCount().orElseThrow());
        return sampleInt64(
                descriptor, originInclusive, boundExclusive, randomGenerator, label, length);
    }

    /**
     * Validates a requested Bernoulli tensor, samples one canonical BOOL carrier, and imports it
     * once.
     *
     * <p>The probability must be finite and in {@code [0.0, 1.0]}. Every row-major element calls
     * {@link RandomGenerator#nextDouble()} exactly once, including at either probability endpoint,
     * so source advancement does not depend on an endpoint optimization. A byte is one exactly
     * when the draw is strictly less than the probability and zero otherwise. The source and
     * carrier remain transient, and the result always has BOOL type with gradients disabled.</p>
     *
     * <p>The caller owns generator configuration, state, advancement, and safe thread access.
     * Equivalent output requires an equivalent implementation and initial state, identical
     * arguments, and no interfering use; no cross-algorithm, provider, Java-version,
     * seed-expansion, or concurrent-use guarantee is made. Pre-sampling failures consume no call
     * or identifier. A source exception preserves earlier calls but creates no destination or
     * identifier. Blank-label and exhaustion effects occur after all calls and destination
     * allocation through the one BOOL flat import, and no state is rolled back.</p>
     *
     * @param shape non-null shape already checked by the public factory; must be fully static
     * @param probability finite success probability in the closed interval {@code [0.0, 1.0]}
     * @param randomGenerator exact non-null transient caller-owned source
     * @param label non-null optional label delegated unchanged to BOOL flat import
     * @return the exact tensor returned by one BOOL flat-import call; never {@code null}
     * @throws IllegalArgumentException if shape, count, or probability validation fails, or
     *     delegated label validation rejects blank text
     * @throws ArithmeticException if checked logical-count or dense-layout arithmetic overflows
     * @throws IllegalStateException if identifier space is exhausted after all samples are drawn
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    static Tensor randomBernoulli(
            Shape shape,
            double probability,
            RandomGenerator randomGenerator,
            Optional<String> label) {
        TensorDescriptor descriptor = bernoulliDescriptor(shape, probability);
        int length = Math.toIntExact(shape.knownElementCount().orElseThrow());
        return sampleBernoulli(descriptor, probability, randomGenerator, label, length);
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
     * Validates uniform metadata and creates the canonical dense descriptor before sampling or
     * carrier allocation.
     *
     * <p>Checks run in shape, checked count, Java-array limit, floating type, finite lower bound,
     * finite upper bound, and strict bound order. The exact distribution-specific messages are
     * {@code uniform random tensor creation requires a fully static shape: <shape>},
     * {@code uniform random tensor element count exceeds Java array limit: required=<required>,
     * maximum=2147483647}, {@code uniform random creation requires floating data type:
     * <dataType>}, {@code uniform random lower bound must be finite: <lower>},
     * {@code uniform random upper bound must be finite: <upper>}, and
     * {@code uniform random lower bound must be less than upper bound: lower=<lower>,
     * upper=<upper>}. Descriptor construction then validates gradient eligibility.</p>
     *
     * @param shape requested logical shape
     * @param dataType requested result type
     * @param lowerBoundInclusive requested binary64 inclusive lower bound
     * @param upperBoundExclusive requested binary64 exclusive upper bound
     * @param requiresGrad explicit gradient intent
     * @return a non-null validated dense-contiguous descriptor retaining {@code shape}
     * @throws IllegalArgumentException if the shape is dynamic, count exceeds the Java-array
     *     limit, type is non-floating, either bound is non-finite, or bounds are not strictly
     *     ordered
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     */
    private static TensorDescriptor uniformDescriptor(
            Shape shape,
            DataType dataType,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            boolean requiresGrad) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "uniform random tensor creation requires a fully static shape: " + shape);
        }
        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "uniform random tensor element count exceeds Java array limit: required="
                            + elementCount
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "uniform random creation requires floating data type: " + dataType);
        }
        if (!Double.isFinite(lowerBoundInclusive)) {
            throw new IllegalArgumentException(
                    "uniform random lower bound must be finite: " + lowerBoundInclusive);
        }
        if (!Double.isFinite(upperBoundExclusive)) {
            throw new IllegalArgumentException(
                    "uniform random upper bound must be finite: " + upperBoundExclusive);
        }
        if (!(lowerBoundInclusive < upperBoundExclusive)) {
            throw new IllegalArgumentException(
                    "uniform random lower bound must be less than upper bound: lower="
                            + lowerBoundInclusive
                            + ", upper="
                            + upperBoundExclusive);
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(
                dataType, shape, Optional.of(layout), requiresGrad);
    }

    /**
     * Validates INT32 random metadata and creates the canonical non-differentiable descriptor.
     *
     * @param shape requested logical shape
     * @param originInclusive inclusive signed 32-bit lower bound
     * @param boundExclusive exclusive signed 32-bit upper bound
     * @return a non-null validated dense INT32 descriptor retaining {@code shape}
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java-array limit, or
     *     origin is not strictly less than bound
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     */
    private static TensorDescriptor integralDescriptor(
            Shape shape, int originInclusive, int boundExclusive) {
        validateIntegralShape(shape);
        if (originInclusive >= boundExclusive) {
            throw new IllegalArgumentException(
                    "integral random origin must be less than bound: origin="
                            + originInclusive
                            + ", bound="
                            + boundExclusive);
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(DataType.INT32, shape, Optional.of(layout), false);
    }

    /**
     * Validates INT64 random metadata and creates the canonical non-differentiable descriptor.
     *
     * @param shape requested logical shape
     * @param originInclusive inclusive signed 64-bit lower bound
     * @param boundExclusive exclusive signed 64-bit upper bound
     * @return a non-null validated dense INT64 descriptor retaining {@code shape}
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java-array limit, or
     *     origin is not strictly less than bound
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     */
    private static TensorDescriptor integralDescriptor(
            Shape shape, long originInclusive, long boundExclusive) {
        validateIntegralShape(shape);
        if (originInclusive >= boundExclusive) {
            throw new IllegalArgumentException(
                    "integral random origin must be less than bound: origin="
                            + originInclusive
                            + ", bound="
                            + boundExclusive);
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(DataType.INT64, shape, Optional.of(layout), false);
    }

    /**
     * Validates the shared integral shape and Java-array count boundary before bound validation.
     *
     * <p>Dynamic shape reports
     * {@code integral random tensor creation requires a fully static shape: <shape>}. A count
     * above the Java-array limit reports {@code integral random tensor element count exceeds Java
     * array limit: required=<required>, maximum=2147483647}. Checked count overflow propagates
     * before either message can be superseded by bound validation.</p>
     *
     * @param shape requested logical shape
     * @throws IllegalArgumentException if shape is dynamic or count exceeds the Java-array limit
     * @throws ArithmeticException if checked element-count arithmetic overflows
     */
    private static void validateIntegralShape(Shape shape) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "integral random tensor creation requires a fully static shape: " + shape);
        }
        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "integral random tensor element count exceeds Java array limit: required="
                            + elementCount
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
    }

    /**
     * Validates Bernoulli metadata and creates the canonical dense non-differentiable descriptor.
     *
     * <p>Validation checks fully static shape, checked logical count, the Java-array limit, and
     * finite closed-interval probability in that order. Descriptor creation then fixes BOOL type
     * and false gradient intent; callers cannot select a numeric type or request gradients because
     * Bernoulli creation is logical leaf data rather than numeric truthiness or conversion.</p>
     *
     * @param shape requested logical shape
     * @param probability requested binary64 success probability
     * @return a non-null validated dense BOOL descriptor retaining {@code shape}
     * @throws IllegalArgumentException if shape is dynamic, count exceeds the Java-array limit,
     *     or probability is non-finite or outside {@code [0.0, 1.0]}
     * @throws ArithmeticException if checked element-count or layout arithmetic overflows
     */
    private static TensorDescriptor bernoulliDescriptor(Shape shape, double probability) {
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "bernoulli random tensor creation requires a fully static shape: " + shape);
        }
        long elementCount = shape.knownElementCount().orElseThrow();
        if (elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "bernoulli random tensor element count exceeds Java array limit: required="
                            + elementCount
                            + ", maximum="
                            + Integer.MAX_VALUE);
        }
        if (!Double.isFinite(probability) || probability < 0.0d || probability > 1.0d) {
            throw new IllegalArgumentException(
                    "bernoulli probability must be finite and in [0.0, 1.0]: " + probability);
        }
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        return new TensorDescriptor(DataType.BOOL, shape, Optional.of(layout), false);
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

    /**
     * Samples bounded binary64 values into one source carrier and delegates once to matching flat
     * import.
     *
     * <p>Each loop iteration calls the exact supplied source's bounded binary64 method once with
     * the unchanged bounds. For a conforming source, every stored value is in the binary64
     * half-open interval. No sample is post-validated, and neither source nor carrier is retained.
     * The method performs one FLOAT64 flat import after all calls complete.</p>
     *
     * @param descriptor non-null validated dense FLOAT64 result descriptor
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one FLOAT64 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleUniformFloat64(
            TensorDescriptor descriptor,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        double[] source = new double[length];
        for (int index = 0; index < length; index++) {
            source[index] = randomGenerator.nextDouble(
                    lowerBoundInclusive, upperBoundExclusive);
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples bounded binary64 values, narrows each once to binary32, and delegates one source
     * carrier to matching flat import.
     *
     * <p>The half-open promise applies to each conforming binary64 source result. Binary32
     * narrowing may equal the narrowed upper bound or round downward; no clamping or resampling is
     * performed. Neither source nor carrier is retained, and one FLOAT32 flat import follows all
     * bounded calls.</p>
     *
     * @param descriptor non-null validated dense FLOAT32 result descriptor
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one FLOAT32 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleUniformFloat32(
            TensorDescriptor descriptor,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        float[] source = new float[length];
        for (int index = 0; index < length; index++) {
            source[index] = (float) randomGenerator.nextDouble(
                    lowerBoundInclusive, upperBoundExclusive);
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples bounded binary64 values, narrows to binary32, converts to BFLOAT16 bits, and
     * delegates once to raw-BFLOAT16 flat import.
     *
     * <p>The half-open promise applies before conversion. Each binary64 result narrows once to
     * binary32 and then passes to {@link BFloat16Bits#fromFloat(float)}, so the stored value may
     * equal the narrowed upper bound or round downward. No clamping, resampling, or post-validation
     * occurs. Neither source nor carrier is retained, and one raw-BFLOAT16 flat import follows all
     * bounded calls.</p>
     *
     * @param descriptor non-null validated dense BFLOAT16 result descriptor
     * @param lowerBoundInclusive finite inclusive binary64 lower bound
     * @param upperBoundExclusive finite exclusive binary64 upper bound
     * @param randomGenerator non-null transient caller-owned source; invoked exactly once per
     *     element and never retained
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact source-carrier length
     * @return the non-null tensor returned by the one raw-BFLOAT16 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; completed source calls
     *     and any delegated identifier allocation are not rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleUniformBFloat16(
            TensorDescriptor descriptor,
            double lowerBoundInclusive,
            double upperBoundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        short[] source = new short[length];
        for (int index = 0; index < length; index++) {
            source[index] = BFloat16Bits.fromFloat((float) randomGenerator.nextDouble(
                    lowerBoundInclusive, upperBoundExclusive));
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples one direct bounded INT32 result per element and imports the sole typed carrier.
     *
     * @param descriptor non-null validated dense INT32 result descriptor
     * @param originInclusive inclusive signed 32-bit lower bound
     * @param boundExclusive exclusive signed 32-bit upper bound
     * @param randomGenerator non-null transient caller-owned source; invoked once per element
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact carrier length
     * @return the non-null tensor returned by the one INT32 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; no state is rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleInt32(
            TensorDescriptor descriptor,
            int originInclusive,
            int boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        int[] source = new int[length];
        for (int index = 0; index < length; index++) {
            source[index] = randomGenerator.nextInt(originInclusive, boundExclusive);
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples one direct bounded INT64 result per element and imports the sole typed carrier.
     *
     * @param descriptor non-null validated dense INT64 result descriptor
     * @param originInclusive inclusive signed 64-bit lower bound
     * @param boundExclusive exclusive signed 64-bit upper bound
     * @param randomGenerator non-null transient caller-owned source; invoked once per element
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact carrier length
     * @return the non-null tensor returned by the one INT64 flat-import call
     * @throws RuntimeException if a source call or delegated import throws; no state is rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleInt64(
            TensorDescriptor descriptor,
            long originInclusive,
            long boundExclusive,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        long[] source = new long[length];
        for (int index = 0; index < length; index++) {
            source[index] = randomGenerator.nextLong(originInclusive, boundExclusive);
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }

    /**
     * Samples one unbounded binary64 draw per element into a canonical BOOL carrier and delegates
     * once to BOOL flat import.
     *
     * <p>The strict {@code draw < probability} comparison writes byte one for success and byte
     * zero otherwise. Calls are not skipped when probability is zero or one, so source advancement
     * remains independent of endpoint optimization. Custom non-conforming draws are compared
     * directly without post-validation. Neither the source nor the carrier is retained.</p>
     *
     * @param descriptor non-null validated dense BOOL result descriptor
     * @param probability finite success probability in the closed interval {@code [0.0, 1.0]}
     * @param randomGenerator non-null transient caller-owned source; invoked once per element
     * @param label non-null optional label delegated unchanged to flat import
     * @param length non-negative logical element count and exact carrier length
     * @return the non-null tensor returned by the one BOOL flat-import call
     * @throws RuntimeException if a source call or delegated import throws; no state is rolled back
     * @throws OutOfMemoryError if source or destination carrier allocation fails
     */
    private static Tensor sampleBernoulli(
            TensorDescriptor descriptor,
            double probability,
            RandomGenerator randomGenerator,
            Optional<String> label,
            int length) {
        byte[] source = new byte[length];
        for (int index = 0; index < length; index++) {
            double draw = randomGenerator.nextDouble();
            source[index] = draw < probability ? (byte) 1 : (byte) 0;
        }
        return TensorFactory.fromFlatArray(descriptor, label, source);
    }
}
