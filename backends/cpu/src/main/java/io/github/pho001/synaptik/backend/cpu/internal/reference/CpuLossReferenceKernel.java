package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import java.util.Objects;

/**
 * Frozen scalar oracle for the CPU loss generated-code tests.
 *
 * <p>This owner is intentionally not reachable from CPU preparation or generated classes.  It
 * expresses the ordered scalar traversal against already-decoded values so generated-artifact
 * tests can compare numerical behavior without making a reference implementation a production
 * fallback.  The binary64 overloads use binary64 accumulator operations.  The binary32 overloads
 * deliberately keep every accumulator, contribution, and stable-log-sum-exp narrowing in
 * {@code float}, while calling {@link StrictMath} through its available binary64 signatures;
 * that is the same typed clean-Java oracle shape used by FLOAT32 and BFLOAT16 generated bodies.</p>
 */
public final class CpuLossReferenceKernel {
    /** Creates a stateless loss oracle. */
    public CpuLossReferenceKernel() { }

    /**
     * Evaluates exact-shape squared error in increasing logical element order.
     *
     * @param prediction non-null decoded prediction elements
     * @param target non-null decoded target elements of the same length
     * @param reduction non-null complete-domain reduction
     * @return a new non-null per-element loss array for {@code NONE}, otherwise a new one-element
     *     reduction array
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the arrays do not have equal length
     */
    public static double[] meanSquaredError(double[] prediction, double[] target,
            LossReduction reduction) {
        Objects.requireNonNull(prediction, "prediction");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (prediction.length != target.length) throw new IllegalArgumentException("MSE shape differs");
        if (reduction == LossReduction.NONE) {
            var result = new double[prediction.length];
            for (int element = 0; element < result.length; element++) {
                double difference = prediction[element] - target[element];
                result[element] = difference * difference;
            }
            return result;
        }
        double sum = 0.0d;
        for (int element = 0; element < prediction.length; element++) {
            double difference = prediction[element] - target[element];
            sum += difference * difference;
        }
        return new double[] {reduction == LossReduction.MEAN ? sum / prediction.length : sum};
    }

    /**
     * Evaluates exact-shape squared error with binary32 arithmetic in increasing logical order.
     *
     * @param prediction non-null decoded binary32 prediction elements
     * @param target non-null decoded binary32 target elements with the same length as
     *     {@code prediction}
     * @param reduction non-null complete-domain reduction
     * @return a new non-null per-element binary32 loss array for {@code NONE}, or a new
     *     one-element binary32 reduction array otherwise
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the arrays have different lengths
     */
    public static float[] meanSquaredError(float[] prediction, float[] target,
            LossReduction reduction) {
        Objects.requireNonNull(prediction, "prediction");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (prediction.length != target.length) throw new IllegalArgumentException("MSE shape differs");
        if (reduction == LossReduction.NONE) {
            var result = new float[prediction.length];
            for (int element = 0; element < result.length; element++) {
                float difference = prediction[element] - target[element];
                result[element] = difference * difference;
            }
            return result;
        }
        float sum = 0.0f;
        for (int element = 0; element < prediction.length; element++) {
            float difference = prediction[element] - target[element];
            sum += difference * difference;
        }
        return new float[] {reduction == LossReduction.MEAN ? sum / prediction.length : sum};
    }

    /**
     * Evaluates dense categorical cross entropy in sample-major, then class-major order.
     *
     * @param logits non-null decoded contiguous sample-by-class logits
     * @param target non-null decoded contiguous sample-by-class targets
     * @param samples non-negative sample count
     * @param classes non-negative static class extent
     * @param reduction non-null complete sample-domain reduction
     * @return one loss per sample for {@code NONE}, otherwise one reduced loss
     * @throws NullPointerException if an array or {@code reduction} is {@code null}
     * @throws IllegalArgumentException if the dimensions do not match either input
     */
    public static double[] denseCategoricalCrossEntropy(double[] logits, double[] target,
            int samples, int classes, LossReduction reduction) {
        checkSlices(logits, target, samples, classes, reduction);
        if (reduction == LossReduction.NONE) {
            var result = new double[samples];
            for (int sample = 0; sample < samples; sample++) result[sample] = denseSlice(
                    logits, target, sample * classes, classes);
            return result;
        }
        double sum = 0.0d;
        for (int sample = 0; sample < samples; sample++) sum += denseSlice(logits, target,
                sample * classes, classes);
        return new double[] {reduction == LossReduction.MEAN ? sum / samples : sum};
    }

    /**
     * Evaluates dense categorical cross entropy with binary32 stable-log-sum-exp arithmetic.
     *
     * @param logits non-null contiguous decoded binary32 sample-by-class logits
     * @param target non-null contiguous decoded binary32 sample-by-class target weights
     * @param samples non-negative number of non-class samples
     * @param classes non-negative static class extent
     * @param reduction non-null complete sample-domain reduction
     * @return a new non-null binary32 loss array with one value per sample for {@code NONE}, or
     *     one value otherwise
     * @throws NullPointerException if an array or {@code reduction} is {@code null}
     * @throws IllegalArgumentException if dimensions do not match either input
     */
    public static float[] denseCategoricalCrossEntropy(float[] logits, float[] target,
            int samples, int classes, LossReduction reduction) {
        checkSlices(logits, target, samples, classes, reduction);
        if (reduction == LossReduction.NONE) {
            var result = new float[samples];
            for (int sample = 0; sample < samples; sample++) result[sample] = denseSlice(
                    logits, target, sample * classes, classes);
            return result;
        }
        float sum = 0.0f;
        for (int sample = 0; sample < samples; sample++) sum += denseSlice(logits, target,
                sample * classes, classes);
        return new float[] {reduction == LossReduction.MEAN ? sum / samples : sum};
    }

    /**
     * Evaluates index categorical cross entropy, checking an ignore value before target bounds.
     *
     * @param logits non-null decoded contiguous sample-by-class logits
     * @param target non-null class index per sample
     * @param classes non-negative static class extent
     * @param ignoreIndex nullable exact ignored value
     * @param reduction non-null complete sample-domain reduction
     * @return one sample loss for {@code NONE}, otherwise one reduced loss
     * @throws NullPointerException if {@code logits}, {@code target}, or {@code reduction} is
     *     {@code null}
     * @throws IllegalArgumentException if dimensions disagree or a non-ignored index is outside
     *     the class extent
     */
    public static double[] indexCategoricalCrossEntropy(double[] logits, long[] target,
            int classes, Long ignoreIndex, LossReduction reduction) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (classes < 0 || logits.length != Math.multiplyExact(target.length, classes))
            throw new IllegalArgumentException("index loss dimensions differ");
        if (reduction == LossReduction.NONE) {
            var result = new double[target.length];
            for (int sample = 0; sample < target.length; sample++) result[sample] = indexSlice(
                    logits, sample * classes, classes, target[sample], ignoreIndex);
            return result;
        }
        double sum = 0.0d; long count = 0;
        for (int sample = 0; sample < target.length; sample++) {
            long value = target[sample];
            if (ignoreIndex != null && value == ignoreIndex) continue;
            sum += indexSlice(logits, sample * classes, classes, value, null);
            count++;
        }
        return new double[] {reduction == LossReduction.MEAN ? sum / count : sum};
    }

    /**
     * Evaluates index categorical cross entropy with binary32 stable-log-sum-exp arithmetic.
     *
     * @param logits non-null contiguous decoded binary32 sample-by-class logits
     * @param target non-null exact class index per sample
     * @param classes non-negative static class extent
     * @param ignoreIndex nullable exact ignored index, tested before bounds or logits evaluation
     * @param reduction non-null complete sample-domain reduction
     * @return a new non-null binary32 loss array with one value per sample for {@code NONE}, or
     *     one value otherwise
     * @throws NullPointerException if {@code logits}, {@code target}, or {@code reduction} is
     *     {@code null}
     * @throws IllegalArgumentException if dimensions disagree or a non-ignored index is outside
     *     the class extent
     */
    public static float[] indexCategoricalCrossEntropy(float[] logits, long[] target,
            int classes, Long ignoreIndex, LossReduction reduction) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (classes < 0 || logits.length != Math.multiplyExact(target.length, classes))
            throw new IllegalArgumentException("index loss dimensions differ");
        if (reduction == LossReduction.NONE) {
            var result = new float[target.length];
            for (int sample = 0; sample < target.length; sample++) result[sample] = indexSlice(
                    logits, sample * classes, classes, target[sample], ignoreIndex);
            return result;
        }
        float sum = 0.0f;
        long count = 0;
        for (int sample = 0; sample < target.length; sample++) {
            long value = target[sample];
            if (ignoreIndex != null && value == ignoreIndex) continue;
            sum += indexSlice(logits, sample * classes, classes, value, null);
            count++;
        }
        return new float[] {reduction == LossReduction.MEAN ? sum / count : sum};
    }

    private static void checkSlices(double[] logits, double[] target, int samples, int classes,
            LossReduction reduction) {
        Objects.requireNonNull(logits, "logits"); Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (samples < 0 || classes < 0 || logits.length != Math.multiplyExact(samples, classes)
                || target.length != logits.length) throw new IllegalArgumentException("dense loss dimensions differ");
    }

    private static double denseSlice(double[] logits, double[] target, int base, int classes) {
        double lse = logSumExp(logits, base, classes); double loss = 0.0d;
        for (int clazz = 0; clazz < classes; clazz++) {
            double weight = target[base + clazz];
            if (weight != 0.0d) loss += weight * (lse - logits[base + clazz]);
        }
        return loss;
    }

    private static double indexSlice(double[] logits, int base, int classes, long target,
            Long ignoreIndex) {
        if (ignoreIndex != null && target == ignoreIndex) return 0.0d;
        if (target < 0 || target >= classes) throw new IllegalArgumentException("index target out of bounds");
        return logSumExp(logits, base, classes) - logits[base + (int) target];
    }

    private static float denseSlice(float[] logits, float[] target, int base, int classes) {
        float lse = logSumExp(logits, base, classes);
        float loss = 0.0f;
        for (int clazz = 0; clazz < classes; clazz++) {
            float weight = target[base + clazz];
            if (weight != 0.0f) loss += weight * (lse - logits[base + clazz]);
        }
        return loss;
    }

    private static float indexSlice(float[] logits, int base, int classes, long target,
            Long ignoreIndex) {
        if (ignoreIndex != null && target == ignoreIndex) return 0.0f;
        if (target < 0 || target >= classes) throw new IllegalArgumentException("index target out of bounds");
        return logSumExp(logits, base, classes) - logits[base + (int) target];
    }

    private static double logSumExp(double[] values, int base, int classes) {
        double maximum = Double.NEGATIVE_INFINITY;
        for (int clazz = 0; clazz < classes; clazz++) maximum = Math.max(maximum, values[base + clazz]);
        double sum = 0.0d;
        for (int clazz = 0; clazz < classes; clazz++) sum += StrictMath.exp(values[base + clazz] - maximum);
        return maximum + StrictMath.log(sum);
    }

    private static float logSumExp(float[] values, int base, int classes) {
        float maximum = Float.NEGATIVE_INFINITY;
        for (int clazz = 0; clazz < classes; clazz++) {
            float value = values[base + clazz];
            if (value > maximum) maximum = value;
        }
        float sum = 0.0f;
        for (int clazz = 0; clazz < classes; clazz++) {
            float shifted = values[base + clazz] - maximum;
            sum += (float) StrictMath.exp((double) shifted);
        }
        return maximum + (float) StrictMath.log((double) sum);
    }

    private static void checkSlices(float[] logits, float[] target, int samples, int classes,
            LossReduction reduction) {
        Objects.requireNonNull(logits, "logits");
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(reduction, "reduction");
        if (samples < 0 || classes < 0 || logits.length != Math.multiplyExact(samples, classes)
                || target.length != logits.length) throw new IllegalArgumentException("dense loss dimensions differ");
    }
}
