package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * Stage-A numerical gate for the two categorical CPU loss forms in task 0008O.
 *
 * <p>The generated entry is the current CPU implementation.  {@code direct} below is a frozen,
 * typed clean-Java oracle with the same sample range, increasing logical sample/class traversal,
 * stable max/sum/log-sum-exp passes, and FLOAT32 narrowing points.  {@code vectorMap} is candidate
 * two only: it performs real preferred-species loads and an identity map, then folds lanes in
 * increasing logical order.  It is deliberately not a lane reduction, packing path, helper, or
 * production fallback.  Therefore raw-bit equality is required for every finite and IEEE special
 * control covered here.  Candidate three/four remain stopped by the permission table in
 * {@link CpuStableReductionVectorNumericalSpikeTest}; 0022A/0022B permit reassociation only
 * subject to later conformance tolerances.  Because no such tolerance is current, they do not
 * permit this exact-current raw-bit gate to advance candidates three or four.</p>
 */
final class CpuStableReductionLossStageATest {
    private enum Floating { F32, F64 }
    private enum Carrier { AA, SS, AS, SA }

    @Test void denseGeneratedDirectAndOrderedVectorMapMatchAllAxesWidthsCarriersAndRanges()
            throws Throwable {
        for (Floating floating : Floating.values()) for (Carrier carrier : Carrier.values()) {
            int lanes = lanes(floating);
            for (int classes : new int[] {1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3}) {
                for (int axis = 0; axis != 3; axis++) {
                    Shape shape = shape(axis, classes);
                    int samples = 4;
                    double[] logits = logits(samples * classes);
                    double[] target = denseTarget(samples * classes);
                    for (LossReduction reduction : LossReduction.values()) {
                        Object prediction = carrier(floating, segment(carrier, 0), logits);
                        Object weights = carrier(floating, segment(carrier, 1), target);
                        Object output = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE
                                ? samples : 1]);
                        var plan = densePlan(floating, carrier, shape, axis, reduction);
                        var generated = artifact(plan);
                        long start = reduction == LossReduction.NONE ? 1 : 0;
                        long end = reduction == LossReduction.NONE ? 3 : 1;
                        invoke(generated, plan, prediction, weights, output, start, end);
                        Object directOutput = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE ? samples : 1]);
                        Trace expected = directTyped(floating, prediction, weights, directOutput, axis, classes, samples,
                                reduction, null, start, end);
                        assertRaw(floating, expected.output(), values(floating, output), "dense/" + carrier
                                + "/axis=" + axis + "/classes=" + classes + "/" + reduction);
                        if (axis == 2) {
                            Object candidateOutput = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE ? samples : 1]);
                            Trace candidate = vectorMap(floating, prediction, weights, candidateOutput, axis, classes, samples,
                                    reduction, null, start, end);
                            assertCandidate(floating, expected, candidate, "dense candidate two", classes);
                            assertRaw(floating, values(floating, output), candidate.output(), "dense generated/candidate two");
                        }
                        else assertIndexedRejected(floating, prediction, weights, axis, classes, samples,
                                reduction, null, start, end);
                    }
                }
            }
        }
    }

    @Test void indexGeneratedDirectAndOrderedVectorMapMatchIgnoreSpecialsAndParallelRanges()
            throws Throwable {
        for (Floating floating : Floating.values()) for (Carrier carrier : Carrier.values()) {
            int lanes = lanes(floating);
            for (int classes : new int[] {1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3}) {
                for (int axis = 0; axis != 3; axis++) for (LossReduction reduction : LossReduction.values()) {
                    Shape shape = shape(axis, classes);
                    double[] logits = logits(4 * classes);
                    long[] indexes = {-1, 0, classes - 1L, 0};
                    Object prediction = carrier(floating, segment(carrier, 0), logits);
                    Object index = indexCarrier(segment(carrier, 1), indexes);
                    Object output = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE ? 4 : 1]);
                    var plan = indexPlan(floating, carrier, shape, axis, reduction, true);
                    var generated = artifact(plan);
                    long start = reduction == LossReduction.NONE ? 0 : 0;
                    long end = reduction == LossReduction.NONE ? 4 : 1;
                    invoke(generated, plan, prediction, index, output, start, end);
                    Object directOutput = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE ? 4 : 1]);
                    Trace expected = directTyped(floating, prediction, index, directOutput, axis, classes, 4, reduction, indexes,
                            start, end);
                    assertRaw(floating, expected.output(), values(floating, output), "index generated");
                    if (axis == 2) {
                        Object candidateOutput = carrier(floating, segment(carrier, 2), new double[reduction == LossReduction.NONE ? 4 : 1]);
                        Trace candidate = vectorMap(floating, prediction, index, candidateOutput, axis, classes, 4,
                                reduction, indexes, start, end);
                        assertCandidate(floating, expected, candidate, "index candidate two", classes);
                        assertRaw(floating, values(floating, output), candidate.output(), "index generated/candidate two");
                    }
                    else assertIndexedRejected(floating, prediction, index, axis, classes, 4,
                            reduction, indexes, start, end);
                    if (reduction == LossReduction.NONE) {
                        Object partitioned = carrier(floating, segment(carrier, 2), new double[4]);
                        invoke(generated, plan, prediction, index, partitioned, 0, 2);
                        invoke(generated, plan, prediction, index, partitioned, 2, 4);
                        assertRaw(floating, expected.output(), values(floating, partitioned),
                                "deterministic caller-owned complete samples");
                    }
                }
            }
        }
    }

    @Test void categoricalSpecialAndEmptyControlsKeepCurrentCpuClasses() {
        for (Floating floating : Floating.values()) {
            // Dense zero target skips 0 * NaN/Infinity contribution exactly as the emitter does.
            assertRaw(floating, new double[] {0}, direct(floating,
                    new double[] {Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY},
                    new double[] {0, -0.0}, 2, 2, 1, LossReduction.SUM, null, 0, 1).output(), "zero target");
            // Ignore is decided before logits access; all ignored MEAN retains current NaN (0/0).
            assertRaw(floating, new double[] {Double.NaN}, direct(floating, new double[0], null,
                    2, 0, 2, LossReduction.MEAN, new long[] {-1, -1}, 0, 1).output(), "all ignored");
            assertThrows(IllegalArgumentException.class, () -> direct(floating, new double[] {0}, null,
                    2, 1, 1, LossReduction.NONE, new long[] {1}, 0, 1));
            assertRaw(floating, new double[] {Double.NaN}, direct(floating,
                    new double[] {Double.NaN, 0}, null, 2, 2, 1, LossReduction.NONE,
                    new long[] {1}, 0, 1).output(), "NaN");
            assertRaw(floating, new double[] {Double.POSITIVE_INFINITY}, direct(floating,
                    new double[] {0, Double.NEGATIVE_INFINITY}, null, 2, 2, 1, LossReduction.NONE,
                    new long[] {1}, 0, 1).output(), "infinity");
        }
    }

    @Test void candidateTwoConsumesTheDeclaredTypedCarriersAndCannotBeAScalarStandIn() {
        for (Floating floating : Floating.values()) for (boolean segment : new boolean[] {false, true}) {
            int classes = lanes(floating) + 1;
            Object logitsCarrier = carrier(floating, segment, logits(classes));
            Object targetCarrier = carrier(floating, !segment, denseTarget(classes));
            // Deliberately change the bound carrier after fixture creation.  Candidate two receives
            // no fixture array, and this catches an accidental ignored carrier or scalar stand-in.
            put(floating, logitsCarrier, 0, 7d);
            put(floating, targetCarrier, 0, 1d);
            double[] boundLogits = values(floating, logitsCarrier);
            double[] boundTarget = values(floating, targetCarrier);
            Trace expected = direct(floating, boundLogits, boundTarget, 2, classes, 1,
                    LossReduction.NONE, null, 0, 1);
            Object candidateOutput = carrier(floating, !segment, new double[1]);
            Trace actual = vectorMap(floating, logitsCarrier, targetCarrier, candidateOutput, 2, classes, 1,
                    LossReduction.NONE, null, 0, 1);
            assertCandidate(floating, expected, actual, "carrier-consumption/" + floating + "/" + segment,
                    classes);
        }
    }

    /* Direct oracle: no fixture-shaped constants, helpers, allocation in its hot sample/class
       loops, or tolerance.  The arrays returned by the test harness are output ownership, not
       candidate packing. */
    private record Trace(double[] maximum, double[] shiftedSum, double[] logSumExp,
            double[] contribution, double[] output, int vectorBlocks) {}

    private static Trace direct(Floating f, double[] logits, double[] dense, int axis, int classes, int samples,
            LossReduction reduction, long[] indexes, long start, long end) {
        double[] out = new double[reduction == LossReduction.NONE ? samples : 1];
        double[] maximums = new double[samples], sums = new double[samples], lses = new double[samples];
        double[] contributions = new double[samples];
        double total = 0; long included = 0;
        long first = reduction == LossReduction.NONE ? start : 0;
        long last = reduction == LossReduction.NONE ? end : samples;
        for (long sample = first; sample < last; sample++) {
            long selected = indexes == null ? Long.MIN_VALUE : indexes[(int) sample];
            if (indexes != null && selected == -1) { if (reduction == LossReduction.NONE) out[(int) sample] = 0; continue; }
            if (indexes != null && (selected < 0 || selected >= classes)) throw new IllegalArgumentException("index");
            double maximum = f == Floating.F32 ? Float.NEGATIVE_INFINITY : Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) maximum = max(f, maximum, logits[address(axis, classes, (int) sample, c)]);
            double sum = zero(f);
            for (int c = 0; c < classes; c++) sum = add(f, sum, exp(f, sub(f, logits[address(axis, classes, (int) sample, c)], maximum)));
            double lse = add(f, maximum, log(f, sum));
            double loss = zero(f);
            if (indexes == null) for (int c = 0; c < classes; c++) {
                int at = address(axis, classes, (int) sample, c); double weight = cast(f, dense[at]);
                if (weight != 0) loss = add(f, loss, mul(f, weight, sub(f, lse, logits[at])));
            } else loss = sub(f, lse, logits[address(axis, classes, (int) sample, (int) selected)]);
            maximums[(int) sample] = maximum; sums[(int) sample] = sum; lses[(int) sample] = lse;
            contributions[(int) sample] = loss;
            if (reduction == LossReduction.NONE) out[(int) sample] = loss;
            else { total = add(f, total, loss); included++; }
        }
        if (reduction != LossReduction.NONE) out[0] = reduction == LossReduction.MEAN
                ? div(f, total, indexes == null ? samples : included) : total;
        return new Trace(maximums, sums, lses, contributions, out, 0);
    }

    /** Typed-carrier direct entry used by generated/direct/C2 comparisons; it writes the bound output carrier. */
    private static Trace directTyped(Floating f, Object logitsCarrier, Object targetCarrier, Object outputCarrier,
            int axis, int classes, int samples, LossReduction reduction, long[] indexes, long start, long end) {
        // This is intentionally not `values(...)` plus the array oracle: generated, direct, and
        // C2 must all consume the same bound array/segment carriers.  In particular, index
        // targets are read from their declared INT32 carrier rather than a fixture long[].
        double[] out = new double[reduction == LossReduction.NONE ? samples : 1];
        double[] maximums = new double[samples], sums = new double[samples], lses = new double[samples];
        double[] contributions = new double[samples];
        double total = zero(f); long included = 0;
        long first = reduction == LossReduction.NONE ? start : 0;
        long last = reduction == LossReduction.NONE ? end : samples;
        for (long sample = first; sample < last; sample++) {
            int selected = indexes == null ? Integer.MIN_VALUE : indexValue(targetCarrier, (int) sample);
            if (indexes != null && selected == -1) {
                if (reduction == LossReduction.NONE) out[(int) sample] = zero(f);
                continue;
            }
            if (indexes != null && (selected < 0 || selected >= classes)) {
                throw new IllegalArgumentException("index");
            }
            double maximum = f == Floating.F32 ? Float.NEGATIVE_INFINITY : Double.NEGATIVE_INFINITY;
            for (int c = 0; c < classes; c++) {
                maximum = max(f, maximum, value(f, logitsCarrier, address(axis, classes, (int) sample, c)));
            }
            double sum = zero(f);
            for (int c = 0; c < classes; c++) {
                sum = add(f, sum, exp(f, sub(f,
                        value(f, logitsCarrier, address(axis, classes, (int) sample, c)), maximum)));
            }
            double lse = add(f, maximum, log(f, sum));
            double loss = zero(f);
            if (indexes == null) {
                for (int c = 0; c < classes; c++) {
                    int at = address(axis, classes, (int) sample, c);
                    double weight = value(f, targetCarrier, at);
                    if (weight != 0) loss = add(f, loss, mul(f, weight,
                            sub(f, lse, value(f, logitsCarrier, at))));
                }
            } else {
                loss = sub(f, lse, value(f, logitsCarrier,
                        address(axis, classes, (int) sample, selected)));
            }
            maximums[(int) sample] = maximum;
            sums[(int) sample] = sum;
            lses[(int) sample] = lse;
            contributions[(int) sample] = loss;
            if (reduction == LossReduction.NONE) out[(int) sample] = loss;
            else { total = add(f, total, loss); included++; }
        }
        if (reduction != LossReduction.NONE) out[0] = reduction == LossReduction.MEAN
                ? div(f, total, indexes == null ? samples : included) : total;
        for (int i = 0; i < out.length; i++) put(f, outputCarrier, i, out[i]);
        return new Trace(maximums, sums, lses, contributions, values(f, outputCarrier), 0);
    }

    /** Candidate 2 has real preferred-species loads/map; scalar folds retain class order. */
    private static Trace vectorMap(Floating f, Object logits, Object denseOrIndex, Object output, int axis, int classes,
            int samples, LossReduction reduction, long[] indexes, long start, long end) {
        if (axis != 2) throw new IllegalArgumentException("no proved indexed Vector API access");
        return f == Floating.F32
                ? vectorMapF32(logits, denseOrIndex, output, classes, samples, reduction, indexes, start, end)
                : vectorMapF64(logits, denseOrIndex, output, classes, samples, reduction, indexes, start, end);
    }

    /* The carrier choices are decoded once, before the sample/class hot loops.  Each full class
       block is loaded and identity-mapped once for each pass; lane extraction then preserves the
       scalar increasing-class fold. */
    private static Trace vectorMapF32(Object logitsCarrier, Object targetCarrier, Object outputCarrier, int classes, int samples,
            LossReduction reduction, long[] indexes, long start, long end) {
        float[] logitsArray = logitsCarrier instanceof float[] a ? a : null;
        MemorySegment logitsSegment = logitsArray == null ? (MemorySegment) logitsCarrier : null;
        float[] denseArray = indexes == null && targetCarrier instanceof float[] a ? a : null;
        MemorySegment denseSegment = indexes == null && denseArray == null ? (MemorySegment) targetCarrier : null;
        int[] indexArray = indexes != null && targetCarrier instanceof int[] a ? a : null;
        MemorySegment indexSegment = indexes != null && indexArray == null ? (MemorySegment) targetCarrier : null;
        double[] maximums = new double[samples], sums = new double[samples], lses = new double[samples], contributions = new double[samples];
        double[] out = new double[reduction == LossReduction.NONE ? samples : 1];
        float total = 0f; long included = 0; int blocks = 0, lanes = FloatVector.SPECIES_PREFERRED.length();
        long first = reduction == LossReduction.NONE ? start : 0, last = reduction == LossReduction.NONE ? end : samples;
        for (long sample = first; sample < last; sample++) {
            int selected = indexes == null ? Integer.MIN_VALUE : indexArray != null ? indexArray[(int) sample]
                    : indexSegment.get(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, sample * 4);
            if (indexes != null && selected == -1) { if (reduction == LossReduction.NONE) out[(int) sample] = 0; continue; }
            if (indexes != null && (selected < 0 || selected >= classes)) throw new IllegalArgumentException("index");
            int base = (int) sample * classes, full = classes / lanes * lanes; float maximum = Float.NEGATIVE_INFINITY;
            for (int c = 0; c < full; c += lanes) { FloatVector block = (logitsArray != null ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, logitsArray, base + c) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 4, java.nio.ByteOrder.nativeOrder())).add(0f); blocks++; for (int lane = 0; lane < lanes; lane++) maximum = Math.max(maximum, block.lane(lane)); }
            for (int c = full; c < classes; c++) maximum = Math.max(maximum, logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, (long) (base + c) * 4));
            float sum = 0f;
            for (int c = 0; c < full; c += lanes) { FloatVector block = (logitsArray != null ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, logitsArray, base + c) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 4, java.nio.ByteOrder.nativeOrder())).add(0f); blocks++; for (int lane = 0; lane < lanes; lane++) sum += (float) StrictMath.exp(block.lane(lane) - maximum); }
            for (int c = full; c < classes; c++) { float value = logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, (long) (base + c) * 4); sum += (float) StrictMath.exp(value - maximum); }
            float lse = maximum + (float) StrictMath.log(sum), loss = 0f;
            if (indexes == null) {
                for (int c = 0; c < full; c += lanes) { FloatVector values = (logitsArray != null ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, logitsArray, base + c) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 4, java.nio.ByteOrder.nativeOrder())).add(0f); FloatVector weights = (denseArray != null ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, denseArray, base + c) : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, denseSegment, (long) (base + c) * 4, java.nio.ByteOrder.nativeOrder())).add(0f); blocks += 2; for (int lane = 0; lane < lanes; lane++) { float weight = weights.lane(lane); if (weight != 0f) loss += weight * (lse - values.lane(lane)); } }
                for (int c = full; c < classes; c++) { float value = logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, (long) (base + c) * 4); float weight = denseArray != null ? denseArray[base + c] : denseSegment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, (long) (base + c) * 4); if (weight != 0f) loss += weight * (lse - value); }
            } else { float value = logitsArray != null ? logitsArray[base + selected] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED, (long) (base + selected) * 4); loss = lse - value; }
            maximums[(int) sample] = maximum; sums[(int) sample] = sum; lses[(int) sample] = lse; contributions[(int) sample] = loss;
            if (reduction == LossReduction.NONE) out[(int) sample] = loss; else { total += loss; included++; }
        }
        if (reduction != LossReduction.NONE) out[0] = reduction == LossReduction.MEAN ? total / (indexes == null ? samples : included) : total;
        for (int i = 0; i < out.length; i++) put(Floating.F32, outputCarrier, i, out[i]);
        return new Trace(maximums, sums, lses, contributions, values(Floating.F32, outputCarrier), blocks);
    }

    private static Trace vectorMapF64(Object logitsCarrier, Object targetCarrier, Object outputCarrier, int classes, int samples,
            LossReduction reduction, long[] indexes, long start, long end) {
        double[] logitsArray = logitsCarrier instanceof double[] a ? a : null;
        MemorySegment logitsSegment = logitsArray == null ? (MemorySegment) logitsCarrier : null;
        double[] denseArray = indexes == null && targetCarrier instanceof double[] a ? a : null;
        MemorySegment denseSegment = indexes == null && denseArray == null ? (MemorySegment) targetCarrier : null;
        int[] indexArray = indexes != null && targetCarrier instanceof int[] a ? a : null;
        MemorySegment indexSegment = indexes != null && indexArray == null ? (MemorySegment) targetCarrier : null;
        double[] maximums = new double[samples], sums = new double[samples], lses = new double[samples], contributions = new double[samples];
        double[] out = new double[reduction == LossReduction.NONE ? samples : 1];
        double total = 0d; long included = 0; int blocks = 0, lanes = DoubleVector.SPECIES_PREFERRED.length();
        long first = reduction == LossReduction.NONE ? start : 0, last = reduction == LossReduction.NONE ? end : samples;
        for (long sample = first; sample < last; sample++) {
            int selected = indexes == null ? Integer.MIN_VALUE : indexArray != null ? indexArray[(int) sample]
                    : indexSegment.get(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED, sample * 4);
            if (indexes != null && selected == -1) { if (reduction == LossReduction.NONE) out[(int) sample] = 0d; continue; }
            if (indexes != null && (selected < 0 || selected >= classes)) throw new IllegalArgumentException("index");
            int base = (int) sample * classes, full = classes / lanes * lanes; double maximum = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < full; c += lanes) { DoubleVector block = (logitsArray != null ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, logitsArray, base + c) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 8, java.nio.ByteOrder.nativeOrder())).add(0d); blocks++; for (int lane = 0; lane < lanes; lane++) maximum = Math.max(maximum, block.lane(lane)); }
            for (int c = full; c < classes; c++) maximum = Math.max(maximum, logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) (base + c) * 8));
            double sum = 0d;
            for (int c = 0; c < full; c += lanes) { DoubleVector block = (logitsArray != null ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, logitsArray, base + c) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 8, java.nio.ByteOrder.nativeOrder())).add(0d); blocks++; for (int lane = 0; lane < lanes; lane++) sum += StrictMath.exp(block.lane(lane) - maximum); }
            for (int c = full; c < classes; c++) { double value = logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) (base + c) * 8); sum += StrictMath.exp(value - maximum); }
            double lse = maximum + StrictMath.log(sum), loss = 0d;
            if (indexes == null) {
                for (int c = 0; c < full; c += lanes) { DoubleVector values = (logitsArray != null ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, logitsArray, base + c) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, logitsSegment, (long) (base + c) * 8, java.nio.ByteOrder.nativeOrder())).add(0d); DoubleVector weights = (denseArray != null ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, denseArray, base + c) : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, denseSegment, (long) (base + c) * 8, java.nio.ByteOrder.nativeOrder())).add(0d); blocks += 2; for (int lane = 0; lane < lanes; lane++) { double weight = weights.lane(lane); if (weight != 0d) loss += weight * (lse - values.lane(lane)); } }
                for (int c = full; c < classes; c++) { double value = logitsArray != null ? logitsArray[base + c] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) (base + c) * 8); double weight = denseArray != null ? denseArray[base + c] : denseSegment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) (base + c) * 8); if (weight != 0d) loss += weight * (lse - value); }
            } else { double value = logitsArray != null ? logitsArray[base + selected] : logitsSegment.get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) (base + selected) * 8); loss = lse - value; }
            maximums[(int) sample] = maximum; sums[(int) sample] = sum; lses[(int) sample] = lse; contributions[(int) sample] = loss;
            if (reduction == LossReduction.NONE) out[(int) sample] = loss; else { total += loss; included++; }
        }
        if (reduction != LossReduction.NONE) out[0] = reduction == LossReduction.MEAN ? total / (indexes == null ? samples : included) : total;
        for (int i = 0; i < out.length; i++) put(Floating.F64, outputCarrier, i, out[i]);
        return new Trace(maximums, sums, lses, contributions, values(Floating.F64, outputCarrier), blocks);
    }

    private static void assertCandidate(Floating f, Trace expected, Trace actual, String name, int classes) {
        assertRaw(f, expected.maximum(), actual.maximum(), name + "/max");
        assertRaw(f, expected.shiftedSum(), actual.shiftedSum(), name + "/exp-sum");
        assertRaw(f, expected.logSumExp(), actual.logSumExp(), name + "/log-sum-exp");
        assertRaw(f, expected.contribution(), actual.contribution(), name + "/contribution");
        assertRaw(f, expected.output(), actual.output(), name + "/final");
        if (classes >= lanes(f)) assertTrue(actual.vectorBlocks() > 0,
                name + " must execute preferred-species block loads rather than a scalar stand-in");
    }
    private static void assertIndexedRejected(Floating f, Object logits, Object target, int axis,
            int classes, int samples, LossReduction reduction, long[] indexes, long start, long end) {
        assertThrows(IllegalArgumentException.class, () -> vectorMap(f, logits, target, carrier(f, false, new double[reduction == LossReduction.NONE ? samples : 1]), axis, classes,
                samples, reduction, indexes, start, end), "indexed class access is an explicit scalar-only control");
    }

    private static double[] values(Floating f, Object carrier) {
        int length = carrier instanceof float[] a ? a.length : carrier instanceof double[] a ? a.length
                : (int) (f == Floating.F32 ? ((MemorySegment) carrier).byteSize() / 4
                        : ((MemorySegment) carrier).byteSize() / 8);
        double[] out = new double[length]; int lanes = lanes(f), full = length / lanes * lanes;
        for (int base = 0; base < full; base += lanes) {
            if (f == Floating.F32) { FloatVector v = carrier instanceof float[] a
                    ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, a, base)
                    : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, (MemorySegment) carrier,
                            (long) base * 4, java.nio.ByteOrder.nativeOrder());
                for (int lane = 0; lane < lanes; lane++) out[base + lane] = v.lane(lane);
            } else { DoubleVector v = carrier instanceof double[] a
                    ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, a, base)
                    : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, (MemorySegment) carrier,
                            (long) base * 8, java.nio.ByteOrder.nativeOrder());
                for (int lane = 0; lane < lanes; lane++) out[base + lane] = v.lane(lane); }
        }
        for (int i = full; i < length; i++) out[i] = f == Floating.F32 ? carrier instanceof float[] a ? a[i]
                : ((MemorySegment) carrier).get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (long) i * 4) : carrier instanceof double[] a ? a[i]
                : ((MemorySegment) carrier).get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED, (long) i * 8);
        return out;
    }

    /** One represented FLOAT32/FLOAT64 carrier read for the direct typed oracle. */
    private static double value(Floating f, Object carrier, int index) {
        if (f == Floating.F32) return carrier instanceof float[] array ? array[index]
                : ((MemorySegment) carrier).get(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                        (long) index * Float.BYTES);
        return carrier instanceof double[] array ? array[index]
                : ((MemorySegment) carrier).get(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED,
                        (long) index * Double.BYTES);
    }

    /** Reads an index target from the declared array or segment carrier. */
    private static int indexValue(Object carrier, int index) {
        return carrier instanceof int[] array ? array[index]
                : ((MemorySegment) carrier).get(java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED,
                        (long) index * Integer.BYTES);
    }

    private static Object carrier(Floating f, boolean segment, double[] v) {
        if (f == Floating.F32) { float[] a = new float[v.length]; for (int i = 0; i < a.length; i++) a[i] = (float) v[i];
            return segment ? MemorySegment.ofArray(a) : a; }
        return segment ? MemorySegment.ofArray(v.clone()) : v.clone();
    }
    private static void put(Floating f, Object carrier, int index, double value) {
        if (f == Floating.F32) {
            if (carrier instanceof float[] array) array[index] = (float) value;
            else ((MemorySegment) carrier).set(java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED,
                    (long) index * 4, (float) value);
        } else if (carrier instanceof double[] array) array[index] = value;
        else ((MemorySegment) carrier).set(java.lang.foreign.ValueLayout.JAVA_DOUBLE_UNALIGNED,
                (long) index * 8, value);
    }
    private static Object indexCarrier(boolean segment, long[] v) { int[] a = new int[v.length]; for (int i=0;i<a.length;i++) a[i]=(int)v[i]; return segment ? MemorySegment.ofArray(a) : a; }
    private static boolean segment(Carrier c, int role) { return switch (c) { case AA -> false; case SS -> true; case AS -> role != 0; case SA -> role == 0; }; }
    private static int lanes(Floating f) { return f == Floating.F32 ? FloatVector.SPECIES_PREFERRED.length() : DoubleVector.SPECIES_PREFERRED.length(); }
    private static Shape shape(int axis, int classes) { return axis == 0 ? Shape.of(classes, 2, 2) : axis == 1 ? Shape.of(2, classes, 2) : Shape.of(2, 2, classes); }
    private static int address(int axis, int classes, int sample, int clazz) { return axis == 0 ? clazz * 4 + sample : axis == 1 ? (sample / 2) * classes * 2 + clazz * 2 + sample % 2 : sample * classes + clazz; }
    private static double[] logits(int n) { double[] v=new double[n]; double[] s={-0.0,0.0,Double.MIN_VALUE,-Double.MIN_VALUE,1000,-1000,Double.MAX_VALUE,-Double.MAX_VALUE,1,-1}; for(int i=0;i<n;i++)v[i]=s[i%s.length]; return v; }
    private static double[] denseTarget(int n) { double[] v=new double[n]; for(int i=0;i<n;i++)v[i]=(i%3==0)?0:(i%3==1?1:-0.5); return v; }
    private static double cast(Floating f,double x){return f==Floating.F32?(float)x:x;} private static double zero(Floating f){return f==Floating.F32?0f:0d;} private static double max(Floating f,double a,double b){return f==Floating.F32?Math.max((float)a,(float)b):Math.max(a,b);} private static double add(Floating f,double a,double b){return f==Floating.F32?(float)a+(float)b:a+b;} private static double sub(Floating f,double a,double b){return f==Floating.F32?(float)a-(float)b:a-b;} private static double mul(Floating f,double a,double b){return f==Floating.F32?(float)a*(float)b:a*b;} private static double div(Floating f,double a,long b){return f==Floating.F32?(float)a/b:a/b;} private static double exp(Floating f,double x){return f==Floating.F32?(float)StrictMath.exp((float)x):StrictMath.exp(x);} private static double log(Floating f,double x){return f==Floating.F32?(float)StrictMath.log((float)x):StrictMath.log(x);}
    private static void assertRaw(Floating f,double[] expected,double[] actual,String name){assertEquals(expected.length,actual.length,name);for(int i=0;i<expected.length;i++)assertEquals(f==Floating.F32?Float.floatToRawIntBits((float)expected[i]):Double.doubleToRawLongBits(expected[i]),f==Floating.F32?Float.floatToRawIntBits((float)actual[i]):Double.doubleToRawLongBits(actual[i]),name+"/"+i);}
    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan densePlan(Floating f,Carrier c,Shape s,int axis,LossReduction r){DataType d=f==Floating.F32?DataType.FLOAT32:DataType.FLOAT64; return plan(new Operation(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,new DenseCategoricalCrossEntropyWithLogitsAttrs(axis,r)),List.of(CpuScatterLoweringTest.desc(d,s),CpuScatterLoweringTest.desc(d,s)),CpuScatterLoweringTest.desc(d,r==LossReduction.NONE?Shape.of(2,2):Shape.scalar()),carriers(f,c,false));}
    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan indexPlan(Floating f,Carrier c,Shape s,int axis,LossReduction r,boolean ignore){DataType d=f==Floating.F32?DataType.FLOAT32:DataType.FLOAT64; Shape t=Shape.of(2,2); return plan(new Operation(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,new IndexCategoricalCrossEntropyWithLogitsAttrs(axis,r,ignore?Optional.of(ScalarValue.int32(-1)):Optional.empty())),List.of(CpuScatterLoweringTest.desc(d,s),CpuScatterLoweringTest.desc(DataType.INT32,t)),CpuScatterLoweringTest.desc(d,r==LossReduction.NONE?t:Shape.scalar()),carriers(f,c,true));}
    private static List<CarrierAccess> carriers(Floating f,Carrier c,boolean index){CarrierAccess x=f==Floating.F32?CarrierAccess.FLOAT_ARRAY:CarrierAccess.DOUBLE_ARRAY; CarrierAccess xs=CarrierAccess.MEMORY_SEGMENT; CarrierAccess y=index?CarrierAccess.INT_ARRAY:x; CarrierAccess ys=index?CarrierAccess.MEMORY_SEGMENT:xs; return switch(c){case AA->List.of(x,y,x);case SS->List.of(xs,ys,xs);case AS->List.of(x,ys,xs);case SA->List.of(xs,y,x);};}
    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan plan(Operation op,List<TensorDescriptor> in,TensorDescriptor out,List<CarrierAccess> c){PrepareContext<CpuPartitionAnalysisInputs>b=CpuScatterLoweringTest.context(op,List.of(0,1),in,out);return new CpuPartitionPreparer().analyze(new PrepareContext<>(b.partition(),b.nodes(),b.values(),b.memoryRequirements(),Map.of(),new CpuPartitionAnalysisInputs(false,c))).plan();}
    private static CpuGeneratedKernel artifact(io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan p){var r=p.units().getFirst().portablePlan();var g=new CpuClassFileKernelGenerator();return g.defineClassBytes(r.specialization(),g.generateClassBytes(r.specialization(),r.kernelIr()));}
    private static void invoke(CpuGeneratedKernel a,io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan p,Object x,Object y,Object z,long s,long e)throws Throwable{CpuLossIr l=(CpuLossIr)p.units().getFirst().portablePlan().portableKernelIr();a.entryPoint().invokeWithArguments(x,y,z,l.geometry().pack(new long[]{0,0,0}),s,e);}
}
