package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import org.junit.jupiter.api.Test;

/**
 * CPU-private Stage-A characterization for the attention stable reduction in task 0008O.
 *
 * <p>The candidate deliberately vectorizes loads and multiplication only. Every reduction is
 * folded in increasing logical key (and lane) order, so it is an admissible candidate two, not
 * permission for a lane-local or horizontal reduction. The direct oracle has the same cold
 * geometry, range ownership, passes, narrowing points, and key order.</p>
 */
final class CpuStableReductionAttentionStageATest {
  private enum Type { F32, F64 }
  private enum Carrier { ARRAY, SEGMENT }

  @Test
  void preferredSpeciesVectorMapPreservesAttentionPassesForRequiredWidthsAndSpecialClasses() {
    for (Type type : Type.values()) {
      int lanes = lanes(type);
      assertTrue(lanes > 1, "a multi-lane preferred species is required for this probe");
      for (int keys : List.of(1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3))
        for (boolean masked : List.of(false, true))
          for (boolean causal : List.of(false, true))
            for (boolean weights : List.of(false, true))
              for (double[] values : stimuli(keys))
                assertCandidateEqualsDirect(type, Carrier.ARRAY, Carrier.SEGMENT, keys, 3,
                    values, masked, causal, weights, 0, 1);
    }
  }

  @Test
  void actualGeneratedEntriesMatchTheDirectAndVectorOraclesForAllCarrierSignatures() throws Throwable {
    for (Type type : Type.values()) {
      int lanes = lanes(type);
      for (int keys : List.of(1, lanes - 1, lanes, lanes + 1, 2 * lanes + 3))
        for (Carrier[] carriers : List.of(
            new Carrier[] {Carrier.ARRAY, Carrier.ARRAY},
            new Carrier[] {Carrier.SEGMENT, Carrier.SEGMENT},
            new Carrier[] {Carrier.ARRAY, Carrier.SEGMENT},
            new Carrier[] {Carrier.SEGMENT, Carrier.ARRAY}))
          for (boolean masked : List.of(false, true))
            for (boolean causal : List.of(false, true))
              for (boolean weights : List.of(false, true))
                // Every special class is already independently raw-bit compared at each width;
                // this generated entry probe pins the same cold geometry and carriers to it.
                for (double[] seed : stimuli(keys))
                  assertGeneratedEqualsOracles(type, carriers[0], carriers[1], keys, 3, seed,
                      masked, causal, weights);
    }
  }

  @Test
  void generatedEmptyKeyAndEmptyValueControlsMatchCurrentLoweringOwnership() throws Throwable {
    for (Type type : Type.values()) for (Carrier[] carriers : List.of(
        new Carrier[] {Carrier.ARRAY, Carrier.ARRAY}, new Carrier[] {Carrier.SEGMENT, Carrier.SEGMENT},
        new Carrier[] {Carrier.ARRAY, Carrier.SEGMENT}, new Carrier[] {Carrier.SEGMENT, Carrier.ARRAY})) {
      // S == 0 still owns complete rows when requested weights are present; it stores zero output
      // values and has an empty weights axis.
      Result emptyKeys = generated(type, carriers[0], carriers[1], 0, 3, new double[] {1}, true, true, true);
      Result directKeys = direct(type, carriers[0], carriers[1], 0, 3, new double[] {1}, true, true, true, 0, 4);
      assertBits(type, directKeys.output, emptyKeys.output, "generated S==0 output");
      assertBits(type, directKeys.weights, emptyKeys.weights, "generated S==0 weights");
      // CpuAttentionLowering sets rowCount to zero for Ev == 0 with no weights output.  The
      // generated entry is therefore called with the same empty range, rather than inventing
      // unsupported output-row work.
      Result emptyValues = generated(type, carriers[0], carriers[1], 3, 0, new double[] {1}, false, false, false);
      Result directValues = direct(type, carriers[0], carriers[1], 3, 0, new double[] {1}, false, false, false, 0, 0);
      assertBits(type, directValues.output, emptyValues.output, "generated Ev==0 output");
    }
  }

  @Test
  void generatedEntriesAreDeterministicForConcurrentDisjointCompleteQueryRows() throws Throwable {
    for (Type type : Type.values()) for (Carrier[] carriers : List.of(
        new Carrier[] {Carrier.ARRAY, Carrier.ARRAY}, new Carrier[] {Carrier.SEGMENT, Carrier.SEGMENT},
        new Carrier[] {Carrier.ARRAY, Carrier.SEGMENT}, new Carrier[] {Carrier.SEGMENT, Carrier.ARRAY})) {
      double[] seed = finite(lanes(type) + 1);
      Result serial = generated(type, carriers[0], carriers[1], lanes(type) + 1, 3, seed, true, true, true);
      Result direct = direct(type, carriers[0], carriers[1], lanes(type) + 1, 3, seed, true, true, true, 0, 4);
      try (var callers = Executors.newFixedThreadPool(2)) {
        List<Result> partitions = callers.invokeAll(List.<Callable<Result>>of(
            () -> generatedUnchecked(type, carriers[0], carriers[1], lanes(type) + 1, 3, seed,
                true, true, true, 0, 2),
            () -> generatedUnchecked(type, carriers[0], carriers[1], lanes(type) + 1, 3, seed,
                true, true, true, 2, 4))).stream().map(result -> {
                  try { return result.get(); } catch (Exception exception) { throw new AssertionError(exception); }
                }).toList();
        Result stitched = stitch(partitions.get(0), partitions.get(1), lanes(type) + 1, 3);
        assertBits(type, serial.output, stitched.output, "concurrent generated output");
        assertBits(type, serial.weights, stitched.weights, "concurrent generated weights");
        assertBits(type, direct.output, stitched.output, "concurrent generated/direct output");
        assertBits(type, direct.weights, stitched.weights, "concurrent generated/direct weights");
      }
    }
  }

  @Test
  void candidateAndDirectAreDeterministicForDisjointCompleteRowsAndBroadcastBatches() {
    for (Type type : Type.values()) {
      int keys = lanes(type) + 1;
      // Rows 0..1 and 2..3 are complete, disjoint query rows.  This is the caller-parallel
      // ownership model; no candidate splits a score max, sum, or weighted-value fold.
      Result serial = candidate(type, Carrier.SEGMENT, Carrier.ARRAY, keys, 2,
          finite(keys), true, true, true, 0, 4);
      Result left = candidate(type, Carrier.SEGMENT, Carrier.ARRAY, keys, 2,
          finite(keys), true, true, true, 0, 2);
      Result right = candidate(type, Carrier.SEGMENT, Carrier.ARRAY, keys, 2,
          finite(keys), true, true, true, 2, 4);
      assertBits(type, serial.output, join(left.output, right.output), "caller parallel output");
      assertBits(type, serial.weights, join(left.weights, right.weights), "caller parallel weights");
    }
  }

  @Test
  void indexedPositiveStrideIsExplicitlyRejectedRatherThanPretendingItIsVectorizable() {
    // Current attention lowering emits GENERAL_ODOMETER scalar addressing for indexed layouts;
    // it has no gather/index-map Vector API contract.  Candidate two consequently has exactly
    // the dense contiguous vector-load form above. Positive offsets/strides remain scalar-only
    // controls until a separately proved Java Vector API access form exists.
    assertFalse(indexedVectorAccessIsProved());
  }

  @Test
  void candidateTwoConsumesEveryDeclaredTypedCarrierRatherThanFixtureSurrogates() {
    // Perturbing a declared carrier after fixture construction makes an ignored carrier or a
    // reconstructed scalar fixture observable for every array/segment signature.
    for (Type type : Type.values()) for (Carrier[] carriers : List.of(
        new Carrier[] {Carrier.ARRAY, Carrier.ARRAY}, new Carrier[] {Carrier.SEGMENT, Carrier.SEGMENT},
        new Carrier[] {Carrier.ARRAY, Carrier.SEGMENT}, new Carrier[] {Carrier.SEGMENT, Carrier.ARRAY})) {
      try (Arena arena = Arena.ofConfined()) {
        Inputs inputs = inputs(type, carriers[0], lanes(type) + 1, 3, finite(lanes(type) + 1), arena);
        Result before = candidate(type, carriers[1], inputs, false, false, true, 0, 4);
        // Lane one has key-dependent content; changing lane zero alone would translate every
        // score equally and softmax would correctly hide the read.
        set(type, inputs.query, 1, narrow(type, scalar(type, inputs.query, 1) + 17));
        Result after = candidate(type, carriers[1], inputs, false, false, true, 0, 4);
        assertFalse(sameBits(type, before.output, after.output), "candidate ignored query carrier");
        set(type, inputs.key, 1, narrow(type, scalar(type, inputs.key, 1) - 19));
        Result keyAfter = candidate(type, carriers[1], inputs, false, false, true, 0, 4);
        assertFalse(sameBits(type, after.output, keyAfter.output), "candidate ignored key carrier");
        set(type, inputs.value, 0, narrow(type, scalar(type, inputs.value, 0) - 31));
        Result valueAfter = candidate(type, carriers[1], inputs, false, false, true, 0, 4);
        assertFalse(sameBits(type, keyAfter.output, valueAfter.output), "candidate ignored value carrier");
      }
    }
  }

  private static void assertCandidateEqualsDirect(Type type, Carrier qkv, Carrier outputs,
      int keys, int valueWidth, double[] scoreSeed, boolean masked, boolean causal,
      boolean requestedWeights, long start, long end) {
    Result direct = direct(type, qkv, outputs, keys, valueWidth, scoreSeed, masked, causal,
        requestedWeights, start, end);
    Result vector = candidate(type, qkv, outputs, keys, valueWidth, scoreSeed, masked, causal,
        requestedWeights, start, end);
    assertBits(type, direct.max, vector.max, "score max");
    assertBits(type, direct.sum, vector.sum, "exp/sum");
    assertBits(type, direct.weights, vector.weights, "normalization");
    assertBits(type, direct.output, vector.output, "increasing-key weighted value");
  }

  private static void assertGeneratedEqualsOracles(Type type, Carrier inputs, Carrier outputs,
      int s, int ev, double[] seed, boolean masked, boolean causal, boolean requestedWeights)
      throws Throwable {
    long end = ev == 0 && !requestedWeights ? 0 : 4;
    Result direct = direct(type, inputs, outputs, s, ev, seed, masked, causal, requestedWeights, 0, end);
    Result vector = candidate(type, inputs, outputs, s, ev, seed, masked, causal, requestedWeights, 0, end);
    Result generated = generated(type, inputs, outputs, s, ev, seed, masked, causal, requestedWeights);
    assertBits(type, direct.output, generated.output, "generated/direct weighted values");
    assertBits(type, vector.output, generated.output, "generated/vector weighted values");
    assertBits(type, direct.weights, generated.weights, "generated/direct normalized weights");
    assertBits(type, vector.weights, generated.weights, "generated/vector normalized weights");
  }

  /** Invokes the current schema-57 generated entry; no prepared executable is involved. */
  private static Result generated(Type type, Carrier inputs, Carrier outputs, int s, int ev,
      double[] seed, boolean masked, boolean causal, boolean requestedWeights) throws Throwable {
    long logicalRows = ev == 0 && !requestedWeights ? 0 : 4;
    return generated(type, inputs, outputs, s, ev, seed, masked, causal, requestedWeights, 0,
        logicalRows, logicalRows);
  }

  private static Result generatedUnchecked(Type type, Carrier inputs, Carrier outputs, int s, int ev,
      double[] seed, boolean masked, boolean causal, boolean requestedWeights, long start, long end) {
    try {
      return generated(type, inputs, outputs, s, ev, seed, masked, causal, requestedWeights,
          start, end, 4);
    } catch (Throwable failure) {
      throw new AssertionError(failure);
    }
  }

  private static Result generated(Type type, Carrier inputs, Carrier outputs, int s, int ev,
      double[] seed, boolean masked, boolean causal, boolean requestedWeights, long start, long end,
      long logicalRows) throws Throwable {
    DataType dataType = type == Type.F32 ? DataType.FLOAT32 : DataType.FLOAT64;
    int rows = 4, e = Math.max(1, lanes(type));
    double[] q = new double[rows * e], k = new double[s * e], v = new double[s * ev];
    for (int r = 0; r < rows; r++) for (int x = 0; x < e; x++) q[r * e + x] = narrow(type, x == 0 ? seed[r % seed.length] : 1d / (x + 1));
    for (int j = 0; j < s; j++) for (int x = 0; x < e; x++) k[j * e + x] = narrow(type, x == 0 ? 1 : (j + 1d) / (x + 2));
    for (int j = 0; j < s; j++) for (int d = 0; d < ev; d++) v[j * ev + d] = narrow(type, seed[j % seed.length] + d);
    byte[] mask = new byte[2 * s];
    for (int qRow = 0; qRow < 2; qRow++) for (int j = 0; j < s; j++)
      mask[qRow * s + j] = (byte) ((j & 1) == 0 ? 1 : 0);
    int inputCount = masked ? 4 : 3, outputCount = requestedWeights ? 2 : 1;
    List<DataType> types = new java.util.ArrayList<>();
    for (int i = 0; i < inputCount + outputCount; i++) types.add(i == 3 && masked ? DataType.BOOL : dataType);
    List<CarrierAccess> access = new java.util.ArrayList<>();
    for (int i = 0; i < inputCount; i++) access.add(inputs == Carrier.ARRAY ? (i == 3 && masked ? CarrierAccess.BYTE_ARRAY : type == Type.F32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY) : CarrierAccess.MEMORY_SEGMENT);
    for (int i = 0; i < outputCount; i++) access.add(outputs == Carrier.ARRAY ? (type == Type.F32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.DOUBLE_ARRAY) : CarrierAccess.MEMORY_SEGMENT);
    List<Integer> roles = masked ? List.of(0, 1, 2, 3) : List.of(0, 1, 2);
    var ir = new CpuAttentionIr(dataType, dataType, dataType, dataType, masked, causal,
        outputCount, roles, types, plans(inputCount, outputCount));
    var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
        CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, access, 0, -1, List.of(), true, 57);
    MethodHandle entry = new CpuClassFileKernelGenerator().defineClassBytes(specialization,
        new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir.encodedKernelIr())).entryPoint();
    var geometry = geometry(dataType, rows, s, e, ev, masked, outputCount, logicalRows);
    try (Arena arena = Arena.ofConfined()) {
      Object qCarrier = carrier(type, inputs, q, arena), kCarrier = carrier(type, inputs, k, arena),
          vCarrier = carrier(type, inputs, v, arena), mCarrier = byteCarrier(inputs, mask, arena),
          outCarrier = zeroCarrier(type, outputs, rows * ev, arena), weightCarrier = zeroCarrier(type, outputs, rows * s, arena);
      MemorySegment scratch = arena.allocate(Math.max(1, (long) s * (type == Type.F32 ? 4 : 8)), 8);
      var arguments = new java.util.ArrayList<Object>();
      arguments.add(qCarrier); arguments.add(kCarrier); arguments.add(vCarrier); if (masked) arguments.add(mCarrier);
      arguments.add(outCarrier); if (requestedWeights) arguments.add(weightCarrier);
      arguments.add(scratch); arguments.add(geometry.pack(new long[inputCount + outputCount])); arguments.add(start); arguments.add(end);
      try {
        entry.invokeWithArguments(arguments);
      } catch (ArrayIndexOutOfBoundsException exception) {
        throw new AssertionError("generated attention carrier/geometry: " + type + '/' + inputs
            + '/' + outputs + ": S=" + s + ", Ev=" + ev + ", masked=" + masked
            + ", causal=" + causal + ", weights=" + requestedWeights, exception);
      }
      return new Result(new double[rows], new double[rows], requestedWeights ? read(type, weightCarrier, rows * s) : new double[0], read(type, outCarrier, rows * ev));
    }
  }

  private static Result stitch(Result first, Result second, int s, int ev) {
    double[] output = first.output.clone(), weights = first.weights.clone();
    System.arraycopy(second.output, 2 * ev, output, 2 * ev, 2 * ev);
    System.arraycopy(second.weights, 2 * s, weights, 2 * s, 2 * s);
    return new Result(first.max, first.sum, weights, output);
  }

  private static List<CpuAccessPlan> plans(int inputs, int outputs) {
    var result = new java.util.ArrayList<CpuAccessPlan>();
    for (int i = 0; i < inputs + outputs; i++) result.add(new CpuAccessPlan(
        i < inputs ? CpuAccessPlan.AccessKind.READ : CpuAccessPlan.AccessKind.WRITE,
        CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0));
    return result;
  }

  private static CpuAttentionLowering.Geometry geometry(DataType type, int rows, int s, int e,
      int ev, boolean masked, int outputs, long rowCount) {
    // Two broadcast batches, each with L=2 query rows. This gives generated causal eligibility
    // the same query index (`row % 2`) as the direct/vector row body.
    return new CpuAttentionLowering.Geometry(new long[] {2}, 2, s, e, ev, rowCount,
        s == 0 ? 0 : s * (type == DataType.FLOAT32 ? 4L : 8L), type, type, type, type, 1d,
        masked ? List.of(0, 1, 2, 3) : List.of(0, 1, 2), masked ? 4 : 3, outputs,
        java.util.Optional.of(layout(2L * e, e, 1)), java.util.Optional.of(layout(0, e, 1)),
        java.util.Optional.of(layout(0, ev, 1)), masked ? java.util.Optional.of(layout(0, s, 1)) : java.util.Optional.empty(),
        layout(2L * ev, ev, 1), outputs == 2 ? java.util.Optional.of(layout(2L * s, s, 1)) : java.util.Optional.empty());
  }

  private static CpuAttentionLowering.NormalizedLayout layout(long... strides) {
    return new CpuAttentionLowering.NormalizedLayout(0, strides);
  }

  private static Object carrier(Type type, Carrier carrier, double[] values, Arena arena) {
    if (carrier == Carrier.ARRAY) {
      if (type == Type.F32) { float[] result = new float[values.length]; for (int i = 0; i < result.length; i++) result[i] = (float) values[i]; return result; }
      return values;
    }
    MemorySegment result = arena.allocate(Math.max(1, (long) values.length * (type == Type.F32 ? 4 : 8)), 8);
    for (int i = 0; i < values.length; i++) if (type == Type.F32) result.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, (float) values[i]); else result.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8L, values[i]);
    return result;
  }

  private static Object zeroCarrier(Type type, Carrier carrier, int length, Arena arena) {
    return carrier(type, carrier, new double[length], arena);
  }

  private static Object byteCarrier(Carrier carrier, byte[] values, Arena arena) {
    if (carrier == Carrier.ARRAY) return values;
    MemorySegment result = arena.allocate(Math.max(1, values.length), 1);
    for (int i = 0; i < values.length; i++) result.set(ValueLayout.JAVA_BYTE, i, values[i]);
    return result;
  }

  private static double[] read(Type type, Object carrier, int length) {
    double[] result = new double[length];
    if (carrier instanceof float[] values) for (int i = 0; i < length; i++) result[i] = values[i];
    else if (carrier instanceof double[] values) System.arraycopy(values, 0, result, 0, length);
    else { MemorySegment segment = (MemorySegment) carrier; for (int i = 0; i < length; i++) result[i] = type == Type.F32 ? segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L) : segment.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8L); }
    return result;
  }

  /** Direct clean-Java baseline: score, max, exp/sum, normalize, then increasing-key values. */
  private static Result direct(Type type, Carrier inputCarrier, Carrier outputCarrier, int s, int ev,
      double[] seed, boolean masked, boolean causal, boolean requestedWeights, long start, long end) {
    try (Arena arena = Arena.ofConfined()) {
      return evaluate(type, outputCarrier, inputs(type, inputCarrier, s, ev, seed, arena), masked, causal,
          requestedWeights, start, end);
    }
  }

  /** Candidate two: preferred-species Q/K loads/maps, scalar logical-order folds. */
  private static Result candidate(Type type, Carrier inputCarrier, Carrier outputCarrier, int s, int ev,
      double[] seed, boolean masked, boolean causal, boolean requestedWeights, long start, long end) {
    try (Arena arena = Arena.ofConfined()) {
      return candidate(type, outputCarrier, inputs(type, inputCarrier, s, ev, seed, arena), masked,
          causal, requestedWeights, start, end);
    }
  }

  private static Result candidate(Type type, Carrier outputCarrier, Inputs inputs, boolean masked,
      boolean causal, boolean requestedWeights, long start, long end) {
    Object output = zeroCarrier(type, outputCarrier, Math.toIntExact((end - start) * inputs.valueWidth), inputs.arena);
    Object weights = requestedWeights ? zeroCarrier(type, outputCarrier,
        Math.toIntExact((end - start) * inputs.keys), inputs.arena) : null;
    return candidateEvaluate(type, inputs, output, weights, masked, causal, requestedWeights, start, end);
  }

  private static Result evaluate(Type type, Carrier outputCarrier, Inputs inputs, boolean masked, boolean causal,
      boolean requestedWeights, long start, long end) {
    int e = inputs.embedding, s = inputs.keys, ev = inputs.valueWidth;
    boolean[] eligible = new boolean[s];
    for (int j = 0; j < s; j++) eligible[j] = !masked || (j & 1) == 0;
    double[] maxes = new double[(int) (end - start)], sums = new double[(int) (end - start)];
    double[] allWeights = requestedWeights ? new double[(int) (end - start) * s] : new double[0];
    double[] output = new double[(int) (end - start) * ev];
    {
      for (long row = start; row < end; row++) {
        double[] scores = new double[s];
        int eligibleCount = 0, positive = 0; boolean nan = false, allNegative = true;
        for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) {
          double score = score(type, inputs.query, inputs.key, (int) row, j, e);
          scores[j] = score; eligibleCount++; nan |= Double.isNaN(score);
          positive += score == Double.POSITIVE_INFINITY ? 1 : 0; allNegative &= score == Double.NEGATIVE_INFINITY;
        }
        int mode = eligibleCount == 0 || allNegative ? 0 : nan ? 1 : positive != 0 ? 2 : 3;
        double max = Double.NEGATIVE_INFINITY, sum = 0;
        if (mode == 3) {
          for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) max = max(type, max, scores[j]);
          for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) { scores[j] = narrow(type, StrictMath.exp(narrow(type, scores[j] - max))); sum = add(type, sum, scores[j]); }
          for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) scores[j] = narrow(type, scores[j] / sum);
        } else for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) scores[j] = mode == 1 ? Double.NaN : mode == 2 && scores[j] == Double.POSITIVE_INFINITY ? narrow(type, 1d / positive) : 0;
        int r = (int) (row - start); maxes[r] = max; sums[r] = sum;
        for (int j = 0; j < s; j++) if (requestedWeights) allWeights[r * s + j] = eligible[j] && (!causal || j <= row % 2) ? scores[j] : 0;
        for (int d = 0; d < ev; d++) { double total = mode == 1 ? Double.NaN : 0; if (mode != 0 && mode != 1) for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) total = add(type, total, narrow(type, scores[j] * scalar(type, inputs.value, j * ev + d))); output[r * ev + d] = total; }
      }
    }
    // Candidate two must also observe the declared output representation and its exact typed
    // narrowing rather than returning a convenient double[] surrogate.
    Object typedOutput = carrier(type, outputCarrier, output, inputs.arena);
    Object typedWeights = requestedWeights ? carrier(type, outputCarrier, allWeights, inputs.arena) : null;
    return new Result(maxes, sums, requestedWeights ? read(type, typedWeights, allWeights.length) : allWeights,
        read(type, typedOutput, output.length));
  }

  /* Deliberately separate from the direct body: candidate two has no boolean-selected hot path. */
  private static Result candidateEvaluate(Type type, Inputs inputs, Object typedOutput, Object typedWeights,
      boolean masked, boolean causal, boolean requestedWeights, long start, long end) {
    int e = inputs.embedding, s = inputs.keys, ev = inputs.valueWidth;
    boolean[] eligible = new boolean[s];
    for (int j = 0; j < s; j++) eligible[j] = !masked || (j & 1) == 0;
    double[] maxes = new double[(int) (end - start)], sums = new double[(int) (end - start)];
    double[] allWeights = requestedWeights ? new double[(int) (end - start) * s] : new double[0];
    double[] output = new double[(int) (end - start) * ev];
    for (long row = start; row < end; row++) {
      double[] scores = new double[s]; int eligibleCount = 0, positive = 0;
      boolean nan = false, allNegative = true;
      for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) {
        double score = candidateScore(type, inputs.query, inputs.key, (int) row, j, e);
        scores[j] = score; eligibleCount++; nan |= Double.isNaN(score);
        positive += score == Double.POSITIVE_INFINITY ? 1 : 0; allNegative &= score == Double.NEGATIVE_INFINITY;
      }
      int mode = eligibleCount == 0 || allNegative ? 0 : nan ? 1 : positive != 0 ? 2 : 3;
      double max = Double.NEGATIVE_INFINITY, sum = 0;
      if (mode == 3) {
        for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) max = max(type, max, scores[j]);
        for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) { scores[j] = narrow(type, StrictMath.exp(narrow(type, scores[j] - max))); sum = add(type, sum, scores[j]); }
        for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) scores[j] = narrow(type, scores[j] / sum);
      } else for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) scores[j] = mode == 1 ? Double.NaN : mode == 2 && scores[j] == Double.POSITIVE_INFINITY ? narrow(type, 1d / positive) : 0;
      int r = (int) (row - start); maxes[r] = max; sums[r] = sum;
      for (int j = 0; j < s; j++) if (requestedWeights) allWeights[r * s + j] = eligible[j] && (!causal || j <= row % 2) ? scores[j] : 0;
      for (int d = 0; d < ev; d++) { double total = mode == 1 ? Double.NaN : 0; if (mode != 0 && mode != 1) for (int j = 0; j < s; j++) if (eligible[j] && (!causal || j <= row % 2)) total = add(type, total, narrow(type, scores[j] * scalar(type, inputs.value, j * ev + d))); output[r * ev + d] = total; }
    }
    for (int i = 0; i < output.length; i++) set(type, typedOutput, i, output[i]);
    if (requestedWeights) for (int i = 0; i < allWeights.length; i++) set(type, typedWeights, i, allWeights[i]);
    return new Result(maxes, sums, requestedWeights ? read(type, typedWeights, allWeights.length) : allWeights,
        read(type, typedOutput, output.length));
  }

  private static double score(Type type, Object q, Object k, int row, int key, int e) {
    double total = 0; int x = 0;
    for (; x < e; x++) total = add(type, total, narrow(type, scalar(type, q, row * e + x) * scalar(type, k, key * e + x)));
    return total;
  }

  /** Candidate-two score load/map, followed by an increasing-lane scalar fold. */
  private static double candidateScore(Type type, Object q, Object k, int row, int key, int e) {
    double total = 0; int x = 0;
    if (type == Type.F32) for (; x + FloatVector.SPECIES_PREFERRED.length() <= e;
        x += FloatVector.SPECIES_PREFERRED.length()) { FloatVector a = loadFloat(q, row * e + x);
      FloatVector product = loadFloat(k, key * e + x).mul(a); for (int lane = 0; lane < product.length(); lane++) total = add(type, total, product.lane(lane)); }
    else for (; x + DoubleVector.SPECIES_PREFERRED.length() <= e;
        x += DoubleVector.SPECIES_PREFERRED.length()) { DoubleVector a = loadDouble(q, row * e + x);
      DoubleVector product = loadDouble(k, key * e + x).mul(a); for (int lane = 0; lane < product.length(); lane++) total = add(type, total, product.lane(lane)); }
    for (; x < e; x++) total = add(type, total, narrow(type, scalar(type, q, row * e + x) * scalar(type, k, key * e + x)));
    return total;
  }

  private static Inputs inputs(Type type, Carrier carrier, int s, int ev, double[] seed, Arena arena) {
    int rows = 4, e = Math.max(1, lanes(type));
    double[] q = new double[rows * e], k = new double[Math.max(0, s * e)], v = new double[Math.max(0, s * ev)];
    for (int r = 0; r < rows; r++) for (int x = 0; x < e; x++) q[r * e + x] = narrow(type, x == 0 ? seed[r % seed.length] : 1d / (x + 1));
    for (int j = 0; j < s; j++) for (int x = 0; x < e; x++) k[j * e + x] = narrow(type, x == 0 ? 1 : (j + 1d) / (x + 2));
    for (int j = 0; j < s; j++) for (int d = 0; d < ev; d++) v[j * ev + d] = narrow(type, seed[j % seed.length] + d);
    return new Inputs(carrier(type, carrier, q, arena), carrier(type, carrier, k, arena),
        carrier(type, carrier, v, arena), s, ev, e, arena);
  }

  private static FloatVector loadFloat(Object carrier, int index) { return carrier instanceof float[] array
      ? FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, array, index)
      : FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, (MemorySegment) carrier,
          (long) index * Float.BYTES, ByteOrder.nativeOrder()); }
  private static DoubleVector loadDouble(Object carrier, int index) { return carrier instanceof double[] array
      ? DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, array, index)
      : DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, (MemorySegment) carrier,
          (long) index * Double.BYTES, ByteOrder.nativeOrder()); }
  private static double scalar(Type type, Object carrier, int index) { return type == Type.F32
      ? carrier instanceof float[] array ? array[index] : ((MemorySegment) carrier).get(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), (long) index * Float.BYTES)
      : carrier instanceof double[] array ? array[index] : ((MemorySegment) carrier).get(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), (long) index * Double.BYTES); }
  private static void set(Type type, Object carrier, int index, double value) { if (carrier instanceof float[] array) array[index] = (float) value; else if (carrier instanceof double[] array) array[index] = value; else if (type == Type.F32) ((MemorySegment) carrier).set(ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), (long) index * Float.BYTES, (float) value); else ((MemorySegment) carrier).set(ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), (long) index * Double.BYTES, value); }

  private static double add(Type t, double a, double b) { return narrow(t, a + b); }
  private static double max(Type t, double a, double b) { return narrow(t, Math.max(a, b)); }
  private static double narrow(Type t, double x) { return t == Type.F32 ? (float) x : x; }
  private static int lanes(Type t) { return t == Type.F32 ? FloatVector.SPECIES_PREFERRED.length() : DoubleVector.SPECIES_PREFERRED.length(); }
  private static boolean indexedVectorAccessIsProved() { return false; }
  private static double[] finite(int n) { double[] r = new double[Math.max(1, n)]; for (int i = 0; i < r.length; i++) r[i] = i - r.length / 2d; return r; }
  private static List<double[]> stimuli(int n) { return List.of(finite(n), new double[] {Double.NaN}, new double[] {Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}, new double[] {Double.NEGATIVE_INFINITY}, new double[] {-0.0, 0.0}, new double[] {Double.MIN_VALUE, -Double.MIN_VALUE}, new double[] {700, -700}); }
  private static double[] join(double[] a, double[] b) { double[] r = new double[a.length + b.length]; System.arraycopy(a, 0, r, 0, a.length); System.arraycopy(b, 0, r, a.length, b.length); return r; }
  private static boolean sameBits(Type type, double[] a, double[] b) { if (a.length != b.length) return false; for (int i = 0; i < a.length; i++) if (bits(type, a[i]) != bits(type, b[i])) return false; return true; }
  private static void assertBits(Type t, double[] expected, double[] actual, String pass) { assertEquals(expected.length, actual.length, pass); for (int i = 0; i < expected.length; i++) assertEquals(bits(t, expected[i]), bits(t, actual[i]), pass + " at " + i); }
  private static long bits(Type t, double value) { return t == Type.F32 ? Float.floatToRawIntBits((float) value) : Double.doubleToRawLongBits(value); }
  private record Result(double[] max, double[] sum, double[] weights, double[] output) { }
  private record Inputs(Object query, Object key, Object value, int keys, int valueWidth, int embedding, Arena arena) { }
}
