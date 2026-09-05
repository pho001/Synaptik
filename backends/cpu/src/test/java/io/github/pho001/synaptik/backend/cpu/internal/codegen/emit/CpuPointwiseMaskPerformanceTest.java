package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.*;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.security.MessageDigest;
import java.util.*;
import jdk.incubator.vector.*;
import org.junit.jupiter.api.*;

/** Opt-in retained generated-versus-direct performance gate for CPU 0008L mask boundaries. */
class CpuPointwiseMaskPerformanceTest {
  private static final String ENABLE = "synaptik.cpu.pointwiseMask.performance",
      DIAGNOSTIC = "synaptik.cpu.pointwiseMask.performanceDiagnostic",
      ROOT = "synaptik.cpu.pointwiseMask.performanceEvidenceRoot";
  private static final int FORKS = 5, WARMUPS = 5, SAMPLES = 9, EXACT_CHUNKS = 8192, OFFSET = 7;
  private static final long MINIMUM_NANOS = 25_000_000L;
  private static final long CALIBRATION_MINIMUM_NANOS = 50_000_000L;
  private static final double LIMIT = 1.15d;
  private static final ByteOrder ORDER = ByteOrder.nativeOrder();
  private static final ValueLayout.OfFloat FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ORDER);
  private static final ValueLayout.OfDouble DOUBLE =
      ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ORDER);
  private static volatile long sink;

  enum Form {
    CLASSIFICATION,
    COMPARISON,
    UNARY_LOGICAL,
    BINARY_LOGICAL,
    WHERE_RELOAD,
    FANOUT
  }

  enum Pattern {
    ALL_ARRAY,
    ALL_SEGMENT,
    NUM_ARRAY_BOOL_SEG,
    NUM_SEG_BOOL_ARRAY,
    BOOL_ARRAY_NUM_SEG,
    BOOL_SEG_NUM_ARRAY,
    INPUT_ARRAY_OUTPUT_SEG,
    INPUT_SEG_OUTPUT_ARRAY
  }

  enum Row {
    F32_CLASSIFICATION(
        DataType.FLOAT32, Form.CLASSIFICATION, Pattern.ALL_ARRAY, false, false, false),
    F64_CLASSIFICATION(
        DataType.FLOAT64, Form.CLASSIFICATION, Pattern.ALL_SEGMENT, true, true, true),
    F32_COMPARISON(DataType.FLOAT32, Form.COMPARISON, Pattern.ALL_SEGMENT, true, true, false),
    F64_COMPARISON(DataType.FLOAT64, Form.COMPARISON, Pattern.ALL_ARRAY, false, false, false),
    F32_UNARY_LOGICAL(
        DataType.FLOAT32, Form.UNARY_LOGICAL, Pattern.NUM_ARRAY_BOOL_SEG, true, false, true),
    F64_UNARY_LOGICAL(
        DataType.FLOAT64, Form.UNARY_LOGICAL, Pattern.NUM_SEG_BOOL_ARRAY, false, true, false),
    F32_BINARY_LOGICAL(
        DataType.FLOAT32, Form.BINARY_LOGICAL, Pattern.ALL_SEGMENT, false, true, true),
    F64_BINARY_LOGICAL(
        DataType.FLOAT64, Form.BINARY_LOGICAL, Pattern.ALL_ARRAY, true, false, false),
    F32_WHERE_RELOAD(
        DataType.FLOAT32, Form.WHERE_RELOAD, Pattern.BOOL_ARRAY_NUM_SEG, true, true, false),
    F64_WHERE_RELOAD(
        DataType.FLOAT64, Form.WHERE_RELOAD, Pattern.BOOL_SEG_NUM_ARRAY, false, false, true),
    F32_FANOUT(DataType.FLOAT32, Form.FANOUT, Pattern.INPUT_ARRAY_OUTPUT_SEG, false, false, true),
    F64_FANOUT(DataType.FLOAT64, Form.FANOUT, Pattern.INPUT_SEG_OUTPUT_ARRAY, true, true, false);
    final DataType type;
    final Form form;
    final Pattern pattern;
    final boolean offset, tail, parallel;

    Row(DataType t, Form f, Pattern p, boolean o, boolean x, boolean q) {
      type = t;
      form = f;
      pattern = p;
      offset = o;
      tail = x;
      parallel = q;
    }

    CpuPartitionPreparationPlan.ExecutionStrategy strategy() {
      return parallel
          ? CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_VECTOR
          : CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR;
    }
  }

  public static void main(String[] a) throws Throwable {
    if (a.length == 2 && a[0].equals("--fork")) fork(root(), Integer.parseInt(a[1]));
    else parent(root());
  }

  @Test
  void exactMatrixAndProtocol() {
    assertEquals(12, Row.values().length);
    assertEquals(6, Arrays.stream(Row.values()).map(r -> r.form).distinct().count());
    assertEquals(5, Arrays.stream(Row.values()).filter(r -> r.parallel).count());
    assertEquals(7, Arrays.stream(Row.values()).filter(r -> !r.parallel).count());
    assertEquals(5, FORKS);
    assertEquals(5, WARMUPS);
    assertEquals(9, SAMPLES);
    assertEquals(25_000_000L, MINIMUM_NANOS);
    assertEquals(50_000_000L, CALIBRATION_MINIMUM_NANOS);
    assertTrue(CALIBRATION_MINIMUM_NANOS >= 2L * MINIMUM_NANOS);
    assertEquals(1.15d, LIMIT);
  }

  @Test
  void generatedArtifactsMatchIndependentVectorOraclesBeforeAndAfterExecution() throws Throwable {
    for (Row r : Row.values()) {
      Work w = new Work(r);
      Expected e = verify(w);
      w.generated();
      same(r, e.outputs, snap(w));
      w.direct();
      same(r, e.outputs, snap(w));
    }
  }

  @Test
  void retainedFiveFreshForkEvidence() throws Exception {
    Assumptions.assumeTrue(Boolean.getBoolean(ENABLE));
    parent(root());
  }

  @Test
  void diagnosticSingleFreshFork() throws Throwable {
    Assumptions.assumeTrue(Boolean.getBoolean(DIAGNOSTIC));
    fork(root(), 0);
  }

  private static Path root() {
    String s = System.getProperty(ROOT);
    if (s == null || s.isBlank()) throw new IllegalArgumentException(ROOT + " is required");
    return Path.of(s).toAbsolutePath();
  }

  private static void parent(Path root) throws Exception {
    Path p = dir(root);
    Files.createDirectories(p);
    Files.writeString(
        p.resolve("protocol.txt"),
        "rows=12\n"
            + "forks=5\n"
            + "warmup_batches=5\n"
            + "randomized_symmetric_ab_ba_pairs=9\n"
            + "adaptive_minimum_ns=25000000\n"
            + "post_warmup_calibration_minimum_ns=50000000\n"
            + "fixed_heap=-Xms1g,-Xmx1g\n"
            + "c2_only=true\n"
            + "retry=false\n"
            + "discard=false\n"
            + "threshold=1.15\n"
            + "rows="
            + Arrays.toString(Row.values())
            + "\n");
    Files.copy(
        source(),
        p.resolve("CpuPointwiseMaskPerformanceTest.java"),
        StandardCopyOption.REPLACE_EXISTING);
    Files.writeString(p.resolve("environment.txt"), System.getProperties().toString());
    String java = Path.of(System.getProperty("java.home"), "bin", "java").toString();
    StringBuilder commands = new StringBuilder();
    for (int f = 0; f < FORKS; f++) {
      List<String> c =
          List.of(
              java,
              "-Xms1g",
              "-Xmx1g",
              "-XX:-TieredCompilation",
              "-Xbatch",
              "--add-modules",
              "jdk.incubator.vector",
              "-cp",
              System.getProperty("java.class.path"),
              "-D" + ROOT + "=" + root,
              getName(),
              "--fork",
              "" + f);
      commands.append(String.join(" ", c)).append('\n');
      Process q =
          new ProcessBuilder(c)
              .redirectOutput(p.resolve("fork-" + f + ".stdout").toFile())
              .redirectError(p.resolve("fork-" + f + ".stderr").toFile())
              .start();
      assertEquals(0, q.waitFor(), "fork " + f);
      validate(p.resolve("raw-fork-" + f + ".csv"), f);
    }
    Files.writeString(p.resolve("commands.txt"), commands);
    aggregate(p);
    manifest(p);
  }

  private static void fork(Path root, int fork) throws Throwable {
    if (fork < 0 || fork >= FORKS) throw new IllegalArgumentException("fork");
    Path p = dir(root);
    Files.createDirectories(p);
    Random rnd = new Random(0x8000cafL + fork);
    StringBuilder raw = new StringBuilder("row,iterations,median_ratio,checksum\n");
    for (Row r : Row.values()) {
      Work w = new Work(r);
      Expected expected = verify(w);
      int n = calibrate(w, expected, 1, MINIMUM_NANOS);
      for (int i = 0; i < WARMUPS; i++) {
        pair(w, n, rnd);
        same(r, expected.outputs, snap(w));
      }
      n = calibrate(w, expected, n, CALIBRATION_MINIMUM_NANOS);
      double[] ratios = new double[SAMPLES];
      StringBuilder samples =
          new StringBuilder(
              "row,sample,iterations,generated_ns,direct_ns,generated_before_ns,direct_after_ns,generated_after_ns,direct_before_ns,ratio\n");
      for (int i = 0; i < SAMPLES; i++) {
        Measurement m = pair(w, n, rnd);
        assertTrue(
            m.generated() >= MINIMUM_NANOS && m.direct() >= MINIMUM_NANOS,
            r + " sample below minimum");
        ratios[i] = m.ratio();
        assertTrue(ratios[i] <= LIMIT, r + " fork " + fork + " sample " + i + "=" + ratios[i]);
        same(r, expected.outputs, snap(w));
        samples
            .append(r)
            .append(',')
            .append(i)
            .append(',')
            .append(n)
            .append(',')
            .append(m.generated())
            .append(',')
            .append(m.direct())
            .append(',')
            .append(m.gb)
            .append(',')
            .append(m.da)
            .append(',')
            .append(m.ga)
            .append(',')
            .append(m.db)
            .append(',')
            .append(m.ratio())
            .append('\n');
      }
      Files.writeString(p.resolve("measurements-fork-" + fork + "-" + r + ".csv"), samples);
      Arrays.sort(ratios);
      assertTrue(ratios[4] <= LIMIT, r + " fork median=" + ratios[4]);
      w.generated();
      same(r, expected.outputs, snap(w));
      w.direct();
      same(r, expected.outputs, snap(w));
      raw.append(r)
          .append(',')
          .append(n)
          .append(',')
          .append(ratios[4])
          .append(',')
          .append(sink)
          .append('\n');
    }
    Files.writeString(p.resolve("raw-fork-" + fork + ".csv"), raw);
  }

  /**
   * Calibrates equal work for both sides. The second call occurs after all five warmup pairs and
   * uses a two-times sample-duration floor to provide headroom against residual throughput change.
   *
   * @param w prepared generated and direct work for one matrix row
   * @param e verified output bytes that every calibration attempt must preserve
   * @param initialIterations positive shared iteration count at which calibration starts
   * @param minimumNanos minimum aggregate duration required independently from each timed side
   * @return the first shared iteration count for which both sides meet {@code minimumNanos}
   * @throws Throwable if generated invocation fails
   */
  private static int calibrate(Work w, Expected e, int initialIterations, long minimumNanos)
      throws Throwable {
    int n = initialIterations;
    for (; ; ) {
      Measurement m = pair(w, n, new Random(1));
      same(w.row, e.outputs, snap(w));
      if (m.generated() >= minimumNanos && m.direct() >= minimumNanos) return n;
      n = Math.multiplyExact(n, 2);
    }
  }

  /** Measures one randomized G-D-D-G or D-G-G-D pair; both directions are retained. */
  private static Measurement pair(Work w, int n, Random r) throws Throwable {
    long gb, da, ga, db;
    if (r.nextBoolean()) {
      gb = time(w, true, n);
      da = time(w, false, n);
      db = time(w, false, n);
      ga = time(w, true, n);
    } else {
      db = time(w, false, n);
      ga = time(w, true, n);
      gb = time(w, true, n);
      da = time(w, false, n);
    }
    return new Measurement(gb, da, ga, db);
  }

  private static long time(Work w, boolean generated, int n) throws Throwable {
    long s = System.nanoTime();
    for (int i = 0; i < n; i++)
      if (generated) w.generated();
      else w.direct();
    return System.nanoTime() - s;
  }

  private static final class Work {
    final Row row;
    final int count, offset, capacity;
    final CpuKernelIr ir;
    final List<DataType> types;
    final List<Object> b;
    final long[] geometry;
    final MethodHandle entry;

    Work(Row row) {
      this.row = row;
      int lanes = lanes(row.type);
      count = EXACT_CHUNKS * lanes + (row.tail ? 3 : 0);
      offset = row.offset ? OFFSET : 0;
      capacity = offset + count + lanes + 11;
      ir = ir(row);
      types = types(ir);
      List<CpuKernelSpecialization.CarrierAccess> cs = carriers(row, types);
      b = new ArrayList<>();
      int boundary = 0;
      for (CpuKernelIr.Value v : ir.values()) {
        if (v.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
        Object x =
            carrier(
                v.dataType(),
                capacity,
                cs.get(boundary) == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        if (v.kind() == CpuKernelIr.Value.Kind.INPUT)
          input(x, v.dataType(), offset, count, boundary);
        else fill(x, v.dataType(), capacity, 90);
        b.add(x);
        boundary++;
      }
      geometry = geometry(types.size(), count, offset);
      var s =
          new CpuKernelSpecialization(
              CpuLoweringFingerprint.fromHex(ir.structuralKey()),
              CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
              row.strategy(),
              types,
              cs,
              bits(row.type),
              -1,
              List.of(),
              false,
              61);
      var g = new CpuClassFileKernelGenerator();
      entry = g.defineClassBytes(s, g.generateClassBytes(s, ir)).entryPoint();
    }

    void generated() throws Throwable {
      if (row.parallel) {
        int split = split();
        invoke(geometry, 0, split);
        invoke(geometry(types.size(), count, offset + split), split, count);
      } else invoke(geometry, 0, count);
      sink += observe(this);
    }

    void invoke(long[] g, long start, long end) throws Throwable {
      switch (row) {
        case F32_CLASSIFICATION ->
            entry.invokeExact((float[]) b.get(0), (byte[]) b.get(1), g, start, end);
        case F64_CLASSIFICATION ->
            entry.invokeExact((MemorySegment) b.get(0), (MemorySegment) b.get(1), g, start, end);
        case F32_COMPARISON ->
            entry.invokeExact(
                (MemorySegment) b.get(0),
                (MemorySegment) b.get(1),
                (MemorySegment) b.get(2),
                g,
                start,
                end);
        case F64_COMPARISON ->
            entry.invokeExact(
                (double[]) b.get(0), (double[]) b.get(1), (byte[]) b.get(2), g, start, end);
        case F32_UNARY_LOGICAL ->
            entry.invokeExact((float[]) b.get(0), (MemorySegment) b.get(1), g, start, end);
        case F64_UNARY_LOGICAL ->
            entry.invokeExact((MemorySegment) b.get(0), (byte[]) b.get(1), g, start, end);
        case F32_BINARY_LOGICAL ->
            entry.invokeExact(
                (MemorySegment) b.get(0),
                (MemorySegment) b.get(1),
                (MemorySegment) b.get(2),
                (MemorySegment) b.get(3),
                g,
                start,
                end);
        case F64_BINARY_LOGICAL ->
            entry.invokeExact(
                (double[]) b.get(0),
                (double[]) b.get(1),
                (double[]) b.get(2),
                (byte[]) b.get(3),
                g,
                start,
                end);
        case F32_WHERE_RELOAD ->
            entry.invokeExact(
                (byte[]) b.get(0),
                (MemorySegment) b.get(1),
                (MemorySegment) b.get(2),
                (MemorySegment) b.get(3),
                g,
                start,
                end);
        case F64_WHERE_RELOAD ->
            entry.invokeExact(
                (MemorySegment) b.get(0),
                (double[]) b.get(1),
                (double[]) b.get(2),
                (double[]) b.get(3),
                g,
                start,
                end);
        case F32_FANOUT ->
            entry.invokeExact(
                (float[]) b.get(0),
                (float[]) b.get(1),
                (MemorySegment) b.get(2),
                (MemorySegment) b.get(3),
                g,
                start,
                end);
        case F64_FANOUT ->
            entry.invokeExact(
                (MemorySegment) b.get(0),
                (MemorySegment) b.get(1),
                (byte[]) b.get(2),
                (double[]) b.get(3),
                g,
                start,
                end);
      }
    }

    void direct() {
      if (row.parallel) {
        int split = split();
        oracle(this, 0, split);
        oracle(this, split, count);
      } else oracle(this, 0, count);
      sink += observe(this);
    }

    int split() {
      return count / 2 / lanes(row.type) * lanes(row.type);
    }
  }

  private static void oracle(Work w, int start, int end) {
    int a = w.offset + start, i = start, n = lanes(w.row.type);
    switch (w.row) {
      case F32_CLASSIFICATION -> {
        float[] x = (float[]) w.b.get(0);
        byte[] o = (byte[]) w.b.get(1);
        for (; i + n <= end; i += n, a += n)
          store(
              FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, x, a)
                  .test(VectorOperators.IS_NAN),
              o,
              a);
        for (; i < end; i++, a++) o[a] = (byte) (Float.isNaN(x[a]) ? 1 : 0);
      }
      case F64_CLASSIFICATION -> {
        MemorySegment x = (MemorySegment) w.b.get(0), o = (MemorySegment) w.b.get(1);
        for (; i + n <= end; i += n, a += n)
          store(
              DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, x, (long) a * 8, ORDER)
                  .test(VectorOperators.IS_FINITE),
              o,
              a);
        for (; i < end; i++, a++)
          o.setAtIndex(
              ValueLayout.JAVA_BYTE, a, (byte) (Double.isFinite(x.getAtIndex(DOUBLE, a)) ? 1 : 0));
      }
      case F32_COMPARISON -> {
        MemorySegment x = (MemorySegment) w.b.get(0),
            y = (MemorySegment) w.b.get(1),
            o = (MemorySegment) w.b.get(2);
        for (; i + n <= end; i += n, a += n)
          store(
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, x, (long) a * 4, ORDER)
                  .compare(
                      VectorOperators.GT,
                      FloatVector.fromMemorySegment(
                          FloatVector.SPECIES_PREFERRED, y, (long) a * 4, ORDER)),
              o,
              a);
        for (; i < end; i++, a++)
          o.setAtIndex(
              ValueLayout.JAVA_BYTE,
              a,
              (byte) (x.getAtIndex(FLOAT, a) > y.getAtIndex(FLOAT, a) ? 1 : 0));
      }
      case F64_COMPARISON -> {
        double[] x = (double[]) w.b.get(0), y = (double[]) w.b.get(1);
        byte[] o = (byte[]) w.b.get(2);
        for (; i + n <= end; i += n, a += n)
          store(
              DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, x, a)
                  .compare(
                      VectorOperators.EQ,
                      DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, y, a)),
              o,
              a);
        for (; i < end; i++, a++) o[a] = (byte) (x[a] == y[a] ? 1 : 0);
      }
      case F32_UNARY_LOGICAL -> {
        float[] x = (float[]) w.b.get(0);
        MemorySegment o = (MemorySegment) w.b.get(1);
        for (; i + n <= end; i += n, a += n)
          store(
              FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, x, a)
                  .test(VectorOperators.IS_FINITE)
                  .not(),
              o,
              a);
        for (; i < end; i++, a++)
          o.setAtIndex(ValueLayout.JAVA_BYTE, a, (byte) (Float.isFinite(x[a]) ? 0 : 1));
      }
      case F64_UNARY_LOGICAL -> {
        MemorySegment x = (MemorySegment) w.b.get(0);
        byte[] o = (byte[]) w.b.get(1);
        for (; i + n <= end; i += n, a += n)
          store(
              DoubleVector.fromMemorySegment(DoubleVector.SPECIES_PREFERRED, x, (long) a * 8, ORDER)
                  .test(VectorOperators.IS_NAN)
                  .not(),
              o,
              a);
        for (; i < end; i++, a++) o[a] = (byte) (Double.isNaN(x.getAtIndex(DOUBLE, a)) ? 0 : 1);
      }
      case F32_BINARY_LOGICAL -> {
        MemorySegment x = (MemorySegment) w.b.get(0),
            y = (MemorySegment) w.b.get(1),
            z = (MemorySegment) w.b.get(2),
            o = (MemorySegment) w.b.get(3);
        for (; i + n <= end; i += n, a += n) {
          var xv =
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, x, (long) a * 4, ORDER);
          var yv =
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, y, (long) a * 4, ORDER);
          var zv =
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, z, (long) a * 4, ORDER);
          store(xv.compare(VectorOperators.GT, yv).and(zv.test(VectorOperators.IS_FINITE)), o, a);
        }
        for (; i < end; i++, a++)
          o.setAtIndex(
              ValueLayout.JAVA_BYTE,
              a,
              (byte)
                  (x.getAtIndex(FLOAT, a) > y.getAtIndex(FLOAT, a)
                          && Float.isFinite(z.getAtIndex(FLOAT, a))
                      ? 1
                      : 0));
      }
      case F64_BINARY_LOGICAL -> {
        double[] x = (double[]) w.b.get(0), y = (double[]) w.b.get(1), z = (double[]) w.b.get(2);
        byte[] o = (byte[]) w.b.get(3);
        for (; i + n <= end; i += n, a += n) {
          var xv = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, x, a);
          var yv = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, y, a);
          var zv = DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, z, a);
          store(xv.compare(VectorOperators.LE, yv).or(zv.test(VectorOperators.IS_NAN)), o, a);
        }
        for (; i < end; i++, a++) o[a] = (byte) (x[a] <= y[a] || Double.isNaN(z[a]) ? 1 : 0);
      }
      case F32_WHERE_RELOAD -> {
        byte[] m = (byte[]) w.b.get(0);
        MemorySegment x = (MemorySegment) w.b.get(1),
            y = (MemorySegment) w.b.get(2),
            o = (MemorySegment) w.b.get(3);
        for (; i + n <= end; i += n, a += n) {
          var xv =
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, x, (long) a * 4, ORDER);
          var yv =
              FloatVector.fromMemorySegment(FloatVector.SPECIES_PREFERRED, y, (long) a * 4, ORDER);
          yv.blend(xv, load32(m, a)).intoMemorySegment(o, (long) a * 4, ORDER);
        }
        for (; i < end; i++, a++)
          o.setAtIndex(FLOAT, a, m[a] == 1 ? x.getAtIndex(FLOAT, a) : y.getAtIndex(FLOAT, a));
      }
      case F64_WHERE_RELOAD -> {
        MemorySegment m = (MemorySegment) w.b.get(0);
        double[] x = (double[]) w.b.get(1), y = (double[]) w.b.get(2), o = (double[]) w.b.get(3);
        for (; i + n <= end; i += n, a += n)
          DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, y, a)
              .blend(DoubleVector.fromArray(DoubleVector.SPECIES_PREFERRED, x, a), load64(m, a))
              .intoArray(o, a);
        for (; i < end; i++, a++) o[a] = m.getAtIndex(ValueLayout.JAVA_BYTE, a) == 1 ? x[a] : y[a];
      }
      case F32_FANOUT -> {
        float[] x = (float[]) w.b.get(0), y = (float[]) w.b.get(1);
        MemorySegment m = (MemorySegment) w.b.get(2), o = (MemorySegment) w.b.get(3);
        for (; i + n <= end; i += n, a += n) {
          var xv = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, x, a);
          var yv = FloatVector.fromArray(FloatVector.SPECIES_PREFERRED, y, a);
          var mask = xv.compare(VectorOperators.GT, yv);
          store(mask, m, a);
          yv.blend(xv, mask).intoMemorySegment(o, (long) a * 4, ORDER);
        }
        for (; i < end; i++, a++) {
          boolean q = x[a] > y[a];
          m.setAtIndex(ValueLayout.JAVA_BYTE, a, (byte) (q ? 1 : 0));
          o.setAtIndex(FLOAT, a, q ? x[a] : y[a]);
        }
      }
      case F64_FANOUT -> {
        MemorySegment x = (MemorySegment) w.b.get(0), y = (MemorySegment) w.b.get(1);
        byte[] m = (byte[]) w.b.get(2);
        double[] o = (double[]) w.b.get(3);
        for (; i + n <= end; i += n, a += n) {
          var xv =
              DoubleVector.fromMemorySegment(
                  DoubleVector.SPECIES_PREFERRED, x, (long) a * 8, ORDER);
          var yv =
              DoubleVector.fromMemorySegment(
                  DoubleVector.SPECIES_PREFERRED, y, (long) a * 8, ORDER);
          var mask = xv.compare(VectorOperators.GT, yv);
          store(mask, m, a);
          yv.blend(xv, mask).intoArray(o, a);
        }
        for (; i < end; i++, a++) {
          double q = x.getAtIndex(DOUBLE, a), r = y.getAtIndex(DOUBLE, a);
          boolean mask = q > r;
          m[a] = (byte) (mask ? 1 : 0);
          o[a] = mask ? q : r;
        }
      }
    }
  }

  private static void store(VectorMask<?> m, byte[] o, int i) {
    VectorSpecies<Byte> s = byteSpecies(m.length());
    VectorMask<Byte> a = VectorMask.fromLong(s, lowBits(m.length()));
    ByteVector.zero(s)
        .blend(ByteVector.broadcast(s, (byte) 1), VectorMask.fromLong(s, m.toLong()))
        .intoArray(o, i, a);
  }

  private static void store(VectorMask<?> m, MemorySegment o, int i) {
    VectorSpecies<Byte> s = byteSpecies(m.length());
    VectorMask<Byte> a = VectorMask.fromLong(s, lowBits(m.length()));
    ByteVector.zero(s)
        .blend(ByteVector.broadcast(s, (byte) 1), VectorMask.fromLong(s, m.toLong()))
        .intoMemorySegment(o, i, ORDER, a);
  }

  private static VectorMask<Float> load32(byte[] m, int i) {
    int n = FloatVector.SPECIES_PREFERRED.length();
    VectorSpecies<Byte> s = byteSpecies(n);
    long x =
        ByteVector.fromArray(s, m, i, VectorMask.fromLong(s, lowBits(n)))
            .compare(VectorOperators.EQ, (byte) 1)
            .toLong();
    return VectorMask.fromLong(FloatVector.SPECIES_PREFERRED, x);
  }

  private static VectorMask<Double> load64(MemorySegment m, int i) {
    int n = DoubleVector.SPECIES_PREFERRED.length();
    VectorSpecies<Byte> s = byteSpecies(n);
    long x =
        ByteVector.fromMemorySegment(s, m, i, ORDER, VectorMask.fromLong(s, lowBits(n)))
            .compare(VectorOperators.EQ, (byte) 1)
            .toLong();
    return VectorMask.fromLong(DoubleVector.SPECIES_PREFERRED, x);
  }

  private static VectorSpecies<Byte> byteSpecies(int n) {
    return n <= 8
        ? ByteVector.SPECIES_64
        : n <= 16
            ? ByteVector.SPECIES_128
            : n <= 32 ? ByteVector.SPECIES_256 : ByteVector.SPECIES_512;
  }

  private static long lowBits(int n) {
    return n == 64 ? -1L : (1L << n) - 1L;
  }

  private static Expected verify(Work w) throws Throwable {
    List<byte[]> before = snap(w);
    w.generated();
    List<byte[]> generated = snap(w);
    restore(w, before);
    w.direct();
    same(w.row, generated, snap(w));
    return new Expected(generated);
  }

  private static List<byte[]> snap(Work w) {
    List<byte[]> r = new ArrayList<>();
    int b = 0;
    for (CpuKernelIr.Value v : w.ir.values()) {
      if (v.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
      if (v.kind() == CpuKernelIr.Value.Kind.OUTPUT) r.add(bytes(w.b.get(b)));
      b++;
    }
    return List.copyOf(r);
  }

  private static void restore(Work w, List<byte[]> saved) {
    int b = 0, o = 0;
    for (CpuKernelIr.Value v : w.ir.values()) {
      if (v.kind() == CpuKernelIr.Value.Kind.VIRTUAL) continue;
      if (v.kind() == CpuKernelIr.Value.Kind.OUTPUT) copy(saved.get(o++), w.b.get(b));
      b++;
    }
  }

  private static void same(Row row, List<byte[]> x, List<byte[]> y) {
    assertEquals(x.size(), y.size(), row.toString());
    for (int i = 0; i < x.size(); i++) assertArrayEquals(x.get(i), y.get(i), row + " output " + i);
  }

  private static byte[] bytes(Object x) {
    MemorySegment s =
        x instanceof byte[] a
            ? MemorySegment.ofArray(a)
            : x instanceof float[] a
                ? MemorySegment.ofArray(a)
                : x instanceof double[] a ? MemorySegment.ofArray(a) : (MemorySegment) x;
    byte[] r = new byte[Math.toIntExact(s.byteSize())];
    MemorySegment.copy(s, 0, MemorySegment.ofArray(r), 0, r.length);
    return r;
  }

  private static void copy(byte[] x, Object y) {
    MemorySegment s =
        y instanceof byte[] a
            ? MemorySegment.ofArray(a)
            : y instanceof float[] a
                ? MemorySegment.ofArray(a)
                : y instanceof double[] a ? MemorySegment.ofArray(a) : (MemorySegment) y;
    MemorySegment.copy(MemorySegment.ofArray(x), 0, s, 0, x.length);
  }

  private static long observe(Work w) {
    int i = w.offset + (int) ((sink & Long.MAX_VALUE) % w.count);
    return switch (w.row.form) {
      case CLASSIFICATION, UNARY_LOGICAL -> byteAt(w.b.get(1), i);
      case COMPARISON -> byteAt(w.b.get(2), i);
      case BINARY_LOGICAL -> byteAt(w.b.get(3), i);
      case WHERE_RELOAD -> numberBits(w.b.get(3), w.row.type, i);
      case FANOUT -> byteAt(w.b.get(2), i) ^ numberBits(w.b.get(3), w.row.type, i);
    };
  }

  private static long byteAt(Object x, int i) {
    return x instanceof byte[] a ? a[i] : ((MemorySegment) x).getAtIndex(ValueLayout.JAVA_BYTE, i);
  }

  private static long numberBits(Object x, DataType t, int i) {
    if (t == DataType.FLOAT32)
      return Float.floatToRawIntBits(
          x instanceof float[] a ? a[i] : ((MemorySegment) x).getAtIndex(FLOAT, i));
    return Double.doubleToRawLongBits(
        x instanceof double[] a ? a[i] : ((MemorySegment) x).getAtIndex(DOUBLE, i));
  }

  private static CpuKernelIr ir(Row row) {
    DataType t = row.type;
    CpuAccessPlan r = plan(CpuAccessPlan.AccessKind.READ), w = plan(CpuAccessPlan.AccessKind.WRITE);
    return switch (row.form) {
      case CLASSIFICATION ->
          kernel(
              List.of(
                  v(0, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(
                  ins(
                      row == Row.F32_CLASSIFICATION
                          ? CpuPointwiseOpcode.IS_NAN
                          : CpuPointwiseOpcode.IS_FINITE,
                      List.of(0),
                      1)),
              List.of(new CpuKernelIr.Store(1, 0)));
      case COMPARISON ->
          kernel(
              List.of(
                  v(0, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(2, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(
                  ins(
                      row == Row.F32_COMPARISON
                          ? CpuPointwiseOpcode.GREATER_THAN
                          : CpuPointwiseOpcode.EQUAL,
                      List.of(0, 1),
                      2)),
              List.of(new CpuKernelIr.Store(2, 0)));
      case UNARY_LOGICAL ->
          kernel(
              List.of(
                  v(0, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL, r),
                  v(2, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(
                  ins(
                      row == Row.F32_UNARY_LOGICAL
                          ? CpuPointwiseOpcode.IS_FINITE
                          : CpuPointwiseOpcode.IS_NAN,
                      List.of(0),
                      1),
                  ins(CpuPointwiseOpcode.LOGICAL_NOT, List.of(1), 2)),
              List.of(new CpuKernelIr.Store(2, 0)));
      case BINARY_LOGICAL ->
          kernel(
              List.of(
                  v(0, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(2, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(3, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL, r),
                  v(4, DataType.BOOL, CpuKernelIr.Value.Kind.VIRTUAL, r),
                  v(5, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(
                  ins(
                      row == Row.F32_BINARY_LOGICAL
                          ? CpuPointwiseOpcode.GREATER_THAN
                          : CpuPointwiseOpcode.LESS_OR_EQUAL,
                      List.of(0, 1),
                      3),
                  ins(
                      row == Row.F32_BINARY_LOGICAL
                          ? CpuPointwiseOpcode.IS_FINITE
                          : CpuPointwiseOpcode.IS_NAN,
                      List.of(2),
                      4),
                  ins(
                      row == Row.F32_BINARY_LOGICAL
                          ? CpuPointwiseOpcode.LOGICAL_AND
                          : CpuPointwiseOpcode.LOGICAL_OR,
                      List.of(3, 4),
                      5)),
              List.of(new CpuKernelIr.Store(5, 0)));
      case WHERE_RELOAD ->
          kernel(
              List.of(
                  v(0, DataType.BOOL, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(2, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(3, t, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(ins(CpuPointwiseOpcode.WHERE, List.of(0, 1, 2), 3)),
              List.of(new CpuKernelIr.Store(3, 0)));
      case FANOUT ->
          kernel(
              List.of(
                  v(0, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(1, t, CpuKernelIr.Value.Kind.INPUT, r),
                  v(2, DataType.BOOL, CpuKernelIr.Value.Kind.OUTPUT, w),
                  v(3, t, CpuKernelIr.Value.Kind.OUTPUT, w)),
              List.of(
                  ins(CpuPointwiseOpcode.GREATER_THAN, List.of(0, 1), 2),
                  ins(CpuPointwiseOpcode.WHERE, List.of(2, 0, 1), 3)),
              List.of(new CpuKernelIr.Store(2, 0), new CpuKernelIr.Store(3, 1)));
    };
  }

  private static CpuKernelIr kernel(
      List<CpuKernelIr.Value> v, List<CpuKernelIr.Instruction> i, List<CpuKernelIr.Store> s) {
    return new CpuKernelIr(v, i, new CpuKernelIr.Loop("start", "end"), s);
  }

  private static CpuKernelIr.Value v(int i, DataType t, CpuKernelIr.Value.Kind k, CpuAccessPlan p) {
    return new CpuKernelIr.Value(i, t, k, p);
  }

  private static CpuKernelIr.Instruction ins(CpuPointwiseOpcode o, List<Integer> i, int x) {
    return new CpuKernelIr.Instruction(o, i, x);
  }

  private static CpuAccessPlan plan(CpuAccessPlan.AccessKind k) {
    return new CpuAccessPlan(
        k, CpuAccessPlan.Regime.DENSE_LINEAR, 1, List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
  }

  private static List<DataType> types(CpuKernelIr ir) {
    return ir.values().stream()
        .filter(v -> v.kind() != CpuKernelIr.Value.Kind.VIRTUAL)
        .map(CpuKernelIr.Value::dataType)
        .toList();
  }

  private static List<CpuKernelSpecialization.CarrierAccess> carriers(
      Row row, List<DataType> types) {
    List<CpuKernelSpecialization.CarrierAccess> r = new ArrayList<>();
    for (int i = 0; i < types.size(); i++) {
      DataType t = types.get(i);
      boolean segment =
          switch (row.pattern) {
            case ALL_ARRAY -> false;
            case ALL_SEGMENT -> true;
            case NUM_ARRAY_BOOL_SEG -> t == DataType.BOOL;
            case NUM_SEG_BOOL_ARRAY -> t != DataType.BOOL;
            case BOOL_ARRAY_NUM_SEG -> t != DataType.BOOL;
            case BOOL_SEG_NUM_ARRAY -> t == DataType.BOOL;
            case INPUT_ARRAY_OUTPUT_SEG -> i >= 2;
            case INPUT_SEG_OUTPUT_ARRAY -> i < 2;
          };
      r.add(
          segment
              ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
              : t == DataType.FLOAT32
                  ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                  : t == DataType.FLOAT64
                      ? CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY
                      : CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY);
    }
    return List.copyOf(r);
  }

  private static Object carrier(DataType t, int n, boolean segment) {
    if (segment) return MemorySegment.ofArray(new byte[Math.multiplyExact(n, t.byteWidth())]);
    return t == DataType.FLOAT32
        ? new float[n]
        : t == DataType.FLOAT64 ? new double[n] : new byte[n];
  }

  private static void input(Object x, DataType t, int offset, int count, int boundary) {
    for (int i = 0; i < count; i++) {
      int q = (i + boundary * 7) % 41;
      double z =
          t == DataType.BOOL
              ? (i & 1)
              : q == 0
                  ? Double.NaN
                  : q == 1
                      ? Double.POSITIVE_INFINITY
                      : q == 2 ? Double.NEGATIVE_INFINITY : q - 20.25;
      put(x, t, offset + i, z);
    }
  }

  private static void fill(Object x, DataType t, int n, double z) {
    for (int i = 0; i < n; i++) put(x, t, i, z);
  }

  private static void put(Object x, DataType t, int i, double z) {
    if (x instanceof float[] a) a[i] = (float) z;
    else if (x instanceof double[] a) a[i] = z;
    else if (x instanceof byte[] a) a[i] = (byte) z;
    else if (t == DataType.FLOAT32) ((MemorySegment) x).setAtIndex(FLOAT, i, (float) z);
    else if (t == DataType.FLOAT64) ((MemorySegment) x).setAtIndex(DOUBLE, i, z);
    else ((MemorySegment) x).setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) z);
  }

  private static int lanes(DataType t) {
    return t == DataType.FLOAT32
        ? FloatVector.SPECIES_PREFERRED.length()
        : DoubleVector.SPECIES_PREFERRED.length();
  }

  private static int bits(DataType t) {
    return t == DataType.FLOAT32
        ? FloatVector.SPECIES_PREFERRED.vectorBitSize()
        : DoubleVector.SPECIES_PREFERRED.vectorBitSize();
  }

  private static long[] geometry(int b, int count, int base) {
    long[] g = new long[2 + 4 * b];
    g[0] = count;
    for (int i = 0; i < b; i++) {
      g[2 + i] = base;
      g[2 + b + i] = 1;
      g[2 + 3 * b + i] = base + count;
    }
    return g;
  }

  private static Path dir(Path root) {
    return root.resolve("cpu-pointwise-mask-performance");
  }

  private static String getName() {
    return CpuPointwiseMaskPerformanceTest.class.getName();
  }

  private static void validate(Path p, int fork) throws Exception {
    List<String> l = Files.readAllLines(p);
    assertEquals(13, l.size());
    for (int i = 1; i < l.size(); i++) {
      String[] v = l.get(i).split(",");
      assertEquals(Row.values()[i - 1].name(), v[0]);
      assertTrue(Double.parseDouble(v[2]) <= LIMIT, "fork " + fork + " " + v[0]);
    }
  }

  private static void aggregate(Path p) throws Exception {
    StringBuilder out = new StringBuilder("row,fork0,fork1,fork2,fork3,fork4,median,accepted\n");
    for (int r = 0; r < 12; r++) {
      double[] x = new double[5];
      for (int f = 0; f < 5; f++)
        x[f] =
            Double.parseDouble(
                Files.readAllLines(p.resolve("raw-fork-" + f + ".csv")).get(r + 1).split(",")[2]);
      double[] s = x.clone();
      Arrays.sort(s);
      assertTrue(s[2] <= LIMIT, Row.values()[r] + " aggregate=" + s[2]);
      out.append(Row.values()[r]);
      for (double z : x) out.append(',').append(z);
      out.append(',').append(s[2]).append(",true\n");
    }
    Files.writeString(p.resolve("aggregate.csv"), out);
  }

  private static void manifest(Path p) throws Exception {
    MessageDigest d = MessageDigest.getInstance("SHA-256");
    StringBuilder out = new StringBuilder();
    try (var paths = Files.walk(p)) {
      for (Path x : paths.filter(Files::isRegularFile).sorted().toList())
        if (!x.getFileName().toString().startsWith("manifest"))
          out.append(HexFormat.of().formatHex(d.digest(Files.readAllBytes(x))))
              .append("  ")
              .append(p.relativize(x))
              .append('\n');
    }
    Files.writeString(p.resolve("manifest.sha256"), out);
    Files.writeString(
        p.resolve("manifest.digest"),
        HexFormat.of().formatHex(d.digest(out.toString().getBytes(StandardCharsets.UTF_8))));
  }

  private static Path source() {
    Path cwd = Path.of(System.getProperty("user.dir")).toAbsolutePath();
    Path a =
        cwd.resolve(
            "src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseMaskPerformanceTest.java");
    if (Files.isRegularFile(a)) return a;
    Path b =
        cwd.resolve(
            "backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuPointwiseMaskPerformanceTest.java");
    if (Files.isRegularFile(b)) return b;
    throw new IllegalStateException("performance source not found from " + cwd);
  }

  private record Expected(List<byte[]> outputs) {}

  private record Measurement(long gb, long da, long ga, long db) {
    long generated() {
      return gb + ga;
    }

    long direct() {
      return da + db;
    }

    double ratio() {
      return Math.sqrt(((double) gb / da) * ((double) ga / db));
    }
  }
}
