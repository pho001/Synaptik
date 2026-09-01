package io.github.pho001.synaptik.backend.cpu.internal.reference;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering.Geometry;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering.NormalizedLayout;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Objects;

/** Optimal clean-Java semantic oracle for the specialized direct attention row algorithm. */
public final class CpuAttentionReferenceKernel {
  private CpuAttentionReferenceKernel() {}

  /**
   * Evaluates complete rows using the same sequential arithmetic and scratch shape as generated
   * execution. This method is test/evidence infrastructure and is never a generated fallback.
   *
   * @param geometry non-null resolved attention geometry
   * @param boundaries non-null unique inputs followed by one or two outputs
   * @param scratch non-null range-private score storage when {@code S > 0}
   * @param causal whether top-left causal eligibility is active
   * @param start inclusive row bound
   * @param end exclusive row bound
   * @throws NullPointerException if an argument is {@code null}
   * @throws IllegalArgumentException if boundary count or range disagrees
   */
  public static void evaluate(
      Geometry geometry,
      List<CpuBufferArgument> boundaries,
      MemorySegment scratch,
      boolean causal,
      long start,
      long end) {
    Objects.requireNonNull(geometry, "geometry");
    boundaries = List.copyOf(boundaries);
    Objects.requireNonNull(scratch, "scratch");
    if (boundaries.size() != geometry.uniqueInputCount() + geometry.outputCount()
        || start < 0
        || end < start
        || end > geometry.rowCount())
      throw new IllegalArgumentException("attention reference invocation disagrees");
    for (long row = start; row < end; row++) {
      if (geometry.resultType() == DataType.FLOAT64)
        row64(geometry, boundaries, scratch, causal, row);
      else row32(geometry, boundaries, scratch, causal, row);
    }
  }

  private static void row32(
      Geometry g, List<CpuBufferArgument> b, MemorySegment scratch, boolean causal, long row) {
    Bases x = bases(g, row);
    long eligible = 0, positive = 0;
    boolean nan = false, allNeg = true;
    for (long j = 0; j < g.keyLength(); j++)
      if (eligible(g, b, causal, x, j)) {
        float score = 0;
        for (long e = 0; e < g.embedding(); e++)
          score +=
              (float)
                  (read(b.get(role(g, 0)), g.queryType(), x.q() + j0(g.query(), e))
                      * read(b.get(role(g, 1)), g.keyType(), x.k() + j1(g.key(), j, e)));
        score *= (float) g.scale();
        scratch.set(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4, score);
        eligible++;
        nan |= Float.isNaN(score);
        if (score == Float.POSITIVE_INFINITY) positive++;
        if (score != Float.NEGATIVE_INFINITY) allNeg = false;
      }
    int mode = eligible == 0 || allNeg ? 0 : nan ? 1 : positive > 0 ? 2 : 3;
    if (mode == 3) {
      float max = Float.NEGATIVE_INFINITY;
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j))
          max = Math.max(max, scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4));
      float sum = 0;
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j)) {
          float w =
              (float) StrictMath.exp(scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4) - max);
          scratch.set(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4, w);
          sum += w;
        }
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j))
          scratch.set(
              ValueLayout.JAVA_FLOAT_UNALIGNED,
              j * 4,
              scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4) / sum);
    } else
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j)) {
          float score = scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4);
          float w =
              mode == 1
                  ? Float.NaN
                  : mode == 2 && score == Float.POSITIVE_INFINITY ? 1f / positive : 0f;
          scratch.set(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4, w);
        }
    if (g.outputCount() == 2)
      for (long j = 0; j < g.keyLength(); j++)
        write(
            b.get(g.uniqueInputCount() + 1),
            g.resultType(),
            x.w() + j0(g.weights().orElseThrow(), j),
            eligible(g, b, causal, x, j)
                ? scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4)
                : 0f);
    for (long d = 0; d < g.valueEmbedding(); d++) {
      float out = mode == 1 ? Float.NaN : 0;
      if (mode != 0 && mode != 1)
        for (long j = 0; j < g.keyLength(); j++)
          if (eligible(g, b, causal, x, j))
            out +=
                scratch.get(ValueLayout.JAVA_FLOAT_UNALIGNED, j * 4)
                    * (float) read(b.get(role(g, 2)), g.valueType(), x.v() + j1(g.value(), j, d));
      write(b.get(g.uniqueInputCount()), g.resultType(), x.o() + j0(g.output(), d), out);
    }
  }

  private static void row64(
      Geometry g, List<CpuBufferArgument> b, MemorySegment scratch, boolean causal, long row) {
    Bases x = bases(g, row);
    long eligible = 0, positive = 0;
    boolean nan = false, allNeg = true;
    for (long j = 0; j < g.keyLength(); j++)
      if (eligible(g, b, causal, x, j)) {
        double score = 0;
        for (long e = 0; e < g.embedding(); e++)
          score +=
              read(b.get(role(g, 0)), g.queryType(), x.q() + j0(g.query(), e))
                  * read(b.get(role(g, 1)), g.keyType(), x.k() + j1(g.key(), j, e));
        score *= g.scale();
        scratch.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8, score);
        eligible++;
        nan |= Double.isNaN(score);
        if (score == Double.POSITIVE_INFINITY) positive++;
        if (score != Double.NEGATIVE_INFINITY) allNeg = false;
      }
    int mode = eligible == 0 || allNeg ? 0 : nan ? 1 : positive > 0 ? 2 : 3;
    if (mode == 3) {
      double max = Double.NEGATIVE_INFINITY;
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j))
          max = Math.max(max, scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8));
      double sum = 0;
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j)) {
          double w = StrictMath.exp(scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8) - max);
          scratch.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8, w);
          sum += w;
        }
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j))
          scratch.set(
              ValueLayout.JAVA_DOUBLE_UNALIGNED,
              j * 8,
              scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8) / sum);
    } else
      for (long j = 0; j < g.keyLength(); j++)
        if (eligible(g, b, causal, x, j)) {
          double score = scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8);
          scratch.set(
              ValueLayout.JAVA_DOUBLE_UNALIGNED,
              j * 8,
              mode == 1
                  ? Double.NaN
                  : mode == 2 && score == Double.POSITIVE_INFINITY ? 1d / positive : 0d);
        }
    if (g.outputCount() == 2)
      for (long j = 0; j < g.keyLength(); j++)
        write(
            b.get(g.uniqueInputCount() + 1),
            g.resultType(),
            x.w() + j0(g.weights().orElseThrow(), j),
            eligible(g, b, causal, x, j)
                ? scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8)
                : 0d);
    for (long d = 0; d < g.valueEmbedding(); d++) {
      double out = mode == 1 ? Double.NaN : 0;
      if (mode != 0 && mode != 1)
        for (long j = 0; j < g.keyLength(); j++)
          if (eligible(g, b, causal, x, j))
            out +=
                scratch.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, j * 8)
                    * read(b.get(role(g, 2)), g.valueType(), x.v() + j1(g.value(), j, d));
      write(b.get(g.uniqueInputCount()), g.resultType(), x.o() + j0(g.output(), d), out);
    }
  }

  private static Bases bases(Geometry g, long row) {
    long qi = row % g.queryLength(), batch = row / g.queryLength();
    long q = base(g.query().orElseThrow(), g, batch),
        k = base(g.key().orElseThrow(), g, batch),
        v = base(g.value().orElseThrow(), g, batch),
        m = g.mask().map(l -> base(l, g, batch)).orElse(0L),
        o = base(g.output(), g, batch),
        w = g.weights().map(l -> base(l, g, batch)).orElse(0L);
    return new Bases(
        q + qi * stride(g.query(), -2),
        k,
        v,
        m + qi * stride(g.mask(), -2),
        o + qi * stride(g.output(), -2),
        w + qi * stride(g.weights(), -2),
        qi);
  }

  private static long base(NormalizedLayout l, Geometry g, long ordinal) {
    long a = l.offset();
    long[] e = g.batchExtents(), s = l.strides();
    for (int axis = e.length - 1; axis >= 0; axis--) {
      long c = ordinal % e[axis];
      ordinal /= e[axis];
      a += c * s[axis];
    }
    return a;
  }

  private static boolean eligible(
      Geometry g, List<CpuBufferArgument> b, boolean causal, Bases x, long j) {
    return (!causal || j <= x.qi())
        && (g.mask().isEmpty()
            || read(b.get(role(g, 3)), DataType.BOOL, x.m() + j0(g.mask().orElseThrow(), j)) != 0);
  }

  private static int role(Geometry g, int role) {
    return g.roleBoundaryPositions().get(role);
  }

  private static long j0(NormalizedLayout l, long x) {
    long[] s = l.strides();
    return x * s[s.length - 1];
  }

  private static long j0(java.util.Optional<NormalizedLayout> l, long x) {
    return l.isEmpty() ? 0 : j0(l.orElseThrow(), x);
  }

  private static long j1(java.util.Optional<NormalizedLayout> l, long a, long z) {
    long[] s = l.orElseThrow().strides();
    return a * s[s.length - 2] + z * s[s.length - 1];
  }

  private static long stride(java.util.Optional<NormalizedLayout> l, int axis) {
    if (l.isEmpty()) return 0;
    long[] s = l.orElseThrow().strides();
    return s[s.length + axis];
  }

  private static long stride(NormalizedLayout l, int axis) {
    long[] s = l.strides();
    return s[s.length + axis];
  }

  private static double read(CpuBufferArgument a, DataType t, long p) {
    long byteAddress = Math.multiplyExact(p, t.byteWidth());
    if (a instanceof CpuBufferArgument.Segment s)
      return switch (t) {
        case FLOAT64 -> s.segment().get(ValueLayout.JAVA_DOUBLE_UNALIGNED, byteAddress);
        case FLOAT32 -> s.segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, byteAddress);
        case BFLOAT16 ->
            BFloat16Bits.toFloat(s.segment().get(ValueLayout.JAVA_SHORT_UNALIGNED, byteAddress));
        case BOOL -> s.segment().get(ValueLayout.JAVA_BYTE, byteAddress);
        default -> throw new AssertionError(t);
      };
    long base = a.byteOffset() / t.byteWidth() + p;
    int i = Math.toIntExact(base);
    return switch (t) {
      case FLOAT64 -> ((CpuBufferArgument.Doubles) a).carrier()[i];
      case FLOAT32 -> ((CpuBufferArgument.Floats) a).carrier()[i];
      case BFLOAT16 -> BFloat16Bits.toFloat(((CpuBufferArgument.Shorts) a).carrier()[i]);
      case BOOL -> ((CpuBufferArgument.Bytes) a).carrier()[i];
      default -> throw new AssertionError(t);
    };
  }

  private static void write(CpuBufferArgument a, DataType t, long p, double value) {
    long byteAddress = Math.multiplyExact(p, t.byteWidth());
    if (a instanceof CpuBufferArgument.Segment s) {
      switch (t) {
        case FLOAT64 -> s.segment().set(ValueLayout.JAVA_DOUBLE_UNALIGNED, byteAddress, value);
        case FLOAT32 ->
            s.segment().set(ValueLayout.JAVA_FLOAT_UNALIGNED, byteAddress, (float) value);
        case BFLOAT16 ->
            s.segment()
                .set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    byteAddress,
                    BFloat16Bits.fromFloat((float) value));
        default -> throw new AssertionError(t);
      }
      return;
    }
    int i = Math.toIntExact(a.byteOffset() / t.byteWidth() + p);
    switch (t) {
      case FLOAT64 -> ((CpuBufferArgument.Doubles) a).carrier()[i] = value;
      case FLOAT32 -> ((CpuBufferArgument.Floats) a).carrier()[i] = (float) value;
      case BFLOAT16 ->
          ((CpuBufferArgument.Shorts) a).carrier()[i] = BFloat16Bits.fromFloat((float) value);
      default -> throw new AssertionError(t);
    }
  }

  private record Bases(long q, long k, long v, long m, long o, long w, long qi) {}
}
