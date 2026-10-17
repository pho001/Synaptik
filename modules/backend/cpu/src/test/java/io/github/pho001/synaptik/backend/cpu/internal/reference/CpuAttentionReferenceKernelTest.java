package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering.Geometry;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering.NormalizedLayout;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuAttentionReferenceKernelTest {
  @Test
  void finiteRowsUseStableSequentialFloatArithmetic() {
    float[] q = {1, 0, 0, 1},
        k = {1, 0, 0, 1},
        v = {2, 4, 6, 8},
        o = new float[4],
        w = new float[4];
    Geometry g = geometry(2, 2, 2, 2, false, 2);
    try (Arena a = Arena.ofConfined()) {
      CpuAttentionReferenceKernel.evaluate(
          g, List.of(f(q), f(k), f(v), f(o), f(w)), a.allocate(8, 8), false, 0, 2);
    }
    float p = (float) (Math.exp(1) / (Math.exp(1) + 1));
    assertArrayEquals(new float[] {p, 1 - p, 1 - p, p}, w, 1e-6f);
    assertArrayEquals(
        new float[] {
          2 * p + 6 * (1 - p), 4 * p + 8 * (1 - p), 2 * (1 - p) + 6 * p, 4 * (1 - p) + 8 * p
        },
        o,
        1e-5f);
  }

  @Test
  void excludedRowsAvoidNumericalReadsAndPositiveInfinityTiesSplit() {
    byte[] mask = {0, 0, 0};
    float[] zeroOut = {-1}, zeroWeights = {-1, -1, -1};
    Geometry excluded = geometry(1, 3, 1, 1, true, 2);
    try (Arena a = Arena.ofConfined()) {
      CpuAttentionReferenceKernel.evaluate(
          excluded,
          List.of(
              f(new float[0]),
              f(new float[0]),
              f(new float[0]),
              bytes(mask),
              f(zeroOut),
              f(zeroWeights)),
          a.allocate(16, 8),
          false,
          0,
          1);
    }
    assertEquals(0f, zeroOut[0]);
    assertArrayEquals(new float[3], zeroWeights);

    float[] out = {0}, weights = new float[3];
    Geometry ties = geometry(1, 3, 1, 1, false, 2);
    try (Arena a = Arena.ofConfined()) {
      CpuAttentionReferenceKernel.evaluate(
          ties,
          List.of(
              f(new float[] {Float.POSITIVE_INFINITY}),
              f(new float[] {1, 1, -1}),
              f(new float[] {2, 6, Float.NaN}),
              f(out),
              f(weights)),
          a.allocate(16, 8),
          false,
          0,
          1);
    }
    assertArrayEquals(new float[] {.5f, .5f, 0}, weights);
    assertTrue(Float.isNaN(out[0]));
  }

  private static Geometry geometry(long l, long s, long e, long ev, boolean masked, int outputs) {
    var q = new NormalizedLayout(0, new long[] {e, 1});
    var k = new NormalizedLayout(0, new long[] {e, 1});
    var v = new NormalizedLayout(0, new long[] {ev, 1});
    var m = new NormalizedLayout(0, new long[] {s, 1});
    var o = new NormalizedLayout(0, new long[] {ev, 1});
    var w = new NormalizedLayout(0, new long[] {s, 1});
    return new Geometry(
        new long[0],
        l,
        s,
        e,
        ev,
        l,
        s == 0 ? 0 : ((s * 4 + 7) & -8L),
        DataType.FLOAT32,
        DataType.FLOAT32,
        DataType.FLOAT32,
        DataType.FLOAT32,
        1,
        List.of(0, 1, 2, 3).subList(0, masked ? 4 : 3),
        masked ? 4 : 3,
        outputs,
        Optional.of(q),
        Optional.of(k),
        Optional.of(v),
        masked ? Optional.of(m) : Optional.empty(),
        o,
        outputs == 2 ? Optional.of(w) : Optional.empty());
  }

  private static CpuBufferArgument f(float[] x) {
    return new CpuBufferArgument.Floats(x, 0, (long) x.length * 4, false);
  }

  private static CpuBufferArgument bytes(byte[] x) {
    return new CpuBufferArgument.Bytes(x, 0, x.length, true);
  }
}
