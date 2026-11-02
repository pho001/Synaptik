package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuAttentionMaskValidatorTest {
  @Test
  void temporalMaskAndBoolRangesResetPrivateCursorAndSkipInactiveBytes() {
    byte[] mask = {9, 9, 0, 9, 1, 9, 2, 9, 1};
    byte[] active = {1, 0};
    long[] extents = {2, 2};
    long[] strides = {4, 2};
    long[] cursor = {71, 71};
    MemorySegment storage = MemorySegment.ofArray(mask);
    MemorySegment activity = MemorySegment.ofArray(active);
    Runnable scan = () -> CpuAttentionMaskValidator.validateTemporal(storage, 2L,
        extents, strides, cursor, 0L, 4L, 0L, true, activity, 0L, 2L, 2L);
    assertDoesNotThrow(scan::run);
    active[1] = 1;
    assertThrows(IllegalArgumentException.class, scan::run);
    mask[6] = 0;
    assertDoesNotThrow(scan::run);
    mask[6] = 2;
    active[1] = 0;
    assertDoesNotThrow(scan::run);
    assertDoesNotThrow(() -> CpuBooleanInputValidator.validateTemporal(storage, 4L,
        extents, strides, cursor, 1L, 4L, 0L, true, activity, 0L, 2L, 2L, "canonical"));
    assertDoesNotThrow(() -> CpuAttentionMaskValidator.validateTemporal(null, 0L,
        null, null, null, 0L, 1L, 0L, false, null, 0L, 0L, 0L));
    assertDoesNotThrow(() -> CpuAttentionMaskValidator.validateTemporal(null, 0L,
        new long[] {0}, new long[] {1}, new long[1],
        0L, 0L, 0L, true, null, 0L, 0L, 0L));
  }

  @Test
  void validatesEachDistinctStridedLogicalByteForArraysAndSegments() {
    var plan =
        new CpuAccessPlan(
            CpuAccessPlan.AccessKind.READ,
            CpuAccessPlan.Regime.GENERAL_ODOMETER,
            2,
            List.of(CpuAccessPlan.AxisRole.STRIDED, CpuAccessPlan.AxisRole.BROADCAST),
            0);
    var binding =
        CpuAccessPlan.Binding.create(plan, new long[] {2, 1}, 1, new long[] {2, 0}, 2, 0, 2, 4);
    byte[] valid = {9, 1, 9, 0, 9};
    assertDoesNotThrow(
        () ->
            CpuAttentionMaskValidator.validate(
                new CpuBufferArgument.Bytes(valid, 0, valid.length, true), binding));
    valid[3] = 2;
    assertThrows(
        IllegalArgumentException.class,
        () ->
            CpuAttentionMaskValidator.validate(
                new CpuBufferArgument.Bytes(valid, 0, valid.length, true), binding));
    try (Arena arena = Arena.ofConfined()) {
      var segment = arena.allocate(5);
      segment.set(ValueLayout.JAVA_BYTE, 1, (byte) 1);
      segment.set(ValueLayout.JAVA_BYTE, 3, (byte) -1);
      assertThrows(
          IllegalArgumentException.class,
          () ->
              CpuAttentionMaskValidator.validate(
                  new CpuBufferArgument.Segment(
                      io.github.pho001.synaptik.model.datatype.DataType.BOOL, segment, 5, true),
                  binding));
    }
  }
}
