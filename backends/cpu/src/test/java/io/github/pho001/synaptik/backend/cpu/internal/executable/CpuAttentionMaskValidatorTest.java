package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuAttentionMaskValidatorTest {
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
