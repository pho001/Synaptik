package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Cold canonical-byte validator for one distinct logical attention-mask domain.
 *
 * <p>Binding invokes this validator before generated work or output writes so a BOOL mask is
 * checked once per represented logical coordinate, rather than once per broadcast use.
 */
public final class CpuAttentionMaskValidator {
  private CpuAttentionMaskValidator() {}

  /**
   * Validates each logical mask coordinate exactly once in row-major order.
   *
   * @param argument non-null byte-array or segment mask carrier
   * @param binding non-null complete logical mask binding
   * @throws NullPointerException if an argument is {@code null}
   * @throws IllegalArgumentException if a represented byte is neither {@code 0} nor {@code 1}
   * @throws ArithmeticException if address traversal overflows
   */
  public static void validate(CpuBufferArgument argument, CpuAccessPlan.Binding binding) {
    Objects.requireNonNull(argument, "argument");
    Objects.requireNonNull(binding, "binding");
    long[] extents = binding.extents().stream().mapToLong(Long::longValue).toArray();
    long[] strides = binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
    long[] coordinates = new long[extents.length];
    long address = binding.baseElementOffset();
    for (long logical = 0; logical < binding.elementCount(); logical++) {
      byte value = read(argument, address);
      if (value != 0 && value != 1)
        throw new IllegalArgumentException("attention BOOL mask must use canonical bytes");
      for (int axis = coordinates.length - 1; axis >= 0; axis--) {
        coordinates[axis]++;
        address = Math.addExact(address, strides[axis]);
        if (coordinates[axis] < extents[axis]) break;
        coordinates[axis] = 0;
        address = Math.subtractExact(address, Math.multiplyExact(extents[axis], strides[axis]));
      }
    }
  }

  private static byte read(CpuBufferArgument argument, long address) {
    if (argument instanceof CpuBufferArgument.Bytes bytes)
      return bytes.carrier()[Math.toIntExact(bytes.byteOffset() + address)];
    if (argument instanceof CpuBufferArgument.Segment segment)
      return segment.segment().get(ValueLayout.JAVA_BYTE, address);
    throw new IllegalArgumentException("attention mask requires a byte carrier");
  }
}
