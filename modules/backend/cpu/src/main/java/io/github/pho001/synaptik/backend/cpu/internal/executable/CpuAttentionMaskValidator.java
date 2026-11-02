package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.MemorySegment;

/**
 * Cold-bound canonical-byte validator for one distinct logical attention-mask domain.
 *
 * <p>Binding retains the distinct mask domain without reading its values. Execution scans after
 * producers and before generated work or output writes, once per logical coordinate rather than
 * once per broadcast use.
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
    bind(argument, binding).run();
  }

  /**
   * Binds a sequential allocation-free mask scan without reading producer-owned values.
   *
   * @param argument non-null already carrier-validated borrowed BOOL storage
   * @param binding non-null complete distinct logical mask domain
   * @return non-null action that rejects noncanonical bytes before attention writes or submission
   * @throws NullPointerException if an argument is null
   * @throws IllegalArgumentException if the carrier is not byte-addressable BOOL storage
   */
  public static Runnable bind(CpuBufferArgument argument, CpuAccessPlan.Binding binding) {
    return CpuBooleanInputValidator.bind(argument, binding,
        "attention BOOL mask must use canonical bytes");
  }

  /**
   * Checks the entire eligible distinct mask domain before any attention numerical range.
   * This forwards to the ordinary canonical BOOL loop without allocations or broadcast rereads.
   *
   * @param storage borrowed prebound mask storage
   * @param startAddress byte address of logical start before temporal displacement
   * @param extents immutable cold distinct-mask extents
   * @param strides immutable cold mask byte strides
   * @param coordinates invocation-private rank-sized cursor, reset on every call
   * @param start inclusive distinct-mask ordinal
   * @param end exclusive ordinal for the entire unit validation domain
   * @param temporalOffset additional mask byte displacement for this step
   * @param ownerPresent structural occurrence presence, independent of empty domains
   * @param activity borrowed row bytes, or null for all-active
   * @param activityOffset byte address of row zero
   * @param rowStride mask ordinals per original row coordinate, or zero for a shared byte
   * @param rowExtent proved row extent, unused for a shared byte
   * @throws IllegalArgumentException if an eligible represented byte is not zero or one
   * @throws ArithmeticException if checked address arithmetic overflows
   */
  public static void validateTemporal(MemorySegment storage, long startAddress,
      long[] extents, long[] strides, long[] coordinates, long start, long end,
      long temporalOffset, boolean ownerPresent, MemorySegment activity,
      long activityOffset, long rowStride, long rowExtent) {
    CpuBooleanInputValidator.validateTemporal(storage, startAddress, extents, strides, coordinates,
        start, end, temporalOffset, ownerPresent, activity, activityOffset, rowStride, rowExtent,
        "attention BOOL mask must use canonical bytes");
  }
}
