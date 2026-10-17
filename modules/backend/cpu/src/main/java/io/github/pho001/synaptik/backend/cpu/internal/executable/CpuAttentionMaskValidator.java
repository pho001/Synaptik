package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;

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
}
