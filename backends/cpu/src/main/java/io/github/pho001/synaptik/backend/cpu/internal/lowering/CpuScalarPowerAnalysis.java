package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/**
 * Classifies one already-validated exact floating scalar exponent into a CPU realization plan.
 *
 * <p>The classification compares the represented BFLOAT16, FLOAT32, or FLOAT64 bits directly.
 * Both signed
 * zeros select {@link CpuKernelIr.PowerRealization#POSITIVE_ONE}; the direct power contract
 * returns positive one for every base, including NaN, signed zero, and infinities. Exact positive
 * one selects {@code IDENTITY}. Exact positive two selects {@code SQUARE}, and exact negative one
 * selects {@code RECIPROCAL}. BFLOAT16 realizations decode the represented base, perform the
 * FLOAT32-domain operation, and apply the required BFLOAT16 result encoding; identity therefore
 * preserves signed zero and infinity but canonicalizes a NaN at the producing operation boundary.
 * FLOAT32 and FLOAT64 square and reciprocal use one primitive operation in their result type.</p>
 *
 * <p>Every other bit pattern, including fractional and other integral values, infinities, and all
 * NaN payloads, selects {@code DIRECT}. Multi-step multiplication or reciprocal chains would add
 * rounded intermediate results and therefore lack a universal finite-rounding, overflow,
 * underflow, and subnormal-transition proof. This stateless analysis neither reads Tensor storage
 * nor inspects graphs, routes, configuration, or generated code.</p>
 */
public final class CpuScalarPowerAnalysis {
    private static final long BFLOAT16_SIGN = 0x8000L;
    private static final long BFLOAT16_ONE = 0x3f80L;
    private static final long BFLOAT16_TWO = 0x4000L;
    private static final long BFLOAT16_NEGATIVE_ONE = 0xbf80L;
    private static final long FLOAT32_SIGN = 0x8000_0000L;
    private static final long FLOAT32_ONE = 0x3f80_0000L;
    private static final long FLOAT32_TWO = 0x4000_0000L;
    private static final long FLOAT32_NEGATIVE_ONE = 0xbf80_0000L;
    private static final long FLOAT64_SIGN = 0x8000_0000_0000_0000L;
    private static final long FLOAT64_ONE = 0x3ff0_0000_0000_0000L;
    private static final long FLOAT64_TWO = 0x4000_0000_0000_0000L;
    private static final long FLOAT64_NEGATIVE_ONE = 0xbff0_0000_0000_0000L;

    /** Creates a stateless exact-bit classifier. */
    public CpuScalarPowerAnalysis() {
    }

    /**
     * Selects the unique proved realization for one exact floating scalar immediate.
     *
     * @param exponent non-null exact BFLOAT16, FLOAT32, or FLOAT64 exponent retained by canonical
     *     CPU IR
     * @return the non-null deterministic realization selected from the exact represented bits
     * @throws NullPointerException if {@code exponent} is {@code null}
     * @throws IllegalArgumentException if the exponent type is not BFLOAT16, FLOAT32, or FLOAT64
     */
    public CpuKernelIr.PowerRealization analyze(CpuKernelIr.ScalarImmediate exponent) {
        Objects.requireNonNull(exponent, "exponent");
        long bits = exponent.bits();
        if (exponent.dataType() == DataType.BFLOAT16) {
            bits &= 0xffffL;
            if (bits == 0L || bits == BFLOAT16_SIGN) {
                return CpuKernelIr.PowerRealization.POSITIVE_ONE;
            }
            if (bits == BFLOAT16_ONE) return CpuKernelIr.PowerRealization.IDENTITY;
            if (bits == BFLOAT16_TWO) return CpuKernelIr.PowerRealization.SQUARE;
            if (bits == BFLOAT16_NEGATIVE_ONE) return CpuKernelIr.PowerRealization.RECIPROCAL;
            return CpuKernelIr.PowerRealization.DIRECT;
        }
        if (exponent.dataType() == DataType.FLOAT32) {
            bits &= 0xffff_ffffL;
            if (bits == 0L || bits == FLOAT32_SIGN) {
                return CpuKernelIr.PowerRealization.POSITIVE_ONE;
            }
            if (bits == FLOAT32_ONE) return CpuKernelIr.PowerRealization.IDENTITY;
            if (bits == FLOAT32_TWO) return CpuKernelIr.PowerRealization.SQUARE;
            if (bits == FLOAT32_NEGATIVE_ONE) return CpuKernelIr.PowerRealization.RECIPROCAL;
            return CpuKernelIr.PowerRealization.DIRECT;
        }
        if (exponent.dataType() == DataType.FLOAT64) {
            if (bits == 0L || bits == FLOAT64_SIGN) {
                return CpuKernelIr.PowerRealization.POSITIVE_ONE;
            }
            if (bits == FLOAT64_ONE) return CpuKernelIr.PowerRealization.IDENTITY;
            if (bits == FLOAT64_TWO) return CpuKernelIr.PowerRealization.SQUARE;
            if (bits == FLOAT64_NEGATIVE_ONE) return CpuKernelIr.PowerRealization.RECIPROCAL;
            return CpuKernelIr.PowerRealization.DIRECT;
        }
        throw new IllegalArgumentException(
                "scalar power exponent must be BFLOAT16, FLOAT32, or FLOAT64");
    }
}
