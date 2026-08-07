package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/**
 * Classifies one already-validated exact floating scalar exponent into a CPU realization plan.
 *
 * <p>The classification compares the represented FLOAT32 or FLOAT64 bits directly. Both signed
 * zeros select {@link CpuKernelIr.PowerRealization#POSITIVE_ONE}; the direct power contract
 * returns positive one for every base, including NaN, signed zero, and infinities. Exact positive
 * one selects {@code IDENTITY}, which preserves every represented base classification and zero
 * sign. Exact positive two selects {@code SQUARE}; one primitive typed multiplication is the one
 * correctly rounded operation for the mathematical square. Exact negative one selects
 * {@code RECIPROCAL}; one primitive typed division of positive one by the base is the one
 * correctly rounded reciprocal and preserves the required zero and infinity signs.</p>
 *
 * <p>Every other bit pattern, including fractional and other integral values, infinities, and all
 * NaN payloads, selects {@code DIRECT}. Multi-step multiplication or reciprocal chains would add
 * rounded intermediate results and therefore lack a universal finite-rounding, overflow,
 * underflow, and subnormal-transition proof. This stateless analysis neither reads Tensor storage
 * nor inspects graphs, routes, configuration, or generated code.</p>
 */
public final class CpuScalarPowerAnalysis {
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
     * @param exponent non-null exact FLOAT32 or FLOAT64 exponent retained by canonical CPU IR
     * @return the non-null deterministic realization selected from the exact represented bits
     * @throws NullPointerException if {@code exponent} is {@code null}
     * @throws IllegalArgumentException if the exponent type is not FLOAT32 or FLOAT64
     */
    public CpuKernelIr.PowerRealization analyze(CpuKernelIr.ScalarImmediate exponent) {
        Objects.requireNonNull(exponent, "exponent");
        long bits = exponent.bits();
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
        throw new IllegalArgumentException("scalar power exponent must be FLOAT32 or FLOAT64");
    }
}
