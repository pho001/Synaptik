package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import java.lang.foreign.MemorySegment;
import static java.lang.foreign.ValueLayout.*;

/**
 * Direct scalar F32 oracle with fixed feature/hidden facts and invocation-only B.
 * Exact reductions use the same signed five-limb, single-rounding algorithm and 48-byte state
 * as generated execution, not a float or double accumulator. No Model/Runtime work is timed.
 */
final class CpuOrdinaryDynamicBatchCleanJavaOracle {
    private CpuOrdinaryDynamicBatchCleanJavaOracle() {}

    static void matmul4x3(MemorySegment x, MemorySegment w, MemorySegment y, int b) {
        for (int row = 0; row < b; row++) for (int col = 0; col < 3; col++) {
            int leftAddress = row * 4, rightAddress = col;
            float sum = 0;
            for (int term = 0; term < 4; term++) {
                sum += x.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * w.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 1; rightAddress += 3;
            }
            y.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 3 + col), sum);
        }
    }

    static void dx4x3(MemorySegment dy, MemorySegment w, MemorySegment dx, int b) {
        for (int row = 0; row < b; row++) for (int col = 0; col < 4; col++) {
            int leftAddress = row * 3, rightAddress = col * 3;
            float sum = 0;
            for (int term = 0; term < 3; term++) {
                sum += dy.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * w.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 1; rightAddress += 1;
            }
            dx.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 4 + col), sum);
        }
    }

    static void dw4x3(MemorySegment x, MemorySegment dy, MemorySegment dw, int b) {
        for (int row = 0; row < 4; row++) for (int col = 0; col < 3; col++) {
            int leftAddress = row, rightAddress = col;
            float sum = 0;
            for (int term = 0; term < b; term++) {
                sum += x.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * dy.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 4; rightAddress += 3;
            }
            dw.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 3 + col), sum);
        }
    }

    static void matmul128x128(MemorySegment x, MemorySegment w, MemorySegment y, int b) {
        for (int row = 0; row < b; row++) for (int col = 0; col < 128; col++) {
            int leftAddress = row * 128, rightAddress = col;
            float sum = 0;
            for (int term = 0; term < 128; term++) {
                sum += x.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * w.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 1; rightAddress += 128;
            }
            y.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 128 + col), sum);
        }
    }

    static void dx128x128(MemorySegment dy, MemorySegment w, MemorySegment dx, int b) {
        for (int row = 0; row < b; row++) for (int col = 0; col < 128; col++) {
            int leftAddress = row * 128, rightAddress = col * 128;
            float sum = 0;
            for (int term = 0; term < 128; term++) {
                sum += dy.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * w.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 1; rightAddress += 1;
            }
            dx.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 128 + col), sum);
        }
    }

    static void dw128x128(MemorySegment x, MemorySegment dy, MemorySegment dw, int b) {
        for (int row = 0; row < 128; row++) for (int col = 0; col < 128; col++) {
            int leftAddress = row, rightAddress = col;
            float sum = 0;
            for (int term = 0; term < b; term++) {
                sum += x.get(JAVA_FLOAT_UNALIGNED, 4L * leftAddress)
                        * dy.get(JAVA_FLOAT_UNALIGNED, 4L * rightAddress);
                leftAddress += 128; rightAddress += 128;
            }
            dw.set(JAVA_FLOAT_UNALIGNED, 4L * (row * 128 + col), sum);
        }
    }

    /** Each cell reads a fixed-stride contributor stream; caller supplies exact domain and cells. */
    static void reduce(MemorySegment input, MemorySegment output, MemorySegment scratch,
            int cells, int domain, int cellStep, int termStep, boolean mean) {
        for (int cell = 0; cell < cells; cell++) {
            for (int limb = 0; limb < 6; limb++) scratch.setAtIndex(JAVA_LONG, limb, 0);
            long flags = 0;
            int address = cell * cellStep;
            for (int term = 0; term < domain; term++, address += termStep) {
                int bits = Float.floatToRawIntBits(input.get(JAVA_FLOAT_UNALIGNED, 4L * (address)));
                boolean negative = bits < 0;
                int exponent = bits >>> 23 & 255;
                long fraction = bits & 0x7fffff;
                if (exponent == 255) { flags |= fraction != 0 ? 1 : negative ? 4 : 2; continue; }
                if (exponent == 0 && fraction == 0) { flags |= negative ? 16 : 8; continue; }
                flags |= 32;
                long significand = exponent == 0 ? fraction : fraction | 0x800000;
                int shift = exponent == 0 ? 0 : exponent - 1;
                int word = shift >>> 6, offset = shift & 63;
                long carry = negative ? 1 : 0;
                for (int limb = 0; limb < 5; limb++) {
                    long part = limb == word ? significand << offset
                            : offset != 0 && limb == word + 1 ? significand >>> (64 - offset) : 0;
                    if (negative) part = ~part;
                    long withCarry = part + carry;
                    carry = Long.compareUnsigned(withCarry, part) < 0 ? 1 : 0;
                    long old = scratch.getAtIndex(JAVA_LONG, limb + 1);
                    long added = old + withCarry;
                    if (Long.compareUnsigned(added, old) < 0) carry = 1;
                    scratch.setAtIndex(JAVA_LONG, limb + 1, added);
                }
            }
            scratch.set(JAVA_LONG, 0, flags);
            output.set(JAVA_FLOAT_UNALIGNED, 4L * (cell), finish(scratch, flags, domain, mean));
        }
    }

    // Once per output, never once per contributor. Signed magnitude/division/rounding mirrors
    // the inlined generated finish; JIT inlining is not an assumed performance guarantee.
    private static float finish(MemorySegment state, long flags, int domain, boolean mean) {
        if ((flags & 1) != 0 || (flags & 6) == 6) return Float.intBitsToFloat(0x7fc00000);
        if ((flags & 2) != 0) return Float.POSITIVE_INFINITY;
        if ((flags & 4) != 0) return Float.NEGATIVE_INFINITY;
        if (mean && domain == 0) return Float.intBitsToFloat(0x7fc00000);
        boolean negative = state.getAtIndex(JAVA_LONG, 5) < 0;
        if (negative) {
            long carry = 1;
            for (int limb = 1; limb <= 5; limb++) {
                long old = ~state.getAtIndex(JAVA_LONG, limb), sum = old + carry;
                if (Long.compareUnsigned(sum, old) >= 0) carry = 0;
                state.setAtIndex(JAVA_LONG, limb, sum);
            }
        }
        long remainder = 0;
        if (mean) for (int limb = 5; limb >= 1; limb--) {
            long word = state.getAtIndex(JAVA_LONG, limb), quotient = 0;
            for (int bit = 63; bit >= 0; bit--) {
                long next = (remainder << 1) | (word >>> bit & 1);
                if (Long.compareUnsigned(next, domain) >= 0) { next -= domain; quotient |= 1L << bit; }
                remainder = next;
            }
            state.setAtIndex(JAVA_LONG, limb, quotient);
        }
        int topWord = 5;
        while (topWord > 1 && state.getAtIndex(JAVA_LONG, topWord) == 0) topWord--;
        long top = state.getAtIndex(JAVA_LONG, topWord);
        if (top == 0 && remainder == 0) return Float.intBitsToFloat(flags == 16 ? 0x80000000 : 0);
        int bitLength = (topWord - 1) * 64 + 64 - Long.numberOfLeadingZeros(top);
        int unbiased = bitLength - 150, shift = Math.max(0, bitLength - 24);
        int word = shift >>> 6, offset = shift & 63;
        long quotient = state.getAtIndex(JAVA_LONG, word + 1) >>> offset;
        if (offset != 0 && word + 1 < 5) quotient |= state.getAtIndex(JAVA_LONG, word + 2) << (64 - offset);
        boolean guard = false, sticky = false;
        if (shift != 0) {
            int bit = shift - 1, guardWord = bit >>> 6, guardOffset = bit & 63;
            guard = (state.getAtIndex(JAVA_LONG, guardWord + 1) >>> guardOffset & 1) != 0;
            for (int limb = 0; limb < guardWord; limb++) sticky |= state.getAtIndex(JAVA_LONG, limb + 1) != 0;
            if (guardOffset != 0) sticky |= (state.getAtIndex(JAVA_LONG, guardWord + 1) & ((1L << guardOffset) - 1)) != 0;
            sticky |= remainder != 0;
        } else if (mean) {
            int comparison = Long.compareUnsigned(remainder << 1, domain);
            guard = comparison >= 0; sticky = comparison > 0;
        }
        if (guard && (sticky || (quotient & 1) != 0)) quotient++;
        int sign = negative ? 0x80000000 : 0;
        if (unbiased < -126) return Float.intBitsToFloat(sign | (int) quotient);
        if (quotient == 0x1000000) { quotient >>>= 1; unbiased++; }
        if (unbiased > 127) return Float.intBitsToFloat(sign | 0x7f800000);
        return Float.intBitsToFloat(sign | ((unbiased + 127) << 23) | (int) (quotient & 0x7fffff));
    }

    /** Typed 34-limb counterpart; exact contributor state and rational rounding never narrow. */
    /** Each cell reads a fixed-stride contributor stream; caller supplies exact domain and cells. */
    static void reduceFloat64(MemorySegment input, MemorySegment output, MemorySegment scratch,
            int cells, int domain, int cellStep, int termStep, boolean mean) {
        for (int cell = 0; cell < cells; cell++) {
            for (int limb = 0; limb < 35; limb++) scratch.setAtIndex(JAVA_LONG, limb, 0);
            long flags = 0;
            int address = cell * cellStep;
            for (int term = 0; term < domain; term++, address += termStep) {
                long bits = Double.doubleToRawLongBits(input.get(JAVA_DOUBLE_UNALIGNED, 8L * address));
                boolean negative = bits < 0;
                int exponent = (int)(bits >>> 52 & 2047);
                long fraction = bits & 0xfffffffffffffL;
                if (exponent == 2047) { flags |= fraction != 0 ? 1 : negative ? 4 : 2; continue; }
                if (exponent == 0 && fraction == 0) { flags |= negative ? 16 : 8; continue; }
                flags |= 32;
                long significand = exponent == 0 ? fraction : fraction | 0x10000000000000L;
                int shift = exponent == 0 ? 0 : exponent - 1;
                int word = shift >>> 6, offset = shift & 63;
                long carry = negative ? 1 : 0;
                for (int limb = 0; limb < 34; limb++) {
                    long part = limb == word ? significand << offset
                            : offset != 0 && limb == word + 1 ? significand >>> (64 - offset) : 0;
                    if (negative) part = ~part;
                    long withCarry = part + carry;
                    carry = Long.compareUnsigned(withCarry, part) < 0 ? 1 : 0;
                    long old = scratch.getAtIndex(JAVA_LONG, limb + 1);
                    long added = old + withCarry;
                    if (Long.compareUnsigned(added, old) < 0) carry = 1;
                    scratch.setAtIndex(JAVA_LONG, limb + 1, added);
                }
            }
            scratch.set(JAVA_LONG, 0, flags);
            output.set(JAVA_DOUBLE_UNALIGNED, 8L * cell, finish64(scratch, flags, domain, mean));
        }
    }

    // Once per output, never once per contributor. Signed magnitude/division/rounding mirrors
    // the inlined generated finish; JIT inlining is not an assumed performance guarantee.
    private static double finish64(MemorySegment state, long flags, int domain, boolean mean) {
        if ((flags & 1) != 0 || (flags & 6) == 6) return Double.longBitsToDouble(0x7ff8000000000000L);
        if ((flags & 2) != 0) return Double.POSITIVE_INFINITY;
        if ((flags & 4) != 0) return Double.NEGATIVE_INFINITY;
        if (mean && domain == 0) return Double.longBitsToDouble(0x7ff8000000000000L);
        boolean negative = state.getAtIndex(JAVA_LONG, 34) < 0;
        if (negative) {
            long carry = 1;
            for (int limb = 1; limb <= 34; limb++) {
                long old = ~state.getAtIndex(JAVA_LONG, limb), sum = old + carry;
                if (Long.compareUnsigned(sum, old) >= 0) carry = 0;
                state.setAtIndex(JAVA_LONG, limb, sum);
            }
        }
        long remainder = 0;
        if (mean) for (int limb = 34; limb >= 1; limb--) {
            long word = state.getAtIndex(JAVA_LONG, limb), quotient = 0;
            for (int bit = 63; bit >= 0; bit--) {
                long next = (remainder << 1) | (word >>> bit & 1);
                if (Long.compareUnsigned(next, domain) >= 0) { next -= domain; quotient |= 1L << bit; }
                remainder = next;
            }
            state.setAtIndex(JAVA_LONG, limb, quotient);
        }
        int topWord = 34;
        while (topWord > 1 && state.getAtIndex(JAVA_LONG, topWord) == 0) topWord--;
        long top = state.getAtIndex(JAVA_LONG, topWord);
        if (top == 0 && remainder == 0) return Double.longBitsToDouble(flags == 16 ? Long.MIN_VALUE : 0);
        int bitLength = (topWord - 1) * 64 + 64 - Long.numberOfLeadingZeros(top);
        int unbiased = bitLength - 1075, shift = Math.max(0, bitLength - 53);
        int word = shift >>> 6, offset = shift & 63;
        long quotient = state.getAtIndex(JAVA_LONG, word + 1) >>> offset;
        if (offset != 0 && word + 1 < 34) quotient |= state.getAtIndex(JAVA_LONG, word + 2) << (64 - offset);
        boolean guard = false, sticky = false;
        if (shift != 0) {
            int bit = shift - 1, guardWord = bit >>> 6, guardOffset = bit & 63;
            guard = (state.getAtIndex(JAVA_LONG, guardWord + 1) >>> guardOffset & 1) != 0;
            for (int limb = 0; limb < guardWord; limb++) sticky |= state.getAtIndex(JAVA_LONG, limb + 1) != 0;
            if (guardOffset != 0) sticky |= (state.getAtIndex(JAVA_LONG, guardWord + 1) & ((1L << guardOffset) - 1)) != 0;
            sticky |= remainder != 0;
        } else if (mean) {
            int comparison = Long.compareUnsigned(remainder << 1, domain);
            guard = comparison >= 0; sticky = comparison > 0;
        }
        if (guard && (sticky || (quotient & 1) != 0)) quotient++;
        long sign = negative ? Long.MIN_VALUE : 0;
        if (unbiased < -1022) return Double.longBitsToDouble(sign | quotient);
        if (quotient == 0x20000000000000L) { quotient >>>= 1; unbiased++; }
        if (unbiased > 1023) return Double.longBitsToDouble(sign | 0x7ff0000000000000L);
        return Double.longBitsToDouble(sign | ((long)(unbiased + 1023) << 52) | (quotient & 0xfffffffffffffL));
    }
}

