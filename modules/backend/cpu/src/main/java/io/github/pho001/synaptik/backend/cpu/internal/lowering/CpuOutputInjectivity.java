package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import java.util.Arrays;
import java.util.Objects;

/**
 * Shared cold proof that a resolved non-negative CPU output layout has distinct logical writes.
 * The rank-sized proof combines separated address spans with disjoint stride residue classes.
 * Layouts outside that sufficient proof retain an exact, bounded primitive-address fallback;
 * larger unproved layouts fail closed. No proof allocates in proportion to an unbounded domain.
 * Storage offsets are deliberately absent: translating every address preserves injectivity.
 * Callers independently validate offsets, referenced spans, carriers and physical alias policies.
 */
public final class CpuOutputInjectivity {
    private static final int EXACT_ELEMENT_LIMIT = 1_000_000;

    private CpuOutputInjectivity() { }

    /**
     * Proves unique addresses for coordinates in the supplied rectangular logical domain.
     * Singleton strides contribute no addresses, and an empty domain is vacuously injective.
     * For an already proved offset set with stride gcd G, adding stride S and extent E is safe
     * when S exceeds its complete address span, or E is at most G/gcd(G,S): all nonzero shifts
     * then occupy different residues modulo G, even when their address intervals overlap.
     * A false result means either actual overlap or a large layout outside the sufficient proof.
     *
     * @param extents non-null non-negative logical extents; borrowed and not modified
     * @param strides non-null non-negative element strides, one per extent; not modified
     * @return true only when every logical coordinate selects a distinct relative address
     * @throws NullPointerException if either array is null
     * @throws IllegalArgumentException if ranks differ or an extent or stride is negative
     * @throws ArithmeticException if the nonempty logical count or relative span overflows long
     */
    public static boolean isInjective(long[] extents, long[] strides) {
        Objects.requireNonNull(extents, "extents");
        Objects.requireNonNull(strides, "strides");
        if (extents.length != strides.length) throw new IllegalArgumentException("layout ranks differ");
        boolean empty = false;
        for (int axis = 0; axis < extents.length; axis++) {
            if (extents[axis] < 0 || strides[axis] < 0) throw new IllegalArgumentException(
                    "injectivity requires non-negative extents and strides");
            empty |= extents[axis] == 0;
        }
        if (empty) return true;

        long count = 1;
        long span = 0;
        int[] axes = new int[extents.length];
        int active = 0;
        for (int axis = 0; axis < extents.length; axis++) {
            count = Math.multiplyExact(count, extents[axis]);
            span = Math.addExact(span, Math.multiplyExact(extents[axis] - 1, strides[axis]));
            if (extents[axis] <= 1) continue;
            int insertion = active;
            while (insertion > 0 && strides[axes[insertion - 1]] > strides[axis]) {
                axes[insertion] = axes[insertion - 1];
                insertion--;
            }
            axes[insertion] = axis;
            active++;
        }
        Math.addExact(span, 1);

        long coveredSpan = 0;
        long strideGcd = 0;
        boolean proved = true;
        for (int index = 0; index < active; index++) {
            int axis = axes[index];
            long stride = strides[axis];
            if (stride == 0) return false;
            long nextGcd = gcd(strideGcd, stride);
            if (stride <= coveredSpan && extents[axis] > strideGcd / nextGcd) {
                proved = false;
                break;
            }
            coveredSpan = Math.addExact(coveredSpan,
                    Math.multiplyExact(extents[axis] - 1, stride));
            strideGcd = nextGcd;
        }
        if (proved) return true;
        if (count > EXACT_ELEMENT_LIMIT) return false;

        long[] addresses = new long[(int) count];
        long[] coordinates = new long[extents.length];
        for (int ordinal = 0; ordinal < addresses.length; ordinal++) {
            long address = 0;
            for (int axis = 0; axis < extents.length; axis++) {
                address += coordinates[axis] * strides[axis];
            }
            addresses[ordinal] = address;
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                if (++coordinates[axis] < extents[axis]) break;
                coordinates[axis] = 0;
            }
        }
        Arrays.sort(addresses);
        for (int index = 1; index < addresses.length; index++) {
            if (addresses[index - 1] == addresses[index]) return false;
        }
        return true;
    }

    private static long gcd(long left, long right) {
        while (right != 0) {
            long remainder = left % right;
            left = right;
            right = remainder;
        }
        return left;
    }
}
