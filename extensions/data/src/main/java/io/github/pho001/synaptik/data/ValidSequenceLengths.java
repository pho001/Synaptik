package io.github.pho001.synaptik.data;

import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable host metadata declaring one non-negative position-independent valid count per original batch row.
 *
 * <p>A valid count counts logical time steps, including ordinary zero-valued data. It does not
 * identify a padding sentinel or prove tensor contents. This value stores no time extent; a
 * {@link PaddedSequenceBatch} checks upper bounds against its borrowed tensor's shape.
 * Construction scans only the copied host lengths in O(B) time for B rows, not tensor values.
 * Instances may be shared; callers coordinate source-array mutation during construction.</p>
 */
public final class ValidSequenceLengths {
    private final long[] lengths;

    /**
     * Retains the factory's already copied and validated array without another copy.
     *
     * @param lengths non-null exclusively owned array of non-negative lengths in row order
     */
    private ValidSequenceLengths(long[] lengths) {
        this.lengths = lengths;
    }

    /**
     * Snapshots ordered lengths, then checks the owned copy in row order for negative entries.
     *
     * @param lengths non-null caller-owned lengths in time steps; each must be non-negative;
     *     empty selects a zero-row batch; coordinate mutation until this call returns
     * @return a non-null immutable value independent of subsequent source-array mutations
     * @throws NullPointerException if {@code lengths} is null, with message {@code lengths}
     * @throws IllegalArgumentException if a copied entry is negative, identifying its first row
     */
    public static ValidSequenceLengths of(long... lengths) {
        long[] copied = Objects.requireNonNull(lengths, "lengths").clone();
        for (int row = 0; row < copied.length; row++) {
            if (copied[row] < 0) {
                throw new IllegalArgumentException(
                        "Length at row " + row + " must be non-negative: " + copied[row]);
            }
        }
        return new ValidSequenceLengths(copied);
    }

    /**
     * Reports how many original rows have declared counts.
     *
     * @return non-negative host-array length, with zero denoting an empty batch
     */
    public int batchSize() {
        return lengths.length;
    }

    /**
     * Reads the count for one original row without wrapping negative indices.
     *
     * @param row zero-based row satisfying {@code 0 <= row < batchSize()}
     * @return non-negative position-independent valid count in logical time steps
     * @throws IndexOutOfBoundsException if {@code row} is outside the original batch
     */
    public long length(int row) {
        return lengths[Objects.checkIndex(row, lengths.length)];
    }

    /**
     * Exports a mutable copy without exposing this value's owned array.
     *
     * @return a non-null fresh array in original row order, including for an empty batch
     */
    public long[] toLongArray() {
        return lengths.clone();
    }

    /**
     * Compares lengths structurally in original row order, independently of tensor data.
     *
     * @param other candidate value, possibly null
     * @return true exactly for this instance or another value with the same ordered lengths
     */
    @Override
    public boolean equals(Object other) {
        return this == other || (other instanceof ValidSequenceLengths validLengths
                && Arrays.equals(lengths, validLengths.lengths));
    }

    /**
     * Hashes the ordered immutable lengths, not any caller-owned source array.
     *
     * @return a structural hash consistent with {@link #equals(Object)}
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(lengths);
    }

    /**
     * Describes only the declared host counts, never tensor contents or storage.
     *
     * @return non-null diagnostic text containing ordered lengths; not a serialization format
     */
    @Override
    public String toString() {
        return "ValidSequenceLengths" + Arrays.toString(lengths);
    }
}
