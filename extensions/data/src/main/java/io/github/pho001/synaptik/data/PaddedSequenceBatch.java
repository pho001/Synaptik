package io.github.pho001.synaptik.data;

import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.Objects;

/**
 * Declares valid intervals for an existing, already ordered and padded sequence tensor.
 *
 * <p>Only descriptor/shape metadata and host lengths are inspected. Construction is O(B) for B
 * rows, independent of the time extent and element count. It never inspects storage or its
 * liveness, reads or copies values, creates expressions, transposes, materializes, or executes.
 * Trailing dimensions are opaque; dtype, gradient flags, provenance and physical layout remain
 * Model concerns. Declared intervals do not prove contents: zero is ordinary data, arbitrary
 * padding regions pass, and padding lengths {@code T - length(row)} are derived rather than stored.
 * RIGHT selects {@code [0,L)}; LEFT selects {@code [T-L,T)} for each row count L.</p>
 *
 * <p>Immutability covers the copied lengths and fixed component references, not tensor contents
 * or the tensor's mutable borrowed-storage association. Callers retain Model lifetime and access
 * obligations; this wrapper owns no resource and promises no readiness, liveness or thread safety
 * for tensor access. Record-generated equality and hashing use Tensor object identity, structural
 * lengths and both enum identities, never tensor contents. Record-generated diagnostic text contains only
 * component metadata and is not a serialization format.</p>
 *
 * @param values non-null borrowed tensor retained exactly, with static leading time/batch axes
 * @param validLengths non-null immutable counts retained exactly in original batch-row order
 * @param axisOrder non-null logical leading-axis order retained exactly, not physical layout
 * @param paddingSide non-null side on which padding precedes or follows valid data
 */
public record PaddedSequenceBatch(
        Tensor values, ValidSequenceLengths validLengths, SequenceAxisOrder axisOrder, PaddingSide paddingSide) {
    /**
     * Validates interval geometry without consulting or changing tensor storage.
     *
     * <p>Checks nulls in parameter order, then rank, static leading axes, batch count, and upper
     * bounds in row order. Extents remain long; no element products or byte spans are calculated.
     * Zero rows accept empty lengths for any T, and T=0 requires every length to be zero.
     * Trailing zero or dynamic dimensions do not constrain counts.</p>
     *
     * @param values non-null borrowed tensor of rank at least two; never mutated or copied
     * @param validLengths non-null immutable lengths, with count B and each length at most T
     * @param axisOrder non-null order interpreting {@code [T,B,...]} or {@code [B,T,...]}
     * @param paddingSide non-null side declaring a valid prefix (RIGHT) or suffix (LEFT)
     * @throws NullPointerException if a component is null, checked in declaration order with
     *     its parameter name as the message
     * @throws IllegalArgumentException if rank is below two, either leading axis is not static,
     *     B differs from the length count, or a row exceeds T; diagnostics identify the mismatch
     *     or the first offending row and T
     */
    public PaddedSequenceBatch {
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(validLengths, "validLengths");
        Objects.requireNonNull(axisOrder, "axisOrder");
        Objects.requireNonNull(paddingSide, "paddingSide");
        Shape shape = values.descriptor().shape();
        if (shape.rank() < 2) {
            throw new IllegalArgumentException("Sequence values require rank >= 2: " + shape.rank());
        }
        int timeAxis = axisOrder == SequenceAxisOrder.TIME_MAJOR ? 0 : 1;
        int batchAxis = 1 - timeAxis;
        if (!(shape.dimension(timeAxis) instanceof StaticDimension time)) {
            throw new IllegalArgumentException("Time axis " + timeAxis + " must be static");
        }
        if (!(shape.dimension(batchAxis) instanceof StaticDimension batch)) {
            throw new IllegalArgumentException("Batch axis " + batchAxis + " must be static");
        }
        if (batch.size() != (long) validLengths.batchSize()) {
            throw new IllegalArgumentException("Batch extent B=" + batch.size()
                    + " differs from validLengths count=" + validLengths.batchSize());
        }
        for (int row = 0; row < validLengths.batchSize(); row++) {
            long length = validLengths.length(row);
            if (length > time.size()) {
                throw new IllegalArgumentException("Length at row " + row + " is " + length
                        + ", exceeding T=" + time.size());
            }
        }
    }

    /**
     * Starts reusable fluent construction without reading the tensor's shape.
     *
     * @param values non-null borrowed tensor fixed for the resulting builder; never mutated
     * @return a fresh non-null builder with no axis order, side or lengths selected
     * @throws NullPointerException if {@code values} is null, with message {@code values}
     */
    public static Builder builder(Tensor values) {
        return new Builder(Objects.requireNonNull(values, "values"));
    }

    /**
     * Returns the borrowed tensor without snapshotting its contents or storage association.
     *
     * @return the exact non-null tensor supplied at construction, with unchanged Model obligations
     */
    public Tensor values() {
        return values;
    }

    /**
     * Returns the immutable valid counts associated with original batch rows.
     *
     * @return the exact non-null lengths supplied at construction, safe to share as metadata
     */
    public ValidSequenceLengths validLengths() {
        return validLengths;
    }

    /**
     * Returns the declaration of logical leading axes, without implying physical contiguity.
     *
     * @return the exact non-null axis-order constant supplied at construction
     */
    public SequenceAxisOrder axisOrder() {
        return axisOrder;
    }

    /**
     * Identifies the padding region without inspecting values.
     * @return the exact non-null side selected at construction
     */
    public PaddingSide paddingSide() {
        return paddingSide;
    }

    /**
     * Reusable, caller-coordinated construction state for one fixed borrowed tensor.
     *
     * <p>Not thread-safe. Lengths, side and order require explicit selection; empty lengths are a
     * selection, not a missing value. Setters return this same builder and the last successful
     * setter wins. Lengths are snapshotted immediately, and failed setters/builds preserve prior
     * selections. Each successful build creates a fresh batch; later changes cannot alter earlier
     * batches. The builder owns no resource and closes nothing.</p>
     */
    public static final class Builder {
        private final Tensor values;
        private ValidSequenceLengths validLengths;
        private SequenceAxisOrder axisOrder;
        private PaddingSide paddingSide;

        /**
         * Retains the factory-checked tensor without reading descriptor or storage metadata.
         *
         * @param values non-null borrowed tensor fixed for this builder
         */
        private Builder(Tensor values) {
            this.values = values;
        }

        /**
         * Selects logical {@code [T,B,...]} axes without transposing values.
         *
         * @return this exact builder, replacing any earlier order selection
         */
        public Builder timeMajor() {
            axisOrder = SequenceAxisOrder.TIME_MAJOR;
            return this;
        }

        /**
         * Selects logical {@code [B,T,...]} axes without transposing values.
         *
         * @return this exact builder, replacing any earlier order selection
         */
        public Builder batchMajor() {
            axisOrder = SequenceAxisOrder.BATCH_MAJOR;
            return this;
        }

        /**
         * Declares padding before the valid suffix, without moving values.
         * @return this exact builder, replacing any earlier side selection
         */
        public Builder leftPadded() {
            paddingSide = PaddingSide.LEFT;
            return this;
        }

        /**
         * Declares padding after the valid prefix, without moving values.
         * @return this exact builder, replacing any earlier side selection
         */
        public Builder rightPadded() {
            paddingSide = PaddingSide.RIGHT;
            return this;
        }

        /**
         * Immediately snapshots and validates valid counts before replacing the prior selection.
         *
         * @param lengths non-null caller-owned non-negative time-step counts in original row
         *     order; empty explicitly selects zero rows; coordinate mutation during this call
         * @return this exact builder after successful replacement of its immutable lengths
         * @throws NullPointerException if {@code lengths} is null, with message {@code lengths};
         *     any previous selection remains unchanged
         * @throws IllegalArgumentException if a copied length is negative, identifying the first
         *     offending row; any previous selection remains unchanged
         */
        public Builder validLengths(long... lengths) {
            validLengths = ValidSequenceLengths.of(lengths);
            return this;
        }

        /**
         * Constructs a fresh batch through the canonical geometry validator without recopying
         * immutable lengths or changing this builder's selections.
         *
         * @return a non-null fresh batch retaining this tensor, counts, axis order and side
         * @throws IllegalStateException if any selection is missing, before geometry validation
         * @throws IllegalArgumentException if selected geometry fails the record constructor's
         *     rank, static-axis, batch-count or per-row upper-bound checks
         */
        public PaddedSequenceBatch build() {
            if (axisOrder == null || validLengths == null || paddingSide == null) {
                throw new IllegalStateException("Select axisOrder, paddingSide and validLengths before build");
            }
            return new PaddedSequenceBatch(values, validLengths, axisOrder, paddingSide);
        }
    }
}
