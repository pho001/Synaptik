package io.github.pho001.synaptik.data;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable configuration for padding ordered long sequences to their batch maximum.
 *
 * <p>Each call creates one contiguous, offset-zero, rank-two INT64 non-gradient leaf with
 * GC-owned writable heap storage. No input list or row is retained and no close protocol is
 * introduced. Padding values remain ordinary data when present inside a valid interval.
 * Configuration can be shared; callers coordinate input mutation throughout each synchronous
 * call. No transactional snapshot is promised.</p>
 */
public final class LongSequenceBatcher {
    private final long paddingValue;
    private final PaddingSide paddingSide;
    private final SequenceAxisOrder axisOrder;

    /**
     * Copies complete builder selections into an independent configuration.
     * @param builder non-null builder with all selections present
     */
    private LongSequenceBatcher(Builder builder) {
        paddingValue = builder.paddingValue;
        paddingSide = builder.paddingSide;
        axisOrder = builder.axisOrder;
    }

    /**
     * Starts explicit configuration without default padding, side or order.
     * @return a fresh non-null caller-coordinated builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Copies rows into one directly allocated Model destination, padding without truncation.
     *
     * <p>B is list size and T is maximum row length. Empty input yields B=0,T=0; all-empty
     * rows yield T=0. Geometry is checked incrementally before batch-sized metadata allocation.
     * Row traversal is sequential, so linked lists do not incur repeated indexed-access costs.
     * The element count must fit a Java array; byte span is checked as long, not capped at int.
     * Later input mutation cannot change the result. Model allocation failures propagate.</p>
     *
     * @param sequences non-null ordered list of non-null caller-owned rows; coordinate list
     *     and row mutation until this call returns
     * @return a fresh non-null padded batch with original row counts and no retained input arrays
     * @throws NullPointerException if the list is null (message sequences), or a row is null
     *     (message identifies its index)
     * @throws IllegalArgumentException if checked geometry overflows or exceeds Integer.MAX_VALUE
     *     elements, before output allocation
     * @throws IllegalStateException if Model tensor identifiers are exhausted
     * @throws OutOfMemoryError if JVM heap allocation fails
     */
    public PaddedSequenceBatch batch(List<long[]> sequences) {
        Objects.requireNonNull(sequences, "sequences");
        int batchSize = sequences.size();
        int time = 0;
        int row = 0;
        for (long[] sequence : sequences) {
            Objects.requireNonNull(sequence, "sequences row " + row);
            time = Math.max(time, sequence.length);
            try {
                long elements = Math.multiplyExact((long) batchSize, time);
                if (elements > Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("Batch element count exceeds Java array limit: " + elements);
                }
                Math.multiplyExact(elements, (long) Long.BYTES);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException("Batch geometry overflows long", overflow);
            }
            row++;
        }
        long[] lengths = new long[batchSize];
        Shape shape = axisOrder == SequenceAxisOrder.TIME_MAJOR
                ? Shape.of(time, batchSize) : Shape.of(batchSize, time);
        var descriptor = new TensorDescriptor(DataType.INT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var values = TensorFactory.allocate(descriptor);
        var destination = values.hostStorage().orElseThrow().segment();
        row = 0;
        for (long[] sequence : sequences) {
            Objects.requireNonNull(sequence, "sequences row " + row);
            lengths[row] = sequence.length;
            int start = paddingSide == PaddingSide.LEFT ? time - sequence.length : 0;
            for (int step = 0; step < time; step++) {
                int source = step - start;
                long value = source >= 0 && source < sequence.length ? sequence[source] : paddingValue;
                long index = axisOrder == SequenceAxisOrder.TIME_MAJOR
                        ? (long) step * batchSize + row : (long) row * time + step;
                destination.setAtIndex(ValueLayout.JAVA_LONG, index, value);
            }
            row++;
        }
        return new PaddedSequenceBatch(values, ValidSequenceLengths.of(lengths), axisOrder, paddingSide);
    }

    /**
     * Reusable, non-thread-safe configuration state. All three choices must be explicit;
     * last selections win and later changes cannot alter previously built configurations.
     */
    public static final class Builder {
        private long paddingValue;
        private boolean hasPaddingValue;
        private PaddingSide paddingSide;
        private SequenceAxisOrder axisOrder;

        /** Creates unselected configuration state for the factory. */
        private Builder() { }

        /**
         * Selects the exact long written outside valid intervals.
         * @param value padding value, including zero or either long extreme
         * @return this exact builder, replacing the previous value
         */
        public Builder paddingValue(long value) {
            paddingValue = value;
            hasPaddingValue = true;
            return this;
        }

        /**
         * Selects padding before each valid suffix.
         * @return this exact builder, replacing the previous side
         */
        public Builder leftPadded() {
            paddingSide = PaddingSide.LEFT;
            return this;
        }

        /**
         * Selects padding after each valid prefix.
         * @return this exact builder, replacing the previous side
         */
        public Builder rightPadded() {
            paddingSide = PaddingSide.RIGHT;
            return this;
        }

        /**
         * Selects output shape {@code [T,B]}.
         * @return this exact builder, replacing the previous order
         */
        public Builder timeMajor() {
            axisOrder = SequenceAxisOrder.TIME_MAJOR;
            return this;
        }

        /**
         * Selects output shape {@code [B,T]}.
         * @return this exact builder, replacing the previous order
         */
        public Builder batchMajor() {
            axisOrder = SequenceAxisOrder.BATCH_MAJOR;
            return this;
        }

        /**
         * Freezes all current selections into a fresh independent configuration.
         * @return a fresh non-null immutable batcher
         * @throws IllegalStateException if padding value, side or order has not been selected
         */
        public LongSequenceBatcher build() {
            if (!hasPaddingValue || paddingSide == null || axisOrder == null) {
                throw new IllegalStateException("Select paddingValue, paddingSide and axisOrder before build");
            }
            return new LongSequenceBatcher(this);
        }
    }
}
