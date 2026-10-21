package io.github.pho001.synaptik.testing.architecture;

import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.data.PaddingSide;
import io.github.pho001.synaptik.data.LongSequenceBatcher;
import io.github.pho001.synaptik.data.SequenceAxisOrder;
import io.github.pho001.synaptik.data.ValidSequenceLengths;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.util.List;

/**
 * Compile-only witness of the public Data API and exported Model declarations without an
 * inherited test or NN classpath. This proves visibility, not materialization or execution.
 */
final class DataApiConsumer {
    /** Prevents instantiation of this compile-only declaration witness. */
    private DataApiConsumer() { }

    /**
     * Exercises fluent construction and the public nested Builder from outside the Data package.
     *
     * @param values non-null borrowed tensor of Shape {@code [3,4,...]}
     * @return a non-null borrowed batch declaring prefixes {@code [3,2,1]}
     * @throws NullPointerException if {@code values} is null
     * @throws IllegalArgumentException if the supplied tensor has incompatible prefix geometry
     */
    static PaddedSequenceBatch fluent(Tensor values) {
        PaddedSequenceBatch.Builder builder = PaddedSequenceBatch.builder(values).rightPadded();
        return builder.batchMajor().validLengths(3, 2, 1).build();
    }

    /**
     * Exercises direct construction and every metadata accessor through the Data-only classpath.
     *
     * @param values non-null borrowed tensor of Shape {@code [4,3,...]}
     * @return a non-null batch retaining exact components and original row prefixes
     * @throws NullPointerException if {@code values} is null
     * @throws IllegalArgumentException if the supplied tensor has incompatible prefix geometry
     * @throws AssertionError if the constructed prefixes disagree with their array export
     */
    static PaddedSequenceBatch direct(Tensor values) {
        ValidSequenceLengths lengths = ValidSequenceLengths.of(3, 2, 1);
        PaddedSequenceBatch batch = new PaddedSequenceBatch(values, lengths, SequenceAxisOrder.TIME_MAJOR, PaddingSide.RIGHT);
        Tensor borrowed = batch.values();
        ValidSequenceLengths prefixes = batch.validLengths();
        SequenceAxisOrder order = batch.axisOrder();
        long[] exported = prefixes.toLongArray();
        if (prefixes.batchSize() != exported.length || prefixes.length(0) != exported[0]) {
            throw new AssertionError("Prefix accessors must agree with their array export");
        }
        PaddingSide side = batch.paddingSide();
        return new PaddedSequenceBatch(borrowed, prefixes, order, side);
    }

    /**
     * Exercises integral allocation through Data and transitive public Model storage.
     * @param rows non-null ordered list of non-null caller-owned long arrays
     * @return fresh non-null left-padded batch-major INT64 values
     * @throws NullPointerException if the list or a row is null
     * @throws IllegalArgumentException if output geometry exceeds the Java element limit
     * @throws IllegalStateException if Model tensor identifiers are exhausted
     * @throws OutOfMemoryError if heap allocation fails
     */
    static PaddedSequenceBatch integral(List<long[]> rows) {
        LongSequenceBatcher.Builder builder = LongSequenceBatcher.builder();
        return builder.paddingValue(Long.MIN_VALUE).leftPadded().batchMajor().build().batch(rows);
    }

    /**
     * Names a Model Tensor through Data's API export without reading storage or changing ownership.
     *
     * @param tensor borrowed Tensor reference, possibly null; never mutated or retained
     * @return the exact supplied reference, including null
     */
    static Tensor identity(Tensor tensor) {
        return tensor;
    }

    /**
     * Reads Model's immutable logical Shape without materialization or graph execution.
     *
     * @param tensor non-null borrowed Tensor whose descriptor is inspected
     * @return the non-null immutable Shape reference held by the Tensor descriptor
     * @throws NullPointerException if {@code tensor} is null
     */
    static Shape shape(Tensor tensor) {
        return tensor.descriptor().shape();
    }

    /**
     * Reads Model's logical element type without accessing host or backend storage.
     *
     * @param tensor non-null borrowed Tensor whose descriptor is inspected
     * @return the non-null immutable data type held by the Tensor descriptor
     * @throws NullPointerException if {@code tensor} is null
     */
    static DataType dataType(Tensor tensor) {
        return tensor.descriptor().dataType();
    }
}
