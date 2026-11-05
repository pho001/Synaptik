package io.github.pho001.synaptik.tools.cli;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Optional;

/**
 * Separate reusable native input/target stores. Refill only on the workflow thread after the
 * previous synchronous Model call returns; close only after the last call returns.
 */
final class NativeBatch implements AutoCloseable {
    private final Arena arena;
    private final MemorySegment inputMemory;
    private final MemorySegment targetMemory;
    private final int batch;
    private final int features;
    private final int outputs;
    private final Tensor input;
    private final Tensor target;

    NativeBatch(CliModel.Signature signature, boolean training) {
        arena = Arena.ofShared();
        try {
            batch = signature.batch();
            features = signature.features();
            outputs = signature.outputs();
            inputMemory = arena.allocate(Math.multiplyExact(Math.multiplyExact((long) batch, features), 4L), 4);
            input = wrap(signature.input(), inputMemory);
            if (training) {
                targetMemory = arena.allocate(Math.multiplyExact(Math.multiplyExact((long) batch, outputs), 4L), 4);
                target = wrap(signature.target(), targetMemory);
            } else {
                targetMemory = null;
                target = null;
            }
        } catch (RuntimeException | Error failure) {
            try {
                arena.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            throw failure;
        }
    }

    private static Tensor wrap(TensorDescriptor descriptor, MemorySegment segment) {
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, segment.byteSize() / 4, segment)));
    }

    /**
     * Refills one exact real batch without intermediate batch arrays or repeated values.
     * @param cursor current stable-file traversal with input columns followed by optional targets
     * @throws IOException for file failures
     * @throws IllegalArgumentException if the file shrank or changed geometry
     */
    void fill(NumericCsv.Cursor cursor) throws IOException {
        for (int row = 0; row < batch; row++) {
            if (!cursor.next()) throw new IllegalArgumentException("CSV ended before the validated batch count");
            long base = (long) row * features;
            for (int column = 0; column < features; column++) {
                inputMemory.set(ValueLayout.JAVA_FLOAT, (base + column) * 4L, cursor.value(column));
            }
            if (targetMemory != null) {
                for (int column = 0; column < outputs; column++) {
                    long element = (long) row * outputs + column;
                    targetMemory.set(ValueLayout.JAVA_FLOAT, element * 4L, cursor.value(features + column));
                }
            }
        }
    }

    Tensor input() { return input; }
    Tensor target() { return target; }

    @Override
    public void close() { arena.close(); }
}
