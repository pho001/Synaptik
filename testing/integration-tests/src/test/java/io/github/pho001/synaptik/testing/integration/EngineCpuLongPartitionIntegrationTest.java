package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public numerical execution of a CPU-only partition longer than the bounded fusion domain. */
final class EngineCpuLongPartitionIntegrationTest {
    @Test void ninePublishedCpuNodesExecuteInOnePreparedRun() {
        Shape shape = Shape.of(4);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            float[] source = {-1f, 3f, 0f, 2f};
            MemorySegment inputBytes = arena.allocate(4L * Float.BYTES, Float.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(source), 0, inputBytes, 0,
                    inputBytes.byteSize());
            Tensor input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT32, 4, inputBytes)));
            var outputs = new ArrayList<Tensor>();
            Tensor current = input;
            for (int index = 0; index < 9; index++) {
                current = current.add(ScalarValue.float32(2f));
                outputs.add(current);
            }
            var compiled = engine.compile(outputs);
            assertEquals(List.of("cpu"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
            assertEquals(9, EngineMixedOwnerTestAccess.compileArtifacts(compiled).graph()
                    .nodes().size());
            try (var prepared = engine.prepare(compiled);
                    var result = engine.run(prepared, List.of(input))) {
                assertEquals(9, result.resultCount());
                assertArrayEquals(new float[] {17f, 21f, 18f, 20f}, values(result));
            }
        }
    }

    @Test void publishedScalarChainThenSortAndLatePointwiseTailExecutesAcrossRuns() {
        Shape shape = Shape.of(4);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            float[] source = {-1f, 3f, 0f, 2f};
            MemorySegment inputBytes = arena.allocate(4L * Float.BYTES, Float.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(source), 0, inputBytes, 0,
                    inputBytes.byteSize());
            Tensor input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                    new MemorySegmentStorage(DataType.FLOAT32, 4, inputBytes)));
            var outputs = new ArrayList<Tensor>();
            Tensor current = input;
            for (int index = 0; index < 9; index++) {
                current = current.add(ScalarValue.float32(2f));
                outputs.add(current);
            }
            current = current.sort(0).add(ScalarValue.float32(2f))
                    .mul(ScalarValue.float32(2f));
            outputs.add(current);
            var compiled = engine.compile(outputs);
            assertEquals(1, compiled.inputs().size());
            assertEquals(List.of("cpu"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
            assertEquals(12, EngineMixedOwnerTestAccess.compileArtifacts(compiled).graph()
                    .nodes().size());
            try (var prepared = engine.prepare(compiled);
                    var first = engine.run(prepared, List.of(input));
                    var second = engine.run(prepared, List.of(input))) {
                assertEquals(10, first.resultCount());
                assertArrayEquals(new float[] {38f, 40f, 44f, 46f}, values(first));
                assertArrayEquals(new float[] {38f, 40f, 44f, 46f}, values(second));
            }
        }
    }

    private static float[] values(io.github.pho001.synaptik.engine.RunResult result) {
        var publication = result.publications().getLast();
        var bytes = result.materialize(publication, 4L * Float.BYTES).bytes();
        float[] values = new float[4];
        for (int index = 0; index < values.length; index++) values[index] = bytes.getFloat();
        return values;
    }
}
