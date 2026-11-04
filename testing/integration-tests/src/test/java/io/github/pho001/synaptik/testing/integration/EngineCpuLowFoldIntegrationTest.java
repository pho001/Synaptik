package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public low UNFOLD preserves exact raw storage bits and leaves its input unchanged. */
class EngineCpuLowFoldIntegrationTest {
    @Test void actualLowUnfoldRemainsRaw() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) for (boolean image : List.of(false, true)) {
            try (var arena = Arena.ofShared(); var engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
                Shape shape = image ? Shape.of(2, 2, 3, 5) : Shape.of(2, 5, 2);
                var input = tensor(arena, type, shape, true, 1);
                var source = ((MemorySegmentStorage) input.hostStorage().orElseThrow()).segment();
                short[] raw = {(short) 0x8000, 1, (short) 0x8001, 0};
                for (int i = 0; i < source.byteSize() / 2; i++) source.setAtIndex(ValueLayout.JAVA_SHORT, i, raw[i % raw.length]);
                byte[] unchanged = source.toArray(ValueLayout.JAVA_BYTE);
                var window = new Window2dAttrs(2, 3, 1, 1, 1, 1, 1, 1, true);
                var unfolded = image ? input.unfold2d(window) : input.unfold(1, 3, 1);
                var forward = engine.compile(List.of(unfolded));
                try (var prepared = engine.prepare(forward); var result = engine.run(prepared, List.of(input))) {
                    long bytes = unfolded.descriptor().shape().knownElementCount().orElseThrow() * 2;
                    var output = result.materialize(result.publications().getFirst(), bytes).bytes();
                    var observed = new java.util.HashSet<Short>();
                    while (output.hasRemaining()) {
                        short value = output.getShort();
                        assertTrue(value == 0 || value == (short) 0x8000 || value == 1 || value == (short) 0x8001);
                        observed.add(value);
                    }
                    assertTrue(observed.contains((short) 0x8000));
                    assertTrue(observed.contains((short) 1));
                    assertArrayEquals(unchanged, source.toArray(ValueLayout.JAVA_BYTE));
                }
            }
        }
    }


    private static Tensor tensor(Arena arena, DataType type, Shape shape, boolean gradient, int scale) {
        long count = shape.knownElementCount().orElseThrow(); var storage = arena.allocate(count * 2, 2);
        for (int i = 0; i < count; i++) storage.setAtIndex(ValueLayout.JAVA_SHORT, i,
                encode(type, scale * (i % 17 - 8) / 7f));
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient),
                Optional.empty(), Optional.of(new MemorySegmentStorage(type, count, storage)));
    }
    private static short encode(DataType type, float value) { return type == DataType.BFLOAT16 ? BFloat16Bits.fromFloat(value) : Float.floatToFloat16(value); }
}
