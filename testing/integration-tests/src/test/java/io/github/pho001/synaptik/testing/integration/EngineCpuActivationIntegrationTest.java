package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU sessions publish activation extremes and low named-site cancellation unchanged. */
final class EngineCpuActivationIntegrationTest {
    @Test void doubleErfAndGeluPublishHugeFiniteValuesAndSpecialClasses() {
        double[] values = {-1e100, 1e100, -Double.MAX_VALUE, Double.MAX_VALUE, -0d, 0d,
                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, Double.NaN};
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor input = tensor(arena, DataType.FLOAT64, MemorySegment.ofArray(values), values.length);
            for (boolean gelu : List.of(false, true)) {
                ByteBuffer bytes = run(engine, input, gelu ? input.gelu() : input.erf(), values.length * 8L);
                for (double x : values) {
                    double actual = bytes.getDouble();
                    double expected = gelu ? x <= 0 ? Math.copySign(0d, x) : x
                            : x == 0 ? x : Math.copySign(1d, x);
                    if (Double.isNaN(x)) assertTrue(Double.isNaN(actual));
                    else assertEquals(Double.doubleToRawLongBits(expected), Double.doubleToRawLongBits(actual));
                }
            }
        }
    }

    @Test void bothLowTypesPublishNegativeZeroGeluAtMinusSix() {
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            short[] values = type == DataType.BFLOAT16 ? new short[] {(short) 0xc0c0, (short) 0x40c0,
                    (short) 0x8000, 0, (short) 0xff80, (short) 0x7f80, (short) 0x7fc0}
                    : new short[] {(short) 0xc600, (short) 0x4600, (short) 0x8000, 0,
                            (short) 0xfc00, (short) 0x7c00, (short) 0x7e00};
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Tensor input = tensor(arena, type, MemorySegment.ofArray(values), values.length);
                for (boolean tanh : List.of(false, true)) {
                    ByteBuffer bytes = run(engine, input,
                            tanh ? input.geluTanhApproximation() : input.gelu(), values.length * 2L);
                    short[] expected = {(short) 0x8000, values[1], (short) 0x8000, 0,
                            (short) 0x8000, values[5]};
                    for (short value : expected) assertEquals(value, bytes.getShort());
                    short nan = bytes.getShort();
                    float decoded = type == DataType.BFLOAT16 ? Float.intBitsToFloat(nan << 16)
                            : Float.float16ToFloat(nan);
                    assertTrue(Float.isNaN(decoded));
                }
            }
        }
    }

    private static ByteBuffer run(Engine engine, Tensor input, Tensor output, long size) {
        try (var session = engine.prepare(engine.compile(List.of(output)));
                var result = engine.run(session, List.of(input))) {
            assertEquals(1, result.resultCount());
            return result.materialize(result.publications().getFirst(), size).bytes();
        }
    }

    private static Tensor tensor(Arena arena, DataType type, MemorySegment source, int count) {
        MemorySegment storage = arena.allocate(source.byteSize(), type.byteWidth());
        storage.copyFrom(source);
        Shape shape = Shape.of(count);
        var descriptor = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return TensorFactory.create(descriptor, Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, count, storage)));
    }
}
