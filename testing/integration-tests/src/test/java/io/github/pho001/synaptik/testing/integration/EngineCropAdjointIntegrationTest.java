package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Raw crop inference and later physical view completion must compose with selected cotangents. */
final class EngineCropAdjointIntegrationTest {
    @Test
    void croppedViewPublishesTheSelectedInputIntervalAndScattersItsCotangent() {
        try (Engine engine = Engine.standard()) {
            for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
                Tensor input = tensor(type, true, 1, 2, 3, 4);
                Tensor crop = input.cropToShape(Shape.of(2), Shape.of(1));
                Tensor seed = tensor(type, false, 3, 7);
                var compiled = engine.compile(List.of(crop), List.of(seed), List.of(input));
                try (var prepared = engine.prepare(compiled); var result = engine.run(prepared, List.of(input, seed))) {
                    assertEquals(2, result.publications().size());
                    compare(result.materialize(result.publications().get(0), 1024), 2, 3);
                    compare(result.materialize(result.publications().get(1), 1024), 0, 3, 7, 0);
                }
            }
        }
    }

    private static Tensor tensor(DataType type, boolean gradient, double... values) {
        Shape shape = Shape.of(values.length);
        var descriptor = new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
        if (type == DataType.FLOAT64) return TensorFactory.fromFlatArray(descriptor, Optional.empty(), values);
        float[] raw = new float[values.length];
        for (int i = 0; i < raw.length; i++) raw[i] = (float) values[i];
        return TensorFactory.fromFlatArray(descriptor, Optional.empty(), raw);
    }

    private static void compare(HostTensorValue value, double... expected) {
        var bytes = value.bytes();
        assertEquals(expected.length * value.dataType().byteWidth(), bytes.remaining());
        for (double lane : expected) assertEquals(lane,
                value.dataType() == DataType.FLOAT64 ? bytes.getDouble() : bytes.getFloat(), 0);
    }
}
