package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class Conv3dTest {
    @Test
    void bindsExpectedStateAndDelegatesOneOccurrence() {
        Conv3d layer = new Conv3d(6, 2, 3, 4, 1, 2, 1, 0, 1, 2, 1, 1, 1, 2,
                false, DataType.BFLOAT16, ParameterInitialization.ones(), 9L);
        Tensor input = tensor(Shape.of(3, 4, 5, 7, 9), DataType.BFLOAT16);
        Tensor result = layer.forward(input);
        var provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertEquals(Shape.of(6, 2, 2, 3, 4), layer.weight().value().descriptor().shape()),
                () -> assertEquals(Shape.of(3, 6, 4, 4, 10), result.descriptor().shape()),
                () -> assertSame(Conv3dKind.CONV3D, provenance.operation().kind()),
                () -> assertEquals(new Conv3dAttrs(1, 2, 1, 0, 1, 2, 1, 1, 1, 2), provenance.operation().attrs()),
                () -> assertSame(input, provenance.inputs().getFirst()),
                () -> assertSame(layer.weight().value(), provenance.inputs().get(1)),
                () -> assertTrue(layer.bias().isEmpty()));
    }

    @Test
    void rejectsWrongRankTypeAndChannelsWithoutPublication() {
        Conv3d layer = new Conv3d(4, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2,
                true, DataType.FLOAT32, ParameterInitialization.zeros(), 0L);
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 4, 3, 3), DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 4, 3, 3, 3), DataType.FLOAT64)));
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 3, 3, 3, 3), DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 0, 3, 3, 3), DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 4, 1, 3, 3), DataType.FLOAT32)));
        assertThrows(IllegalStateException.class, layer::weight);
    }

    @Test
    void compatibleBatchAndAllSpatialExtentsMayVaryAfterBinding() {
        Conv3d layer = new Conv3d(6, 2, 3, 2, 1, 2, 1, 0, 1, 0, 1, 1, 1, 2,
                false, DataType.FLOAT32, ParameterInitialization.zeros(), 0L);
        layer.forward(tensor(Shape.of(1, 4, 4, 7, 6), DataType.FLOAT32));
        assertEquals(Shape.of(5, 6, 7, 5, 10),
                layer.forward(tensor(Shape.of(5, 4, 8, 9, 11), DataType.FLOAT32))
                        .descriptor().shape());
    }

    private static Tensor tensor(Shape shape, DataType type) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), false));
    }
}
