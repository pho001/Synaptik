package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class Conv2dTest {
    @Test
    void bindsExpectedStateAndDelegatesOneBiasedOccurrence() {
        Conv2d layer = new Conv2d(8, 3, 5, 2, 1, 1, 2, 1, 1, 2, true,
                DataType.FLOAT64, ParameterInitialization.zeros(), 3L);
        Tensor input = tensor(Shape.of(2, 4, 11, 13), DataType.FLOAT64);
        Tensor result = layer.forward(input);
        var provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertEquals(Shape.of(8, 2, 3, 5), layer.weight().value().descriptor().shape()),
                () -> assertEquals(Shape.of(2, 8, 6, 13), result.descriptor().shape()),
                () -> assertSame(Conv2dKind.CONV2D, provenance.operation().kind()),
                () -> assertEquals(new Conv2dAttrs(2, 1, 1, 2, 1, 1, 2), provenance.operation().attrs()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertSame(layer.weight().value(), provenance.inputs().get(1)),
                () -> assertSame(layer.bias().orElseThrow().value(), provenance.inputs().get(2)));
    }

    @Test
    void validatesGroupsChannelsAndStaticSpatialFit() {
        assertThrows(IllegalArgumentException.class, () -> new Conv2d(5, 3, 3, 1, 1, 0, 0,
                1, 1, 2, false, DataType.FLOAT32, ParameterInitialization.ones(), 0L));
        Conv2d layer = new Conv2d(4, 5, 5, 1, 1, 0, 0, 1, 1, 2, false,
                DataType.FLOAT32, ParameterInitialization.ones(), 0L);
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 3, 8, 8), DataType.FLOAT32)));
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(1, 4, 3, 8), DataType.FLOAT32)));
        assertThrows(IllegalStateException.class, layer::weight);
    }

    @Test
    void rejectsWrongTypeRankAndNonPositiveOrDynamicChannelsBeforeBinding() {
        Conv2d layer = new Conv2d(4, 3, 3, 1, 1, 0, 0, 1, 1, 2, false,
                DataType.FLOAT32, ParameterInitialization.ones(), 0L);
        Tensor dynamicChannels = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), new DynamicDimension("C"),
                        new StaticDimension(5), new StaticDimension(5)),
                Optional.empty(), false));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> layer.forward(tensor(Shape.of(1, 4, 5), DataType.FLOAT32))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> layer.forward(tensor(Shape.of(1, 4, 5, 5), DataType.FLOAT64))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> layer.forward(tensor(Shape.of(1, 0, 5, 5), DataType.FLOAT32))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> layer.forward(dynamicChannels)),
                () -> assertThrows(IllegalStateException.class, layer::weight));
    }

    @Test
    void compatibleBatchAndBothSpatialExtentsMayVaryAfterBinding() {
        Conv2d layer = new Conv2d(4, 3, 2, 2, 1, 1, 0, 1, 1, 2, false,
                DataType.FLOAT32, ParameterInitialization.zeros(), 0L);
        layer.forward(tensor(Shape.of(1, 4, 7, 8), DataType.FLOAT32));
        assertEquals(Shape.of(5, 4, 6, 12),
                layer.forward(tensor(Shape.of(5, 4, 11, 13), DataType.FLOAT32))
                        .descriptor().shape());
    }

    private static Tensor tensor(Shape shape, DataType type) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), false));
    }
}
