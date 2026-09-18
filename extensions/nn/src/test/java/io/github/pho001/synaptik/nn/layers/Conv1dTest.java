package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class Conv1dTest {
    @Test
    void convolutionLayerPublicSurfacesRemainRankSpecificAndMinimal() {
        assertAll(
                () -> assertLayerSurface(Conv1d.class, 10),
                () -> assertLayerSurface(Conv2d.class, 14),
                () -> assertLayerSurface(Conv3d.class, 18));
    }

    @Test
    void bindsGroupedStateAndDelegatesThroughVisibleComposition() {
        Conv1d layer = new Conv1d(6, 3, 2, 1, 1, 2, true, DataType.FLOAT32,
                ParameterInitialization.ones(), 7L);
        Tensor input = tensor(Shape.of(4, 4, 9), DataType.FLOAT32);
        Tensor result = layer.forward(input);
        Tensor convolution = result.provenance().orElseThrow().inputs().getFirst();

        assertAll(
                () -> assertEquals(Shape.of(6, 2, 3), layer.weight().value().descriptor().shape()),
                () -> assertEquals(Shape.of(6), layer.bias().orElseThrow().value().descriptor().shape()),
                () -> assertEquals(List.of("weight", "bias"),
                        layer.parameters().stream().map(parameter -> parameter.name()).toList()),
                () -> assertEquals(Shape.of(4, 6, 5), result.descriptor().shape()),
                () -> assertSame(AxisTransformKind.SQUEEZE, result.provenance().orElseThrow().operation().kind()),
                () -> assertSame(Conv2dKind.CONV2D, convolution.provenance().orElseThrow().operation().kind()),
                () -> assertSame(layer.weight().value(), convolution.provenance().orElseThrow()
                        .inputs().get(1).provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(layer.bias().orElseThrow().value(), convolution.provenance().orElseThrow().inputs().get(2)));
    }

    @Test
    void rejectsIncompatibleDescriptorsBeforeBindingAndAllowsVariableSpatialShape() {
        Conv1d layer = new Conv1d(4, 3, 1, 0, 1, 1, false, DataType.FLOAT32,
                ParameterInitialization.zeros(), 1L);
        assertThrows(IllegalArgumentException.class,
                () -> layer.forward(tensor(Shape.of(2, 3), DataType.FLOAT32)));
        assertThrows(IllegalStateException.class, layer::weight);
        layer.forward(tensor(Shape.of(2, 3, 7), DataType.FLOAT32));
        assertEquals(Shape.of(5, 4, 11),
                layer.forward(tensor(Shape.of(5, 3, 13), DataType.FLOAT32)).descriptor().shape());
        assertTrue(layer.bias().isEmpty());
    }

    private static Tensor tensor(Shape shape, DataType type) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), false));
    }

    private static void assertLayerSurface(Class<?> type, int constructorParameterCount) {
        assertAll(
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertSame(UnaryTensorModule.class, type.getSuperclass()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(1, type.getDeclaredConstructors().length),
                () -> assertEquals(constructorParameterCount,
                        type.getDeclaredConstructors()[0].getParameterCount()),
                () -> assertTrue(Modifier.isPublic(
                        type.getDeclaredConstructors()[0].getModifiers())),
                () -> assertEquals(Set.of("weight", "bias", "forward"),
                        java.util.Arrays.stream(type.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(java.lang.reflect.Method::getName)
                                .collect(java.util.stream.Collectors.toSet())),
                () -> assertTrue(java.util.Arrays.stream(type.getDeclaredFields())
                        .noneMatch(field -> Modifier.isPublic(field.getModifiers())
                                || Modifier.isProtected(field.getModifiers()))),
                () -> assertTrue(java.util.Arrays.stream(type.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isProtected(method.getModifiers()))));
    }
}
