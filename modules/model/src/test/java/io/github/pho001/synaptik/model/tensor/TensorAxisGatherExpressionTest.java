package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorAxisGatherExpressionTest {
    @Test
    void exposesOnlyFinalAxisGatherAndUnchangedNdMethods() throws Exception {
        Method gather = Tensor.class.getDeclaredMethod("gather", Tensor.class, int.class);
        Method elements =
                Tensor.class.getDeclaredMethod("gatherElements", Tensor.class, int.class);
        assertTrue(Modifier.isPublic(gather.getModifiers()));
        assertTrue(Modifier.isPublic(elements.getModifiers()));
        assertEquals(Tensor.class, gather.getReturnType());
        assertEquals(Tensor.class, elements.getReturnType());

        Set<String> names = Arrays.stream(Tensor.class.getDeclaredMethods())
                .map(Method::getName)
                .filter(name -> name.startsWith("gather"))
                .collect(Collectors.toSet());
        assertEquals(Set.of("gather", "gatherElements", "gatherNd"), names);

        assertEquals(
                List.of(
                        "create", "gather", "gatherElements", "gatherShape",
                        "validateGatherElements", "validateIndexType"),
                Arrays.stream(TensorAxisGatherExpressions.class.getDeclaredMethods())
                        .map(Method::getName)
                        .sorted()
                        .toList());
        assertEquals(0, TensorAxisGatherExpressions.class.getDeclaredFields().length);
    }

    @Test
    void gatherReplacesAxisWithScalarOrdinaryZeroAndDynamicIndexShapes() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension query = new DynamicDimension("query");
        StaticDimension tail = new StaticDimension(4);
        Tensor data = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, new StaticDimension(3), tail),
                true);

        Tensor scalar = data.gather(tensor(DataType.INT32, Shape.scalar(), false), 1);
        Tensor ordinary = data.gather(tensor(DataType.INT64, Shape.of(5, 6), false), -2);
        Tensor zero = data.gather(tensor(DataType.INT32, Shape.of(0), false), 1);
        Shape dynamicIndices = Shape.ofDimensions(query);
        Tensor dynamic = data.gather(tensor(DataType.INT64, dynamicIndices, false), 1);

        assertEquals(Shape.ofDimensions(batch, tail), scalar.descriptor().shape());
        assertEquals(Shape.ofDimensions(batch, new StaticDimension(5), new StaticDimension(6), tail),
                ordinary.descriptor().shape());
        assertEquals(Shape.ofDimensions(batch, new StaticDimension(0), tail),
                zero.descriptor().shape());
        assertSame(query, dynamic.descriptor().shape().dimensions().get(1));
        assertSame(batch, dynamic.descriptor().shape().dimensions().get(0));
        assertSame(tail, dynamic.descriptor().shape().dimensions().get(2));
        assertOperation(dynamic, AxisGatherKind.GATHER, 1, data);
    }

    @Test
    void gatherElementsRetainsExactAlignedIndicesShape() {
        DynamicDimension batch = new DynamicDimension("batch");
        DynamicDimension selected = new DynamicDimension("selected");
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), new StaticDimension(4));
        Shape indicesShape = Shape.ofDimensions(batch, selected, new StaticDimension(4));
        Tensor data = tensor(DataType.FLOAT64, dataShape, true);
        Tensor indices = tensor(DataType.INT32, indicesShape, false);

        Tensor result = data.gatherElements(indices, -2);
        assertSame(indicesShape, result.descriptor().shape());
        assertOperation(result, AxisGatherKind.GATHER_ELEMENTS, 1, data);

        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class,
                () -> data.gatherElements(tensor(DataType.INT32, Shape.of(2, 3), false), 1));
        assertEquals(
                "gatherElements indices rank must match data rank: expected=3, actual=2",
                rank.getMessage());
        IllegalArgumentException dimension = assertThrows(
                IllegalArgumentException.class,
                () -> data.gatherElements(
                        tensor(DataType.INT32, Shape.ofDimensions(
                                new DynamicDimension("other"), selected, new StaticDimension(4)), false),
                        1));
        assertTrue(dimension.getMessage().startsWith(
                "gatherElements indices dimension at axis 0 must match data:"));
    }

    @Test
    void validatesTypesBeforeAxesAndCreatesFreshUnresolvedMetadata() {
        Tensor data = tensor(DataType.INT64, Shape.of(2, 3), false);
        for (DataType type : DataType.values()) {
            if (type == DataType.INT32 || type == DataType.INT64) {
                continue;
            }
            IllegalArgumentException gather = assertThrows(
                    IllegalArgumentException.class,
                    () -> data.gather(tensor(type, Shape.of(4), false), 9));
            assertEquals(
                    "gather indices data type must be INT32 or INT64: " + type,
                    gather.getMessage());
            IllegalArgumentException elements = assertThrows(
                    IllegalArgumentException.class,
                    () -> data.gatherElements(tensor(type, Shape.of(2, 4), false), 9));
            assertEquals(
                    "gatherElements indices data type must be INT32 or INT64: " + type,
                    elements.getMessage());
        }
        Tensor indices = tensor(DataType.INT32, Shape.of(4), false);
        Tensor first = data.gather(indices, 1);
        Tensor second = data.gather(indices, 1);
        assertNotSame(first, second);
        assertNotEquals(first.id(), second.id());
        assertSame(DataType.INT64, first.descriptor().dataType());
        assertFalse(first.descriptor().requiresGrad());
        assertTrue(first.descriptor().layout().isEmpty());
        assertTrue(first.label().isEmpty());
        assertTrue(first.hostStorage().isEmpty());
        assertEquals(0, first.provenance().orElseThrow().outputIndex());
        assertEquals(1, first.provenance().orElseThrow().producer().outputCount());
        assertThrows(IndexOutOfBoundsException.class, () -> data.gather(indices, 2));
    }

    private static void assertOperation(
            Tensor result, AxisGatherKind kind, int axis, Tensor data) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertSame(kind, provenance.operation().kind());
        assertEquals(new IndexAxisAttrs(axis), provenance.operation().attrs());
        assertSame(data, provenance.inputs().getFirst());
        assertEquals(2, provenance.inputs().size());
        assertEquals(0, provenance.outputIndex());
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }
}
