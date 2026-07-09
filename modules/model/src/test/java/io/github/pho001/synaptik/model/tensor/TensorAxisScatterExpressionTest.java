package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorAxisScatterExpressionTest {
    @Test
    void exposesOnlyScatterElementsAndUnchangedNdSurface() {
        Set<String> names = Arrays.stream(Tensor.class.getDeclaredMethods())
                .map(Method::getName)
                .filter(name -> name.startsWith("scatter"))
                .collect(Collectors.toSet());
        assertEquals(Set.of("scatterElements", "scatterNd"), names);
        assertEquals(2, Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("scatterElements"))
                .count());
        assertEquals(3, Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("scatterNd"))
                .count());
        assertEquals(
                List.of(
                        "create", "scatterElements", "scatterElements",
                        "validateIndexType", "validateMatchingDataType",
                        "validateScatterElementsShape"),
                Arrays.stream(TensorAxisScatterExpressions.class.getDeclaredMethods())
                        .map(Method::getName)
                        .sorted()
                        .toList());
        assertEquals(0, TensorAxisScatterExpressions.class.getDeclaredFields().length);
    }

    @Test
    void retainsShortAndExplicitScatterElementsBehavior() {
        DynamicDimension batch = new DynamicDimension("batch");
        Shape dataShape = Shape.ofDimensions(batch, new StaticDimension(3), new StaticDimension(4));
        Shape updateShape = Shape.ofDimensions(batch, new StaticDimension(7), new StaticDimension(4));
        Tensor data = tensor(DataType.FLOAT32, dataShape, true);
        Tensor indices = tensor(DataType.INT64, updateShape, false);
        Tensor updates = tensor(DataType.FLOAT32, updateShape, false);

        Tensor replacement = data.scatterElements(indices, updates, -2);
        Tensor addition = data.scatterElements(indices, updates, 1, ScatterReduction.ADD);
        assertOperation(replacement, new ScatterElementsAttrs(1, ScatterReduction.NONE), data, indices, updates);
        assertOperation(addition, new ScatterElementsAttrs(1, ScatterReduction.ADD), data, indices, updates);
        assertSame(dataShape, replacement.descriptor().shape());
        assertSame(DataType.FLOAT32, replacement.descriptor().dataType());
        assertTrue(replacement.descriptor().requiresGrad());
        assertTrue(replacement.descriptor().layout().isEmpty());
        assertTrue(replacement.hostStorage().isEmpty());
        assertTrue(replacement.label().isEmpty());
        assertNotSame(replacement, addition);
        assertNotEquals(replacement.id(), addition.id());
    }

    @Test
    void preservesValidationOrderAndExactFailures() {
        Tensor data = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor wrongIndices = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor updates = tensor(DataType.BOOL, Shape.of(2, 4), false);
        assertEquals(
                "scatterElements indices data type must be INT32 or INT64: FLOAT32",
                assertThrows(IllegalArgumentException.class,
                        () -> data.scatterElements(wrongIndices, updates, 9)).getMessage());
        Tensor indices = tensor(DataType.INT32, Shape.of(2, 4), false);
        assertEquals(
                "scatterElements BOOL data supports only NONE reduction: ADD",
                assertThrows(IllegalArgumentException.class,
                        () -> data.scatterElements(indices, updates, 9, ScatterReduction.ADD))
                        .getMessage());
        assertThrows(IndexOutOfBoundsException.class,
                () -> data.scatterElements(indices, updates, 2));
        assertThrows(NullPointerException.class,
                () -> data.scatterElements(null, updates, 1));
    }

    private static void assertOperation(
            Tensor result,
            ScatterElementsAttrs attrs,
            Tensor data,
            Tensor indices,
            Tensor updates) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertSame(AxisScatterKind.SCATTER_ELEMENTS, provenance.operation().kind());
        assertEquals(attrs, provenance.operation().attrs());
        assertEquals(List.of(data, indices, updates), provenance.inputs());
        assertEquals(0, provenance.outputIndex());
        assertEquals(1, provenance.producer().outputCount());
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.empty(), requiresGrad));
    }
}
