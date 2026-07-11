package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class TensorTopKExpressionTest {
    @Test
    void helperMethodsAndResultExposeOnlyTheRequiredSurface() throws Exception {
        var apply = TensorTopKExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, long.class, int.class, boolean.class, boolean.class);
        var helperMethods = Arrays.stream(TensorTopKExpressions.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic()).toList();
        var constructors = TensorTopKExpressions.class.getDeclaredConstructors();
        var components = TopKResult.class.getRecordComponents();
        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorTopKExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorTopKExpressions.class.getModifiers())),
                () -> assertEquals(Set.of(), Set.of(TensorTopKExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorTopKExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorTopKExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(List.of(apply), helperMethods),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertSame(TopKResult.class, apply.getReturnType()),
                () -> assertTensorMethod("topK", long.class, int.class),
                () -> assertTensorMethod("topK", long.class, int.class,
                        boolean.class, boolean.class),
                () -> assertTrue(TopKResult.class.isRecord()),
                () -> assertEquals(List.of("values", "indices"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertEquals(List.of(Tensor.class, Tensor.class),
                        Arrays.stream(components).map(component -> component.getType()).toList()),
                () -> assertEquals(0, TopKResult.class.getDeclaredClasses().length));
    }

    @Test
    void defaultsToLargestSortedAndNormalizesAxisExactlyOnce() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 4), true);
        TopKResult defaults = input.topK(3, -1);
        TopKResult explicit = input.topK(2, 0, false, false);
        assertAll(
                () -> assertEquals(new TopKAttrs(1, 3, true, true),
                        defaults.values().provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(new TopKAttrs(0, 2, false, false),
                        explicit.indices().provenance().orElseThrow().operation().attrs()));
    }

    @Test
    void acceptsEveryCurrentTypeAndDerivesExactOutputMetadata() {
        Shape inputShape = Shape.of(2, 4);
        for (DataType type : DataType.values()) {
            for (boolean requiresGrad : type.isFloating() ? List.of(false, true) : List.of(false)) {
                TopKResult result = tensor(type, inputShape, requiresGrad).topK(3, 1);
                Tensor values = result.values();
                Tensor indices = result.indices();
                assertAll(
                        () -> assertSame(type, values.descriptor().dataType()),
                        () -> assertEquals(requiresGrad, values.descriptor().requiresGrad()),
                        () -> assertSame(DataType.INT64, indices.descriptor().dataType()),
                        () -> assertFalse(indices.descriptor().requiresGrad()),
                        () -> assertSame(values.descriptor().shape(), indices.descriptor().shape()),
                        () -> assertEquals(Shape.of(2, 3), values.descriptor().shape()),
                        () -> assertTrue(values.descriptor().layout().isEmpty()),
                        () -> assertTrue(indices.descriptor().layout().isEmpty()),
                        () -> assertTrue(values.label().isEmpty()),
                        () -> assertTrue(indices.label().isEmpty()),
                        () -> assertTrue(values.hostStorage().isEmpty()),
                        () -> assertTrue(indices.hostStorage().isEmpty()));
            }
        }
    }

    @Test
    void replacesOnlySelectedDimensionAndDefersNonStaticBounds() {
        Dimension left = new DynamicDimension("N");
        Dimension selectedDynamic = new DynamicDimension("K");
        Dimension expression = DimensionExpressions.addConstant(new DynamicDimension("M"), 2);
        Shape dynamicShape = Shape.ofDimensions(left, selectedDynamic, expression);
        Shape dynamicResult = tensor(DataType.INT64, dynamicShape, false)
                .topK(Long.MAX_VALUE, 1).values().descriptor().shape();
        Shape expressionShape = Shape.ofDimensions(left, expression);
        Shape expressionResult = tensor(DataType.BOOL, expressionShape, false)
                .topK(0, -1).values().descriptor().shape();
        assertAll(
                () -> assertSame(left, dynamicResult.dimensions().get(0)),
                () -> assertEquals(new StaticDimension(Long.MAX_VALUE),
                        dynamicResult.dimensions().get(1)),
                () -> assertNotSame(selectedDynamic, dynamicResult.dimensions().get(1)),
                () -> assertSame(expression, dynamicResult.dimensions().get(2)),
                () -> assertSame(left, expressionResult.dimensions().get(0)),
                () -> assertEquals(new StaticDimension(0), expressionResult.dimensions().get(1)),
                () -> assertNotSame(expression, expressionResult.dimensions().get(1)));
    }

    @Test
    void handlesZeroEmptySingletonAndMaximumStaticExtents() {
        assertAll(
                () -> assertEquals(Shape.of(2, 0), tensor(DataType.BOOL, Shape.of(2, 0), false)
                        .topK(0, 1).values().descriptor().shape()),
                () -> assertEquals(Shape.of(1), tensor(DataType.INT32, Shape.of(1), false)
                        .topK(1, 0).values().descriptor().shape()),
                () -> assertEquals(Shape.of(Long.MAX_VALUE),
                        tensor(DataType.FLOAT64, Shape.of(Long.MAX_VALUE), true)
                                .topK(Long.MAX_VALUE, 0).values().descriptor().shape()));
    }

    @Test
    void createsExactlyTwoIndexedWrappersUnderOneExactProducer() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(5), true);
        long before = next.get();
        TopKResult result = input.topK(3, 0, true, false);
        Tensor values = result.values();
        Tensor indices = result.indices();
        TensorProvenance valuesOrigin = values.provenance().orElseThrow();
        TensorProvenance indicesOrigin = indices.provenance().orElseThrow();
        TensorProducer producer = valuesOrigin.producer();
        assertAll(
                () -> assertEquals(before, values.id().value()),
                () -> assertEquals(before + 1, indices.id().value()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertSame(producer, indicesOrigin.producer()),
                () -> assertSame(TopKKind.TOP_K, producer.operation().kind()),
                () -> assertEquals(new TopKAttrs(0, 3, true, false),
                        producer.operation().attrs()),
                () -> assertEquals(List.of(input), producer.inputs()),
                () -> assertEquals(2, producer.outputCount()),
                () -> assertEquals(0, valuesOrigin.outputIndex()),
                () -> assertEquals(1, indicesOrigin.outputIndex()),
                () -> assertSame(values.descriptor(), producer.outputDescriptors().get(0)),
                () -> assertSame(indices.descriptor(), producer.outputDescriptors().get(1)));
    }

    @Test
    void repeatedCallsCreateIndependentOccurrencesWithoutMutatingInput() {
        Shape shape = Shape.of(4);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT64, shape, Optional.of(LayoutDescriptor.contiguous(shape)), true);
        Tensor input = TensorFactory.allocate(descriptor, Optional.of("input"));
        TopKResult first = input.topK(2, 0);
        TopKResult second = input.topK(2, 0);
        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotSame(first.values(), second.values()),
                () -> assertNotEquals(first.values().id(), second.values().id()),
                () -> assertNotSame(first.values().descriptor(), second.values().descriptor()),
                () -> assertNotSame(first.values().descriptor().shape(),
                        second.values().descriptor().shape()),
                () -> assertNotSame(first.values().provenance().orElseThrow().producer(),
                        second.values().provenance().orElseThrow().producer()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertTrue(input.hostStorage().isPresent()),
                () -> assertEquals(Optional.of("input"), input.label()));
    }

    @Test
    void validatesInExactOrderAndConsumesNoIdsForLocalFailures() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor scalar = tensor(DataType.INT64, Shape.scalar(), false);
        Tensor vector = tensor(DataType.BOOL, Shape.of(2), false);
        long before = next.get();
        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> TensorTopKExpressions.apply(null, -1, 0, true, true)).getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0",
                        assertThrows(IndexOutOfBoundsException.class,
                                () -> scalar.topK(-1, 0)).getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 1",
                        assertThrows(IndexOutOfBoundsException.class,
                                () -> vector.topK(-1, 2)).getMessage()),
                () -> assertEquals("k must be non-negative: -1",
                        assertThrows(IllegalArgumentException.class,
                                () -> vector.topK(-1, 0)).getMessage()),
                () -> assertEquals(
                        "k must not exceed selected static extent: k=3, axis=0, extent=2",
                        assertThrows(IllegalArgumentException.class,
                                () -> vector.topK(3, 0)).getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void resultRejectsNullInOrderAndRetainsExactReferencesWithValueSemantics() {
        Tensor values = tensor(DataType.FLOAT32, Shape.of(0), true);
        Tensor indices = tensor(DataType.INT64, Shape.of(0), false);
        TopKResult result = new TopKResult(values, indices);
        assertAll(
                () -> assertEquals("values", assertThrows(NullPointerException.class,
                        () -> new TopKResult(null, null)).getMessage()),
                () -> assertEquals("indices", assertThrows(NullPointerException.class,
                        () -> new TopKResult(values, null)).getMessage()),
                () -> assertSame(values, result.values()),
                () -> assertSame(indices, result.indices()),
                () -> assertEquals(result, new TopKResult(values, indices)),
                () -> assertEquals(result.hashCode(), new TopKResult(values, indices).hashCode()),
                () -> assertEquals("TopKResult[values=" + values + ", indices=" + indices + "]",
                        result.toString()));
    }

    private static void assertTensorMethod(String name, Class<?>... parameterTypes) throws Exception {
        var method = Tensor.class.getDeclaredMethod(name, parameterTypes);
        assertAll(
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertSame(TopKResult.class, method.getReturnType()));
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
    }
}
