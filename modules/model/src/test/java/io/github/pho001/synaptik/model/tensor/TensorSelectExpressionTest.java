package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorSelectExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(100_000);

    @Test
    void exposesOneExactPublicMethodAndFiveMethodStatelessHelper() throws Exception {
        Method select = Tensor.class.getDeclaredMethod("select", int.class, long.class);
        var constructor = TensorSelectExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(TensorSelectExpressions.class.getDeclaredMethods());
        Method apply = TensorSelectExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, int.class, long.class);
        Method normalizeIndex = TensorSelectExpressions.class.getDeclaredMethod(
                "normalizeIndex", Dimension.class, int.class, long.class);
        Method removeAxis = TensorSelectExpressions.class.getDeclaredMethod(
                "removeAxis", Shape.class, int.class);
        Method resolveViewLayout = TensorSelectExpressions.class.getDeclaredMethod(
                "resolveViewLayout", TensorDescriptor.class, Shape.class, int.class, long.class);
        Method create = TensorSelectExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, Optional.class,
                SelectAttrs.class);

        assertAll(
                () -> assertSame(Tensor.class, select.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, long.class),
                        Arrays.asList(select.getParameterTypes())),
                () -> assertFalse(select.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(select.getModifiers())),
                () -> assertFalse(Modifier.isStatic(select.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(select.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorSelectExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorSelectExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorSelectExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorSelectExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorSelectExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(5, methods.size()),
                () -> assertEquals(
                        Set.of("apply", "normalizeIndex", "removeAxis", "resolveViewLayout",
                                "create"),
                        methods.stream().map(Method::getName).collect(Collectors.toSet())),
                () -> assertEquals(1, methods.stream()
                        .filter(method -> !Modifier.isPrivate(method.getModifiers())).count()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertSame(Tensor.class, apply.getReturnType()),
                () -> assertSame(long.class, normalizeIndex.getReturnType()),
                () -> assertSame(Shape.class, removeAxis.getReturnType()),
                () -> assertSame(Optional.class, resolveViewLayout.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(normalizeIndex.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(removeAxis.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolveViewLayout.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())));
    }

    @Test
    void normalizesPositiveAndNegativeRequestsAndRecordsExactSemantics() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true);

        Tensor positive = input.select(1, 2);
        Tensor negative = input.select(-2, -1);
        TensorProvenance positiveProvenance = positive.provenance().orElseThrow();
        TensorProvenance negativeProvenance = negative.provenance().orElseThrow();
        SelectAttrs positiveAttrs = (SelectAttrs) positiveProvenance.operation().attrs();
        SelectAttrs negativeAttrs = (SelectAttrs) negativeProvenance.operation().attrs();

        assertAll(
                () -> assertEquals(Shape.of(2, 4), positive.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4), negative.descriptor().shape()),
                () -> assertEquals(new SelectAttrs(1, 2), positiveAttrs),
                () -> assertEquals(new SelectAttrs(1, 2), negativeAttrs),
                () -> assertSame(SelectKind.SELECT, positiveProvenance.operation().kind()),
                () -> assertSame(SelectKind.SELECT, negativeProvenance.operation().kind()),
                () -> assertSame(positiveAttrs, positiveProvenance.operation().attrs()),
                () -> assertEquals(List.of(input), positiveProvenance.inputs()),
                () -> assertSame(input, positiveProvenance.inputs().getFirst()),
                () -> assertEquals(List.of(input), negativeProvenance.inputs()),
                () -> assertTrue(positive.label().isEmpty()),
                () -> assertTrue(positive.hostStorage().isEmpty()),
                () -> assertTrue(negative.label().isEmpty()),
                () -> assertTrue(negative.hostStorage().isEmpty()));
    }

    @Test
    void rejectsNullAxesAndStaticIndicesWithExactMessagesWithoutConsumingIdentity()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true);
        Tensor zeroExtent = tensor(DataType.INT32, Shape.of(2, 0), Optional.empty(), false);
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorSelectExpressions.apply(null, 0, 0));
        IndexOutOfBoundsException lowAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(-3, 0));
        IndexOutOfBoundsException minimumAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(Integer.MIN_VALUE, 0));
        IndexOutOfBoundsException highAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(2, 0));
        IndexOutOfBoundsException scalarAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> scalar.select(0, 0));
        IndexOutOfBoundsException negativeLow = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(1, -4));
        IndexOutOfBoundsException highIndex = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(1, 3));
        IndexOutOfBoundsException maximumIndex = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(1, Long.MAX_VALUE));
        IndexOutOfBoundsException minimumIndex = assertThrows(
                IndexOutOfBoundsException.class, () -> input.select(1, Long.MIN_VALUE));
        IndexOutOfBoundsException zero = assertThrows(
                IndexOutOfBoundsException.class, () -> zeroExtent.select(1, 0));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("Axis -3 is outside shape rank 2", lowAxis.getMessage()),
                () -> assertEquals(
                        "Axis -2147483648 is outside shape rank 2", minimumAxis.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 2", highAxis.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalarAxis.getMessage()),
                () -> assertEquals(
                        "select index -4 is outside axis 1 extent 3", negativeLow.getMessage()),
                () -> assertEquals(
                        "select index 3 is outside axis 1 extent 3", highIndex.getMessage()),
                () -> assertEquals(
                        "select index 9223372036854775807 is outside axis 1 extent 3",
                        maximumIndex.getMessage()),
                () -> assertEquals(
                        "select index -9223372036854775808 is outside axis 1 extent 3",
                        minimumIndex.getMessage()),
                () -> assertEquals(
                        "select index 0 is outside axis 1 extent 0", zero.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void acceptsDynamicNonNegativeIndexRemovesSelectedAxisAndPreservesOtherReferences() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension channels = new StaticDimension(4);
        DynamicDimension width = new DynamicDimension("width");
        Shape shape = Shape.ofDimensions(batch, channels, width);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);

        Tensor selectedDynamic = input.select(0, Long.MAX_VALUE);
        Shape dynamicResultShape = selectedDynamic.descriptor().shape();
        SelectAttrs attrs = (SelectAttrs) selectedDynamic.provenance().orElseThrow()
                .operation().attrs();
        Tensor selectedStatic = input.select(1, -1);
        Shape staticResultShape = selectedStatic.descriptor().shape();
        IllegalArgumentException negativeDynamic = assertThrows(
                IllegalArgumentException.class, () -> input.select(-1, -1));

        assertAll(
                () -> assertEquals(new SelectAttrs(0, Long.MAX_VALUE), attrs),
                () -> assertEquals(2, dynamicResultShape.rank()),
                () -> assertSame(channels, dynamicResultShape.dimensions().get(0)),
                () -> assertSame(width, dynamicResultShape.dimensions().get(1)),
                () -> assertTrue(selectedDynamic.descriptor().layout().isEmpty()),
                () -> assertSame(batch, staticResultShape.dimensions().get(0)),
                () -> assertSame(width, staticResultShape.dimensions().get(1)),
                () -> assertEquals(
                        "select index -1 cannot be normalized against dynamic axis 2",
                        negativeDynamic.getMessage()));
    }

    @Test
    void supportsRankOneScalarAndEveryDataTypeEligibilityChoice() {
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, Shape.of(1), Optional.empty(), requiresGrad);

                Tensor result = input.select(-1, -1);

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(Shape.scalar(), result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertEquals(new SelectAttrs(0, 0), result.provenance()
                                .map(TensorProvenance::operation)
                                .map(Operation::attrs)
                                .orElseThrow()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void derivesSelectedStrideRemovalAndCheckedOffsetForResolvedLayoutKinds() {
        Shape shape = Shape.of(2, 3, 4);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(
                shape, new long[] {12, 4, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(
                shape, new long[] {20, 5, 1}, 3, true);
        LayoutDescriptor zeroSelectedStride = LayoutDescriptor.of(
                shape, new long[] {12, 0, 1}, 2, true);

        LayoutDescriptor denseResult = selectedLayout(shape, dense, 1, 2);
        LayoutDescriptor offsetResult = selectedLayout(shape, offset, 1, 2);
        LayoutDescriptor stridedResult = selectedLayout(shape, strided, 1, 2);
        LayoutDescriptor zeroStrideResult = selectedLayout(shape, zeroSelectedStride, 1, 2);

        assertAll(
                () -> assertArrayEquals(new long[] {12, 1}, denseResult.strides()),
                () -> assertEquals(8, denseResult.storageOffset()),
                () -> assertSame(LayoutKind.STRIDED, denseResult.kind()),
                () -> assertEquals(24, denseResult.referencedElementSpan()),
                () -> assertTrue(denseResult.isView()),
                () -> assertNotSame(dense, denseResult),
                () -> assertArrayEquals(new long[] {12, 1}, offsetResult.strides()),
                () -> assertEquals(13, offsetResult.storageOffset()),
                () -> assertEquals(29, offsetResult.referencedElementSpan()),
                () -> assertNotSame(offset, offsetResult),
                () -> assertArrayEquals(new long[] {20, 1}, stridedResult.strides()),
                () -> assertEquals(13, stridedResult.storageOffset()),
                () -> assertEquals(37, stridedResult.referencedElementSpan()),
                () -> assertNotSame(strided, stridedResult),
                () -> assertArrayEquals(new long[] {12, 1}, zeroStrideResult.strides()),
                () -> assertEquals(2, zeroStrideResult.storageOffset()),
                () -> assertEquals(18, zeroStrideResult.referencedElementSpan()),
                () -> assertNotSame(zeroSelectedStride, zeroStrideResult));
    }

    @Test
    void leavesUnresolvedAndEmptyResultGeometryUnresolved() {
        Tensor staticUnresolved = tensor(
                DataType.INT64, Shape.of(2, 3, 4), Optional.empty(), false);
        Shape emptyShape = Shape.of(2, 3, 0);
        LayoutDescriptor emptyLayout = LayoutDescriptor.contiguous(emptyShape);
        Tensor emptyInput = tensor(
                DataType.FLOAT64, emptyShape, Optional.of(emptyLayout), true);

        Tensor unresolved = staticUnresolved.select(1, 1);
        Tensor empty = emptyInput.select(1, 1);

        assertAll(
                () -> assertEquals(Shape.of(2, 4), unresolved.descriptor().shape()),
                () -> assertTrue(unresolved.descriptor().layout().isEmpty()),
                () -> assertEquals(Shape.of(2, 0), empty.descriptor().shape()),
                () -> assertTrue(empty.descriptor().layout().isEmpty()));
    }

    @Test
    void repeatedSameCoordinateAndNestedRequestsRemainFreshExplicitExpressions() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true);

        Tensor first = input.select(1, 2);
        Tensor second = input.select(1, 2);
        Tensor nested = first.select(0, 1);

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(first, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(first.id(), nested.id()),
                () -> assertSame(input, first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(input, second.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().getFirst()),
                () -> assertEquals(Shape.of(4), nested.descriptor().shape()),
                () -> assertEquals(new SelectAttrs(0, 1), nested.provenance()
                        .map(TensorProvenance::operation)
                        .map(Operation::attrs)
                        .orElseThrow()));
    }

    @Test
    void leavesInputMetadataStorageLivenessAndValuesUntouched() {
        Shape shape = Shape.of(2, 3, 4);
        LayoutDescriptor inputLayout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 24, arena.allocate(96, 1));
        float[] values = new float[24];
        for (int index = 0; index < values.length; index++) {
            values[index] = index + 0.5f;
        }
        storage.segment().copyFrom(java.lang.foreign.MemorySegment.ofArray(values));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.empty(),
                Optional.of(storage));
        float[] before = storage.segment().toArray(JAVA_FLOAT);

        Tensor result = input.select(1, 2);
        float[] after = storage.segment().toArray(JAVA_FLOAT);
        arena.close();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputLayout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertFalse(storage.isAlive()),
                () -> assertArrayEquals(before, after),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void successfulCallConsumesExactlyOneIdentityAfterLocalConstruction() throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 3, 4),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3, 4))),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        input.select(1, 2);

        assertEquals(before + 1, next.get());
    }

    private static LayoutDescriptor selectedLayout(
            Shape shape, LayoutDescriptor layout, int axis, long index) {
        return tensor(DataType.FLOAT32, shape, Optional.of(layout), false)
                .select(axis, index)
                .descriptor()
                .layout()
                .orElseThrow();
    }

    private static List<Boolean> validGradientChoices(DataType dataType) {
        return dataType.isDifferentiable() ? List.of(false, true) : List.of(false);
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
