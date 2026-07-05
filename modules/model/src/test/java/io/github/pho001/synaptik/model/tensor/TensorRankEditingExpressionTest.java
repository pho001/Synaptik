package io.github.pho001.synaptik.model.tensor;

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
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
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
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorRankEditingExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(80_000);

    @Test
    void exposesExactlyTwoPublicMethodsAndNineMethodStatelessHelper() throws Exception {
        Method expandDims = Tensor.class.getDeclaredMethod("expandDims", int.class);
        Method squeeze = Tensor.class.getDeclaredMethod("squeeze", int.class);
        var constructor = TensorRankEditingExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(
                TensorRankEditingExpressions.class.getDeclaredMethods());
        Method normalize = TensorRankEditingExpressions.class.getDeclaredMethod(
                "normalizeInsertionAxis", int.class, int.class);
        Method validate = TensorRankEditingExpressions.class.getDeclaredMethod(
                "validateSqueezableDimension", Shape.class, int.class);
        Method insert = TensorRankEditingExpressions.class.getDeclaredMethod(
                "insertSingleton", Shape.class, int.class);
        Method remove = TensorRankEditingExpressions.class.getDeclaredMethod(
                "removeSingleton", Shape.class, int.class);
        Method resolveInserted = TensorRankEditingExpressions.class.getDeclaredMethod(
                "resolveInsertedLayout", TensorDescriptor.class, Shape.class, Shape.class,
                int.class);
        Method resolveSqueezed = TensorRankEditingExpressions.class.getDeclaredMethod(
                "resolveSqueezedLayout", TensorDescriptor.class, Shape.class, int.class);
        Method create = TensorRankEditingExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, Optional.class,
                AxisTransformKind.class, int.class);

        assertAll(
                () -> assertSame(Tensor.class, expandDims.getReturnType()),
                () -> assertEquals(List.of(int.class),
                        Arrays.asList(expandDims.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(expandDims.getModifiers())),
                () -> assertFalse(Modifier.isStatic(expandDims.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(expandDims.getModifiers())),
                () -> assertFalse(expandDims.isVarArgs()),
                () -> assertSame(Tensor.class, squeeze.getReturnType()),
                () -> assertEquals(List.of(int.class),
                        Arrays.asList(squeeze.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(squeeze.getModifiers())),
                () -> assertFalse(Modifier.isStatic(squeeze.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(squeeze.getModifiers())),
                () -> assertFalse(squeeze.isVarArgs()),
                () -> assertTrue(Modifier.isFinal(
                        TensorRankEditingExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorRankEditingExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorRankEditingExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorRankEditingExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorRankEditingExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(9, methods.size()),
                () -> assertEquals(
                        Set.of("expandDims", "squeeze", "normalizeInsertionAxis",
                                "validateSqueezableDimension", "insertSingleton",
                                "removeSingleton", "resolveInsertedLayout",
                                "resolveSqueezedLayout", "create"),
                        methods.stream().map(Method::getName).collect(Collectors.toSet())),
                () -> assertEquals(2, methods.stream()
                        .filter(method -> !Modifier.isPrivate(method.getModifiers())).count()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertSame(int.class, normalize.getReturnType()),
                () -> assertSame(void.class, validate.getReturnType()),
                () -> assertSame(Shape.class, insert.getReturnType()),
                () -> assertSame(Shape.class, remove.getReturnType()),
                () -> assertSame(Optional.class, resolveInserted.getReturnType()),
                () -> assertSame(Optional.class, resolveSqueezed.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(normalize.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(validate.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(insert.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(remove.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolveInserted.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolveSqueezed.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())));
    }

    @Test
    void preservesEveryDataTypeEligibilityAndExactSemantics() {
        Shape shape = Shape.of(2, 1, 3);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, shape, Optional.empty(), requiresGrad);

                Tensor expanded = input.expandDims(1);
                Tensor squeezed = input.squeeze(1);
                TensorProvenance expandedProvenance = expanded.provenance().orElseThrow();
                TensorProvenance squeezedProvenance = squeezed.provenance().orElseThrow();

                assertAll(
                        () -> assertSame(dataType, expanded.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2, 1, 1, 3),
                                expanded.descriptor().shape()),
                        () -> assertEquals(requiresGrad,
                                expanded.descriptor().requiresGrad()),
                        () -> assertSame(AxisTransformKind.EXPAND_DIMS,
                                expandedProvenance.operation().kind()),
                        () -> assertEquals(new AxisTransformAttrs(1),
                                expandedProvenance.operation().attrs()),
                        () -> assertEquals(List.of(input), expandedProvenance.inputs()),
                        () -> assertSame(dataType, squeezed.descriptor().dataType()),
                        () -> assertEquals(Shape.of(2, 3), squeezed.descriptor().shape()),
                        () -> assertEquals(requiresGrad,
                                squeezed.descriptor().requiresGrad()),
                        () -> assertSame(AxisTransformKind.SQUEEZE,
                                squeezedProvenance.operation().kind()),
                        () -> assertEquals(new AxisTransformAttrs(1),
                                squeezedProvenance.operation().attrs()),
                        () -> assertSame(input, squeezedProvenance.inputs().getFirst()),
                        () -> assertTrue(expanded.label().isEmpty()),
                        () -> assertTrue(expanded.hostStorage().isEmpty()),
                        () -> assertTrue(squeezed.label().isEmpty()),
                        () -> assertTrue(squeezed.hostStorage().isEmpty()),
                        () -> assertNotSame(input, expanded),
                        () -> assertNotSame(input, squeezed));
            }
        }
    }

    @Test
    void normalizesInsertionBoundariesAndPreservesExactDimensionReferences() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension width = new StaticDimension(4);
        Shape shape = Shape.ofDimensions(batch, width);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);

        Tensor negativeStart = input.expandDims(-3);
        Tensor positiveStart = input.expandDims(0);
        Tensor negativeEnd = input.expandDims(-1);
        Tensor positiveEnd = input.expandDims(2);
        Tensor scalarNegative = tensor(
                DataType.INT64, Shape.scalar(), Optional.empty(), false).expandDims(-1);
        Tensor scalarPositive = tensor(
                DataType.BOOL, Shape.scalar(), Optional.empty(), false).expandDims(0);

        assertAll(
                () -> assertEquals(new AxisTransformAttrs(0), attrs(negativeStart)),
                () -> assertEquals(new AxisTransformAttrs(0), attrs(positiveStart)),
                () -> assertEquals(new AxisTransformAttrs(2), attrs(negativeEnd)),
                () -> assertEquals(new AxisTransformAttrs(2), attrs(positiveEnd)),
                () -> assertEquals(Shape.of(1), scalarNegative.descriptor().shape()),
                () -> assertEquals(new AxisTransformAttrs(0), attrs(scalarNegative)),
                () -> assertEquals(Shape.of(1), scalarPositive.descriptor().shape()),
                () -> assertSame(batch,
                        negativeStart.descriptor().shape().dimensions().get(1)),
                () -> assertSame(width,
                        negativeStart.descriptor().shape().dimensions().get(2)),
                () -> assertSame(batch,
                        negativeEnd.descriptor().shape().dimensions().get(0)),
                () -> assertSame(width,
                        negativeEnd.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(new StaticDimension(1),
                        negativeEnd.descriptor().shape().dimensions().get(2)));
    }

    @Test
    void rejectsInvalidInsertionAxesWithExactMessagesAndNoIdentityConsumption() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorRankEditingExpressions.expandDims(null, 0));
        IndexOutOfBoundsException below = assertThrows(
                IndexOutOfBoundsException.class, () -> input.expandDims(-4));
        IndexOutOfBoundsException minimum = assertThrows(
                IndexOutOfBoundsException.class, () -> input.expandDims(Integer.MIN_VALUE));
        IndexOutOfBoundsException above = assertThrows(
                IndexOutOfBoundsException.class, () -> input.expandDims(3));
        IndexOutOfBoundsException maximum = assertThrows(
                IndexOutOfBoundsException.class, () -> input.expandDims(Integer.MAX_VALUE));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals(
                        "Axis -4 is outside insertion range for shape rank 2",
                        below.getMessage()),
                () -> assertEquals(
                        "Axis -2147483648 is outside insertion range for shape rank 2",
                        minimum.getMessage()),
                () -> assertEquals(
                        "Axis 3 is outside insertion range for shape rank 2",
                        above.getMessage()),
                () -> assertEquals(
                        "Axis 2147483647 is outside insertion range for shape rank 2",
                        maximum.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void squeezesPositiveNegativeAndRankOneAxesWithExactDimensionIdentity() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension singleton = new StaticDimension(1);
        StaticDimension width = new StaticDimension(4);
        Shape shape = Shape.ofDimensions(batch, singleton, width);
        Tensor input = tensor(DataType.FLOAT64, shape, Optional.empty(), true);

        Tensor positive = input.squeeze(1);
        Tensor negative = input.squeeze(-2);
        Tensor scalar = tensor(
                DataType.INT32, Shape.of(1), Optional.empty(), false).squeeze(0);

        assertAll(
                () -> assertEquals(Shape.ofDimensions(batch, width),
                        positive.descriptor().shape()),
                () -> assertSame(batch, positive.descriptor().shape().dimensions().get(0)),
                () -> assertSame(width, positive.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(new AxisTransformAttrs(1), attrs(positive)),
                () -> assertEquals(positive.descriptor().shape(), negative.descriptor().shape()),
                () -> assertEquals(new AxisTransformAttrs(1), attrs(negative)),
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertEquals(new AxisTransformAttrs(0), attrs(scalar)));
    }

    @Test
    void rejectsInvalidOrUnprovedSqueezeAxesWithExactMessagesAndNoIdentityConsumption()
            throws Exception {
        DynamicDimension dynamic = new DynamicDimension("items");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(
                        new StaticDimension(1),
                        new StaticDimension(0),
                        new StaticDimension(2),
                        dynamic),
                Optional.empty(),
                true);
        Tensor scalar = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorRankEditingExpressions.squeeze(null, 0));
        IndexOutOfBoundsException low = assertThrows(
                IndexOutOfBoundsException.class, () -> input.squeeze(-5));
        IndexOutOfBoundsException high = assertThrows(
                IndexOutOfBoundsException.class, () -> input.squeeze(4));
        IndexOutOfBoundsException scalarFailure = assertThrows(
                IndexOutOfBoundsException.class, () -> scalar.squeeze(0));
        IllegalArgumentException zero = assertThrows(
                IllegalArgumentException.class, () -> input.squeeze(1));
        IllegalArgumentException nonOne = assertThrows(
                IllegalArgumentException.class, () -> input.squeeze(2));
        IllegalArgumentException unknown = assertThrows(
                IllegalArgumentException.class, () -> input.squeeze(-1));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("Axis -5 is outside shape rank 4", low.getMessage()),
                () -> assertEquals("Axis 4 is outside shape rank 4", high.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalarFailure.getMessage()),
                () -> assertEquals(
                        "cannot squeeze axis 1 of Shape[1, 0, 2, items]: "
                                + "dimension must be statically known as 1",
                        zero.getMessage()),
                () -> assertEquals(
                        "cannot squeeze axis 2 of Shape[1, 0, 2, items]: "
                                + "dimension must be statically known as 1",
                        nonOne.getMessage()),
                () -> assertEquals(
                        "cannot squeeze axis 3 of Shape[1, 0, 2, items]: "
                                + "dimension must be statically known as 1",
                        unknown.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void insertsStridesForEveryResolvedLayoutKindAndAtTheEnd() {
        Shape shape = Shape.of(2, 3);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(shape, new long[] {3, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(shape, new long[] {1, 2}, 4, true);
        LayoutDescriptor broadcast = LayoutDescriptor.of(shape, new long[] {0, 1}, 6, true);

        LayoutDescriptor denseResult = expandLayout(shape, dense, 0);
        LayoutDescriptor offsetResult = expandLayout(shape, offset, 0);
        LayoutDescriptor stridedResult = expandLayout(shape, strided, 1);
        LayoutDescriptor broadcastResult = expandLayout(shape, broadcast, 1);
        LayoutDescriptor endResult = expandLayout(shape, dense, 2);
        LayoutDescriptor scalarResult = expandLayout(
                Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), 0);
        Shape emptyShape = Shape.of(0);
        LayoutDescriptor emptyResult = expandLayout(
                emptyShape, LayoutDescriptor.contiguous(emptyShape), 1);

        assertAll(
                () -> assertLayout(denseResult, new long[] {6, 3, 1}, 0,
                        LayoutKind.DENSE_CONTIGUOUS, 6),
                () -> assertLayout(offsetResult, new long[] {6, 3, 1}, 5,
                        LayoutKind.DENSE_WITH_OFFSET, 11),
                () -> assertLayout(stridedResult, new long[] {1, 6, 2}, 4,
                        LayoutKind.STRIDED, 10),
                () -> assertLayout(broadcastResult, new long[] {0, 3, 1}, 6,
                        LayoutKind.BROADCAST_ZERO_STRIDE, 9),
                () -> assertLayout(endResult, new long[] {3, 1, 1}, 0,
                        LayoutKind.DENSE_CONTIGUOUS, 6),
                () -> assertLayout(scalarResult, new long[] {1}, 0,
                        LayoutKind.DENSE_CONTIGUOUS, 1),
                () -> assertLayout(emptyResult, new long[] {1, 1}, 0,
                        LayoutKind.DENSE_CONTIGUOUS, 0),
                () -> assertNotSame(dense, denseResult),
                () -> assertNotSame(offset, offsetResult),
                () -> assertNotSame(strided, stridedResult),
                () -> assertNotSame(broadcast, broadcastResult));
    }

    @Test
    void removesStridesForEveryResolvedLayoutKindAndPreservesOffset() {
        Shape shape = Shape.of(2, 1, 3);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(shape, new long[] {3, 3, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(shape, new long[] {1, 7, 2}, 4, true);
        LayoutDescriptor broadcast = LayoutDescriptor.of(shape, new long[] {0, 3, 1}, 6, true);

        LayoutDescriptor denseResult = squeezeLayout(shape, dense, 1);
        LayoutDescriptor offsetResult = squeezeLayout(shape, offset, -2);
        LayoutDescriptor stridedResult = squeezeLayout(shape, strided, 1);
        LayoutDescriptor broadcastResult = squeezeLayout(shape, broadcast, 1);
        Shape rankOneShape = Shape.of(1);
        LayoutDescriptor scalarResult = squeezeLayout(
                rankOneShape, LayoutDescriptor.contiguous(rankOneShape), 0);

        assertAll(
                () -> assertLayout(denseResult, new long[] {3, 1}, 0,
                        LayoutKind.DENSE_CONTIGUOUS, 6),
                () -> assertLayout(offsetResult, new long[] {3, 1}, 5,
                        LayoutKind.DENSE_WITH_OFFSET, 11),
                () -> assertLayout(stridedResult, new long[] {1, 2}, 4,
                        LayoutKind.STRIDED, 10),
                () -> assertLayout(broadcastResult, new long[] {0, 1}, 6,
                        LayoutKind.BROADCAST_ZERO_STRIDE, 9),
                () -> assertLayout(scalarResult, new long[0], 0,
                        LayoutKind.DENSE_CONTIGUOUS, 1),
                () -> assertNotSame(dense, denseResult),
                () -> assertNotSame(offset, offsetResult),
                () -> assertNotSame(strided, stridedResult),
                () -> assertNotSame(broadcast, broadcastResult));
    }

    @Test
    void leavesUnresolvedGeometryUnresolvedAndChecksInsertedStrideBeforeIdentity() throws Exception {
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(1));
        Tensor dynamic = tensor(DataType.FLOAT32, dynamicShape, Optional.empty(), true);
        Tensor staticUnresolved = tensor(
                DataType.INT32, Shape.of(1, 3), Optional.empty(), false);
        Shape overflowShape = Shape.of(2, 0);
        LayoutDescriptor overflowLayout = LayoutDescriptor.of(
                overflowShape, new long[] {Long.MAX_VALUE, 1}, 0, true);
        Tensor overflow = tensor(
                DataType.FLOAT64, overflowShape, Optional.of(overflowLayout), false);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        ArithmeticException failure = assertThrows(
                ArithmeticException.class, () -> overflow.expandDims(0));

        assertAll(
                () -> assertTrue(dynamic.expandDims(1).descriptor().layout().isEmpty()),
                () -> assertTrue(staticUnresolved.expandDims(0)
                        .descriptor().layout().isEmpty()),
                () -> assertTrue(staticUnresolved.squeeze(0)
                        .descriptor().layout().isEmpty()),
                () -> assertEquals("long overflow", failure.getMessage()),
                () -> assertEquals(before + 3, next.get()));
    }

    @Test
    void repeatedNestedInverseLikeAndDeadStorageCallsRemainFreshWithoutInspection() {
        Shape shape = Shape.of(2, 1);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 2, arena.allocate(8, 1));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.empty(),
                Optional.of(storage));
        arena.close();

        Tensor first = input.expandDims(1);
        Tensor second = input.expandDims(1);
        Tensor inverseLike = first.squeeze(1);
        Tensor nested = first.expandDims(-1);

        assertAll(
                () -> assertFalse(storage.isAlive()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(input, inverseLike),
                () -> assertNotSame(first, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertSame(input,
                        first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first,
                        inverseLike.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()));
    }

    @Test
    void successfulCallConsumesOneIdentityAndExhaustionOccursOnlyAtFinalDelegation()
            throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 1))),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        input.squeeze(1);
        assertEquals(before + 1, next.get());

        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.expandDims(0));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static AxisTransformAttrs attrs(Tensor tensor) {
        return (AxisTransformAttrs) tensor.provenance().orElseThrow().operation().attrs();
    }

    private static LayoutDescriptor expandLayout(
            Shape inputShape, LayoutDescriptor inputLayout, int axis) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .expandDims(axis)
                .descriptor()
                .layout()
                .orElseThrow();
    }

    private static LayoutDescriptor squeezeLayout(
            Shape inputShape, LayoutDescriptor inputLayout, int axis) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .squeeze(axis)
                .descriptor()
                .layout()
                .orElseThrow();
    }

    private static void assertLayout(
            LayoutDescriptor layout,
            long[] strides,
            long offset,
            LayoutKind kind,
            long span) {
        assertAll(
                () -> assertArrayEquals(strides, layout.strides()),
                () -> assertEquals(offset, layout.storageOffset()),
                () -> assertSame(kind, layout.kind()),
                () -> assertEquals(span, layout.referencedElementSpan()),
                () -> assertTrue(layout.isView()));
    }

    private static List<Boolean> validGradientChoices(DataType dataType) {
        return dataType.isDifferentiable() ? List.of(false, true) : List.of(false);
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
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
