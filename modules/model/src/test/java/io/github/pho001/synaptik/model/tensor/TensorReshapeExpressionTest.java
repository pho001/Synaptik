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
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
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

class TensorReshapeExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(50_000);

    @Test
    void exposesExactlyTwoPublicOverloadsAndSixMethodStatelessHelper() throws Exception {
        Method raw = Tensor.class.getDeclaredMethod("reshape", long[].class);
        Method exact = Tensor.class.getDeclaredMethod("reshape", Shape.class);
        var constructor = TensorReshapeExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(TensorReshapeExpressions.class.getDeclaredMethods());
        Method rawApply = TensorReshapeExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, long[].class);
        Method exactApply = TensorReshapeExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Shape.class);
        Method normalize = TensorReshapeExpressions.class.getDeclaredMethod(
                "normalizeRequestedShape", Shape.class, long[].class);
        Method validate = TensorReshapeExpressions.class.getDeclaredMethod(
                "validateTargetShape", Shape.class, Shape.class);
        Method resolve = TensorReshapeExpressions.class.getDeclaredMethod(
                "resolveViewLayout", TensorDescriptor.class, Shape.class);
        Method create = TensorReshapeExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, Optional.class);

        assertAll(
                () -> assertSame(Tensor.class, raw.getReturnType()),
                () -> assertTrue(raw.isVarArgs()),
                () -> assertEquals(List.of(long[].class), Arrays.asList(raw.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(raw.getModifiers())),
                () -> assertFalse(Modifier.isStatic(raw.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(raw.getModifiers())),
                () -> assertSame(Tensor.class, exact.getReturnType()),
                () -> assertFalse(exact.isVarArgs()),
                () -> assertEquals(List.of(Shape.class), Arrays.asList(exact.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(exact.getModifiers())),
                () -> assertFalse(Modifier.isStatic(exact.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(exact.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorReshapeExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorReshapeExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorReshapeExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorReshapeExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorReshapeExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(6, methods.size()),
                () -> assertEquals(
                        Set.of("apply", "normalizeRequestedShape", "validateTargetShape",
                                "resolveViewLayout", "create"),
                        methods.stream().map(Method::getName).collect(Collectors.toSet())),
                () -> assertEquals(2,
                        methods.stream().filter(method -> method.getName().equals("apply")).count()),
                () -> assertEquals(2,
                        methods.stream().filter(method -> !Modifier.isPrivate(method.getModifiers()))
                                .count()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertSame(Tensor.class, rawApply.getReturnType()),
                () -> assertSame(Tensor.class, exactApply.getReturnType()),
                () -> assertSame(Shape.class, normalize.getReturnType()),
                () -> assertSame(void.class, validate.getReturnType()),
                () -> assertSame(Optional.class, resolve.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(normalize.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(validate.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolve.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())));
    }

    @Test
    void acceptsEveryDataTypeAndValidGradientChoiceWithExactSemantics() {
        Shape inputShape = Shape.of(2, 3);
        Shape targetShape = Shape.of(3, 2);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, inputShape, Optional.empty(), requiresGrad);

                Tensor result = input.reshape(targetShape);
                TensorProvenance provenance = result.provenance().orElseThrow();
                TargetShapeAttrs attrs = (TargetShapeAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(targetShape, result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertSame(ShapeTransformKind.RESHAPE,
                                provenance.operation().kind()),
                        () -> assertSame(targetShape, attrs.targetShape()),
                        () -> assertEquals(List.of(input), provenance.inputs()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void rawRequestsSupportScalarZeroExtentsAndDefensiveArrayOwnership() {
        Tensor scalarInput = tensor(DataType.INT64, Shape.of(1), Optional.empty(), false);
        Tensor scalar = scalarInput.reshape();
        Tensor emptyInput = tensor(DataType.BOOL, Shape.of(2, 0, 4), Optional.empty(), false);
        long[] request = {0, 8};

        Tensor empty = emptyInput.reshape(request);
        Shape normalized = empty.descriptor().shape();
        request[0] = 7;

        assertAll(
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertEquals(Shape.of(0, 8), normalized),
                () -> assertArrayEquals(new long[] {0, 8}, normalized.toLongArray()),
                () -> assertSame(normalized,
                        ((TargetShapeAttrs) empty.provenance().orElseThrow()
                                .operation().attrs()).targetShape()));
    }

    @Test
    void infersOneSentinelAtEveryRepresentativePositionWithoutMutatingRequest() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true);
        long[] middleRequest = {3, -1, 2};

        Tensor leading = input.reshape(-1, 3, 2);
        Tensor middle = input.reshape(middleRequest);
        Tensor trailing = input.reshape(2, 3, -1);

        assertAll(
                () -> assertEquals(Shape.of(4, 3, 2), leading.descriptor().shape()),
                () -> assertEquals(Shape.of(3, 4, 2), middle.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3, 4), trailing.descriptor().shape()),
                () -> assertArrayEquals(new long[] {3, -1, 2}, middleRequest));
    }

    @Test
    void supportsUnambiguousInferredZeroAndRejectsZeroProductAmbiguity() {
        Tensor empty = tensor(DataType.INT32, Shape.of(0, 3), Optional.empty(), false);

        Tensor inferredZero = empty.reshape(-1, 2);
        IllegalArgumentException ambiguity = assertThrows(
                IllegalArgumentException.class, () -> empty.reshape(0, -1));
        IllegalArgumentException irrelevantOverflow = assertThrows(
                IllegalArgumentException.class,
                () -> empty.reshape(Long.MAX_VALUE, 2, 0, -1));

        assertAll(
                () -> assertEquals(Shape.of(0, 2), inferredZero.descriptor().shape()),
                () -> assertEquals(
                        "cannot infer -1 when known requested dimensions have product zero",
                        ambiguity.getMessage()),
                () -> assertEquals(
                        "cannot infer -1 when known requested dimensions have product zero",
                        irrelevantOverflow.getMessage()));
    }

    @Test
    void rejectsRawRequestFailuresWithExactMessagesAndPrecedence() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("n"), new StaticDimension(3));
        Tensor dynamicInput = tensor(DataType.FLOAT32, dynamic, Optional.empty(), true);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorReshapeExpressions.apply(null, new long[] {1}));
        NullPointerException nullRequest = assertThrows(
                NullPointerException.class,
                () -> TensorReshapeExpressions.apply(input, (long[]) null));
        IllegalArgumentException negativeWins = assertThrows(
                IllegalArgumentException.class, () -> input.reshape(-2, -1, -1));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class, () -> input.reshape(-1, 6, -1));
        IllegalArgumentException dynamicInference = assertThrows(
                IllegalArgumentException.class, () -> dynamicInput.reshape(-1, 3));
        IllegalArgumentException indivisible = assertThrows(
                IllegalArgumentException.class, () -> input.reshape(4, -1));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("requestedShape", nullRequest.getMessage()),
                () -> assertEquals(
                        "requestedShape[0] must be non-negative or -1: -2",
                        negativeWins.getMessage()),
                () -> assertEquals(
                        "requestedShape must contain at most one -1", duplicate.getMessage()),
                () -> assertEquals(
                        "cannot infer -1 from dynamic input shape Shape[n, 3]",
                        dynamicInference.getMessage()),
                () -> assertEquals(
                        "cannot infer reshape dimension: input element count 6 is not divisible by known requested product 4",
                        indivisible.getMessage()));
    }

    @Test
    void exactShapeOverloadRetainsReferenceDefersDynamicEqualityAndValidatesKnownCounts() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true);
        DynamicDimension symbol = new DynamicDimension("items");
        Shape dynamicTarget = Shape.ofDimensions(symbol, new StaticDimension(2));

        Tensor dynamicResult = input.reshape(dynamicTarget);
        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorReshapeExpressions.apply(null, dynamicTarget));
        NullPointerException nullTarget = assertThrows(
                NullPointerException.class, () -> input.reshape((Shape) null));
        IllegalArgumentException mismatch = assertThrows(
                IllegalArgumentException.class, () -> input.reshape(Shape.of(5)));

        assertAll(
                () -> assertSame(dynamicTarget, dynamicResult.descriptor().shape()),
                () -> assertSame(symbol,
                        dynamicResult.descriptor().shape().dimensions().getFirst()),
                () -> assertSame(dynamicTarget,
                        ((TargetShapeAttrs) dynamicResult.provenance().orElseThrow()
                                .operation().attrs()).targetShape()),
                () -> assertTrue(dynamicResult.descriptor().layout().isEmpty()),
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("targetShape", nullTarget.getMessage()),
                () -> assertEquals(
                        "reshape element count mismatch: input=6, target=5",
                        mismatch.getMessage()));
    }

    @Test
    void resolvedContiguousInputsProduceNewCanonicalSameOffsetViewLayouts() {
        Shape inputShape = Shape.of(2, 3);
        Shape targetShape = Shape.of(3, 2);
        LayoutDescriptor zeroOffset = LayoutDescriptor.contiguous(inputShape);
        LayoutDescriptor offset = LayoutDescriptor.of(
                inputShape, new long[] {3, 1}, 5, true);

        Tensor zeroResult = tensor(
                DataType.FLOAT32, inputShape, Optional.of(zeroOffset), true)
                .reshape(targetShape);
        Tensor offsetResult = tensor(
                DataType.FLOAT32, inputShape, Optional.of(offset), true)
                .reshape(targetShape);
        LayoutDescriptor zeroView = zeroResult.descriptor().layout().orElseThrow();
        LayoutDescriptor offsetView = offsetResult.descriptor().layout().orElseThrow();

        assertAll(
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS, zeroView.kind()),
                () -> assertArrayEquals(new long[] {2, 1}, zeroView.strides()),
                () -> assertEquals(0, zeroView.storageOffset()),
                () -> assertTrue(zeroView.isView()),
                () -> assertNotSame(zeroOffset, zeroView),
                () -> assertEquals(LayoutKind.DENSE_WITH_OFFSET, offsetView.kind()),
                () -> assertArrayEquals(new long[] {2, 1}, offsetView.strides()),
                () -> assertEquals(5, offsetView.storageOffset()),
                () -> assertTrue(offsetView.isView()),
                () -> assertNotSame(offset, offsetView));
    }

    @Test
    void resolvesScalarAndZeroExtentViewGeometry() {
        Tensor scalar = tensor(
                DataType.FLOAT32,
                Shape.of(1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(1))),
                false).reshape(Shape.scalar());
        Shape emptyInputShape = Shape.of(0);
        Tensor empty = tensor(
                DataType.BOOL,
                emptyInputShape,
                Optional.of(LayoutDescriptor.contiguous(emptyInputShape)),
                false).reshape(2, 0, 4);

        LayoutDescriptor scalarLayout = scalar.descriptor().layout().orElseThrow();
        LayoutDescriptor emptyLayout = empty.descriptor().layout().orElseThrow();
        assertAll(
                () -> assertArrayEquals(new long[0], scalarLayout.strides()),
                () -> assertEquals(1, scalarLayout.referencedElementSpan()),
                () -> assertTrue(scalarLayout.isView()),
                () -> assertArrayEquals(new long[] {0, 4, 1}, emptyLayout.strides()),
                () -> assertEquals(0, emptyLayout.referencedElementSpan()),
                () -> assertTrue(emptyLayout.isView()));
    }

    @Test
    void unresolvedStridedBroadcastAndDynamicGeometryRemainUnresolved() {
        Shape shape = Shape.of(2, 3);
        Shape target = Shape.of(3, 2);
        List<Optional<LayoutDescriptor>> nonViewable = List.of(
                Optional.empty(),
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true)));

        for (Optional<LayoutDescriptor> layout : nonViewable) {
            Tensor result = tensor(DataType.FLOAT32, shape, layout, false).reshape(target);
            assertTrue(result.descriptor().layout().isEmpty());
        }

        Shape dynamicInputShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Shape dynamicTarget = Shape.ofDimensions(
                new DynamicDimension("items"), new StaticDimension(2));
        Tensor dynamicInput = tensor(
                DataType.FLOAT32, dynamicInputShape, Optional.empty(), true);
        Tensor staticInputDynamicTarget = tensor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);

        assertAll(
                () -> assertTrue(dynamicInput.reshape(target).descriptor().layout().isEmpty()),
                () -> assertTrue(staticInputDynamicTarget.reshape(dynamicTarget)
                        .descriptor().layout().isEmpty()));
    }

    @Test
    void sameShapeRepeatedAndNestedRequestsRemainFreshExplicitExpressions() {
        Shape shape = Shape.of(2, 3);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);

        Tensor first = input.reshape(shape);
        Tensor second = input.reshape(shape);
        Tensor nested = first.reshape(shape);

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(second, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(second.id(), nested.id()),
                () -> assertSame(input,
                        first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void leavesInputMetadataStorageLivenessAndValuesUntouched() {
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {2, 1}, 3, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 7, arena.allocate(28, 1));
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        java.lang.foreign.MemorySegment.copy(
                java.lang.foreign.MemorySegment.ofArray(values), 0, storage.segment(), 12, 16);
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(shape)),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(12, 16).toArray(
                java.lang.foreign.ValueLayout.JAVA_FLOAT);

        Tensor result = input.reshape(4);
        float[] after = storage.segment().asSlice(12, 16).toArray(
                java.lang.foreign.ValueLayout.JAVA_FLOAT);
        arena.close();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputLayout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertFalse(storage.isAlive()),
                () -> assertArrayEquals(before, after),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void everyEarlyValidationAndArithmeticFailureConsumesNoTensorIdentity() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(IllegalArgumentException.class, () -> input.reshape(-2));
        assertThrows(IllegalArgumentException.class, () -> input.reshape(-1, -1));
        assertThrows(IllegalArgumentException.class, () -> input.reshape(5));
        assertThrows(ArithmeticException.class,
                () -> tensor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false)
                        .reshape(-1, Long.MAX_VALUE, 2));
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("n"));
        assertThrows(ArithmeticException.class,
                () -> tensor(DataType.FLOAT32, dynamic, Optional.empty(), false)
                        .reshape(Shape.of(Long.MAX_VALUE, 2)));
        Shape zero = Shape.of(0);
        assertThrows(ArithmeticException.class,
                () -> tensor(
                                DataType.FLOAT32,
                                zero,
                                Optional.of(LayoutDescriptor.contiguous(zero)),
                                false)
                        .reshape(Shape.of(0, Long.MAX_VALUE, 2)));

        assertEquals(before, next.get());
    }

    @Test
    void propagatesIdentifierExhaustionOnlyAtFinalFactoryDelegation() throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))),
                true);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.reshape(3, 2));

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
