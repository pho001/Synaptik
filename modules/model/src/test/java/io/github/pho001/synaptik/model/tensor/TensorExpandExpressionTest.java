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
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
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

class TensorExpandExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(60_000);

    @Test
    void exposesExactlyTwoPublicOverloadsAndSixMethodStatelessHelper() throws Exception {
        Method raw = Tensor.class.getDeclaredMethod("expand", long[].class);
        Method exact = Tensor.class.getDeclaredMethod("expand", Shape.class);
        var constructor = TensorExpandExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(TensorExpandExpressions.class.getDeclaredMethods());
        Method rawApply = TensorExpandExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, long[].class);
        Method exactApply = TensorExpandExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Shape.class);
        Method validate = TensorExpandExpressions.class.getDeclaredMethod(
                "validateExpansion", Shape.class, Shape.class);
        Method resolve = TensorExpandExpressions.class.getDeclaredMethod(
                "resolveViewLayout", TensorDescriptor.class, Shape.class);
        Method derive = TensorExpandExpressions.class.getDeclaredMethod(
                "deriveExpandedStrides", Shape.class, LayoutDescriptor.class, Shape.class);
        Method create = TensorExpandExpressions.class.getDeclaredMethod(
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
                () -> assertTrue(Modifier.isFinal(TensorExpandExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorExpandExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorExpandExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorExpandExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorExpandExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(6, methods.size()),
                () -> assertEquals(
                        Set.of("apply", "validateExpansion", "resolveViewLayout",
                                "deriveExpandedStrides", "create"),
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
                () -> assertSame(void.class, validate.getReturnType()),
                () -> assertSame(Optional.class, resolve.getReturnType()),
                () -> assertSame(long[].class, derive.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(validate.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolve.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(derive.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())));
    }

    @Test
    void acceptsEveryDataTypeAndValidGradientChoiceWithExactSemantics() {
        Shape inputShape = Shape.of(1, 3);
        Shape targetShape = Shape.of(2, 3);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, inputShape, Optional.empty(), requiresGrad);

                Tensor result = input.expand(targetShape);
                TensorProvenance provenance = result.provenance().orElseThrow();
                TargetShapeAttrs attrs = (TargetShapeAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(targetShape, result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertSame(ShapeTransformKind.EXPAND,
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
        Tensor scalarInput = tensor(DataType.INT64, Shape.scalar(), Optional.empty(), false);
        Tensor scalar = scalarInput.expand();
        Tensor singleton = tensor(DataType.BOOL, Shape.of(1, 3), Optional.empty(), false);
        long[] request = {2, 0, 3};

        Tensor empty = singleton.expand(request);
        Shape normalized = empty.descriptor().shape();
        request[1] = 7;

        assertAll(
                () -> assertSame(Shape.scalar(), scalar.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 0, 3), normalized),
                () -> assertArrayEquals(new long[] {2, 0, 3}, normalized.toLongArray()),
                () -> assertSame(normalized,
                        ((TargetShapeAttrs) empty.provenance().orElseThrow()
                                .operation().attrs()).targetShape()));
    }

    @Test
    void validatesDirectionalRightAlignedStaticScalarAndLeadingExpansion() {
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), true);
        Tensor vector = tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), true);
        Tensor singleton = tensor(DataType.FLOAT32, Shape.of(1, 3), Optional.empty(), true);

        assertAll(
                () -> assertEquals(Shape.of(2, 0, 4),
                        scalar.expand(2, 0, 4).descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3),
                        vector.expand(2, 3).descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3),
                        singleton.expand(2, 3).descriptor().shape()),
                () -> assertEquals(Shape.of(5, 2, 3),
                        singleton.expand(5, 2, 3).descriptor().shape()));
    }

    @Test
    void acceptsEveryDeferredAlignedCompatibilityCategoryAndRetainsExactTargets() {
        Dimension named = new DynamicDimension("named");
        Dimension equalNamed = new DynamicDimension("named");
        Dimension otherNamed = new DynamicDimension("other");
        Dimension expression = DimensionExpressions.addConstant(
                new DynamicDimension("base"), 2);
        Dimension constrainedUnknown = DimensionExpressions.unknown(
                1, Optional.of(new StaticDimension(8)));
        Dimension otherUnknown = DimensionExpressions.unknown(0, Optional.empty());

        Shape equalUnresolvedTarget = Shape.ofDimensions(equalNamed);
        Shape unresolvedToUnresolvedTarget = Shape.ofDimensions(otherNamed);
        Shape unresolvedToStaticTarget = Shape.of(4);
        Shape expressionToUnknownTarget = Shape.ofDimensions(otherUnknown);
        Shape constrainedUnknownToStaticTarget = Shape.of(6);
        Shape staticToUnresolvedTarget = Shape.ofDimensions(new DynamicDimension("target"));
        Shape singletonToExpressionTarget = Shape.ofDimensions(expression);

        Tensor equalUnresolved = tensor(
                DataType.FLOAT64, Shape.ofDimensions(named), Optional.empty(), true)
                .expand(equalUnresolvedTarget);
        Tensor unresolvedToUnresolved = tensor(
                DataType.FLOAT64, Shape.ofDimensions(named), Optional.empty(), true)
                .expand(unresolvedToUnresolvedTarget);
        Tensor unresolvedToStatic = tensor(
                DataType.FLOAT64, Shape.ofDimensions(named), Optional.empty(), true)
                .expand(unresolvedToStaticTarget);
        Tensor expressionToUnknown = tensor(
                DataType.FLOAT64, Shape.ofDimensions(expression), Optional.empty(), true)
                .expand(expressionToUnknownTarget);
        Tensor constrainedUnknownToStatic = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(constrainedUnknown),
                Optional.empty(),
                true)
                .expand(constrainedUnknownToStaticTarget);
        Tensor staticToUnresolved = tensor(
                DataType.FLOAT64, Shape.of(4), Optional.empty(), true)
                .expand(staticToUnresolvedTarget);
        Tensor singletonToExpression = tensor(
                DataType.FLOAT64, Shape.of(1), Optional.empty(), true)
                .expand(singletonToExpressionTarget);

        List<Tensor> results = List.of(
                equalUnresolved,
                unresolvedToUnresolved,
                unresolvedToStatic,
                expressionToUnknown,
                constrainedUnknownToStatic,
                staticToUnresolved,
                singletonToExpression);
        List<Shape> targets = List.of(
                equalUnresolvedTarget,
                unresolvedToUnresolvedTarget,
                unresolvedToStaticTarget,
                expressionToUnknownTarget,
                constrainedUnknownToStaticTarget,
                staticToUnresolvedTarget,
                singletonToExpressionTarget);

        for (int index = 0; index < results.size(); index++) {
            Tensor result = results.get(index);
            Shape target = targets.get(index);
            assertAll(
                    () -> assertSame(target, result.descriptor().shape()),
                    () -> assertSame(
                            target,
                            ((TargetShapeAttrs) result.provenance().orElseThrow()
                                    .operation().attrs()).targetShape()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()));
        }
    }

    @Test
    void acceptsUnresolvedLeadingAxesWithoutAddingAlignedObligations() {
        Shape targetShape = Shape.ofDimensions(
                new DynamicDimension("leading"), new StaticDimension(3));
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), true);

        Tensor result = input.expand(targetShape);

        assertAll(
                () -> assertSame(targetShape, result.descriptor().shape()),
                () -> assertSame(
                        targetShape,
                        ((TargetShapeAttrs) result.provenance().orElseThrow()
                                .operation().attrs()).targetShape()),
                () -> assertTrue(result.descriptor().layout().isEmpty()));
    }

    @Test
    void rejectsNullNegativeRankAndAlignedFailuresWithExactPrecedenceAndMessages() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);

        NullPointerException nullRawInput = assertThrows(
                NullPointerException.class,
                () -> TensorExpandExpressions.apply(null, new long[] {2, 3}));
        NullPointerException nullRequest = assertThrows(
                NullPointerException.class,
                () -> TensorExpandExpressions.apply(input, (long[]) null));
        NullPointerException nullExactInput = assertThrows(
                NullPointerException.class,
                () -> TensorExpandExpressions.apply(null, Shape.of(2, 3)));
        NullPointerException nullTarget = assertThrows(
                NullPointerException.class, () -> input.expand((Shape) null));
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class, () -> input.expand(2, -1, 3));
        IllegalArgumentException rank = assertThrows(
                IllegalArgumentException.class, () -> input.expand(3));
        IllegalArgumentException firstMismatch = assertThrows(
                IllegalArgumentException.class, () -> input.expand(4, 5));
        IllegalArgumentException noShrink = assertThrows(
                IllegalArgumentException.class, () -> input.expand(1, 3));
        IllegalArgumentException zeroMismatch = assertThrows(
                IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.of(0, 3), Optional.empty(), true)
                        .expand(1, 3));

        assertAll(
                () -> assertEquals("input", nullRawInput.getMessage()),
                () -> assertEquals("requestedShape", nullRequest.getMessage()),
                () -> assertEquals("input", nullExactInput.getMessage()),
                () -> assertEquals("targetShape", nullTarget.getMessage()),
                () -> assertEquals(
                        "Static dimension size must be non-negative: -1", negative.getMessage()),
                () -> assertEquals(
                        "expand target rank 1 must be at least input rank 2", rank.getMessage()),
                () -> assertEquals(
                        "cannot expand input shape Shape[2, 3] to target shape Shape[4, 5] at target axis 0",
                        firstMismatch.getMessage()),
                () -> assertEquals(
                        "cannot expand input shape Shape[2, 3] to target shape Shape[1, 3] at target axis 0",
                        noShrink.getMessage()),
                () -> assertEquals(
                        "cannot expand input shape Shape[0, 3] to target shape Shape[1, 3] at target axis 0",
                        zeroMismatch.getMessage()));
    }

    @Test
    void derivesZeroAndPreservedStridesFromEveryResolvedInputLayoutKind() {
        Shape baseShape = Shape.of(1, 3);
        Shape expandedShape = Shape.of(2, 4, 3);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(baseShape);
        LayoutDescriptor offset = LayoutDescriptor.of(
                baseShape, new long[] {3, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(
                baseShape, new long[] {7, 2}, 4, true);
        Shape broadcastShape = Shape.of(1, 2, 3);
        LayoutDescriptor broadcast = LayoutDescriptor.of(
                broadcastShape, new long[] {0, 0, 1}, 6, true);

        LayoutDescriptor denseResult = expandLayout(baseShape, dense, expandedShape);
        LayoutDescriptor offsetResult = expandLayout(baseShape, offset, expandedShape);
        LayoutDescriptor stridedResult = expandLayout(baseShape, strided, expandedShape);
        LayoutDescriptor broadcastResult = expandLayout(
                broadcastShape, broadcast, Shape.of(4, 2, 3));

        assertAll(
                () -> assertArrayEquals(new long[] {0, 0, 1}, denseResult.strides()),
                () -> assertEquals(0, denseResult.storageOffset()),
                () -> assertEquals(LayoutKind.BROADCAST_ZERO_STRIDE, denseResult.kind()),
                () -> assertEquals(3, denseResult.referencedElementSpan()),
                () -> assertTrue(denseResult.isView()),
                () -> assertNotSame(dense, denseResult),
                () -> assertArrayEquals(new long[] {0, 0, 1}, offsetResult.strides()),
                () -> assertEquals(5, offsetResult.storageOffset()),
                () -> assertEquals(8, offsetResult.referencedElementSpan()),
                () -> assertNotSame(offset, offsetResult),
                () -> assertArrayEquals(new long[] {0, 0, 2}, stridedResult.strides()),
                () -> assertEquals(4, stridedResult.storageOffset()),
                () -> assertEquals(9, stridedResult.referencedElementSpan()),
                () -> assertNotSame(strided, stridedResult),
                () -> assertArrayEquals(new long[] {0, 0, 1}, broadcastResult.strides()),
                () -> assertEquals(6, broadcastResult.storageOffset()),
                () -> assertEquals(9, broadcastResult.referencedElementSpan()),
                () -> assertSame(LayoutKind.BROADCAST_ZERO_STRIDE, broadcastResult.kind()),
                () -> assertNotSame(broadcast, broadcastResult));
    }

    @Test
    void preservesSingletonStrideForEqualExtentAndResolvesScalarAndEmptyGeometry() {
        Shape singletonShape = Shape.of(1, 3);
        LayoutDescriptor singletonInput = LayoutDescriptor.contiguous(singletonShape);
        LayoutDescriptor identity = expandLayout(singletonShape, singletonInput, singletonShape);
        Shape emptyInputShape = Shape.of(1, 0, 3);
        LayoutDescriptor emptyInput = LayoutDescriptor.contiguous(emptyInputShape);
        LayoutDescriptor empty = expandLayout(
                emptyInputShape, emptyInput, Shape.of(2, 0, 3));
        Shape scalar = Shape.scalar();
        LayoutDescriptor scalarInput = LayoutDescriptor.contiguous(scalar);
        LayoutDescriptor scalarResult = expandLayout(scalar, scalarInput, scalar);

        assertAll(
                () -> assertArrayEquals(new long[] {3, 1}, identity.strides()),
                () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS, identity.kind()),
                () -> assertTrue(identity.isView()),
                () -> assertArrayEquals(new long[] {0, 3, 1}, empty.strides()),
                () -> assertEquals(0, empty.referencedElementSpan()),
                () -> assertArrayEquals(new long[0], scalarResult.strides()),
                () -> assertEquals(1, scalarResult.referencedElementSpan()),
                () -> assertTrue(scalarResult.isView()));
    }

    @Test
    void leavesDynamicTargetAndUnresolvedInputLayoutUnresolved() {
        Shape staticInputShape = Shape.of(1, 3);
        Tensor unresolved = tensor(
                DataType.FLOAT32, staticInputShape, Optional.empty(), true);
        Tensor resolved = tensor(
                DataType.FLOAT32,
                staticInputShape,
                Optional.of(LayoutDescriptor.contiguous(staticInputShape)),
                true);
        Shape dynamicTarget = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        assertAll(
                () -> assertTrue(unresolved.expand(2, 3).descriptor().layout().isEmpty()),
                () -> assertTrue(resolved.expand(dynamicTarget).descriptor().layout().isEmpty()),
                () -> assertSame(dynamicTarget, resolved.expand(dynamicTarget).descriptor().shape()));
    }

    @Test
    void sameShapeRepeatedAndNestedRequestsRemainFreshExplicitExpressions() {
        Shape shape = Shape.of(1, 3);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.empty(), true);

        Tensor first = input.expand(shape);
        Tensor second = input.expand(shape);
        Tensor nested = first.expand(shape);

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
        Shape shape = Shape.of(1, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {3, 1}, 2, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 5, arena.allocate(20, 1));
        float[] values = {1.0f, 2.0f, 3.0f};
        MemorySegment.copy(MemorySegment.ofArray(values), 0, storage.segment(), 8, 12);
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(shape)),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(8, 12).toArray(ValueLayout.JAVA_FLOAT);

        Tensor result = input.expand(4, 3);
        float[] after = storage.segment().asSlice(8, 12).toArray(ValueLayout.JAVA_FLOAT);
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
    void earlyValidationFailuresConsumeNoIdentityAndDeferredSuccessConsumesExactlyOne()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertThrows(IllegalArgumentException.class, () -> input.expand(-1, 2, 3));
        assertThrows(IllegalArgumentException.class, () -> input.expand(3));
        assertThrows(IllegalArgumentException.class, () -> input.expand(2, 4));
        assertEquals(before, next.get());

        Shape deferredTarget = Shape.ofDimensions(
                new DynamicDimension("target"), new StaticDimension(3));
        Tensor deferred = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("source"), new StaticDimension(3)),
                Optional.empty(),
                true)
                .expand(deferredTarget);

        assertAll(
                () -> assertEquals(before, deferred.id().value()),
                () -> assertEquals(Math.addExact(before, 1), next.get()));
    }

    @Test
    void propagatesIdentifierExhaustionOnlyAtFinalFactoryDelegation() throws Exception {
        Shape shape = Shape.of(1, 3);
        Tensor input = tensor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.expand(2, 3));

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

    private static LayoutDescriptor expandLayout(
            Shape inputShape, LayoutDescriptor inputLayout, Shape targetShape) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .expand(targetShape)
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
