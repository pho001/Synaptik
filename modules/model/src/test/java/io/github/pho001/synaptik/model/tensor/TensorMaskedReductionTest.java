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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorMaskedReductionTest {
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(120_000);

    @Test
    void exposesExactHelperAndPublicSurface() throws ReflectiveOperationException {
        Class<?> helper = TensorMaskedReductionExpressions.class;
        int modifiers = helper.getModifiers();
        var constructors = helper.getDeclaredConstructors();
        List<Method> methods = Arrays.asList(helper.getDeclaredMethods());

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertFalse(helper.isRecord()),
                () -> assertEquals(Set.of(), Set.of(helper.getInterfaces())),
                () -> assertEquals(0, helper.getDeclaredFields().length),
                () -> assertEquals(0, helper.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(3, methods.size()),
                () -> assertTrue(methods.stream().noneMatch(Method::isSynthetic)));

        assertHelperMethod(
                "apply",
                false,
                Tensor.class,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                int.class);
        assertHelperMethod("reduceShape", true, Shape.class, Shape.class, int.class);
        assertHelperMethod(
                "create",
                true,
                Tensor.class,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                int.class,
                Shape.class);

        for (String methodName : List.of("sum", "mean")) {
            Method method = Tensor.class.getDeclaredMethod(
                    methodName, int.class, Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())),
                    () -> assertEquals(
                            List.of(int.class, Tensor.class),
                            Arrays.asList(method.getParameterTypes())));
        }
    }

    @Test
    void delegatesToExactKindsWithOneProducerAndOrderedInputs() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor mask = tensor(DataType.BOOL, Shape.of(3, 1), false);

        Tensor sum = input.sum(1, mask);
        Tensor mean = input.mean(-2, mask);

        assertExpression(sum, input, mask, AggregateReductionKind.SUM, 1);
        assertExpression(mean, input, mask, AggregateReductionKind.MEAN, 1);
        for (Tensor result : List.of(sum, mean)) {
            TensorProvenance provenance = result.provenance().orElseThrow();
            assertAll(
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertEquals(1, provenance.producer().outputDescriptors().size()),
                    () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                    () -> assertSame(
                            result.descriptor(),
                            provenance.producer().outputDescriptors().getFirst()));
        }
    }

    @Test
    void acceptsEveryFloatingInputTypeAndOnlyBoolMasks() {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (boolean requiresGrad : List.of(false, true)) {
                Tensor input = tensor(dataType, Shape.of(2, 3), requiresGrad);
                Tensor mask = tensor(DataType.BOOL, Shape.of(3), false);
                for (Tensor result : List.of(input.sum(-1, mask), input.mean(1, mask))) {
                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertEquals(
                                    requiresGrad, result.descriptor().requiresGrad()),
                            () -> assertEquals(Shape.of(2), result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()));
                }
            }
        }

        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), false);
        for (DataType dataType : List.of(
                DataType.FLOAT64,
                DataType.FLOAT32,
                DataType.BFLOAT16,
                DataType.INT32,
                DataType.INT64)) {
            Tensor mask = tensor(dataType, Shape.of(2), false);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> input.sum(0, mask));
            assertEquals(
                    "mask must have BOOL data type, but was " + dataType,
                    failure.getMessage());
        }
    }

    @Test
    void acceptsOrdinaryRightAlignedBroadcastMatrix() {
        Dimension batch = new DynamicDimension("batch");
        Dimension time = new DynamicDimension("time");
        Dimension features = new StaticDimension(4);
        Shape inputShape = Shape.ofDimensions(batch, time, features);
        Tensor input = tensor(DataType.FLOAT32, inputShape, false);

        for (Shape maskShape : List.of(
                Shape.scalar(),
                inputShape,
                Shape.ofDimensions(features),
                Shape.ofDimensions(time, new StaticDimension(1)),
                Shape.ofDimensions(batch, time, new StaticDimension(1)))) {
            Tensor result = input.sum(1, tensor(DataType.BOOL, maskShape, false));
            assertEquals(new MaskedReductionAttrs(1),
                    result.provenance().orElseThrow().operation().attrs());
        }

        Tensor zeroInput = tensor(DataType.FLOAT32, Shape.of(2, 0, 4), false);
        assertEquals(
                Shape.of(2, 4),
                zeroInput.mean(1, tensor(DataType.BOOL, Shape.of(0, 1), false))
                        .descriptor().shape());

        Dimension expression = DimensionExpressions.addConstant(
                new DynamicDimension("N"), 2);
        Tensor expressionInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(expression, features), false);
        Shape equalExpressionMask = Shape.ofDimensions(
                DimensionExpressions.add(
                        new StaticDimension(2), new DynamicDimension("N")),
                new StaticDimension(1));
        assertEquals(
                Shape.ofDimensions(features),
                expressionInput.sum(
                                0, tensor(DataType.BOOL, equalExpressionMask, false))
                        .descriptor().shape());

        Dimension unknown = DimensionExpressions.unknown(0, Optional.empty());
        Tensor unknownInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(unknown, features), false);
        assertEquals(
                Shape.ofDimensions(features),
                unknownInput.mean(
                                0,
                                tensor(
                                        DataType.BOOL,
                                        Shape.ofDimensions(unknown, new StaticDimension(1)),
                                        false))
                        .descriptor().shape());
    }

    @Test
    void rejectsNonRightAlignedAndInputEnlargingMasks() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        Tensor nonRightAligned = tensor(DataType.BOOL, Shape.of(2, 3), false);
        IllegalArgumentException incompatible = assertThrows(
                IllegalArgumentException.class, () -> input.sum(1, nonRightAligned));
        assertEquals(
                "Cannot broadcast dimensions StaticDimension[size=3]"
                        + " and StaticDimension[size=2] at result axis 1 for"
                        + " Shape[2, 3, 4] and Shape[2, 3]",
                incompatible.getMessage());

        Tensor singletonInput = tensor(DataType.FLOAT32, Shape.of(1, 3, 4), false);
        Tensor enlarging = tensor(DataType.BOOL, Shape.of(2, 3, 4), false);
        IllegalArgumentException enlarged = assertThrows(
                IllegalArgumentException.class, () -> singletonInput.mean(1, enlarging));
        assertEquals(
                "mask shape Shape[2, 3, 4] must broadcast exactly to input shape"
                        + " Shape[1, 3, 4], but produced Shape[2, 3, 4]",
                enlarged.getMessage());

        Tensor extraLeading = tensor(DataType.BOOL, Shape.of(1, 2, 3, 4), false);
        IllegalArgumentException leading = assertThrows(
                IllegalArgumentException.class, () -> input.sum(1, extraLeading));
        assertEquals(
                "mask shape Shape[1, 2, 3, 4] must broadcast exactly to input shape"
                        + " Shape[2, 3, 4], but produced Shape[1, 2, 3, 4]",
                leading.getMessage());
    }

    @Test
    void rejectsUnprovableDynamicRelationshipsWithoutConsumingIdentity()
            throws ReflectiveOperationException {
        Dimension nPlusOne = DimensionExpressions.addConstant(new DynamicDimension("N"), 1);
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("batch"), nPlusOne),
                false);
        List<Shape> invalidMasks = List.of(
                Shape.of(3),
                Shape.ofDimensions(new DynamicDimension("other")),
                Shape.ofDimensions(
                        DimensionExpressions.addConstant(new DynamicDimension("N"), 2)),
                Shape.ofDimensions(DimensionExpressions.unknown(0, Optional.empty())));
        AtomicLong nextId = nextTensorIdState();
        long before = nextId.get();

        for (Shape maskShape : invalidMasks) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> input.sum(0, tensor(DataType.BOOL, maskShape, false)));
        }

        Dimension firstUnknown = DimensionExpressions.unknown(0, Optional.empty());
        Tensor unknownInput = tensor(
                DataType.FLOAT32, Shape.ofDimensions(firstUnknown), false);
        Tensor distinctUnknownMask = tensor(
                DataType.BOOL,
                Shape.ofDimensions(DimensionExpressions.unknown(0, Optional.empty())),
                false);
        assertThrows(
                IllegalArgumentException.class,
                () -> unknownInput.mean(0, distinctUnknownMask));

        assertEquals(before, nextId.get());
    }

    @Test
    void explicitRankEditingMakesAlignmentVisibleInProvenance() {
        Dimension batch = new DynamicDimension("batch");
        Dimension time = new DynamicDimension("time");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, time, new StaticDimension(4)),
                true);
        Tensor originalMask = tensor(
                DataType.BOOL, Shape.ofDimensions(batch, time), false);

        assertThrows(IllegalArgumentException.class, () -> input.mean(1, originalMask));

        Tensor alignedMask = originalMask.expandDims(2);
        Tensor result = input.mean(1, alignedMask);
        TensorProvenance resultProvenance = result.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(
                        Shape.ofDimensions(batch, time, new StaticDimension(1)),
                        alignedMask.descriptor().shape()),
                () -> assertTrue(alignedMask.provenance().isPresent()),
                () -> assertSame(input, resultProvenance.inputs().get(0)),
                () -> assertSame(alignedMask, resultProvenance.inputs().get(1)),
                () -> assertNotSame(originalMask, resultProvenance.inputs().get(1)),
                () -> assertEquals(
                        new MaskedReductionAttrs(1),
                        resultProvenance.operation().attrs()));
    }

    @Test
    void removesAxisPreservesReferencesAndLeavesInputsUnchanged() {
        Dimension batch = new DynamicDimension("batch");
        Dimension rows = new StaticDimension(3);
        Dimension columns = new DynamicDimension("columns");
        Shape shape = Shape.ofDimensions(batch, rows, columns);
        float[] inputValues = new float[12];
        byte[] maskValues = {1, 0, 1};
        TensorDescriptor inputDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.empty(),
                true);
        TensorDescriptor maskDescriptor = new TensorDescriptor(
                DataType.BOOL,
                Shape.of(3, 1),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(3, 1))),
                false);
        HostTensorStorage inputStorage = new MemorySegmentStorage(
                DataType.FLOAT32, inputValues.length, MemorySegment.ofArray(inputValues));
        HostTensorStorage maskStorage = new MemorySegmentStorage(
                DataType.BOOL, maskValues.length, MemorySegment.ofArray(maskValues));
        Tensor leaf = tensor(DataType.FLOAT32, shape, false);
        TensorProvenance prior = new TensorProvenance(
                new TensorProducer(
                        new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE),
                        List.of(leaf),
                        List.of(inputDescriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                inputDescriptor,
                Optional.of("input"),
                Optional.of(prior),
                Optional.of(inputStorage));
        Tensor mask = new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                maskDescriptor,
                Optional.of("mask"),
                Optional.empty(),
                Optional.of(maskStorage));

        Tensor first = input.sum(1, mask);
        Tensor second = input.sum(1, mask);
        List<Dimension> resultDimensions = first.descriptor().shape().dimensions();
        Tensor rankOne = tensor(DataType.FLOAT32, Shape.ofDimensions(columns), false);
        Tensor scalarResult = rankOne.sum(
                0, tensor(DataType.BOOL, Shape.scalar(), false));

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertEquals(2, resultDimensions.size()),
                () -> assertSame(batch, resultDimensions.get(0)),
                () -> assertSame(columns, resultDimensions.get(1)),
                () -> assertSame(Shape.scalar(), scalarResult.descriptor().shape()),
                () -> assertSame(inputDescriptor, input.descriptor()),
                () -> assertSame(maskDescriptor, mask.descriptor()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertEquals(Optional.of("mask"), mask.label()),
                () -> assertSame(prior, input.provenance().orElseThrow()),
                () -> assertSame(inputStorage, input.hostStorage().orElseThrow()),
                () -> assertSame(maskStorage, mask.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new float[12], inputValues),
                () -> assertArrayEquals(new byte[] {1, 0, 1}, maskValues),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(first.descriptor().layout().isEmpty()));
    }

    @Test
    void validatesInExactOrderAndConsumesNoIdentity() throws ReflectiveOperationException {
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(2), false);
        Tensor wrongMask = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor incompatible = tensor(DataType.BOOL, Shape.of(3), false);
        AtomicLong nextId = nextTensorIdState();
        long before = nextId.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorMaskedReductionExpressions.apply(null, null, null, 0));
        NullPointerException nullMask = assertThrows(
                NullPointerException.class,
                () -> TensorMaskedReductionExpressions.apply(floating, null, null, 0));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorMaskedReductionExpressions.apply(floating, bool, null, 0));
        IllegalArgumentException invalidKind = assertThrows(
                IllegalArgumentException.class,
                () -> TensorMaskedReductionExpressions.apply(
                        floating, bool, AggregateReductionKind.ARG_MAX, 8));
        IllegalArgumentException invalidInput = assertThrows(
                IllegalArgumentException.class,
                () -> integral.sum(8, wrongMask));
        IllegalArgumentException invalidMask = assertThrows(
                IllegalArgumentException.class,
                () -> floating.sum(8, wrongMask));
        IndexOutOfBoundsException invalidAxis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> floating.sum(1, incompatible));
        IllegalArgumentException invalidBroadcast = assertThrows(
                IllegalArgumentException.class,
                () -> floating.sum(0, incompatible));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("mask", nullMask.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals(
                        "kind must be SUM or MEAN, but was ARG_MAX",
                        invalidKind.getMessage()),
                () -> assertEquals(
                        "input must be a floating data type, but was INT32",
                        invalidInput.getMessage()),
                () -> assertEquals(
                        "mask must have BOOL data type, but was FLOAT32",
                        invalidMask.getMessage()),
                () -> assertEquals("Axis 1 is outside shape rank 1", invalidAxis.getMessage()),
                () -> assertEquals(
                        "Cannot broadcast dimensions StaticDimension[size=2]"
                                + " and StaticDimension[size=3] at result axis 0 for"
                                + " Shape[2] and Shape[3]",
                        invalidBroadcast.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void reportsScalarAxisFailureBeforeBroadcast() throws ReflectiveOperationException {
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor mask = tensor(DataType.BOOL, Shape.of(2), false);
        AtomicLong nextId = nextTensorIdState();
        long before = nextId.get();

        IndexOutOfBoundsException failure = assertThrows(
                IndexOutOfBoundsException.class, () -> scalar.mean(0, mask));

        assertAll(
                () -> assertEquals("Axis 0 is outside shape rank 0", failure.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void propagatesIdentifierExhaustionOnlyAfterValidConstruction()
            throws ReflectiveOperationException {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor mask = tensor(DataType.BOOL, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();

        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class, () -> input.sum(0, mask));

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()),
                    () -> assertTrue(input.provenance().isEmpty()),
                    () -> assertTrue(mask.provenance().isEmpty()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertHelperMethod(
            String name,
            boolean privateMethod,
            Class<?> returnType,
            Class<?>... parameterTypes)
            throws ReflectiveOperationException {
        Method method = TensorMaskedReductionExpressions.class.getDeclaredMethod(
                name, parameterTypes);
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertEquals(privateMethod, Modifier.isPrivate(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
    }

    private static void assertExpression(
            Tensor result,
            Tensor input,
            Tensor mask,
            AggregateReductionKind kind,
            int expectedAxis) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(kind, provenance.operation().kind()),
                () -> assertEquals(
                        new MaskedReductionAttrs(expectedAxis),
                        provenance.operation().attrs()),
                () -> assertEquals(2, provenance.inputs().size()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertSame(mask, provenance.inputs().get(1)));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
