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
        List<Method> methods = Arrays.stream(helper.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();

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
                () -> assertEquals(5, methods.size()));

        assertHelperMethod(
                "apply",
                false,
                Tensor.class,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                int.class);
        assertHelperMethod(
                "resolveMapping",
                true,
                List.class,
                Shape.class,
                Shape.class,
                int.class);
        assertHelperMethod(
                "compatible",
                true,
                boolean.class,
                Dimension.class,
                Dimension.class);
        assertHelperMethod("reduceShape", true, Shape.class, Shape.class, int.class);
        assertHelperMethod(
                "create",
                true,
                Tensor.class,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                int.class,
                List.class,
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
    void delegatesPublicMethodsToExactKindsAndOrderedInputs() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        Tensor mask = tensor(DataType.BOOL, Shape.of(2, 3), false);

        Tensor sum = input.sum(1, mask);
        Tensor mean = input.mean(-2, mask);

        assertExpression(
                sum,
                input,
                mask,
                AggregateReductionKind.SUM,
                new MaskedReductionAttrs(1, List.of(0, 1)));
        assertExpression(
                mean,
                input,
                mask,
                AggregateReductionKind.MEAN,
                new MaskedReductionAttrs(1, List.of(0, 1)));
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
    void resolvesScalarContiguousRightAndNoncontiguousMappings() {
        Dimension batch = new DynamicDimension("batch");
        Dimension time = new DynamicDimension("time");
        Dimension features = new StaticDimension(4);
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, time, features),
                false);

        assertMapping(input.sum(1, tensor(DataType.BOOL, Shape.scalar(), false)), List.of());
        assertMapping(
                input.sum(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(batch, time), false)),
                List.of(0, 1));
        assertMapping(
                input.mean(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(time, features), false)),
                List.of(1, 2));
        assertMapping(
                input.sum(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(batch, features), false)),
                List.of(0, 2));
    }

    @Test
    void prefersAxisCoverageThenDisplacementAndDeterministicAxisOrder() {
        Dimension repeated = new DynamicDimension("N");
        Tensor repeatedInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(repeated, repeated, repeated, repeated),
                false);

        assertMapping(
                repeatedInput.sum(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(repeated), false)),
                List.of(1));
        assertMapping(
                repeatedInput.sum(2, tensor(
                        DataType.BOOL, Shape.ofDimensions(repeated, repeated), false)),
                List.of(0, 2));

        Tensor noCoverage = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        assertMapping(
                noCoverage.mean(1, tensor(DataType.BOOL, Shape.of(2), false)),
                List.of(0));

        Tensor singletonMask = tensor(DataType.BOOL, Shape.of(1, 1), false);
        assertMapping(noCoverage.sum(1, singletonMask), List.of(0, 1));
        assertMapping(noCoverage.sum(1, singletonMask), List.of(0, 1));
    }

    @Test
    void handlesZeroExtentsAndOnlyLocallyProvableDynamicRelationships() {
        Dimension batch = new DynamicDimension("batch");
        Dimension zero = new StaticDimension(0);
        Dimension width = new DynamicDimension("width");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, zero, width),
                false);

        assertMapping(
                input.sum(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(zero), false)),
                List.of(1));
        assertMapping(
                input.mean(1, tensor(
                        DataType.BOOL, Shape.ofDimensions(new StaticDimension(1)), false)),
                List.of(1));
        assertMapping(
                input.sum(0, tensor(
                        DataType.BOOL,
                        Shape.ofDimensions(new DynamicDimension("batch")),
                        false)),
                List.of(0));

        Tensor differentDynamic = tensor(
                DataType.BOOL,
                Shape.ofDimensions(new DynamicDimension("other")),
                false);
        IllegalArgumentException dynamicFailure = assertThrows(
                IllegalArgumentException.class, () -> input.sum(0, differentDynamic));
        Tensor staticNonSingleton = tensor(DataType.BOOL, Shape.of(3), false);
        IllegalArgumentException staticFailure = assertThrows(
                IllegalArgumentException.class, () -> input.sum(0, staticNonSingleton));

        assertAll(
                () -> assertEquals(
                        "mask shape Shape[other] cannot be aligned to input shape"
                                + " Shape[batch, 0, width] for reduction axis 0",
                        dynamicFailure.getMessage()),
                () -> assertEquals(
                        "mask shape Shape[3] cannot be aligned to input shape"
                                + " Shape[batch, 0, width] for reduction axis 0",
                        staticFailure.getMessage()));
    }

    @Test
    void removesAxisAndPreservesUnaffectedDimensionReferences() {
        Dimension batch = new DynamicDimension("batch");
        Dimension rows = new StaticDimension(3);
        Dimension columns = new DynamicDimension("columns");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, rows, columns),
                true);
        Tensor mask = tensor(DataType.BOOL, Shape.ofDimensions(rows), false);

        Tensor result = input.mean(-2, mask);
        List<Dimension> resultDimensions = result.descriptor().shape().dimensions();
        Tensor rankOne = tensor(DataType.FLOAT32, Shape.ofDimensions(columns), false);
        Tensor scalarResult = rankOne.sum(
                0, tensor(DataType.BOOL, Shape.scalar(), false));

        assertAll(
                () -> assertEquals(2, resultDimensions.size()),
                () -> assertSame(batch, resultDimensions.get(0)),
                () -> assertSame(columns, resultDimensions.get(1)),
                () -> assertSame(Shape.scalar(), scalarResult.descriptor().shape()));
    }

    @Test
    void everyResultIsFreshNestableAndLeavesInputsUnchanged() {
        float[] inputValues = {1.0f, 2.0f, 3.0f, 4.0f};
        byte[] maskValues = {1, 0, 1, 0};
        Shape shape = Shape.of(2, 2);
        TensorDescriptor inputDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        TensorDescriptor maskDescriptor = new TensorDescriptor(
                DataType.BOOL,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
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
        Tensor nested = first.mean(
                0, tensor(DataType.BOOL, Shape.scalar(), false));

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first, nested),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(inputDescriptor, input.descriptor()),
                () -> assertSame(maskDescriptor, mask.descriptor()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertEquals(Optional.of("mask"), mask.label()),
                () -> assertSame(prior, input.provenance().orElseThrow()),
                () -> assertTrue(mask.provenance().isEmpty()),
                () -> assertSame(inputStorage, input.hostStorage().orElseThrow()),
                () -> assertSame(maskStorage, mask.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, inputValues),
                () -> assertArrayEquals(new byte[] {1, 0, 1, 0}, maskValues),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(first.descriptor().layout().isEmpty()));
    }

    @Test
    void validatesInExactOrderAndConsumesNoIdentity() throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2), false);
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
        Tensor integral = tensor(DataType.INT32, Shape.of(2), false);
        Tensor wrongMask = tensor(DataType.FLOAT32, Shape.of(2), false);
        IllegalArgumentException invalidInput = assertThrows(
                IllegalArgumentException.class,
                () -> integral.sum(8, wrongMask));
        IllegalArgumentException invalidMask = assertThrows(
                IllegalArgumentException.class,
                () -> floating.sum(8, wrongMask));
        Tensor rankTwoMask = tensor(DataType.BOOL, Shape.of(1, 1), false);
        IndexOutOfBoundsException invalidAxis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> floating.sum(1, rankTwoMask));
        IllegalArgumentException invalidRank = assertThrows(
                IllegalArgumentException.class,
                () -> floating.sum(0, rankTwoMask));

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
                () -> assertEquals(
                        "Axis 1 is outside shape rank 1", invalidAxis.getMessage()),
                () -> assertEquals(
                        "mask rank must not exceed input rank: mask=2, input=1",
                        invalidRank.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void reportsScalarAxisFailureBeforeRankOrAlignment() throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        long before = nextId.get();
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor mask = tensor(DataType.BOOL, Shape.of(1), false);

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
            MaskedReductionAttrs expectedAttrs) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(kind, provenance.operation().kind()),
                () -> assertEquals(expectedAttrs, provenance.operation().attrs()),
                () -> assertEquals(2, provenance.inputs().size()),
                () -> assertSame(input, provenance.inputs().get(0)),
                () -> assertSame(mask, provenance.inputs().get(1)));
    }

    private static void assertMapping(Tensor result, List<Integer> expectedMapping) {
        Object attrs = result.provenance().orElseThrow().operation().attrs();
        assertTrue(attrs instanceof MaskedReductionAttrs);
        assertEquals(expectedMapping, ((MaskedReductionAttrs) attrs).maskInputAxes());
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
