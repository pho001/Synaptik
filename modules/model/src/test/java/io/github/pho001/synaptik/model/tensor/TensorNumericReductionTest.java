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
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
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

class TensorNumericReductionTest {
    private static final List<ReductionFamily> FAMILIES = List.of(
            new ReductionFamily(
                    "sum", AggregateReductionKind.SUM, Tensor::sum, Tensor::sum, Tensor::sum),
            new ReductionFamily(
                    "mean", AggregateReductionKind.MEAN, Tensor::mean, Tensor::mean, Tensor::mean),
            new ReductionFamily(
                    "prod", AggregateReductionKind.PROD, Tensor::prod, Tensor::prod, Tensor::prod),
            new ReductionFamily(
                    "min", AggregateReductionKind.MIN, Tensor::min, Tensor::min, Tensor::min),
            new ReductionFamily(
                    "max", AggregateReductionKind.MAX, Tensor::max, Tensor::max, Tensor::max));
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(80_000);

    @Test
    void exposesExactPublicEntriesAndPackagePrivateHelperShape()
            throws ReflectiveOperationException {
        int modifiers = TensorReductionExpressions.class.getModifiers();
        var constructors = TensorReductionExpressions.class.getDeclaredConstructors();
        List<Method> methods = Arrays.stream(TensorReductionExpressions.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertFalse(TensorReductionExpressions.class.isRecord()),
                () -> assertEquals(
                        Set.of(), Set.of(TensorReductionExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorReductionExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorReductionExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(6, methods.size()));

        assertHelperMethod(
                "applyFull",
                false,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class);
        assertHelperMethod(
                "applyAxis",
                false,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                int.class,
                boolean.class);
        assertHelperMethod(
                "validateKind", true, void.class, AggregateReductionKind.class);
        assertHelperMethod(
                "validateInput", true, void.class, Tensor.class, AggregateReductionKind.class);
        assertHelperMethod(
                "reduceShape", true, Shape.class, Shape.class, int.class, boolean.class);
        assertHelperMethod(
                "create",
                true,
                Tensor.class,
                Tensor.class,
                AggregateReductionKind.class,
                OperationAttrs.class,
                Shape.class);

        for (ReductionFamily family : FAMILIES) {
            Method full = Tensor.class.getDeclaredMethod(family.methodName());
            Method axis = Tensor.class.getDeclaredMethod(family.methodName(), int.class);
            Method retained = Tensor.class.getDeclaredMethod(
                    family.methodName(), int.class, boolean.class);
            for (Method method : List.of(full, axis, retained)) {
                assertAll(
                        () -> assertEquals(Tensor.class, method.getReturnType()),
                        () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                        () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                        () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
            }
        }
    }

    @Test
    void mapsEveryPublicFormToExactKindAttributesAndOneInputProvenance() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);

        for (ReductionFamily family : FAMILIES) {
            Tensor full = family.applyFull(input);
            Tensor removed = family.applyAxis(input, -1);
            Tensor retained = family.applyAxis(input, 0, true);

            assertExpression(full, input, family.kind(), NoOperationAttrs.INSTANCE);
            assertExpression(
                    removed,
                    input,
                    family.kind(),
                    new AxisReductionAttrs(1, false));
            assertExpression(
                    retained,
                    input,
                    family.kind(),
                    new AxisReductionAttrs(0, true));
        }
    }

    @Test
    void acceptsEveryFloatingTypeAndPreservesGradientEligibilityForEveryKind() {
        for (DataType dataType : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            for (boolean requiresGrad : List.of(false, true)) {
                Tensor input = tensor(dataType, Shape.of(2, 3), requiresGrad);
                for (ReductionFamily family : FAMILIES) {
                    for (Tensor result : List.of(
                            family.applyFull(input),
                            family.applyAxis(input, 1),
                            family.applyAxis(input, -2, true))) {
                        assertAll(
                                () -> assertSame(dataType, result.descriptor().dataType()),
                                () -> assertEquals(
                                        requiresGrad, result.descriptor().requiresGrad()),
                                () -> assertTrue(result.descriptor().layout().isEmpty()),
                                () -> assertTrue(result.label().isEmpty()),
                                () -> assertTrue(result.hostStorage().isEmpty()));
                    }
                }
            }
        }
    }

    @Test
    void fullFormsProduceCanonicalScalarAcrossAllSupportedShapeStates() {
        List<Shape> shapes = List.of(
                Shape.scalar(),
                Shape.of(2, 3),
                Shape.of(2, 0, 4),
                Shape.ofDimensions(new DynamicDimension("batch"), new StaticDimension(3)));

        for (Shape shape : shapes) {
            Tensor input = tensor(DataType.FLOAT32, shape, false);
            for (ReductionFamily family : FAMILIES) {
                Tensor result = family.applyFull(input);
                assertSame(Shape.scalar(), result.descriptor().shape());
            }
        }
    }

    @Test
    void axisFormsNormalizeRemoveRetainAndPreserveUnaffectedDimensionReferences() {
        Dimension batch = new DynamicDimension("batch");
        Dimension rows = new StaticDimension(4);
        Dimension columns = new DynamicDimension("columns");
        Dimension zero = new StaticDimension(0);
        Shape inputShape = Shape.ofDimensions(batch, rows, columns, zero);
        Tensor input = tensor(DataType.FLOAT32, inputShape, true);

        for (ReductionFamily family : FAMILIES) {
            Tensor removed = family.applyAxis(input, -2);
            Tensor retained = family.applyAxis(input, 2, true);
            List<Dimension> removedDimensions = removed.descriptor().shape().dimensions();
            List<Dimension> retainedDimensions = retained.descriptor().shape().dimensions();

            assertAll(
                    () -> assertEquals(3, removedDimensions.size()),
                    () -> assertSame(batch, removedDimensions.get(0)),
                    () -> assertSame(rows, removedDimensions.get(1)),
                    () -> assertSame(zero, removedDimensions.get(2)),
                    () -> assertEquals(4, retainedDimensions.size()),
                    () -> assertSame(batch, retainedDimensions.get(0)),
                    () -> assertSame(rows, retainedDimensions.get(1)),
                    () -> assertEquals(new StaticDimension(1), retainedDimensions.get(2)),
                    () -> assertNotSame(columns, retainedDimensions.get(2)),
                    () -> assertSame(zero, retainedDimensions.get(3)),
                    () -> assertEquals(
                            new AxisReductionAttrs(2, false),
                            removed.provenance().orElseThrow().operation().attrs()),
                    () -> assertEquals(
                            new AxisReductionAttrs(2, true),
                            retained.provenance().orElseThrow().operation().attrs()));
        }

        Tensor rankOne = tensor(DataType.FLOAT32, Shape.ofDimensions(columns), false);
        for (ReductionFamily family : FAMILIES) {
            assertSame(Shape.scalar(), family.applyAxis(rankOne, -1).descriptor().shape());
        }
    }

    @Test
    void everyValidCallIsFreshAndLeavesInputMetadataStorageAndContentsUnchanged() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, true);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));

        Tensor firstSum = input.sum(1);
        Tensor secondSum = input.sum(1);
        Tensor nestedProduct = firstSum.prod();
        Tensor firstMinimum = input.min(1);
        Tensor secondMinimum = input.min(1);
        Tensor nestedMaximum = firstMinimum.max();

        assertAll(
                () -> assertNotSame(firstSum, secondSum),
                () -> assertNotEquals(firstSum.id(), secondSum.id()),
                () -> assertNotSame(firstSum, nestedProduct),
                () -> assertSame(
                        input, firstSum.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(
                        firstSum,
                        nestedProduct.provenance().orElseThrow().inputs().getFirst()),
                () -> assertNotSame(firstMinimum, secondMinimum),
                () -> assertNotEquals(firstMinimum.id(), secondMinimum.id()),
                () -> assertNotSame(firstMinimum, nestedMaximum),
                () -> assertSame(
                        input, firstMinimum.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(
                        firstMinimum,
                        nestedMaximum.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new float[] {1.0f, 2.0f, 3.0f, 4.0f}, values),
                () -> assertTrue(firstSum.label().isEmpty()),
                () -> assertTrue(firstSum.hostStorage().isEmpty()),
                () -> assertTrue(firstSum.descriptor().layout().isEmpty()),
                () -> assertTrue(firstMinimum.label().isEmpty()),
                () -> assertTrue(firstMinimum.hostStorage().isEmpty()),
                () -> assertTrue(firstMinimum.descriptor().layout().isEmpty()));
    }

    @Test
    void validatesNullKindTypeAndAxisInExactOrderWithoutConsumingIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2), false);
        long before = nextId.get();

        NullPointerException nullFullInput = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyFull(null, null));
        NullPointerException nullFullKind = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyFull(floating, null));
        NullPointerException nullAxisInput = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyAxis(null, null, 0, false));
        NullPointerException nullAxisKind = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyAxis(floating, null, 0, false));

        assertAll(
                () -> assertEquals("input", nullFullInput.getMessage()),
                () -> assertEquals("kind", nullFullKind.getMessage()),
                () -> assertEquals("input", nullAxisInput.getMessage()),
                () -> assertEquals("kind", nullAxisKind.getMessage()));

        for (AggregateReductionKind kind : List.of(AggregateReductionKind.ARG_MAX)) {
            IllegalArgumentException fullFailure = assertThrows(
                    IllegalArgumentException.class,
                    () -> TensorReductionExpressions.applyFull(floating, kind));
            IllegalArgumentException axisFailure = assertThrows(
                    IllegalArgumentException.class,
                    () -> TensorReductionExpressions.applyAxis(floating, kind, 9, false));
            assertAll(
                    () -> assertEquals(
                            "kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was " + kind,
                            fullFailure.getMessage()),
                    () -> assertEquals(fullFailure.getMessage(), axisFailure.getMessage()));
        }

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalid = tensor(dataType, Shape.of(2), false);
            for (ReductionFamily family : FAMILIES) {
                IllegalArgumentException fullFailure = assertThrows(
                        IllegalArgumentException.class, () -> family.applyFull(invalid));
                IllegalArgumentException axisFailure = assertThrows(
                        IllegalArgumentException.class, () -> family.applyAxis(invalid, 9));
                assertAll(
                        () -> assertEquals(
                                "input must be a floating data type, but was " + dataType,
                                fullFailure.getMessage()),
                        () -> assertEquals(fullFailure.getMessage(), axisFailure.getMessage()));
            }
        }

        IndexOutOfBoundsException highAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> floating.sum(1));
        IndexOutOfBoundsException lowAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> floating.mean(-2));
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        IndexOutOfBoundsException scalarAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> scalar.prod(0, true));

        assertAll(
                () -> assertEquals("Axis 1 is outside shape rank 1", highAxis.getMessage()),
                () -> assertEquals("Axis -2 is outside shape rank 1", lowAxis.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalarAxis.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void keepsAggregateAndBinaryMinimumAndMaximumAsDistinctTypedSemantics() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        Tensor aggregateMinimum = left.min();
        Tensor aggregateMaximum = left.max(1);
        Tensor binaryMinimum = left.minimum(right);
        Tensor binaryMaximum = left.maximum(right);

        assertAll(
                () -> assertSame(
                        AggregateReductionKind.MIN,
                        aggregateMinimum.provenance().orElseThrow().operation().kind()),
                () -> assertSame(
                        AggregateReductionKind.MAX,
                        aggregateMaximum.provenance().orElseThrow().operation().kind()),
                () -> assertSame(
                        BinaryArithmeticKind.MIN,
                        binaryMinimum.provenance().orElseThrow().operation().kind()),
                () -> assertSame(
                        BinaryArithmeticKind.MAX,
                        binaryMaximum.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(
                        List.of(left), aggregateMinimum.provenance().orElseThrow().inputs()),
                () -> assertEquals(
                        List.of(left), aggregateMaximum.provenance().orElseThrow().inputs()),
                () -> assertEquals(
                        List.of(left, right), binaryMinimum.provenance().orElseThrow().inputs()),
                () -> assertEquals(
                        List.of(left, right), binaryMaximum.provenance().orElseThrow().inputs()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction()
            throws ReflectiveOperationException {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), true);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();

        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class, input::sum);

            assertAll(
                    () -> assertEquals(
                            "tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()),
                    () -> assertEquals(Optional.empty(), input.provenance()),
                    () -> assertEquals(Optional.empty(), input.hostStorage()));
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
        Method method = TensorReductionExpressions.class.getDeclaredMethod(name, parameterTypes);
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
            AggregateReductionKind kind,
            Object expectedAttrs) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(kind, provenance.operation().kind()),
                () -> assertEquals(expectedAttrs, provenance.operation().attrs()),
                () -> assertEquals(1, provenance.inputs().size()),
                () -> assertSame(input, provenance.inputs().getFirst()));
        if (expectedAttrs == NoOperationAttrs.INSTANCE) {
            assertSame(NoOperationAttrs.INSTANCE, provenance.operation().attrs());
        }
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

    private record ReductionFamily(
            String methodName,
            AggregateReductionKind kind,
            FullReduction full,
            AxisReduction axis,
            RetainedReduction retained) {
        private Tensor applyFull(Tensor input) {
            return full.apply(input);
        }

        private Tensor applyAxis(Tensor input, int axis) {
            return this.axis.apply(input, axis);
        }

        private Tensor applyAxis(Tensor input, int axis, boolean keepDimensions) {
            return retained.apply(input, axis, keepDimensions);
        }
    }

    @FunctionalInterface
    private interface FullReduction {
        Tensor apply(Tensor input);
    }

    @FunctionalInterface
    private interface AxisReduction {
        Tensor apply(Tensor input, int axis);
    }

    @FunctionalInterface
    private interface RetainedReduction {
        Tensor apply(Tensor input, int axis, boolean keepDimensions);
    }
}
