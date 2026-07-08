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
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
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
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorBooleanReductionTest {
    private static final List<BooleanReductionFamily> FAMILIES = List.of(
            new BooleanReductionFamily(
                    AggregateReductionKind.ALL, Tensor::all, Tensor::all, Tensor::all),
            new BooleanReductionFamily(
                    AggregateReductionKind.ANY, Tensor::any, Tensor::any, Tensor::any));
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(120_000);

    @Test
    void mapsEveryPublicFormToExactKindAttributesAndBoolResult() {
        Tensor input = tensor(DataType.BOOL, Shape.of(2, 3));

        for (BooleanReductionFamily family : FAMILIES) {
            Tensor full = family.full().apply(input);
            Tensor removed = family.axis().apply(input, -1);
            Tensor retained = family.retained().apply(input, 0, true);

            assertExpression(full, input, family.kind(), NoOperationAttrs.INSTANCE);
            assertExpression(removed, input, family.kind(), new AxisReductionAttrs(1, false));
            assertExpression(retained, input, family.kind(), new AxisReductionAttrs(0, true));
            for (Tensor result : List.of(full, removed, retained)) {
                assertAll(
                        () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                        () -> assertFalse(result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()));
            }
        }
    }

    @Test
    void fullAndAxisFormsCoverScalarStaticZeroAndDynamicShapes() {
        List<Shape> shapes = List.of(
                Shape.scalar(),
                Shape.of(2, 3),
                Shape.of(2, 0, 4),
                Shape.ofDimensions(new DynamicDimension("batch"), new StaticDimension(3)));
        for (Shape shape : shapes) {
            for (BooleanReductionFamily family : FAMILIES) {
                assertSame(Shape.scalar(), family.full().apply(tensor(DataType.BOOL, shape))
                        .descriptor().shape());
            }
        }

        Dimension batch = new DynamicDimension("batch");
        Dimension rows = new StaticDimension(4);
        Dimension columns = new DynamicDimension("columns");
        Dimension zero = new StaticDimension(0);
        Tensor input = tensor(
                DataType.BOOL, Shape.ofDimensions(batch, rows, columns, zero));
        for (BooleanReductionFamily family : FAMILIES) {
            List<Dimension> removed = family.axis().apply(input, -2)
                    .descriptor().shape().dimensions();
            List<Dimension> retained = family.retained().apply(input, 2, true)
                    .descriptor().shape().dimensions();
            assertAll(
                    () -> assertEquals(3, removed.size()),
                    () -> assertSame(batch, removed.get(0)),
                    () -> assertSame(rows, removed.get(1)),
                    () -> assertSame(zero, removed.get(2)),
                    () -> assertEquals(4, retained.size()),
                    () -> assertSame(batch, retained.get(0)),
                    () -> assertSame(rows, retained.get(1)),
                    () -> assertEquals(new StaticDimension(1), retained.get(2)),
                    () -> assertNotSame(columns, retained.get(2)),
                    () -> assertSame(zero, retained.get(3)));
        }
        Tensor rankOne = tensor(DataType.BOOL, Shape.ofDimensions(columns));
        for (BooleanReductionFamily family : FAMILIES) {
            assertSame(Shape.scalar(), family.axis().apply(rankOne, -1).descriptor().shape());
        }
    }

    @Test
    void rejectsEveryNonBoolTypeWithKindSpecificMessageBeforeAxisAndIdentity() throws Exception {
        AtomicLong nextId = nextTensorIdState();
        long before = nextId.get();
        for (DataType type : List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            Tensor invalid = tensor(type, Shape.of(2));
            for (BooleanReductionFamily family : FAMILIES) {
                IllegalArgumentException full = assertThrows(
                        IllegalArgumentException.class, () -> family.full().apply(invalid));
                IllegalArgumentException axis = assertThrows(
                        IllegalArgumentException.class, () -> family.axis().apply(invalid, 9));
                String expected = "input must have BOOL data type for "
                        + family.kind() + ", but was " + type;
                assertAll(
                        () -> assertEquals(expected, full.getMessage()),
                        () -> assertEquals(expected, axis.getMessage()));
            }
        }
        assertEquals(before, nextId.get());
    }

    @Test
    void validatesNullKindTypeAndAxisInExactOrder() throws Exception {
        AtomicLong nextId = nextTensorIdState();
        Tensor bool = tensor(DataType.BOOL, Shape.of(2));
        long before = nextId.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyAxis(null, null, 9, false));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorReductionExpressions.applyAxis(bool, null, 9, false));
        IllegalArgumentException unsupported = assertThrows(
                IllegalArgumentException.class,
                () -> TensorReductionExpressions.applyAxis(
                        bool, AggregateReductionKind.ARG_MAX, 9, false));
        IndexOutOfBoundsException invalidAxis = assertThrows(
                IndexOutOfBoundsException.class, () -> bool.all(2));
        IndexOutOfBoundsException scalarAxis = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.BOOL, Shape.scalar()).any(0, true));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals(
                        "kind must be SUM, MEAN, PROD, MIN, MAX, ALL, or ANY, but was ARG_MAX",
                        unsupported.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 1", invalidAxis.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalarAxis.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void aggregateAndElementwiseLogicalSemanticsRemainDistinct() {
        Tensor left = tensor(DataType.BOOL, Shape.of(2));
        Tensor right = tensor(DataType.BOOL, Shape.of(2));
        Tensor all = left.all();
        Tensor any = left.any(0);
        Tensor and = left.logicalAnd(right);
        Tensor or = left.logicalOr(right);

        assertAll(
                () -> assertSame(AggregateReductionKind.ALL,
                        all.provenance().orElseThrow().operation().kind()),
                () -> assertSame(AggregateReductionKind.ANY,
                        any.provenance().orElseThrow().operation().kind()),
                () -> assertSame(BooleanLogicalKind.AND,
                        and.provenance().orElseThrow().operation().kind()),
                () -> assertSame(BooleanLogicalKind.OR,
                        or.provenance().orElseThrow().operation().kind()),
                () -> assertEquals(1, all.provenance().orElseThrow().inputs().size()),
                () -> assertEquals(1, any.provenance().orElseThrow().inputs().size()),
                () -> assertEquals(2, and.provenance().orElseThrow().inputs().size()),
                () -> assertEquals(2, or.provenance().orElseThrow().inputs().size()));
    }

    @Test
    void callsAreFreshAndLeaveInputMetadataStorageAndContentsUnchanged() {
        byte[] values = {0, 1, 1, 0};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.BOOL, shape, Optional.of(layout), false);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.BOOL, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.BOOL, shape);
        TensorProvenance provenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(BooleanLogicalKind.NOT, NoOperationAttrs.INSTANCE),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()), descriptor, Optional.of("input"),
                Optional.of(provenance), Optional.of(storage));

        Tensor first = input.all(1);
        Tensor second = input.all(1);
        Tensor nested = first.any();

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first, nested),
                () -> assertSame(input, first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(provenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new byte[] {0, 1, 1, 0}, values),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(first.descriptor().layout().isEmpty()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidConstruction() throws Exception {
        Tensor input = tensor(DataType.BOOL, Shape.of(2));
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(IllegalStateException.class, input::all);
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertExpression(
            Tensor result, Tensor input, AggregateReductionKind kind, Object attrs) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        assertAll(
                () -> assertSame(kind, provenance.operation().kind()),
                () -> assertEquals(attrs, provenance.operation().attrs()),
                () -> assertEquals(1, provenance.inputs().size()),
                () -> assertSame(input, provenance.inputs().getFirst()));
        if (attrs == NoOperationAttrs.INSTANCE) {
            assertSame(NoOperationAttrs.INSTANCE, provenance.operation().attrs());
        }
    }

    private static Tensor tensor(DataType type, Shape shape) {
        return new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), false),
                Optional.empty(), Optional.empty(), Optional.empty());
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

    private record BooleanReductionFamily(
            AggregateReductionKind kind,
            FullReduction full,
            AxisReduction axis,
            RetainedReduction retained) {
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
