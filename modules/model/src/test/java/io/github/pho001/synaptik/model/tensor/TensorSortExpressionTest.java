package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;




import java.util.Optional;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorSortExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(90_000);

    

    @Test
    void defaultsToAscendingAndNormalizesAxesIntoExactAttributes() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        assertAll(
                () -> assertEquals(new SortAttrs(1, false),
                        input.sort(-1).provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(new SortAttrs(0, true),
                        input.sort(0, true).provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(new SortAttrs(1, false),
                        input.argsort(-1).provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(new SortAttrs(0, true),
                        input.argsort(0, true).provenance().orElseThrow().operation().attrs()));
    }

    @Test
    void acceptsEveryCurrentTypeAndSelectsExactResultMetadata() {
        Shape shape = Shape.of(2, 3);
        for (DataType type : DataType.values()) {
            for (boolean requiresGrad : type.isFloating() ? List.of(false, true) : List.of(false)) {
                Tensor input = tensor(type, shape, requiresGrad);
                Tensor sorted = input.sort(1);
                Tensor indices = input.argsort(1);
                assertAll(
                        () -> assertSame(type, sorted.descriptor().dataType()),
                        () -> assertSame(shape, sorted.descriptor().shape()),
                        () -> assertEquals(requiresGrad, sorted.descriptor().requiresGrad()),
                        () -> assertSame(DataType.INT64, indices.descriptor().dataType()),
                        () -> assertSame(shape, indices.descriptor().shape()),
                        () -> assertFalse(indices.descriptor().requiresGrad()),
                        () -> assertTrue(sorted.descriptor().layout().isEmpty()),
                        () -> assertTrue(indices.descriptor().layout().isEmpty()),
                        () -> assertTrue(sorted.label().isEmpty()),
                        () -> assertTrue(indices.hostStorage().isEmpty()));
            }
        }
    }

    @Test
    void acceptsEmptySingletonDynamicAndExpressionExtentsWithExactShapeReferences() {
        Shape empty = Shape.of(2, 0, 3);
        Shape singleton = Shape.of(1);
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        Shape expression = Shape.ofDimensions(
                DimensionExpressions.addConstant(new DynamicDimension("M"), 2));
        for (Shape shape : List.of(empty, singleton, dynamic, expression)) {
            Tensor input = tensor(DataType.BFLOAT16, shape, true);
            assertAll(
                    () -> assertSame(shape, input.sort(0).descriptor().shape()),
                    () -> assertSame(shape, input.argsort(0).descriptor().shape()));
        }
        assertSame(empty, tensor(DataType.BOOL, empty, false).sort(1).descriptor().shape());
    }

    @Test
    void createsIndependentOneOutputProducersWithExactInputsAndFreshIdentities() {
        Tensor input = tensor(DataType.INT32, Shape.of(4), false);
        Tensor first = input.sort(0);
        Tensor second = input.sort(0);
        Tensor indices = input.argsort(0);
        TensorProvenance firstOrigin = first.provenance().orElseThrow();
        TensorProvenance secondOrigin = second.provenance().orElseThrow();
        TensorProvenance indexOrigin = indices.provenance().orElseThrow();

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(firstOrigin.producer(), secondOrigin.producer()),
                () -> assertNotSame(firstOrigin.producer(), indexOrigin.producer()),
                () -> assertSame(OrderingKind.SORT, firstOrigin.operation().kind()),
                () -> assertSame(OrderingKind.ARGSORT, indexOrigin.operation().kind()),
                () -> assertEquals(0, firstOrigin.outputIndex()),
                () -> assertEquals(1, firstOrigin.producer().outputCount()),
                () -> assertEquals(1, firstOrigin.inputs().size()),
                () -> assertSame(input, firstOrigin.inputs().getFirst()),
                () -> assertSame(first.descriptor(), firstOrigin.outputDescriptor()));
    }

    @Test
    void validatesInputKindThenAxisWithoutConsumingAnIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor vector = tensor(DataType.BOOL, Shape.of(2), false);
        long before = next.get();
        assertAll(
                        () -> assertThrows(NullPointerException.class,
                                () -> TensorSortExpressions.apply(null, null, 4, false)),
                        () -> assertThrows(NullPointerException.class,
                                () -> TensorSortExpressions.apply(vector, null, 4, false)),
                        () -> assertThrows(IndexOutOfBoundsException.class,
                                        () -> vector.sort(2)),
                        () -> assertThrows(IndexOutOfBoundsException.class,
                                        () -> tensor(DataType.INT64, Shape.scalar(), false).argsort(0)),
                        () -> assertEquals(before, next.get()));
    }

    @Test
    void discardsLayoutAndDoesNotMutateInputMetadataOrStorage() {
        int[] values = {3, 1, 2};
        Shape shape = Shape.of(3);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT32, shape, Optional.of(layout), false);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.INT32, values.length, MemorySegment.ofArray(values));
        Tensor input = new Tensor(new TensorId(IDS.getAndIncrement()), descriptor,
                Optional.of("input"), Optional.empty(), Optional.of(storage));

        Tensor result = input.sort(0, true);
        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertEquals(List.of(3, 1, 2), List.of(values[0], values[1], values[2])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction() throws Exception {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(1), true);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> input.sort(0));
            
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
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

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
}
