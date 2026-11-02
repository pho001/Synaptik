package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
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

class TensorArgMaxExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(140_000);

    

    @Test
    void mapsEveryFamilyDefaultAndExplicitPolicyToExactMetadata() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3));
        List<Expected> expected = List.of(
                new Expected(input.argMin(-1), AggregateReductionKind.ARG_MIN,
                        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                new Expected(input.argMin(0, true), AggregateReductionKind.ARG_MIN,
                        new ArgExtremaAttrs(0, true, ArgExtremaTiePolicy.FIRST_INDEX)),
                new Expected(input.argMin(1, false, ArgExtremaTiePolicy.LAST_INDEX),
                        AggregateReductionKind.ARG_MIN,
                        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.LAST_INDEX)),
                new Expected(input.argMax(-1), AggregateReductionKind.ARG_MAX,
                        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                new Expected(input.argMax(0, true), AggregateReductionKind.ARG_MAX,
                        new ArgExtremaAttrs(0, true, ArgExtremaTiePolicy.FIRST_INDEX)),
                new Expected(input.argMax(1, false, ArgExtremaTiePolicy.LAST_INDEX),
                        AggregateReductionKind.ARG_MAX,
                        new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.LAST_INDEX)));

        for (Expected item : expected) {
            Tensor result = item.result();
            ArgExtremaAttrs attrs = (ArgExtremaAttrs)
                    result.provenance().orElseThrow().operation().attrs();
            assertAll(
                    () -> assertSame(DataType.INT64, result.descriptor().dataType()),
                    () -> assertFalse(result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(item.kind(),
                            result.provenance().orElseThrow().operation().kind()),
                    () -> assertEquals(item.attrs(), attrs),
                    () -> assertSame(item.attrs().tiePolicy(), attrs.tiePolicy()),
                    () -> assertEquals(0, result.provenance().orElseThrow().outputIndex()),
                    () -> assertEquals(1, result.provenance().orElseThrow().inputs().size()),
                    () -> assertSame(input,
                            result.provenance().orElseThrow().inputs().getFirst()));
        }
    }

    @Test
    void acceptsAllNumericTypesAndPreservesShapeReferences() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            assertDoesNotThrow(() -> tensor(type, Shape.of(2)).argMin(0));
            assertDoesNotThrow(() -> tensor(type, Shape.of(2)).argMax(0));
        }

        var batch = new DynamicDimension("batch");
        var width = new StaticDimension(4);
        Tensor input = tensor(DataType.INT32, Shape.ofDimensions(batch, width));
        Tensor removed = input.argMin(1);
        Tensor retained = input.argMax(1, true);
        Tensor unselectedZero = tensor(DataType.INT64, Shape.of(0, 2, 3)).argMin(1);
        Tensor dynamicSelected = tensor(DataType.INT64,
                Shape.ofDimensions(batch, width)).argMax(0);
        assertAll(
                () -> assertSame(batch, removed.descriptor().shape().dimensions().getFirst()),
                () -> assertSame(batch, retained.descriptor().shape().dimensions().getFirst()),
                () -> assertEquals(new StaticDimension(1),
                        retained.descriptor().shape().dimensions().get(1)),
                () -> assertNotSame(width, retained.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(Shape.of(0, 3), unselectedZero.descriptor().shape()),
                () -> assertEquals(Shape.of(4), dynamicSelected.descriptor().shape()),
                () -> assertSame(Shape.scalar(),
                        tensor(DataType.INT64, Shape.of(2)).argMin(0).descriptor().shape()));
    }

    @Test
    void validationOrderMessagesAndIdentityEffectsAreExact() throws Exception {
        AtomicLong nextId = nextTensorIdState();
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2));
        Tensor bool = tensor(DataType.BOOL, Shape.of(0));
        long before = nextId.get();

        assertThrows(NullPointerException.class,
                () -> TensorArgExtremaExpressions.apply(null, null, 9, false, null));
        assertThrows(NullPointerException.class,
                () -> TensorArgExtremaExpressions.apply(numeric, null, 9, false, null));
        assertThrows(NullPointerException.class,
                () -> TensorArgExtremaExpressions.apply(
                        numeric, AggregateReductionKind.SUM, 9, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> TensorArgExtremaExpressions.apply(
                        numeric, AggregateReductionKind.SUM, 9, false,
                        ArgExtremaTiePolicy.FIRST_INDEX));
        assertThrows(IllegalArgumentException.class,
                () -> TensorArgExtremaExpressions.apply(
                        bool, AggregateReductionKind.ARG_MIN, 9, false,
                        ArgExtremaTiePolicy.FIRST_INDEX));
        assertThrows(IndexOutOfBoundsException.class,
                () -> numeric.argMax(2));
        assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.INT32, Shape.of(2, 0, 3)).argMin(-2));
        assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.of(0)).argMax(0, true));

        assertAll(
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void repeatedNestedCallsAreFreshAndLeaveInputStateUnchanged() {
        long[] values = {4, 9, 9, 2};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64, shape, Optional.of(layout), false);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.INT64, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.INT64, shape);
        ArgExtremaAttrs originalAttrs =
                new ArgExtremaAttrs(1, false, ArgExtremaTiePolicy.FIRST_INDEX);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(AggregateReductionKind.ARG_MAX, originalAttrs),
                        List.of(leaf), List.of(descriptor)), 0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()), descriptor, Optional.of("input"),
                Optional.of(originalProvenance), Optional.of(storage));

        Tensor first = input.argMin(1, true);
        Tensor second = input.argMin(1, true);
        Tensor nested = first.argMax(1, true, ArgExtremaTiePolicy.LAST_INDEX);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first, nested),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new long[] {4, 9, 9, 2}, values));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction() throws Exception {
        Tensor input = tensor(DataType.INT32, Shape.of(2));
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            assertThrows(IllegalStateException.class, () -> input.argMin(0));
            assertAll(
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
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

    private static Tensor tensor(DataType type, Shape shape) {
        return new Tensor(new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), false),
                Optional.empty(), Optional.empty(), Optional.empty());
    }

    private record Expected(
            Tensor result, AggregateReductionKind kind, ArgExtremaAttrs attrs) {
    }
}
