package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import io.github.pho001.synaptik.model.shape.Dimension;
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
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;

import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;




import java.util.Optional;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorOneHotExpressionTest {
    private static final AtomicLong INPUT_IDS = new AtomicLong(290_000);

    

    @Test
    void acceptsExactlyInt32AndInt64AndCreatesExactBoolMetadata() {
        for (DataType dataType : List.of(DataType.INT32, DataType.INT64)) {
            Tensor result = tensor(dataType, Shape.of(2, 3), false).oneHot(4);
            assertAll(
                    () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                    () -> assertEquals(Shape.of(2, 3, 4), result.descriptor().shape()),
                    () -> assertFalse(result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()));
        }
    }

    @Test
    void rejectsEveryOtherDataTypeBeforeDepthAndWithoutIdAllocation() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        for (DataType dataType : DataType.values()) {
            if (dataType == DataType.INT32 || dataType == DataType.INT64) {
                continue;
            }
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> tensor(dataType, Shape.scalar(), false).oneHot(0));
            
        }
        assertEquals(before, next.get());
    }

    @Test
    void validatesNullThenTypeThenDepthAndFailuresConsumeNoId() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> TensorOneHotExpressions.apply(null, 0));
        IllegalArgumentException typeBeforeDepth = assertThrows(
                IllegalArgumentException.class,
                () -> TensorOneHotExpressions.apply(
                        tensor(DataType.BOOL, Shape.scalar(), false), 0));
        for (long depth : new long[] {0L, -1L, Long.MIN_VALUE}) {
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> tensor(DataType.INT64, Shape.scalar(), false).oneHot(depth));
            
        }

        assertAll(
                        () -> assertEquals(before, next.get()));
    }

    @Test
    void preservesEveryInputDimensionReferenceAndAppendsOneFreshStaticDepth() {
        Dimension zero = new StaticDimension(0);
        Dimension named = new DynamicDimension("batch");
        Dimension expression = DimensionExpressions.add(
                new DynamicDimension("width"), new StaticDimension(2));
        Shape inputShape = Shape.ofDimensions(zero, named, expression);
        Tensor result = tensor(DataType.INT32, inputShape, false).oneHot(7);
        List<Dimension> dimensions = result.descriptor().shape().dimensions();

        assertAll(
                () -> assertEquals(4, dimensions.size()),
                () -> assertSame(zero, dimensions.get(0)),
                () -> assertSame(named, dimensions.get(1)),
                () -> assertSame(expression, dimensions.get(2)),
                () -> assertEquals(new StaticDimension(7), dimensions.get(3)),
                () -> assertNotSame(inputShape.dimension(0), dimensions.get(3)));
    }

    @Test
    void supportsScalarZeroElementAndMaximumDepthWithoutElementCountMaterialization() {
        Tensor scalar = tensor(DataType.INT64, Shape.scalar(), false).oneHot(4);
        Tensor empty = tensor(DataType.INT32, Shape.of(0, 5), false).oneHot(3);
        Tensor maximum = tensor(DataType.INT64, Shape.of(2), false).oneHot(Long.MAX_VALUE);

        assertAll(
                () -> assertEquals(Shape.of(4), scalar.descriptor().shape()),
                () -> assertEquals(Shape.of(0, 5, 3), empty.descriptor().shape()),
                () -> assertEquals(
                        Shape.of(2, Long.MAX_VALUE), maximum.descriptor().shape()),
                () -> assertTrue(maximum.hostStorage().isEmpty()));
    }

    @Test
    void createsOneFreshProducerProvenanceAndIdPerCallWithoutMutatingInput() throws Exception {
        Tensor indices = tensor(DataType.INT64, Shape.of(3), false);
        TensorDescriptor originalDescriptor = indices.descriptor();
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor first = indices.oneHot(3);
        Tensor second = indices.oneHot(3);
        TensorProvenance provenance = first.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(
                        provenance.producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(OneHotKind.ONE_HOT, provenance.operation().kind()),
                () -> assertEquals(new OneHotAttrs(3), provenance.operation().attrs()),
                () -> assertEquals(1, provenance.inputs().size()),
                () -> assertSame(indices, provenance.inputs().getFirst()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()),
                () -> assertSame(originalDescriptor, indices.descriptor()),
                () -> assertTrue(indices.provenance().isEmpty()),
                () -> assertTrue(indices.label().isEmpty()),
                () -> assertTrue(indices.hostStorage().isEmpty()));
    }

    @Test
    void doesNotInspectStoredNegativeOrOutOfRangeIndexValues() {
        Shape shape = Shape.of(3);
        Tensor indices = TensorFactory.fromFlatArray(
                new TensorDescriptor(
                        DataType.INT64,
                        shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)),
                        false),
                Optional.of("unchecked indices"),
                new long[] {-1, 3, Long.MAX_VALUE});

        Tensor result = indices.oneHot(3);

        assertAll(
                () -> assertEquals(Shape.of(3, 3), result.descriptor().shape()),
                () -> assertSame(indices, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(indices.hostStorage().isPresent()),
                () -> assertEquals(Optional.of("unchecked indices"), indices.label()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void propagatesIdentifierExhaustionOnlyAfterValidLocalConstruction() throws Exception {
        Tensor indices = tensor(DataType.INT32, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();

        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> indices.oneHot(2));
            assertAll(
                            () -> assertEquals(Long.MAX_VALUE, next.get()),
                            () -> assertTrue(claimed.get()),
                            () -> assertTrue(indices.provenance().isEmpty()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(INPUT_IDS.getAndIncrement()),
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
