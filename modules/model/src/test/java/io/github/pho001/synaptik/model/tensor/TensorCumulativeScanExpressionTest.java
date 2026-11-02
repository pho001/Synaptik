package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorCumulativeScanExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(160_000);

    

    @Test
    void mapsBothDefaultsAndAllFourExplicitModesToExactKindsAndNormalizedAttributes() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        List<Tensor> results = List.of(
                input.cumSum(-1),
                input.cumSum(1, false, false),
                input.cumSum(1, true, false),
                input.cumSum(1, false, true),
                input.cumSum(1, true, true),
                input.cumProd(-1),
                input.cumProd(1, false, false),
                input.cumProd(1, true, false),
                input.cumProd(1, false, true),
                input.cumProd(1, true, true));
        List<CumulativeScanKind> expectedKinds = List.of(
                CumulativeScanKind.CUM_SUM,
                CumulativeScanKind.CUM_SUM,
                CumulativeScanKind.CUM_SUM,
                CumulativeScanKind.CUM_SUM,
                CumulativeScanKind.CUM_SUM,
                CumulativeScanKind.CUM_PROD,
                CumulativeScanKind.CUM_PROD,
                CumulativeScanKind.CUM_PROD,
                CumulativeScanKind.CUM_PROD,
                CumulativeScanKind.CUM_PROD);
        List<CumulativeScanAttrs> expected = List.of(
                new CumulativeScanAttrs(1, false, false),
                new CumulativeScanAttrs(1, false, false),
                new CumulativeScanAttrs(1, true, false),
                new CumulativeScanAttrs(1, false, true),
                new CumulativeScanAttrs(1, true, true),
                new CumulativeScanAttrs(1, false, false),
                new CumulativeScanAttrs(1, false, false),
                new CumulativeScanAttrs(1, true, false),
                new CumulativeScanAttrs(1, false, true),
                new CumulativeScanAttrs(1, true, true));

        for (int index = 0; index < results.size(); index++) {
            Tensor result = results.get(index);
            CumulativeScanKind expectedKind = expectedKinds.get(index);
            CumulativeScanAttrs expectedAttrs = expected.get(index);
            TensorProvenance provenance = result.provenance().orElseThrow();
            CumulativeScanAttrs attrs = (CumulativeScanAttrs) provenance.operation().attrs();
            assertAll(
                    () -> assertSame(expectedKind, provenance.operation().kind()),
                    () -> assertEquals(expectedAttrs, attrs),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertEquals(1, provenance.inputs().size()),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertEquals(1,
                            provenance.producer().outputDescriptors().size()),
                    () -> assertSame(result.descriptor(), provenance.outputDescriptor()),
                    () -> assertSame(
                            result.descriptor(),
                            provenance.producer().outputDescriptors().getFirst()),
                    () -> assertSame(input.descriptor().shape(), result.descriptor().shape()),
                    () -> assertSame(DataType.FLOAT32, result.descriptor().dataType()),
                    () -> assertTrue(result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()));
        }
    }

    @Test
    void acceptsAllNumericTypesAndPreservesStaticDynamicAndZeroExtentShapes() {
        for (DataType dataType : List.of(
                DataType.FLOAT64,
                DataType.FLOAT32,
                DataType.BFLOAT16,
                DataType.INT32,
                DataType.INT64)) {
            Tensor input = tensor(dataType, Shape.of(2), dataType.isFloating());
            for (Tensor result : List.of(input.cumSum(0), input.cumProd(0))) {
                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(input.descriptor().shape(), result.descriptor().shape()),
                        () -> assertEquals(input.descriptor().requiresGrad(),
                                result.descriptor().requiresGrad()));
            }
        }

        var batch = new DynamicDimension("batch");
        Shape dynamic = Shape.ofDimensions(batch, new DynamicDimension("width"));
        Shape zeroExtent = Shape.of(2, 0, 3);
        Tensor dynamicInput = tensor(DataType.FLOAT64, dynamic, true);
        Tensor zeroInput = tensor(DataType.INT64, zeroExtent, false);
        assertAll(
                () -> assertSame(dynamic, dynamicInput.cumProd(-1).descriptor().shape()),
                () -> assertSame(batch,
                        dynamicInput.cumSum(0).descriptor().shape().dimensions().getFirst()),
                () -> assertSame(zeroExtent, zeroInput.cumProd(1).descriptor().shape()));
    }

    @Test
    void validatesNullTypeAndAxisInDeterministicOrderWithoutConsumingIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2), false);
        long before = next.get();

        assertThrows(NullPointerException.class,
        () -> TensorCumulativeScanExpressions.apply(
                null, null, 9, true, true));
        assertThrows(NullPointerException.class,
        () -> TensorCumulativeScanExpressions.apply(
                numeric, null, 9, true, true));
        assertThrows(IllegalArgumentException.class, () -> bool.cumProd(9));
        assertThrows(IndexOutOfBoundsException.class, () -> numeric.cumProd(1));
        assertThrows(IndexOutOfBoundsException.class, () -> numeric.cumProd(-2));
        IndexOutOfBoundsException scalar = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.INT32, Shape.scalar(), false).cumProd(0));

        assertAll(
                () -> assertEquals(before, next.get()));
    }

    @Test
    void discardsResolvedInputLayoutAndLeavesCompleteInputStateUnchanged() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, false);
        CumulativeScanAttrs originalAttrs = new CumulativeScanAttrs(0, true, false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(CumulativeScanKind.CUM_SUM, originalAttrs),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));

        Tensor result = input.cumProd(1, true, true);

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertEquals(List.of(1.0f, 2.0f, 3.0f, 4.0f),
                        List.of(values[0], values[1], values[2], values[3])),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input,
                        result.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void repeatedCallsAreFreshAndDoNotReturnInput() {
        Tensor input = tensor(DataType.INT64, Shape.of(3), false);
        Tensor first = input.cumProd(0);
        Tensor second = input.cumProd(0, false, false);
        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction() throws Exception {
        Tensor input = tensor(DataType.INT32, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);
            assertThrows(IllegalStateException.class, () -> input.cumProd(0));
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

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
