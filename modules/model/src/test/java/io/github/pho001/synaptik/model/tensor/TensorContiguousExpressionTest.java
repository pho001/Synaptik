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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorContiguousExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(40_000);

    @Test
    void exposesExactlyThePublicMethodAndOneMethodStatelessHelper() throws Exception {
        Method contiguous = Tensor.class.getDeclaredMethod("contiguous");
        Method apply = TensorContiguousExpressions.class.getDeclaredMethod("apply", Tensor.class);
        var constructor = TensorContiguousExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertSame(Tensor.class, contiguous.getReturnType()),
                () -> assertEquals(0, contiguous.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(contiguous.getModifiers())),
                () -> assertFalse(Modifier.isStatic(contiguous.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(contiguous.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        TensorContiguousExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorContiguousExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorContiguousExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorContiguousExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorContiguousExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(List.of("apply"),
                        Arrays.stream(TensorContiguousExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .toList()),
                () -> assertSame(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())));
    }

    @Test
    void acceptsAllDataTypesAndRetainsExactDescriptorFacts() {
        Shape shape = Shape.of(2, 3);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, shape, Optional.empty(), requiresGrad);

                Tensor result = input.contiguous();
                LayoutDescriptor layout = result.descriptor().layout().orElseThrow();
                TensorProvenance provenance = result.provenance().orElseThrow();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertSame(shape, result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS, layout.kind()),
                        () -> assertArrayEquals(new long[] {3, 1}, layout.strides()),
                        () -> assertEquals(0, layout.storageOffset()),
                        () -> assertFalse(layout.isView()),
                        () -> assertEquals(6, layout.referencedElementSpan()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(ContiguousKind.CONTIGUOUS,
                                provenance.operation().kind()),
                        () -> assertSame(NoOperationAttrs.INSTANCE,
                                provenance.operation().attrs()),
                        () -> assertEquals(1, provenance.inputs().size()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void resolvesScalarAndZeroExtentGeometryAndLeavesDynamicShapeUnresolved() {
        Shape scalar = Shape.scalar();
        Shape empty = Shape.of(2, 0, 4);
        DynamicDimension batch = new DynamicDimension("batch");
        Shape dynamic = Shape.ofDimensions(batch, new StaticDimension(4));

        Tensor scalarResult = tensor(DataType.FLOAT32, scalar, Optional.empty(), true).contiguous();
        Tensor emptyResult = tensor(DataType.BOOL, empty, Optional.empty(), false).contiguous();
        Tensor dynamicResult = tensor(
                DataType.INT64, dynamic, Optional.empty(), false).contiguous();

        LayoutDescriptor scalarLayout = scalarResult.descriptor().layout().orElseThrow();
        LayoutDescriptor emptyLayout = emptyResult.descriptor().layout().orElseThrow();
        assertAll(
                () -> assertSame(scalar, scalarResult.descriptor().shape()),
                () -> assertArrayEquals(new long[0], scalarLayout.strides()),
                () -> assertEquals(1, scalarLayout.referencedElementSpan()),
                () -> assertSame(empty, emptyResult.descriptor().shape()),
                () -> assertArrayEquals(new long[] {0, 4, 1}, emptyLayout.strides()),
                () -> assertEquals(0, emptyLayout.referencedElementSpan()),
                () -> assertSame(dynamic, dynamicResult.descriptor().shape()),
                () -> assertSame(batch,
                        dynamicResult.descriptor().shape().dimensions().getFirst()),
                () -> assertTrue(dynamicResult.descriptor().layout().isEmpty()));
    }

    @Test
    void ignoresEveryInputLayoutKindAndAlwaysCreatesNewStaticGeometry() {
        Shape shape = Shape.of(2, 3);
        List<Optional<LayoutDescriptor>> layouts = List.of(
                Optional.empty(),
                Optional.of(LayoutDescriptor.contiguous(shape)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {3, 1}, 4, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)),
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 1}, 0, true)));

        for (Optional<LayoutDescriptor> inputLayout : layouts) {
            Tensor input = tensor(DataType.FLOAT64, shape, inputLayout, true);
            Tensor result = input.contiguous();
            LayoutDescriptor resultLayout = result.descriptor().layout().orElseThrow();

            assertAll(
                    () -> assertEquals(LayoutKind.DENSE_CONTIGUOUS, resultLayout.kind()),
                    () -> assertArrayEquals(new long[] {3, 1}, resultLayout.strides()),
                    () -> assertEquals(0, resultLayout.storageOffset()),
                    () -> assertFalse(resultLayout.isView()),
                    () -> inputLayout.ifPresent(layout -> assertNotSame(layout, resultLayout)));
        }
    }

    @Test
    void leavesLabelProvenanceLiveOrDeadStorageAndRawValuesUntouched() {
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {2, 1}, 3, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 7, arena.allocate(28, 1));
        MemorySegment.copy(
                MemorySegment.ofArray(values), 0, storage.segment(), 12, 16);
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                List.of(leaf));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(12, 16).toArray(
                java.lang.foreign.ValueLayout.JAVA_FLOAT);

        Tensor result = input.contiguous();
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
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input,
                        result.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void repeatedAlreadyContiguousAndNestedRequestsAreAlwaysFresh() {
        Shape shape = Shape.of(3);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        Tensor input = tensor(DataType.FLOAT32, shape, Optional.of(layout), true);

        Tensor first = input.contiguous();
        Tensor second = input.contiguous();
        Tensor nested = first.contiguous();

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(second, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(second.id(), nested.id()),
                () -> assertNotSame(layout,
                        first.descriptor().layout().orElseThrow()),
                () -> assertNotSame(
                        first.descriptor().layout().orElseThrow(),
                        nested.descriptor().layout().orElseThrow()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void nullInputAndStaticLayoutOverflowConsumeNoTensorIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorContiguousExpressions.apply(null));
        Tensor overflowing = tensor(
                DataType.FLOAT32,
                Shape.of(1, Long.MAX_VALUE, 2),
                Optional.empty(),
                false);
        ArithmeticException overflow = assertThrows(
                ArithmeticException.class, overflowing::contiguous);

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("long overflow", overflow.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction() throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, input::contiguous);

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
