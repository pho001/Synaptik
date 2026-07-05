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
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class TensorCastExpressionTest {
    private static final AtomicLong NEXT_INPUT_ID = new AtomicLong(70_000);

    @Test
    void exposesExactPublicEntryAndPackagePrivateHelperShape()
            throws ReflectiveOperationException {
        var constructors = TensorCastExpressions.class.getDeclaredConstructors();
        var methods = Arrays.stream(TensorCastExpressions.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();
        var cast = Tensor.class.getDeclaredMethod("cast", DataType.class);

        assertAll(
                () -> assertEquals(Tensor.class, cast.getReturnType()),
                () -> assertEquals(
                        List.of(DataType.class), Arrays.asList(cast.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(cast.getModifiers())),
                () -> assertFalse(Modifier.isStatic(cast.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(cast.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorCastExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorCastExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isProtected(
                        TensorCastExpressions.class.getModifiers())),
                () -> assertFalse(TensorCastExpressions.class.isRecord()),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(0, TensorCastExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorCastExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, methods.size()),
                () -> assertEquals("apply", methods.getFirst().getName()),
                () -> assertEquals(Tensor.class, methods.getFirst().getReturnType()),
                () -> assertEquals(
                        List.of(Tensor.class, DataType.class),
                        Arrays.asList(methods.getFirst().getParameterTypes())),
                () -> assertTrue(Modifier.isStatic(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isPublic(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isProtected(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isPrivate(methods.getFirst().getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(methods.getFirst().getModifiers())));
    }

    @Test
    void acceptsEverySourceTargetPairWithExactDescriptorOperationAndProvenance() {
        Shape shape = Shape.of(2, 3);

        for (DataType sourceDataType : DataType.values()) {
            for (DataType targetDataType : DataType.values()) {
                Tensor input = tensor(sourceDataType, shape, sourceDataType.isFloating());

                Tensor result = input.cast(targetDataType);
                TensorProvenance provenance = result.provenance().orElseThrow();
                Operation operation = provenance.operation();
                CastAttrs attrs = (CastAttrs) operation.attrs();

                assertAll(
                        () -> assertSame(targetDataType, result.descriptor().dataType()),
                        () -> assertSame(shape, result.descriptor().shape()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertEquals(
                                sourceDataType.isFloating() && targetDataType.isFloating(),
                                result.descriptor().requiresGrad()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertSame(CastKind.CAST, operation.kind()),
                        () -> assertEquals(CastAttrs.class, operation.attrs().getClass()),
                        () -> assertSame(targetDataType, attrs.targetDataType()),
                        () -> assertEquals(1, provenance.inputs().size()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void retainsExactShapeReferenceForScalarZeroStaticAndDynamicInputs() {
        List<Shape> shapes = List.of(
                Shape.scalar(),
                Shape.of(0, 3),
                Shape.of(2, 3),
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(4)));

        for (Shape shape : shapes) {
            Tensor input = tensor(DataType.FLOAT32, shape, true);
            Tensor result = input.cast(DataType.INT64);

            assertAll(
                    () -> assertSame(shape, result.descriptor().shape()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertFalse(result.descriptor().requiresGrad()));
        }
    }

    @Test
    void appliesExactGradientEligibilityMatrix() {
        for (DataType sourceDataType : DataType.values()) {
            for (DataType targetDataType : DataType.values()) {
                for (boolean requested : List.of(false, true)) {
                    if (requested && !sourceDataType.isDifferentiable()) {
                        continue;
                    }
                    Tensor input = tensor(sourceDataType, Shape.scalar(), requested);
                    Tensor result = input.cast(targetDataType);
                    boolean expected = requested
                            && sourceDataType.isFloating()
                            && targetDataType.isFloating();
                    assertEquals(expected, result.descriptor().requiresGrad());
                }
            }
        }
    }

    @Test
    void keepsSameTypeRepeatedAndChainedCastsFreshAndExplicit() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2), true);

        Tensor first = input.cast(DataType.FLOAT32);
        Tensor second = input.cast(DataType.FLOAT32);
        Tensor chained = first.cast(DataType.BFLOAT16);

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(first, chained),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(first.id(), chained.id()),
                () -> assertSame(input, first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(input, second.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first, chained.provenance().orElseThrow().inputs().getFirst()),
                () -> assertNotSame(
                        first.provenance().orElseThrow(),
                        second.provenance().orElseThrow()),
                () -> assertNotSame(
                        first.provenance().orElseThrow().operation().attrs(),
                        second.provenance().orElseThrow().operation().attrs()));
    }

    @Test
    void leavesResolvedInputMetadataStorageAndContentsUnchanged() {
        Shape shape = Shape.of(3);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        float[] values = {-1.0f, 0.0f, 2.0f};
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(layout), true);
        Tensor input = new Tensor(
                new TensorId(NEXT_INPUT_ID.getAndIncrement()),
                descriptor,
                Optional.of("source"),
                Optional.empty(),
                Optional.of(storage));

        Tensor result = input.cast(DataType.FLOAT64);

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("source"), input.label()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new float[] {-1.0f, 0.0f, 2.0f}, values),
                () -> assertSame(shape, result.descriptor().shape()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void acceptsDerivedInputWithoutTraversingOrChangingItsProvenance() {
        Tensor leaf = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor derived = leaf.abs();
        TensorProvenance originalProvenance = derived.provenance().orElseThrow();

        Tensor result = derived.cast(DataType.FLOAT64);

        assertAll(
                () -> assertSame(originalProvenance, derived.provenance().orElseThrow()),
                () -> assertSame(
                        UnaryElementwiseKind.ABS,
                        derived.provenance().orElseThrow().operation().kind()),
                () -> assertSame(derived, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(CastKind.CAST, result.provenance().orElseThrow().operation().kind()));
    }

    @Test
    void validatesInputThenTargetWithoutAllocatingIdentity()
            throws ReflectiveOperationException {
        Tensor input = tensor(DataType.FLOAT32, Shape.scalar(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException publicTarget =
                assertThrows(NullPointerException.class, () -> input.cast(null));
        NullPointerException helperInput = assertThrows(
                NullPointerException.class, () -> TensorCastExpressions.apply(null, null));
        NullPointerException helperTarget = assertThrows(
                NullPointerException.class, () -> TensorCastExpressions.apply(input, null));

        assertAll(
                () -> assertEquals("targetDataType", publicTarget.getMessage()),
                () -> assertEquals("input", helperInput.getMessage()),
                () -> assertEquals("targetDataType", helperTarget.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void propagatesIdentifierExhaustionAfterValidLocalConstruction()
            throws ReflectiveOperationException {
        Tensor input = tensor(DataType.INT32, Shape.of(2), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();

        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class, () -> input.cast(DataType.BOOL));

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
