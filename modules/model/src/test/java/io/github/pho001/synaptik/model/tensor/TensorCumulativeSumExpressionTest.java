package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
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
import io.github.pho001.synaptik.model.operation.scan.CumulativeSumAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeSumKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
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
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorCumulativeSumExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(160_000);

    @Test
    void helperAndPublicSurfaceAreExact() throws Exception {
        var constructors = TensorCumulativeSumExpressions.class.getDeclaredConstructors();
        var methods = TensorCumulativeSumExpressions.class.getDeclaredMethods();
        assertAll(
                () -> assertTrue(Modifier.isFinal(
                        TensorCumulativeSumExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorCumulativeSumExpressions.class.getModifiers())),
                () -> assertFalse(TensorCumulativeSumExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(
                        TensorCumulativeSumExpressions.class.getInterfaces())),
                () -> assertEquals(0,
                        TensorCumulativeSumExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorCumulativeSumExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(3, methods.length),
                () -> assertEquals(
                        Set.of("apply", "validateNumericInput", "create"),
                        Arrays.stream(methods).map(Method::getName).collect(Collectors.toSet())));

        Method apply = TensorCumulativeSumExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, int.class, boolean.class, boolean.class);
        Method validate = TensorCumulativeSumExpressions.class.getDeclaredMethod(
                "validateNumericInput", Tensor.class);
        Method create = TensorCumulativeSumExpressions.class.getDeclaredMethod(
                "create", Tensor.class, Shape.class, CumulativeSumAttrs.class);
        assertAll(
                () -> assertPackagePrivateStatic(apply, Tensor.class),
                () -> assertPrivateStatic(validate, void.class),
                () -> assertPrivateStatic(create, Tensor.class));

        for (Class<?>[] parameters : List.of(
                new Class<?>[] {int.class},
                new Class<?>[] {int.class, boolean.class, boolean.class})) {
            Method method = Tensor.class.getDeclaredMethod("cumSum", parameters);
            assertAll(
                    () -> assertSame(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(parameters),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void mapsDefaultAndAllFourExplicitModesToExactNormalizedAttributes() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        List<Tensor> results = List.of(
                input.cumSum(-1),
                input.cumSum(1, false, false),
                input.cumSum(1, true, false),
                input.cumSum(1, false, true),
                input.cumSum(1, true, true));
        List<CumulativeSumAttrs> expected = List.of(
                new CumulativeSumAttrs(1, false, false),
                new CumulativeSumAttrs(1, false, false),
                new CumulativeSumAttrs(1, true, false),
                new CumulativeSumAttrs(1, false, true),
                new CumulativeSumAttrs(1, true, true));

        for (int index = 0; index < results.size(); index++) {
            Tensor result = results.get(index);
            CumulativeSumAttrs expectedAttrs = expected.get(index);
            TensorProvenance provenance = result.provenance().orElseThrow();
            CumulativeSumAttrs attrs = (CumulativeSumAttrs) provenance.operation().attrs();
            assertAll(
                    () -> assertSame(CumulativeSumKind.CUM_SUM, provenance.operation().kind()),
                    () -> assertEquals(expectedAttrs, attrs),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertEquals(1, provenance.inputs().size()),
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
            Tensor result = input.cumSum(0);
            assertAll(
                    () -> assertSame(dataType, result.descriptor().dataType()),
                    () -> assertSame(input.descriptor().shape(), result.descriptor().shape()),
                    () -> assertEquals(input.descriptor().requiresGrad(),
                            result.descriptor().requiresGrad()));
        }

        var batch = new DynamicDimension("batch");
        Shape dynamic = Shape.ofDimensions(batch, new DynamicDimension("width"));
        Shape zeroExtent = Shape.of(2, 0, 3);
        Tensor dynamicInput = tensor(DataType.FLOAT64, dynamic, true);
        Tensor zeroInput = tensor(DataType.INT64, zeroExtent, false);
        assertAll(
                () -> assertSame(dynamic, dynamicInput.cumSum(-1).descriptor().shape()),
                () -> assertSame(batch,
                        dynamicInput.cumSum(0).descriptor().shape().dimensions().getFirst()),
                () -> assertSame(zeroExtent, zeroInput.cumSum(1).descriptor().shape()));
    }

    @Test
    void validatesNullTypeAndAxisInDeterministicOrderWithoutConsumingIdentity() throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2), false);
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorCumulativeSumExpressions.apply(null, 9, true, true));
        IllegalArgumentException boolBeforeAxis = assertThrows(
                IllegalArgumentException.class, () -> bool.cumSum(9));
        IndexOutOfBoundsException positive = assertThrows(
                IndexOutOfBoundsException.class, () -> numeric.cumSum(1));
        IndexOutOfBoundsException negative = assertThrows(
                IndexOutOfBoundsException.class, () -> numeric.cumSum(-2));
        IndexOutOfBoundsException scalar = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.INT32, Shape.scalar(), false).cumSum(0));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals(
                        "input must have a numeric data type, but was BOOL",
                        boolBeforeAxis.getMessage()),
                () -> assertEquals("Axis 1 is outside shape rank 1", positive.getMessage()),
                () -> assertEquals("Axis -2 is outside shape rank 1", negative.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalar.getMessage()),
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
        CumulativeSumAttrs originalAttrs = new CumulativeSumAttrs(0, true, false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new Operation(CumulativeSumKind.CUM_SUM, originalAttrs), List.of(leaf));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));

        Tensor result = input.cumSum(1, true, true);

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
        Tensor first = input.cumSum(0);
        Tensor second = input.cumSum(0, false, false);
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
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.cumSum(0));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static void assertPackagePrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())));
    }

    private static void assertPrivateStatic(Method method, Class<?> returnType) {
        assertAll(
                () -> assertSame(returnType, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(method.getModifiers())));
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
