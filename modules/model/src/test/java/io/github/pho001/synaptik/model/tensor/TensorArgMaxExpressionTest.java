package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgMaxAttrs;
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

class TensorArgMaxExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(140_000);

    @Test
    void helperAndPublicSurfaceAreExact() throws Exception {
        var constructor = TensorArgMaxExpressions.class.getDeclaredConstructors()[0];
        assertAll(
                () -> assertTrue(Modifier.isFinal(TensorArgMaxExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(TensorArgMaxExpressions.class.getModifiers())),
                () -> assertEquals(0, TensorArgMaxExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorArgMaxExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(4, TensorArgMaxExpressions.class.getDeclaredMethods().length),
                () -> assertEquals(1, TensorArgMaxExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()));
        assertEquals(
                Set.of("apply", "validateNumericInput", "reduceShape", "create"),
                Arrays.stream(TensorArgMaxExpressions.class.getDeclaredMethods())
                        .map(java.lang.reflect.Method::getName).collect(java.util.stream.Collectors.toSet()));

        Method apply = TensorArgMaxExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, int.class, boolean.class, ArgMaxTiePolicy.class);
        Method validate = TensorArgMaxExpressions.class.getDeclaredMethod(
                "validateNumericInput", Tensor.class);
        Method reduce = TensorArgMaxExpressions.class.getDeclaredMethod(
                "reduceShape", Shape.class, int.class, boolean.class);
        Method create = TensorArgMaxExpressions.class.getDeclaredMethod(
                "create", Tensor.class, Shape.class, ArgMaxAttrs.class);
        assertAll(
                () -> assertPackagePrivateStatic(apply, Tensor.class),
                () -> assertPrivateStatic(validate, void.class),
                () -> assertPrivateStatic(reduce, Shape.class),
                () -> assertPrivateStatic(create, Tensor.class));

        for (Class<?>[] parameters : List.of(
                new Class<?>[] {int.class},
                new Class<?>[] {int.class, boolean.class},
                new Class<?>[] {int.class, boolean.class, ArgMaxTiePolicy.class})) {
            Method method = Tensor.class.getDeclaredMethod("argMax", parameters);
            assertAll(
                    () -> assertSame(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void mapsDefaultsAndExplicitPoliciesToExactAttrsAndFixedResult() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3));
        List<Tensor> results = List.of(
                input.argMax(-1),
                input.argMax(0, true),
                input.argMax(1, false, ArgMaxTiePolicy.LAST_INDEX));
        List<ArgMaxAttrs> attrs = List.of(
                new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX),
                new ArgMaxAttrs(0, true, ArgMaxTiePolicy.FIRST_INDEX),
                new ArgMaxAttrs(1, false, ArgMaxTiePolicy.LAST_INDEX));
        for (int i = 0; i < results.size(); i++) {
            Tensor result = results.get(i);
            ArgMaxAttrs actualAttrs = (ArgMaxAttrs)
                    result.provenance().orElseThrow().operation().attrs();
            ArgMaxAttrs expectedAttrs = attrs.get(i);
            assertAll(
                    () -> assertSame(DataType.INT64, result.descriptor().dataType()),
                    () -> assertFalse(result.descriptor().requiresGrad()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertTrue(result.label().isEmpty()),
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(AggregateReductionKind.ARG_MAX,
                            result.provenance().orElseThrow().operation().kind()),
                    () -> assertEquals(expectedAttrs, actualAttrs),
                    () -> assertSame(expectedAttrs.tiePolicy(), actualAttrs.tiePolicy()),
                    () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()));
        }
    }

    @Test
    void acceptsAllNumericTypesRejectsBoolAndPreservesEveryShapeContract() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64)) {
            assertDoesNotThrow(() -> tensor(type, Shape.of(2)).argMax(0));
        }
        IllegalArgumentException bool = assertThrows(
                IllegalArgumentException.class, () -> tensor(DataType.BOOL, Shape.of(2)).argMax(0));
        assertEquals("input must have a numeric data type, but was BOOL", bool.getMessage());

        var batch = new DynamicDimension("batch");
        var width = new StaticDimension(4);
        Tensor input = tensor(DataType.INT32, Shape.ofDimensions(batch, width));
        Tensor removed = input.argMax(1);
        Tensor retained = input.argMax(1, true);
        Tensor zeroRemoved = tensor(DataType.INT64, Shape.of(2, 0, 3)).argMax(1);
        assertAll(
                () -> assertSame(batch, removed.descriptor().shape().dimensions().getFirst()),
                () -> assertSame(batch, retained.descriptor().shape().dimensions().getFirst()),
                () -> assertEquals(new StaticDimension(1), retained.descriptor().shape().dimensions().get(1)),
                () -> assertEquals(Shape.of(2, 3), zeroRemoved.descriptor().shape()),
                () -> assertSame(Shape.scalar(),
                        tensor(DataType.INT64, Shape.of(0)).argMax(0).descriptor().shape()));
    }

    @Test
    void validationOrderAndIdentityEffectsAreExact() throws Exception {
        AtomicLong nextId = nextTensorIdState();
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2));
        Tensor bool = tensor(DataType.BOOL, Shape.of(2));
        long before = nextId.get();
        NullPointerException nullInput = assertThrows(NullPointerException.class,
                () -> TensorArgMaxExpressions.apply(null, 9, false, null));
        NullPointerException nullPolicy = assertThrows(NullPointerException.class,
                () -> TensorArgMaxExpressions.apply(numeric, 9, false, null));
        IndexOutOfBoundsException axis = assertThrows(IndexOutOfBoundsException.class,
                () -> numeric.argMax(2));
        IllegalArgumentException typeBeforeAxis = assertThrows(
                IllegalArgumentException.class,
                () -> bool.argMax(9, false, ArgMaxTiePolicy.FIRST_INDEX));
        IndexOutOfBoundsException scalar = assertThrows(
                IndexOutOfBoundsException.class,
                () -> tensor(DataType.INT32, Shape.scalar()).argMax(0));
        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("tiePolicy", nullPolicy.getMessage()),
                () -> assertEquals("Axis 2 is outside shape rank 1", axis.getMessage()),
                () -> assertEquals(
                        "input must have a numeric data type, but was BOOL",
                        typeBeforeAxis.getMessage()),
                () -> assertEquals("Axis 0 is outside shape rank 0", scalar.getMessage()),
                () -> assertEquals(before, nextId.get()));
    }

    @Test
    void repeatedAndNestedCallsAreFresh() {
        Tensor numeric = tensor(DataType.FLOAT32, Shape.of(2));
        Tensor first = numeric.argMax(0, true);
        Tensor second = numeric.argMax(0, true);
        Tensor nested = first.argMax(0, true, ArgMaxTiePolicy.LAST_INDEX);
        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first, nested),
                () -> assertSame(first, nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void leavesCompleteInputStateAndContentsUnchanged() {
        long[] values = {4, 9, 9, 2};
        Shape shape = Shape.of(2, 2);
        LayoutDescriptor layout = LayoutDescriptor.contiguous(shape);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT64, shape, Optional.of(layout), false);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.INT64, values.length, MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.INT64, shape);
        ArgMaxAttrs originalAttrs =
                new ArgMaxAttrs(1, false, ArgMaxTiePolicy.FIRST_INDEX);
        TensorProvenance originalProvenance = new TensorProvenance(
                new Operation(AggregateReductionKind.ARG_MAX, originalAttrs), List.of(leaf));
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()), descriptor, Optional.of("input"),
                Optional.of(originalProvenance), Optional.of(storage));

        Tensor result = input.argMax(1, true, ArgMaxTiePolicy.LAST_INDEX);

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertSame(layout, input.descriptor().layout().orElseThrow()),
                () -> assertEquals(Optional.of("input"), input.label()),
                () -> assertSame(originalProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(new long[] {4, 9, 9, 2}, values),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertTrue(result.descriptor().layout().isEmpty()));
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
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.argMax(0));
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

    private static Tensor tensor(DataType type, Shape shape) {
        return new Tensor(new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(type, shape, Optional.empty(), false),
                Optional.empty(), Optional.empty(), Optional.empty());
    }
}
