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
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorPrimitiveTakeExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(90_000);

    @Test
    void exposesExactlyOnePublicOverloadAndTwoMethodFieldFreeHelper() throws Exception {
        Method publicTake = Tensor.class.getDeclaredMethod("take", int.class, int[].class);
        Method helperTake = TensorPrimitiveTakeExpressions.class.getDeclaredMethod(
                "take", Tensor.class, int.class, int[].class);
        Method createIndices = TensorPrimitiveTakeExpressions.class.getDeclaredMethod(
                "createIndices", int[].class);
        var constructor = TensorPrimitiveTakeExpressions.class.getDeclaredConstructor();

        assertAll(
                () -> assertSame(Tensor.class, publicTake.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, int[].class),
                        Arrays.asList(publicTake.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(publicTake.getModifiers())),
                () -> assertFalse(Modifier.isStatic(publicTake.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(publicTake.getModifiers())),
                () -> assertFalse(publicTake.isVarArgs()),
                () -> assertTrue(Modifier.isFinal(
                        TensorPrimitiveTakeExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorPrimitiveTakeExpressions.class.getModifiers())),
                () -> assertEquals(
                        0, TensorPrimitiveTakeExpressions.class.getDeclaredFields().length),
                () -> assertEquals(
                        0, TensorPrimitiveTakeExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(
                        1, TensorPrimitiveTakeExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(
                        List.of("createIndices", "take"),
                        Arrays.stream(TensorPrimitiveTakeExpressions.class.getDeclaredMethods())
                                .map(Method::getName)
                                .sorted()
                                .toList()),
                () -> assertPackagePrivateStatic(helperTake, Tensor.class),
                () -> assertPrivateStatic(createIndices, Tensor.class));
    }

    @Test
    void rejectsNullAndEmptyInputsInOrderWithoutAllocationOrIdentityConsumption()
            throws Exception {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullData = assertThrows(
                NullPointerException.class,
                () -> TensorPrimitiveTakeExpressions.take(null, 9, null));
        NullPointerException nullIndices = assertThrows(
                NullPointerException.class,
                () -> TensorPrimitiveTakeExpressions.take(data, 9, null));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class, () -> data.take(9, new int[0]));

        assertAll(
                () -> assertEquals("data", nullData.getMessage()),
                () -> assertEquals("indices", nullIndices.getMessage()),
                () -> assertEquals("take indices must not be empty", empty.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void createsExactDenseIndexTensorAndGatherAxisResultWithOrderedProvenance() {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);
        int[] requested = {2, 0};

        Tensor result = data.take(1, requested);
        TensorProvenance provenance = result.provenance().orElseThrow();
        Tensor generated = provenance.inputs().get(1);
        TensorDescriptor generatedDescriptor = generated.descriptor();
        LayoutDescriptor generatedLayout = generatedDescriptor.layout().orElseThrow();
        Operation operation = provenance.operation();

        requested[0] = 17;
        requested[1] = 19;

        assertAll(
                () -> assertSame(AxisGatherKind.GATHER_AXIS, operation.kind()),
                () -> assertEquals(new IndexAxisAttrs(1), operation.attrs()),
                () -> assertEquals(List.of(data, generated), provenance.inputs()),
                () -> assertSame(data, provenance.inputs().get(0)),
                () -> assertSame(DataType.INT32, generatedDescriptor.dataType()),
                () -> assertEquals(Shape.of(2), generatedDescriptor.shape()),
                () -> assertEquals(
                        LayoutDescriptor.contiguous(Shape.of(2)), generatedLayout),
                () -> assertSame(LayoutKind.DENSE_CONTIGUOUS, generatedLayout.kind()),
                () -> assertFalse(generatedLayout.isView()),
                () -> assertFalse(generatedDescriptor.requiresGrad()),
                () -> assertTrue(generated.label().isEmpty()),
                () -> assertTrue(generated.provenance().isEmpty()),
                () -> assertTrue(generated.hostStorage().isPresent()),
                () -> assertSame(
                        DataType.INT32, generated.hostStorage().orElseThrow().dataType()),
                () -> assertEquals(2, generated.hostStorage().orElseThrow().elementCapacity()),
                () -> assertArrayEquals(new int[] {2, 0}, heapArray(generated)),
                () -> assertSame(DataType.FLOAT32, result.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 2, 4), result.descriptor().shape()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void preservesNegativeAndExtremeIndexValuesWithoutBoundsInspection() {
        Tensor data = tensor(DataType.INT64, Shape.of(2, 3, 4), false);
        int[] requested = {Integer.MIN_VALUE, -1, Integer.MAX_VALUE};

        Tensor result = data.take(-2, requested);
        Tensor generated = result.provenance().orElseThrow().inputs().get(1);

        assertAll(
                () -> assertArrayEquals(requested, heapArray(generated)),
                () -> assertEquals(Shape.of(3), generated.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 3, 4), result.descriptor().shape()),
                () -> assertEquals(
                        new IndexAxisAttrs(1),
                        result.provenance().orElseThrow().operation().attrs()));
    }

    @Test
    void retainsEveryDataTypeAndValidGradientEligibilityFromDataOnly() {
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : dataType.isDifferentiable()
                    ? List.of(false, true)
                    : List.of(false)) {
                Tensor data = tensor(dataType, Shape.of(2, 3), requiresGrad);
                Tensor result = data.take(1, new int[] {1, 0, -1});
                Tensor generated = result.provenance().orElseThrow().inputs().get(1);

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertEquals(Shape.of(2, 3), result.descriptor().shape()),
                        () -> assertSame(DataType.INT32, generated.descriptor().dataType()),
                        () -> assertFalse(generated.descriptor().requiresGrad()));
            }
        }
    }

    @Test
    void repeatedRequestsCreateIndependentIndicesStorageAndFreshIdentities() {
        Tensor data = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        int[] requested = {2, 1, 0};

        Tensor first = data.take(1, requested);
        Tensor second = data.take(1, requested);
        Tensor firstIndices = first.provenance().orElseThrow().inputs().get(1);
        Tensor secondIndices = second.provenance().orElseThrow().inputs().get(1);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(firstIndices, secondIndices),
                () -> assertNotEquals(firstIndices.id(), secondIndices.id()),
                () -> assertNotSame(heapArray(firstIndices), heapArray(secondIndices)),
                () -> assertNotSame(requested, heapArray(firstIndices)),
                () -> assertNotSame(requested, heapArray(secondIndices)),
                () -> assertArrayEquals(requested, heapArray(firstIndices)),
                () -> assertArrayEquals(requested, heapArray(secondIndices)));
    }

    @Test
    void invalidAxisConsumesOnlyGeneratedIndexIdentity() throws Exception {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        IndexOutOfBoundsException failure = assertThrows(
                IndexOutOfBoundsException.class, () -> data.take(2, new int[] {0}));

        assertAll(
                () -> assertEquals("Axis 2 is outside shape rank 2", failure.getMessage()),
                () -> assertEquals(before + 1, next.get()));
    }

    @Test
    void finalResultExhaustionOccursAfterGeneratedIndexClaimsLastIdentity() throws Exception {
        Tensor data = tensor(DataType.FLOAT32, Shape.of(2), true);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(false);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> data.take(0, new int[] {0}));

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

    private static int[] heapArray(Tensor tensor) {
        return (int[]) tensor.hostStorage().orElseThrow().segment().heapBase().orElseThrow();
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

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }
}
