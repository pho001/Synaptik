package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
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
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
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

class TensorPermutationExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(70_000);

    @Test
    void exposesExactlyTwoPublicMethodsAndSixMethodStatelessHelper() throws Exception {
        Method permute = Tensor.class.getDeclaredMethod("permute", int[].class);
        Method transpose = Tensor.class.getDeclaredMethod("transpose");
        var constructor = TensorPermutationExpressions.class.getDeclaredConstructor();
        List<Method> methods = Arrays.asList(
                TensorPermutationExpressions.class.getDeclaredMethods());
        Method apply = TensorPermutationExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, int[].class);
        Method transposeHelper = TensorPermutationExpressions.class.getDeclaredMethod(
                "transpose", Tensor.class);
        Method normalize = TensorPermutationExpressions.class.getDeclaredMethod(
                "normalizePermutation", int.class, int[].class);
        Method permuteShape = TensorPermutationExpressions.class.getDeclaredMethod(
                "permuteShape", Shape.class, int[].class);
        Method resolve = TensorPermutationExpressions.class.getDeclaredMethod(
                "resolveViewLayout", TensorDescriptor.class, Shape.class, int[].class);
        Method create = TensorPermutationExpressions.class.getDeclaredMethod(
                "create", Tensor.class, TensorDescriptor.class, Shape.class, int[].class,
                Optional.class);

        assertAll(
                () -> assertSame(Tensor.class, permute.getReturnType()),
                () -> assertTrue(permute.isVarArgs()),
                () -> assertEquals(List.of(int[].class),
                        Arrays.asList(permute.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(permute.getModifiers())),
                () -> assertFalse(Modifier.isStatic(permute.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(permute.getModifiers())),
                () -> assertSame(Tensor.class, transpose.getReturnType()),
                () -> assertEquals(0, transpose.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(transpose.getModifiers())),
                () -> assertFalse(Modifier.isStatic(transpose.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(transpose.getModifiers())),
                () -> assertTrue(Modifier.isFinal(
                        TensorPermutationExpressions.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        TensorPermutationExpressions.class.getModifiers())),
                () -> assertEquals(0,
                        TensorPermutationExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorPermutationExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1,
                        TensorPermutationExpressions.class.getDeclaredConstructors().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertEquals(0, constructor.getParameterCount()),
                () -> assertEquals(6, methods.size()),
                () -> assertEquals(
                        Set.of("apply", "transpose", "normalizePermutation", "permuteShape",
                                "resolveViewLayout", "create"),
                        methods.stream().map(Method::getName).collect(Collectors.toSet())),
                () -> assertEquals(2, methods.stream()
                        .filter(method -> !Modifier.isPrivate(method.getModifiers())).count()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertSame(Tensor.class, apply.getReturnType()),
                () -> assertSame(Tensor.class, transposeHelper.getReturnType()),
                () -> assertSame(int[].class, normalize.getReturnType()),
                () -> assertSame(Shape.class, permuteShape.getReturnType()),
                () -> assertSame(Optional.class, resolve.getReturnType()),
                () -> assertSame(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(normalize.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(permuteShape.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(resolve.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())));
    }

    @Test
    void preservesEveryDataTypeEligibilityAndExactPermutationSemantics() {
        Shape shape = Shape.of(2, 3, 4);
        for (DataType dataType : DataType.values()) {
            for (boolean requiresGrad : validGradientChoices(dataType)) {
                Tensor input = tensor(dataType, shape, Optional.empty(), requiresGrad);

                Tensor result = input.permute(1, 0, 2);
                TensorProvenance provenance = result.provenance().orElseThrow();
                PermutationAttrs attrs = (PermutationAttrs) provenance.operation().attrs();

                assertAll(
                        () -> assertSame(dataType, result.descriptor().dataType()),
                        () -> assertEquals(Shape.of(3, 2, 4), result.descriptor().shape()),
                        () -> assertEquals(requiresGrad, result.descriptor().requiresGrad()),
                        () -> assertTrue(result.descriptor().layout().isEmpty()),
                        () -> assertSame(AxisTransformKind.PERMUTE,
                                provenance.operation().kind()),
                        () -> assertEquals(List.of(1, 0, 2), attrs.axes()),
                        () -> assertEquals(List.of(input), provenance.inputs()),
                        () -> assertSame(input, provenance.inputs().getFirst()),
                        () -> assertTrue(result.label().isEmpty()),
                        () -> assertTrue(result.hostStorage().isEmpty()),
                        () -> assertNotSame(input, result),
                        () -> assertNotEquals(input.id(), result.id()));
            }
        }
    }

    @Test
    void copiesRawAxesAcceptsScalarAndPreservesExactDimensionReferences() {
        DynamicDimension batch = new DynamicDimension("batch");
        StaticDimension empty = new StaticDimension(0);
        StaticDimension width = new StaticDimension(4);
        Shape mixed = Shape.ofDimensions(batch, empty, width);
        Tensor input = tensor(DataType.FLOAT32, mixed, Optional.empty(), true);
        int[] requestedAxes = {-1, 0, -2};

        Tensor result = input.permute(requestedAxes);
        requestedAxes[0] = 0;
        Shape resultShape = result.descriptor().shape();
        PermutationAttrs attrs = (PermutationAttrs) result.provenance().orElseThrow()
                .operation().attrs();
        Tensor scalar = tensor(
                DataType.INT64, Shape.scalar(), Optional.empty(), false).permute();

        assertAll(
                () -> assertEquals(List.of(2, 0, 1), attrs.axes()),
                () -> assertSame(width, resultShape.dimensions().get(0)),
                () -> assertSame(batch, resultShape.dimensions().get(1)),
                () -> assertSame(empty, resultShape.dimensions().get(2)),
                () -> assertArrayEquals(new long[0],
                        scalar.descriptor().shape().toLongArray()),
                () -> assertEquals(List.of(),
                        ((PermutationAttrs) scalar.provenance().orElseThrow()
                                .operation().attrs()).axes()));
    }

    @Test
    void rejectsInvalidRequestsWithExactPrecedenceMessagesAndNoIdentityConsumption()
            throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), Optional.empty(), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorPermutationExpressions.apply(null, null));
        NullPointerException nullAxes = assertThrows(
                NullPointerException.class, () -> input.permute((int[]) null));
        IllegalArgumentException count = assertThrows(
                IllegalArgumentException.class, () -> input.permute(0, Integer.MIN_VALUE));
        IllegalArgumentException low = assertThrows(
                IllegalArgumentException.class, () -> input.permute(-4, 1, 2));
        IllegalArgumentException minimum = assertThrows(
                IllegalArgumentException.class,
                () -> input.permute(Integer.MIN_VALUE, 1, 2));
        IllegalArgumentException high = assertThrows(
                IllegalArgumentException.class,
                () -> input.permute(Integer.MAX_VALUE, 1, 2));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class, () -> input.permute(0, -3, 2));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("requestedAxes", nullAxes.getMessage()),
                () -> assertEquals(
                        "permutation axis count 2 must equal input rank 3", count.getMessage()),
                () -> assertEquals(
                        "permutation axis -4 at index 0 is outside rank 3", low.getMessage()),
                () -> assertEquals(
                        "permutation axis -2147483648 at index 0 is outside rank 3",
                        minimum.getMessage()),
                () -> assertEquals(
                        "permutation axis 2147483647 at index 0 is outside rank 3",
                        high.getMessage()),
                () -> assertEquals(
                        "permutation contains duplicate normalized axis 0 at index 1",
                        duplicate.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void transposeRequiresRankTwoAndUsesExactPermuteAxes() {
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true);

        Tensor result = input.transpose();
        TensorProvenance provenance = result.provenance().orElseThrow();
        IllegalStateException scalar = assertThrows(
                IllegalStateException.class,
                () -> tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false).transpose());
        IllegalStateException rankThree = assertThrows(
                IllegalStateException.class,
                () -> tensor(DataType.FLOAT32, Shape.of(1, 2, 3), Optional.empty(), true)
                        .transpose());
        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorPermutationExpressions.transpose(null));

        assertAll(
                () -> assertEquals(Shape.of(3, 2), result.descriptor().shape()),
                () -> assertSame(AxisTransformKind.PERMUTE, provenance.operation().kind()),
                () -> assertEquals(List.of(1, 0),
                        ((PermutationAttrs) provenance.operation().attrs()).axes()),
                () -> assertSame(input, provenance.inputs().getFirst()),
                () -> assertEquals(
                        "transpose() requires rank-2 tensor, got rank=0", scalar.getMessage()),
                () -> assertEquals(
                        "transpose() requires rank-2 tensor, got rank=3", rankThree.getMessage()),
                () -> assertEquals("input", nullInput.getMessage()));
    }

    @Test
    void reordersEveryResolvedLayoutKindIntoNewSameOffsetViewGeometry() {
        Shape shape = Shape.of(2, 3);
        LayoutDescriptor dense = LayoutDescriptor.contiguous(shape);
        LayoutDescriptor offset = LayoutDescriptor.of(
                shape, new long[] {3, 1}, 5, true);
        LayoutDescriptor strided = LayoutDescriptor.of(
                shape, new long[] {1, 2}, 4, true);
        LayoutDescriptor broadcast = LayoutDescriptor.of(
                shape, new long[] {0, 1}, 6, true);

        LayoutDescriptor denseResult = permuteLayout(shape, dense, 1, 0);
        LayoutDescriptor offsetResult = permuteLayout(shape, offset, 1, 0);
        LayoutDescriptor stridedResult = permuteLayout(shape, strided, 1, 0);
        LayoutDescriptor broadcastResult = permuteLayout(shape, broadcast, 1, 0);

        assertAll(
                () -> assertArrayEquals(new long[] {1, 3}, denseResult.strides()),
                () -> assertEquals(0, denseResult.storageOffset()),
                () -> assertSame(LayoutKind.STRIDED, denseResult.kind()),
                () -> assertEquals(6, denseResult.referencedElementSpan()),
                () -> assertTrue(denseResult.isView()),
                () -> assertNotSame(dense, denseResult),
                () -> assertArrayEquals(new long[] {1, 3}, offsetResult.strides()),
                () -> assertEquals(5, offsetResult.storageOffset()),
                () -> assertSame(LayoutKind.STRIDED, offsetResult.kind()),
                () -> assertEquals(11, offsetResult.referencedElementSpan()),
                () -> assertNotSame(offset, offsetResult),
                () -> assertArrayEquals(new long[] {2, 1}, stridedResult.strides()),
                () -> assertEquals(4, stridedResult.storageOffset()),
                () -> assertSame(LayoutKind.DENSE_WITH_OFFSET, stridedResult.kind()),
                () -> assertEquals(10, stridedResult.referencedElementSpan()),
                () -> assertNotSame(strided, stridedResult),
                () -> assertArrayEquals(new long[] {1, 0}, broadcastResult.strides()),
                () -> assertEquals(6, broadcastResult.storageOffset()),
                () -> assertSame(LayoutKind.BROADCAST_ZERO_STRIDE, broadcastResult.kind()),
                () -> assertEquals(9, broadcastResult.referencedElementSpan()),
                () -> assertNotSame(broadcast, broadcastResult));
    }

    @Test
    void resolvesScalarAndEmptyGeometryAndLeavesUnresolvedGeometryUnresolved() {
        Shape scalarShape = Shape.scalar();
        LayoutDescriptor scalarInput = LayoutDescriptor.contiguous(scalarShape);
        LayoutDescriptor scalarResult = permuteLayout(scalarShape, scalarInput);
        Shape emptyShape = Shape.of(2, 0, 3);
        LayoutDescriptor emptyInput = LayoutDescriptor.of(
                emptyShape, new long[] {0, 3, 1}, 7, true);
        LayoutDescriptor emptyResult = permuteLayout(emptyShape, emptyInput, 2, 0, 1);
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Tensor dynamic = tensor(DataType.FLOAT32, dynamicShape, Optional.empty(), true);
        Tensor staticUnresolved = tensor(
                DataType.INT32, Shape.of(2, 3), Optional.empty(), false);

        assertAll(
                () -> assertArrayEquals(new long[0], scalarResult.strides()),
                () -> assertEquals(0, scalarResult.storageOffset()),
                () -> assertEquals(1, scalarResult.referencedElementSpan()),
                () -> assertTrue(scalarResult.isView()),
                () -> assertArrayEquals(new long[] {1, 0, 3}, emptyResult.strides()),
                () -> assertEquals(7, emptyResult.storageOffset()),
                () -> assertEquals(0, emptyResult.referencedElementSpan()),
                () -> assertTrue(dynamic.permute(1, 0).descriptor().layout().isEmpty()),
                () -> assertTrue(staticUnresolved.permute(1, 0)
                        .descriptor().layout().isEmpty()));
    }

    @Test
    void identityRepeatedInverseAndNestedCallsRemainFreshExplicitExpressions() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true);

        Tensor first = input.permute(0, 1);
        Tensor second = input.permute(0, 1);
        Tensor swapped = input.permute(1, 0);
        Tensor inverse = swapped.permute(1, 0);
        Tensor nested = first.permute(1, 0);

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotSame(input, inverse),
                () -> assertNotSame(first, nested),
                () -> assertNotEquals(input.id(), first.id()),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotEquals(swapped.id(), inverse.id()),
                () -> assertSame(input,
                        first.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(swapped,
                        inverse.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(first,
                        nested.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void leavesInputMetadataStorageLivenessAndValuesUntouched() {
        Shape shape = Shape.of(2, 3);
        LayoutDescriptor inputLayout = LayoutDescriptor.of(
                shape, new long[] {3, 1}, 2, true);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.of(inputLayout), true);
        Arena arena = Arena.ofConfined();
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 8, arena.allocate(32, 1));
        float[] values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        storage.segment().asSlice(8, 24).copyFrom(
                java.lang.foreign.MemorySegment.ofArray(values));
        Tensor leaf = tensor(DataType.FLOAT32, shape, Optional.empty(), false);
        TensorProvenance originalProvenance = new TensorProvenance(
                new TensorProducer(
                        new Operation(
                                AxisTransformKind.PERMUTE,
                                new PermutationAttrs(List.of(0, 1))),
                        List.of(leaf),
                        List.of(descriptor)),
                0);
        Tensor input = new Tensor(
                new TensorId(IDS.getAndIncrement()),
                descriptor,
                Optional.of("input"),
                Optional.of(originalProvenance),
                Optional.of(storage));
        float[] before = storage.segment().asSlice(8, 24).toArray(JAVA_FLOAT);

        Tensor result = input.transpose();
        float[] after = storage.segment().asSlice(8, 24).toArray(JAVA_FLOAT);
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
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void successfulCallConsumesOneIdentityAndExhaustionOccursOnlyAtFinalDelegation()
            throws Exception {
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.of(2, 3),
                Optional.of(LayoutDescriptor.contiguous(Shape.of(2, 3))),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        input.transpose();
        assertEquals(before + 1, next.get());

        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE);
            claimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.permute(1, 0));

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

    private static LayoutDescriptor permuteLayout(
            Shape inputShape, LayoutDescriptor inputLayout, int... requestedAxes) {
        return tensor(DataType.FLOAT32, inputShape, Optional.of(inputLayout), false)
                .permute(requestedAxes)
                .descriptor()
                .layout()
                .orElseThrow();
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
