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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiFunction;
import org.junit.jupiter.api.Test;

class TensorBinaryArithmeticTest {
    private static final List<BinaryCall> BINARY_CALLS = List.of(
            new BinaryCall("add", BinaryArithmeticKind.ADD, Tensor::add),
            new BinaryCall("sub", BinaryArithmeticKind.SUB, Tensor::sub),
            new BinaryCall("mul", BinaryArithmeticKind.MUL, Tensor::mul),
            new BinaryCall("div", BinaryArithmeticKind.DIV, Tensor::div),
            new BinaryCall("minimum", BinaryArithmeticKind.MIN, Tensor::minimum),
            new BinaryCall("maximum", BinaryArithmeticKind.MAX, Tensor::maximum),
            new BinaryCall("pow", BinaryArithmeticKind.POW, Tensor::pow));
    private static final List<BinaryCall> INTEGRAL_CALLS = List.of(
            new BinaryCall("add", BinaryArithmeticKind.ADD, Tensor::add),
            new BinaryCall("sub", BinaryArithmeticKind.SUB, Tensor::sub),
            new BinaryCall("mul", BinaryArithmeticKind.MUL, Tensor::mul),
            new BinaryCall("minimum", BinaryArithmeticKind.MIN, Tensor::minimum),
            new BinaryCall("maximum", BinaryArithmeticKind.MAX, Tensor::maximum));

    @Test
    void helperAndTensorMethodsHaveExactlyTheRequiredShape() throws ReflectiveOperationException {
        int classModifiers = TensorBinaryExpressions.class.getModifiers();
        var constructors = TensorBinaryExpressions.class.getDeclaredConstructors();
        var methods = TensorBinaryExpressions.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorBinaryExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorBinaryExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorBinaryExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorBinaryExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length));

        long publicTensorMethodCount = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count();
        assertEquals(177, publicTensorMethodCount);

        Method apply = TensorBinaryExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Tensor.class, BinaryArithmeticKind.class);
        assertAll(
                () -> assertEquals(apply, methods[0]),
                () -> assertEquals(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(apply.getModifiers())));

        for (BinaryCall call : BINARY_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName(), Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void mapsEveryPublicMethodToItsExactKindAndOrderedProvenance() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 1), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(1, 3), false);

        for (BinaryCall call : BINARY_CALLS) {
            Tensor result = call.apply(left, right);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Operation operation = provenance.operation();

            assertAll(
                    () -> assertSame(call.kind(), operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                    () -> assertEquals(2, provenance.inputs().size()),
                    () -> assertSame(left, provenance.inputs().get(0)),
                    () -> assertSame(right, provenance.inputs().get(1)),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
        }
    }

    @Test
    void acceptsEveryFloatingPairAndPromotesThroughTheExactHierarchy() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};

        for (BinaryCall call : BINARY_CALLS) {
            for (DataType leftType : floating) {
                for (DataType rightType : floating) {
                    Tensor left = tensor(leftType, Shape.scalar(), false);
                    Tensor right = tensor(rightType, Shape.scalar(), false);

                    Tensor result = call.apply(left, right);

                    assertAll(
                            () -> assertSame(widest(leftType, rightType), result.descriptor().dataType()),
                            () -> assertEquals(Shape.scalar(), result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()));
                }
            }
        }
    }

    @Test
    void acceptsSelectedIntegralKindsForEveryOrderedWidthPair() {
        DataType[] integral = {DataType.INT32, DataType.INT64};

        for (BinaryCall call : INTEGRAL_CALLS) {
            for (DataType leftType : integral) {
                for (DataType rightType : integral) {
                    Tensor left = tensor(leftType, Shape.of(2, 1), false);
                    Tensor right = tensor(rightType, Shape.of(1, 3), false);

                    Tensor result = call.apply(left, right);
                    TensorProvenance provenance = result.provenance().orElseThrow();

                    assertAll(
                            () -> assertSame(
                                    leftType == DataType.INT64 || rightType == DataType.INT64
                                            ? DataType.INT64
                                            : DataType.INT32,
                                    result.descriptor().dataType()),
                            () -> assertEquals(Shape.of(2, 3), result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertFalse(result.descriptor().requiresGrad()),
                            () -> assertSame(call.kind(), provenance.operation().kind()),
                            () -> assertSame(NoOperationAttrs.INSTANCE, provenance.operation().attrs()),
                            () -> assertSame(left, provenance.inputs().get(0)),
                            () -> assertSame(right, provenance.inputs().get(1)),
                            () -> assertEquals(0, provenance.outputIndex()),
                            () -> assertEquals(1, provenance.producer().outputCount()),
                            () -> assertTrue(result.label().isEmpty()),
                            () -> assertTrue(result.hostStorage().isEmpty()));
                }
            }
        }
    }

    @Test
    void integralArithmeticRecordsModularKindsWithoutEvaluatingOverflow() {
        Tensor int32Maximum = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor int32One = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor int64Minimum = tensor(DataType.INT64, Shape.scalar(), false);

        List<Tensor> modularRequests = List.of(
                int32Maximum.add(int32One),
                int32Maximum.sub(int32One),
                int64Minimum.mul(int32One));

        assertAll(
                () -> assertSame(BinaryArithmeticKind.ADD,
                        modularRequests.get(0).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.SUB,
                        modularRequests.get(1).provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        modularRequests.get(2).provenance().orElseThrow().operation().kind()),
                () -> assertSame(DataType.INT32, modularRequests.get(0).descriptor().dataType()),
                () -> assertSame(DataType.INT64, modularRequests.get(2).descriptor().dataType()),
                () -> assertTrue(modularRequests.stream()
                        .allMatch(result -> result.hostStorage().isEmpty())));
    }

    @Test
    void rejectsIntegralDivisionAndPowerBeforeBroadcastAndIdentityAllocation()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor int32 = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor int64Incompatible = tensor(DataType.INT64, Shape.of(4), false);
        long beforeFailures = nextId.get();

        IllegalArgumentException division = assertThrows(
                IllegalArgumentException.class, () -> int32.div(int64Incompatible));
        IllegalArgumentException power = assertThrows(
                IllegalArgumentException.class, () -> int32.pow(int64Incompatible));

        assertAll(
                () -> assertEquals(
                        "DIV does not support integral data types", division.getMessage()),
                () -> assertEquals(
                        "POW does not support integral data types", power.getMessage()),
                () -> assertEquals(beforeFailures, nextId.get()));
    }

    @Test
    void representsAllRequiredStaticAndDynamicBroadcastShapes() {
        DynamicDimension batch = new DynamicDimension("batch");
        List<BroadcastCase> cases = List.of(
                new BroadcastCase(Shape.scalar(), Shape.of(2, 3), Shape.of(2, 3)),
                new BroadcastCase(Shape.of(0, 3), Shape.of(1, 3), Shape.of(0, 3)),
                new BroadcastCase(Shape.of(3), Shape.of(2, 1, 3), Shape.of(2, 1, 3)),
                new BroadcastCase(Shape.of(2, 1, 3), Shape.of(1, 4, 3), Shape.of(2, 4, 3)),
                new BroadcastCase(
                        Shape.ofDimensions(batch, new StaticDimension(1)),
                        Shape.ofDimensions(batch, new StaticDimension(4)),
                        Shape.ofDimensions(batch, new StaticDimension(4))),
                new BroadcastCase(
                        Shape.of(1),
                        Shape.ofDimensions(batch),
                        Shape.ofDimensions(batch)));

        for (BroadcastCase broadcastCase : cases) {
            Tensor result = tensor(DataType.FLOAT32, broadcastCase.left(), false)
                    .add(tensor(DataType.FLOAT32, broadcastCase.right(), false));

            assertAll(
                    () -> assertEquals(broadcastCase.expected(), result.descriptor().shape()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()));
        }
    }

    @Test
    void propagatesGradientEligibilityAsTheExactInputOr() {
        for (boolean leftRequiresGrad : List.of(false, true)) {
            for (boolean rightRequiresGrad : List.of(false, true)) {
                Tensor left = tensor(DataType.FLOAT32, Shape.of(2), leftRequiresGrad);
                Tensor right = tensor(DataType.FLOAT32, Shape.of(2), rightRequiresGrad);

                Tensor result = left.add(right);

                assertEquals(
                        leftRequiresGrad || rightRequiresGrad,
                        result.descriptor().requiresGrad());
            }
        }
    }

    @Test
    void everyValidCallReturnsAFreshUnlabeledStorageFreeUnresolvedTensor() {
        Tensor value = TensorFactory.scalar(2.0f, Optional.of("value"), true);
        Tensor zero = TensorFactory.scalar(0.0f, Optional.of("zero"), false);
        Tensor one = TensorFactory.scalar(1.0f, Optional.of("one"), false);

        Tensor firstAdd = value.add(zero);
        Tensor secondAdd = value.add(zero);
        Tensor multiplyOne = value.mul(one);
        Tensor selfSubtract = value.sub(value);

        assertAll(
                () -> assertNotSame(value, firstAdd),
                () -> assertNotSame(value, multiplyOne),
                () -> assertNotSame(firstAdd, secondAdd),
                () -> assertNotEquals(firstAdd.id(), secondAdd.id()),
                () -> assertTrue(firstAdd.label().isEmpty()),
                () -> assertTrue(firstAdd.hostStorage().isEmpty()),
                () -> assertTrue(firstAdd.descriptor().layout().isEmpty()),
                () -> assertSame(BinaryArithmeticKind.ADD,
                        firstAdd.provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        multiplyOne.provenance().orElseThrow().operation().kind()),
                () -> assertSame(value,
                        selfSubtract.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(value,
                        selfSubtract.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void preservesExpressionIdentityWhenChainingWithoutInterning() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor middle = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(2), false);

        Tensor addition = left.add(middle);
        Tensor product = addition.mul(right);

        assertAll(
                () -> assertSame(BinaryArithmeticKind.ADD,
                        addition.provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        product.provenance().orElseThrow().operation().kind()),
                () -> assertSame(addition, product.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(right, product.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void validatesNullsTypesAndShapesBeforeAllocatingAnIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor incompatible = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("batch")),
                false);
        Tensor staticTwo = tensor(DataType.FLOAT32, Shape.of(2), false);
        long beforeFailures = nextId.get();

        NullPointerException nullLeft = assertThrows(
                NullPointerException.class,
                () -> TensorBinaryExpressions.apply(null, null, null));
        NullPointerException nullRight = assertThrows(
                NullPointerException.class,
                () -> TensorBinaryExpressions.apply(floating, null, null));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorBinaryExpressions.apply(floating, floating, null));
        for (BinaryCall call : BINARY_CALLS) {
            NullPointerException publicNull = assertThrows(
                    NullPointerException.class, () -> call.apply(floating, null));
            assertEquals("right", publicNull.getMessage());
        }

        IllegalArgumentException invalidLeft = assertThrows(
                IllegalArgumentException.class, () -> bool.add(integral));
        IllegalArgumentException invalidRight = assertThrows(
                IllegalArgumentException.class, () -> floating.add(bool));
        IllegalArgumentException mixedLeft = assertThrows(
                IllegalArgumentException.class, () -> integral.add(floating));
        IllegalArgumentException mixedRight = assertThrows(
                IllegalArgumentException.class, () -> floating.add(integral));
        IllegalArgumentException staticShape = assertThrows(
                IllegalArgumentException.class, () -> floating.add(incompatible));
        IllegalArgumentException dynamicShape = assertThrows(
                IllegalArgumentException.class, () -> dynamic.add(staticTwo));

        assertAll(
                () -> assertEquals("left", nullLeft.getMessage()),
                () -> assertEquals("right", nullRight.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals(
                        "left must be a numeric data type, but was BOOL",
                        invalidLeft.getMessage()),
                () -> assertEquals(
                        "right must be a numeric data type, but was BOOL",
                        invalidRight.getMessage()),
                () -> assertEquals(
                        "numeric data types must share a category, but were INT32 and FLOAT32",
                        mixedLeft.getMessage()),
                () -> assertEquals(
                        "numeric data types must share a category, but were FLOAT32 and INT32",
                        mixedRight.getMessage()),
                () -> assertTrue(staticShape.getMessage().contains("at result axis 1")),
                () -> assertTrue(dynamicShape.getMessage().contains("at result axis 0")),
                () -> assertEquals(beforeFailures, nextId.get()));
    }

    @Test
    void doesNotMutateOrRetainInputMetadataOrStorageAsOutputStorage() {
        float[] leftValues = {1.0f, 2.0f};
        float[] rightValues = {3.0f, 4.0f};
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        HostTensorStorage leftStorage = new MemorySegmentStorage(
                DataType.FLOAT32, 2, MemorySegment.ofArray(leftValues));
        HostTensorStorage rightStorage = new MemorySegmentStorage(
                DataType.FLOAT32, 2, MemorySegment.ofArray(rightValues));
        Tensor leaf = TensorFactory.create(
                descriptor, Optional.of("leaf"), Optional.of(leftStorage));
        Tensor right = TensorFactory.create(
                descriptor, Optional.of("right"), Optional.of(rightStorage));
        Operation inputOperation =
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        Tensor left = TensorFactory.createDerived(
                descriptor, Optional.of("left"), inputOperation, List.of(leaf, right));
        TensorProvenance inputProvenance = left.provenance().orElseThrow();
        left.replaceHostStorage(leftStorage);

        Tensor result = left.div(right);

        assertAll(
                () -> assertSame(descriptor, left.descriptor()),
                () -> assertSame(descriptor, right.descriptor()),
                () -> assertEquals(Optional.of("left"), left.label()),
                () -> assertEquals(Optional.of("right"), right.label()),
                () -> assertSame(inputProvenance, left.provenance().orElseThrow()),
                () -> assertTrue(right.provenance().isEmpty()),
                () -> assertSame(leftStorage, left.hostStorage().orElseThrow()),
                () -> assertSame(rightStorage, right.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertArrayEquals(new float[] {1.0f, 2.0f}, leftValues),
                () -> assertArrayEquals(new float[] {3.0f, 4.0f}, rightValues));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static DataType widest(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private record BinaryCall(
            String methodName,
            BinaryArithmeticKind kind,
            BiFunction<Tensor, Tensor, Tensor> function) {
        private Tensor apply(Tensor left, Tensor right) {
            return function.apply(left, right);
        }
    }

    private record BroadcastCase(Shape left, Shape right, Shape expected) {
    }
}
