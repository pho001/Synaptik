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
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
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

class TensorBinaryComparisonTest {
    private static final List<ComparisonCall> COMPARISON_CALLS = List.of(
            new ComparisonCall(
                    "greaterThan", BinaryComparisonKind.GREATER_THAN, Tensor::greaterThan),
            new ComparisonCall(
                    "greaterOrEqual",
                    BinaryComparisonKind.GREATER_OR_EQUAL,
                    Tensor::greaterOrEqual),
            new ComparisonCall("lessThan", BinaryComparisonKind.LESS_THAN, Tensor::lessThan),
            new ComparisonCall(
                    "lessOrEqual", BinaryComparisonKind.LESS_OR_EQUAL, Tensor::lessOrEqual),
            new ComparisonCall("equalTo", BinaryComparisonKind.EQUAL, Tensor::equalTo),
            new ComparisonCall("notEqualTo", BinaryComparisonKind.NOT_EQUAL, Tensor::notEqualTo));

    @Test
    void helperAndTensorMethodsHaveExactlyTheRequiredShape() throws ReflectiveOperationException {
        int classModifiers = TensorComparisonExpressions.class.getModifiers();
        var constructors = TensorComparisonExpressions.class.getDeclaredConstructors();
        var methods = TensorComparisonExpressions.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorComparisonExpressions.class.isRecord()),
                () -> assertEquals(
                        Set.of(), Set.of(TensorComparisonExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorComparisonExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorComparisonExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length));

        Method apply = TensorComparisonExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, Tensor.class, BinaryComparisonKind.class);
        assertAll(
                () -> assertEquals(apply, methods[0]),
                () -> assertEquals(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(apply.getModifiers())));

        for (ComparisonCall call : COMPARISON_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName(), Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(Tensor.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void mapsEveryPublicMethodToItsExactKindAndOrderedProvenance() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 1), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(1, 3), false);

        for (ComparisonCall call : COMPARISON_CALLS) {
            Tensor result = call.apply(left, right);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Operation operation = provenance.operation();

            assertAll(
                    () -> assertSame(call.kind(), operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                    () -> assertEquals(2, provenance.inputs().size()),
                    () -> assertSame(left, provenance.inputs().get(0)),
                    () -> assertSame(right, provenance.inputs().get(1)));
        }

        Tensor reverse = right.greaterThan(left);
        assertAll(
                () -> assertSame(right, reverse.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(left, reverse.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void acceptsEveryFloatingPairAndAlwaysProducesBool() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};

        for (ComparisonCall call : COMPARISON_CALLS) {
            for (DataType leftType : floating) {
                for (DataType rightType : floating) {
                    Tensor left = tensor(leftType, Shape.scalar(), true);
                    Tensor right = tensor(rightType, Shape.scalar(), true);

                    Tensor result = call.apply(left, right);

                    assertAll(
                            () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                            () -> assertEquals(Shape.scalar(), result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertFalse(result.descriptor().requiresGrad()));
                }
            }
        }
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
            Tensor result = tensor(DataType.FLOAT32, broadcastCase.left(), true)
                    .greaterThan(tensor(DataType.FLOAT64, broadcastCase.right(), true));

            assertAll(
                    () -> assertEquals(broadcastCase.expected(), result.descriptor().shape()),
                    () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                    () -> assertTrue(result.descriptor().layout().isEmpty()),
                    () -> assertFalse(result.descriptor().requiresGrad()));
        }
    }

    @Test
    void alwaysDisablesGradientEligibilityForEveryInputCombination() {
        for (boolean leftRequiresGrad : List.of(false, true)) {
            for (boolean rightRequiresGrad : List.of(false, true)) {
                Tensor left = tensor(DataType.FLOAT32, Shape.of(2), leftRequiresGrad);
                Tensor right = tensor(DataType.FLOAT64, Shape.of(2), rightRequiresGrad);

                assertFalse(left.lessOrEqual(right).descriptor().requiresGrad());
            }
        }
    }

    @Test
    void everyValidCallReturnsAFreshUnlabeledStorageFreeUnresolvedTensor() {
        Tensor left = TensorFactory.scalar(2.0f, Optional.of("left"), true);
        Tensor right = TensorFactory.scalar(1.0f, Optional.of("right"), true);

        Tensor firstEqual = left.equalTo(right);
        Tensor secondEqual = left.equalTo(right);
        Tensor symmetricEqual = right.equalTo(left);
        Tensor selfNotEqual = left.notEqualTo(left);

        assertAll(
                () -> assertNotSame(left, firstEqual),
                () -> assertNotSame(right, firstEqual),
                () -> assertNotSame(firstEqual, secondEqual),
                () -> assertNotSame(firstEqual, symmetricEqual),
                () -> assertNotEquals(firstEqual.id(), secondEqual.id()),
                () -> assertTrue(firstEqual.label().isEmpty()),
                () -> assertTrue(firstEqual.hostStorage().isEmpty()),
                () -> assertTrue(firstEqual.descriptor().layout().isEmpty()),
                () -> assertSame(left, firstEqual.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(right, firstEqual.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(right, symmetricEqual.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(left, symmetricEqual.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(left, selfNotEqual.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(left, selfNotEqual.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void preservesExpressionIdentityWhenChainingWithoutInterning() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor middle = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor right = tensor(DataType.FLOAT32, Shape.of(2), false);

        Tensor less = left.lessThan(middle);
        Tensor greater = middle.greaterThan(right);

        assertAll(
                () -> assertSame(BinaryComparisonKind.LESS_THAN,
                        less.provenance().orElseThrow().operation().kind()),
                () -> assertSame(BinaryComparisonKind.GREATER_THAN,
                        greater.provenance().orElseThrow().operation().kind()),
                () -> assertSame(left, less.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(middle, less.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(middle, greater.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(right, greater.provenance().orElseThrow().inputs().get(1)));
    }

    @Test
    void validatesNullsTypesAndShapesBeforeAllocatingAnIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor integral = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor bool = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor incompatible = tensor(DataType.FLOAT64, Shape.of(2, 4), false);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("batch")),
                false);
        Tensor staticTwo = tensor(DataType.FLOAT32, Shape.of(2), false);
        long beforeFailures = nextId.get();

        NullPointerException nullLeft = assertThrows(
                NullPointerException.class,
                () -> TensorComparisonExpressions.apply(null, null, null));
        NullPointerException nullRight = assertThrows(
                NullPointerException.class,
                () -> TensorComparisonExpressions.apply(floating, null, null));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorComparisonExpressions.apply(floating, floating, null));
        for (ComparisonCall call : COMPARISON_CALLS) {
            NullPointerException publicNull = assertThrows(
                    NullPointerException.class, () -> call.apply(floating, null));
            assertEquals("right", publicNull.getMessage());
        }

        IllegalArgumentException invalidLeft = assertThrows(
                IllegalArgumentException.class, () -> integral.equalTo(bool));
        IllegalArgumentException invalidRight = assertThrows(
                IllegalArgumentException.class, () -> floating.equalTo(bool));
        IllegalArgumentException staticShape = assertThrows(
                IllegalArgumentException.class, () -> floating.notEqualTo(incompatible));
        IllegalArgumentException dynamicShape = assertThrows(
                IllegalArgumentException.class, () -> dynamic.greaterThan(staticTwo));

        assertAll(
                () -> assertEquals("left", nullLeft.getMessage()),
                () -> assertEquals("right", nullRight.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals(
                        "left must be a floating data type, but was INT32",
                        invalidLeft.getMessage()),
                () -> assertEquals(
                        "right must be a floating data type, but was BOOL",
                        invalidRight.getMessage()),
                () -> assertTrue(staticShape.getMessage().contains("at result axis 1")),
                () -> assertTrue(dynamicShape.getMessage().contains("at result axis 0")),
                () -> assertEquals(beforeFailures, nextId.get()));
    }

    @Test
    void doesNotMutateOrRetainInputMetadataOrStorageAsOutputStorage() {
        float[] leftValues = {1.0f, 2.0f};
        double[] rightValues = {3.0, 4.0};
        Shape shape = Shape.of(2);
        TensorDescriptor leftDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        TensorDescriptor rightDescriptor = new TensorDescriptor(
                DataType.FLOAT64,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        HostTensorStorage leftStorage = new MemorySegmentStorage(
                DataType.FLOAT32, 2, MemorySegment.ofArray(leftValues));
        HostTensorStorage rightStorage = new MemorySegmentStorage(
                DataType.FLOAT64, 2, MemorySegment.ofArray(rightValues));
        Tensor leaf = TensorFactory.create(
                leftDescriptor, Optional.of("leaf"), Optional.of(leftStorage));
        Tensor right = TensorFactory.create(
                rightDescriptor, Optional.of("right"), Optional.of(rightStorage));
        TensorProvenance inputProvenance = new TensorProvenance(
                new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                List.of(leaf, right));
        Tensor left = TensorFactory.createDerived(
                leftDescriptor, Optional.of("left"), inputProvenance);
        left.replaceHostStorage(leftStorage);

        Tensor result = left.greaterOrEqual(right);

        assertAll(
                () -> assertSame(leftDescriptor, left.descriptor()),
                () -> assertSame(rightDescriptor, right.descriptor()),
                () -> assertTrue(left.descriptor().requiresGrad()),
                () -> assertTrue(right.descriptor().requiresGrad()),
                () -> assertEquals(Optional.of("left"), left.label()),
                () -> assertEquals(Optional.of("right"), right.label()),
                () -> assertSame(inputProvenance, left.provenance().orElseThrow()),
                () -> assertTrue(right.provenance().isEmpty()),
                () -> assertSame(leftStorage, left.hostStorage().orElseThrow()),
                () -> assertSame(rightStorage, right.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertFalse(result.descriptor().requiresGrad()),
                () -> assertArrayEquals(new float[] {1.0f, 2.0f}, leftValues),
                () -> assertArrayEquals(new double[] {3.0, 4.0}, rightValues));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private record ComparisonCall(
            String methodName,
            BinaryComparisonKind kind,
            BiFunction<Tensor, Tensor, Tensor> function) {
        private Tensor apply(Tensor left, Tensor right) {
            return function.apply(left, right);
        }
    }

    private record BroadcastCase(Shape left, Shape right, Shape expected) {
    }
}
