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
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
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

class TensorBooleanLogicalTest {
    private static final List<BinaryLogicalCall> BINARY_CALLS = List.of(
            new BinaryLogicalCall("logicalAnd", BooleanLogicalKind.AND, Tensor::logicalAnd),
            new BinaryLogicalCall("logicalOr", BooleanLogicalKind.OR, Tensor::logicalOr));

    @Test
    void helperAndTensorMethodsHaveExactlyTheRequiredShape() throws ReflectiveOperationException {
        int classModifiers = TensorLogicalExpressions.class.getModifiers();
        var constructors = TensorLogicalExpressions.class.getDeclaredConstructors();
        var methods = Arrays.stream(TensorLogicalExpressions.class.getDeclaredMethods())
                .sorted(java.util.Comparator.comparing(Method::getName))
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorLogicalExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorLogicalExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorLogicalExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorLogicalExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(3, methods.size()));

        Method applyBinary = TensorLogicalExpressions.class.getDeclaredMethod(
                "applyBinary", Tensor.class, Tensor.class, BooleanLogicalKind.class);
        Method applyUnary = TensorLogicalExpressions.class.getDeclaredMethod(
                "applyUnary", Tensor.class, BooleanLogicalKind.class);
        Method create = TensorLogicalExpressions.class.getDeclaredMethod(
                "create", Shape.class, BooleanLogicalKind.class, List.class);
        assertAll(
                () -> assertEquals(List.of(applyBinary, applyUnary, create), methods),
                () -> assertPackagePrivateStaticTensorMethod(applyBinary),
                () -> assertPackagePrivateStaticTensorMethod(applyUnary),
                () -> assertEquals(Tensor.class, create.getReturnType()),
                () -> assertTrue(Modifier.isPrivate(create.getModifiers())),
                () -> assertTrue(Modifier.isStatic(create.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(create.getModifiers())));

        for (BinaryLogicalCall call : BINARY_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName(), Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(Tensor.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        Method logicalNot = Tensor.class.getDeclaredMethod("logicalNot");
        assertAll(
                () -> assertEquals(Tensor.class, logicalNot.getReturnType()),
                () -> assertEquals(0, logicalNot.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(logicalNot.getModifiers())),
                () -> assertFalse(Modifier.isStatic(logicalNot.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(logicalNot.getModifiers())));
    }

    @Test
    void mapsEveryPublicMethodToItsExactKindAndProvenance() {
        Tensor left = tensor(DataType.BOOL, Shape.of(2, 1), false);
        Tensor right = tensor(DataType.BOOL, Shape.of(1, 3), false);

        for (BinaryLogicalCall call : BINARY_CALLS) {
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

        Tensor negated = left.logicalNot();
        assertAll(
                () -> assertSame(BooleanLogicalKind.NOT,
                        negated.provenance().orElseThrow().operation().kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE,
                        negated.provenance().orElseThrow().operation().attrs()),
                () -> assertEquals(1, negated.provenance().orElseThrow().inputs().size()),
                () -> assertSame(left, negated.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void representsAllRequiredStaticAndDynamicBinaryBroadcastShapes() {
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
                new BroadcastCase(Shape.of(1), Shape.ofDimensions(batch), Shape.ofDimensions(batch)));

        for (BroadcastCase broadcastCase : cases) {
            for (BinaryLogicalCall call : BINARY_CALLS) {
                Tensor result = call.apply(
                        tensor(DataType.BOOL, broadcastCase.left(), false),
                        tensor(DataType.BOOL, broadcastCase.right(), false));

                assertLogicalResult(result, broadcastCase.expected());
            }
        }
    }

    @Test
    void unaryNotRetainsTheExactShapeReferenceForEveryShapeKind() {
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(1));
        for (Shape shape : List.of(Shape.scalar(), Shape.of(2, 0, 3), Shape.of(2, 3), dynamic)) {
            Tensor input = tensor(DataType.BOOL, shape, false);

            Tensor result = input.logicalNot();

            assertAll(
                    () -> assertSame(shape, result.descriptor().shape()),
                    () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                    () -> assertLogicalResult(result, shape));
        }
    }

    @Test
    void comparisonMasksChainDirectlyIntoEveryLogicalOperation() {
        Tensor first = tensor(DataType.FLOAT32, Shape.of(2, 1), true)
                .lessThan(tensor(DataType.FLOAT64, Shape.of(1, 3), false));
        Tensor second = tensor(DataType.BFLOAT16, Shape.of(2, 1), false)
                .greaterOrEqual(tensor(DataType.FLOAT32, Shape.of(1, 3), true));

        Tensor conjunction = first.logicalAnd(second);
        Tensor disjunction = first.logicalOr(second);
        Tensor negation = first.logicalNot();

        assertAll(
                () -> assertLogicalResult(conjunction, Shape.of(2, 3)),
                () -> assertLogicalResult(disjunction, Shape.of(2, 3)),
                () -> assertSame(first.descriptor().shape(), negation.descriptor().shape()),
                () -> assertSame(first, conjunction.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(second, conjunction.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(first, negation.provenance().orElseThrow().inputs().getFirst()));
    }

    @Test
    void everyValidCallIsFreshOrderedAndNeverCanonicalized() {
        Tensor input = TensorFactory.fromFlatArray(
                boolDescriptor(Shape.of(2), true), Optional.of("input"), new byte[] {0, 1});
        Tensor other = TensorFactory.fromFlatArray(
                boolDescriptor(Shape.of(2), true), Optional.of("other"), new byte[] {1, 0});

        Tensor firstAnd = input.logicalAnd(other);
        Tensor secondAnd = input.logicalAnd(other);
        Tensor reverseAnd = other.logicalAnd(input);
        Tensor selfOr = input.logicalOr(input);
        Tensor firstNot = input.logicalNot();
        Tensor doubleNot = firstNot.logicalNot();

        assertAll(
                () -> assertNotSame(firstAnd, secondAnd),
                () -> assertNotEquals(firstAnd.id(), secondAnd.id()),
                () -> assertSame(input, firstAnd.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(other, firstAnd.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(other, reverseAnd.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(input, reverseAnd.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(input, selfOr.provenance().orElseThrow().inputs().get(0)),
                () -> assertSame(input, selfOr.provenance().orElseThrow().inputs().get(1)),
                () -> assertNotSame(input, firstNot),
                () -> assertNotSame(input, doubleNot),
                () -> assertNotSame(firstNot, doubleNot),
                () -> assertSame(firstNot, doubleNot.provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(firstAnd.label().isEmpty()),
                () -> assertTrue(firstAnd.hostStorage().isEmpty()),
                () -> assertFalse(firstAnd.descriptor().requiresGrad()));
    }

    @Test
    void rejectsAllFiveNonBoolTypesAtEveryInputPositionWithoutAllocatingIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor bool = tensor(DataType.BOOL, Shape.of(2), false);
        long beforeFailures = nextId.get();
        List<DataType> nonBoolean = List.of(
                DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64);

        for (DataType dataType : nonBoolean) {
            Tensor invalid = tensorWithoutFactory(dataType, Shape.of(2));
            for (BinaryLogicalCall call : BINARY_CALLS) {
                IllegalArgumentException invalidLeft = assertThrows(
                        IllegalArgumentException.class, () -> call.apply(invalid, bool));
                IllegalArgumentException invalidRight = assertThrows(
                        IllegalArgumentException.class, () -> call.apply(bool, invalid));
                assertAll(
                        () -> assertEquals(
                                "left must have BOOL data type, but was " + dataType,
                                invalidLeft.getMessage()),
                        () -> assertEquals(
                                "right must have BOOL data type, but was " + dataType,
                                invalidRight.getMessage()));
            }
            IllegalArgumentException invalidUnary = assertThrows(
                    IllegalArgumentException.class, invalid::logicalNot);
            assertEquals(
                    "input must have BOOL data type, but was " + dataType,
                    invalidUnary.getMessage());
        }

        assertEquals(beforeFailures, nextId.get());
    }

    @Test
    void validatesNullKindAndShapeFailuresInExactOrderBeforeIdentityAllocation()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor bool = tensor(DataType.BOOL, Shape.of(2, 3), false);
        Tensor incompatible = tensor(DataType.BOOL, Shape.of(2, 4), false);
        Tensor dynamic = tensor(
                DataType.BOOL, Shape.ofDimensions(new DynamicDimension("batch")), false);
        Tensor staticTwo = tensor(DataType.BOOL, Shape.of(2), false);
        Tensor nonBool = tensorWithoutFactory(DataType.INT32, Shape.of(2, 3));
        long beforeFailures = nextId.get();

        NullPointerException nullLeft = assertThrows(
                NullPointerException.class,
                () -> TensorLogicalExpressions.applyBinary(null, null, null));
        NullPointerException nullRight = assertThrows(
                NullPointerException.class,
                () -> TensorLogicalExpressions.applyBinary(bool, null, null));
        NullPointerException nullBinaryKind = assertThrows(
                NullPointerException.class,
                () -> TensorLogicalExpressions.applyBinary(bool, bool, null));
        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorLogicalExpressions.applyUnary(null, null));
        NullPointerException nullUnaryKind = assertThrows(
                NullPointerException.class,
                () -> TensorLogicalExpressions.applyUnary(bool, null));
        IllegalArgumentException binaryKind = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLogicalExpressions.applyBinary(
                        nonBool, nonBool, BooleanLogicalKind.NOT));
        IllegalArgumentException unaryAnd = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLogicalExpressions.applyUnary(nonBool, BooleanLogicalKind.AND));
        IllegalArgumentException unaryOr = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLogicalExpressions.applyUnary(nonBool, BooleanLogicalKind.OR));
        IllegalArgumentException staticShape = assertThrows(
                IllegalArgumentException.class, () -> bool.logicalAnd(incompatible));
        IllegalArgumentException dynamicShape = assertThrows(
                IllegalArgumentException.class, () -> dynamic.logicalOr(staticTwo));

        for (BinaryLogicalCall call : BINARY_CALLS) {
            NullPointerException publicNull = assertThrows(
                    NullPointerException.class, () -> call.apply(bool, null));
            assertEquals("right", publicNull.getMessage());
        }

        assertAll(
                () -> assertEquals("left", nullLeft.getMessage()),
                () -> assertEquals("right", nullRight.getMessage()),
                () -> assertEquals("kind", nullBinaryKind.getMessage()),
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullUnaryKind.getMessage()),
                () -> assertEquals(
                        "binary logical expression kind must be AND or OR, but was NOT",
                        binaryKind.getMessage()),
                () -> assertEquals(
                        "unary logical expression kind must be NOT, but was AND",
                        unaryAnd.getMessage()),
                () -> assertEquals(
                        "unary logical expression kind must be NOT, but was OR",
                        unaryOr.getMessage()),
                () -> assertTrue(staticShape.getMessage().contains("at result axis 1")),
                () -> assertTrue(dynamicShape.getMessage().contains("at result axis 0")),
                () -> assertEquals(beforeFailures, nextId.get()));
    }

    @Test
    void preservesInputMetadataProvenanceLabelsStorageAndContents() {
        byte[] leftValues = {0, 1};
        byte[] rightValues = {1, 0};
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = boolDescriptor(shape, true);
        HostTensorStorage leftStorage = new MemorySegmentStorage(
                DataType.BOOL, 2, MemorySegment.ofArray(leftValues));
        HostTensorStorage rightStorage = new MemorySegmentStorage(
                DataType.BOOL, 2, MemorySegment.ofArray(rightValues));
        Tensor leaf = TensorFactory.create(
                descriptor, Optional.of("leaf"), Optional.of(leftStorage));
        Tensor right = TensorFactory.create(
                descriptor, Optional.of("right"), Optional.of(rightStorage));
        TensorProvenance inputProvenance = new TensorProvenance(
                new Operation(BooleanLogicalKind.NOT, NoOperationAttrs.INSTANCE), List.of(leaf));
        Tensor left = TensorFactory.createDerived(
                descriptor, Optional.of("left"), inputProvenance);
        left.replaceHostStorage(leftStorage);

        Tensor binaryResult = left.logicalAnd(right);
        Tensor unaryResult = left.logicalNot();

        assertAll(
                () -> assertSame(descriptor, left.descriptor()),
                () -> assertSame(descriptor, right.descriptor()),
                () -> assertEquals(Optional.of("left"), left.label()),
                () -> assertEquals(Optional.of("right"), right.label()),
                () -> assertSame(inputProvenance, left.provenance().orElseThrow()),
                () -> assertTrue(right.provenance().isEmpty()),
                () -> assertSame(leftStorage, left.hostStorage().orElseThrow()),
                () -> assertSame(rightStorage, right.hostStorage().orElseThrow()),
                () -> assertTrue(binaryResult.hostStorage().isEmpty()),
                () -> assertTrue(unaryResult.hostStorage().isEmpty()),
                () -> assertFalse(binaryResult.descriptor().requiresGrad()),
                () -> assertFalse(unaryResult.descriptor().requiresGrad()),
                () -> assertArrayEquals(new byte[] {0, 1}, leftValues),
                () -> assertArrayEquals(new byte[] {1, 0}, rightValues));
    }

    private static void assertPackagePrivateStaticTensorMethod(Method method) {
        assertAll(
                () -> assertEquals(Tensor.class, method.getReturnType()),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isProtected(method.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(method.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
    }

    private static void assertLogicalResult(Tensor result, Shape expectedShape) {
        assertAll(
                () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                () -> assertEquals(expectedShape, result.descriptor().shape()),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertFalse(result.descriptor().requiresGrad()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static Tensor tensorWithoutFactory(DataType dataType, Shape shape) {
        return new Tensor(
                new TensorId(0),
                new TensorDescriptor(dataType, shape, Optional.empty(), false),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static TensorDescriptor boolDescriptor(Shape shape, boolean resolved) {
        return new TensorDescriptor(
                DataType.BOOL,
                shape,
                resolved ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),
                false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private record BinaryLogicalCall(
            String methodName,
            BooleanLogicalKind kind,
            BiFunction<Tensor, Tensor, Tensor> function) {
        private Tensor apply(Tensor left, Tensor right) {
            return function.apply(left, right);
        }
    }

    private record BroadcastCase(Shape left, Shape right, Shape expected) {
    }
}
