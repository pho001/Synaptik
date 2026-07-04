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
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import org.junit.jupiter.api.Test;

class TensorUnaryElementwiseTest {
    private static final List<UnaryCall> UNARY_CALLS = List.of(
            new UnaryCall("abs", UnaryElementwiseKind.ABS, Tensor::abs),
            new UnaryCall("neg", UnaryElementwiseKind.NEG, Tensor::neg),
            new UnaryCall("inv", UnaryElementwiseKind.INV, Tensor::inv),
            new UnaryCall("log", UnaryElementwiseKind.LOG, Tensor::log),
            new UnaryCall("exp", UnaryElementwiseKind.EXP, Tensor::exp),
            new UnaryCall("erf", UnaryElementwiseKind.ERF, Tensor::erf),
            new UnaryCall("sqrt", UnaryElementwiseKind.SQRT, Tensor::sqrt),
            new UnaryCall("floor", UnaryElementwiseKind.FLOOR, Tensor::floor),
            new UnaryCall("ceil", UnaryElementwiseKind.CEIL, Tensor::ceil),
            new UnaryCall("sign", UnaryElementwiseKind.SIGN, Tensor::sign),
            new UnaryCall("relu", UnaryElementwiseKind.RELU, Tensor::relu),
            new UnaryCall("sigmoid", UnaryElementwiseKind.SIGMOID, Tensor::sigmoid),
            new UnaryCall("tanh", UnaryElementwiseKind.TANH, Tensor::tanh),
            new UnaryCall("fastExp", UnaryElementwiseKind.FAST_EXP, Tensor::fastExp),
            new UnaryCall("fastTanh", UnaryElementwiseKind.FAST_TANH, Tensor::fastTanh));

    @Test
    void helperAndTensorMethodsHaveExactlyTheRequiredShape() throws ReflectiveOperationException {
        int classModifiers = TensorUnaryExpressions.class.getModifiers();
        var constructors = TensorUnaryExpressions.class.getDeclaredConstructors();
        var methods = TensorUnaryExpressions.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorUnaryExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorUnaryExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorUnaryExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorUnaryExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length));

        Method apply = TensorUnaryExpressions.class.getDeclaredMethod(
                "apply", Tensor.class, UnaryElementwiseKind.class);
        assertAll(
                () -> assertEquals(apply, methods[0]),
                () -> assertEquals(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(apply.getModifiers())));

        for (UnaryCall call : UNARY_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName());
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(0, method.getParameterCount()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void mapsEveryPublicMethodToItsExactKindAndOneInputProvenance() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        for (UnaryCall call : UNARY_CALLS) {
            Tensor result = call.apply(input);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Operation operation = provenance.operation();

            assertAll(
                    () -> assertSame(call.kind(), operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                    () -> assertEquals(1, provenance.inputs().size()),
                    () -> assertSame(input, provenance.inputs().getFirst()));
        }
    }

    @Test
    void acceptsEveryFloatingTypeAndRetainsExactShapeReferenceAcrossShapeStates() {
        Shape scalar = Shape.scalar();
        Shape empty = Shape.of(2, 0, 4);
        Shape ordinary = Shape.of(2, 3);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        for (DataType dataType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            for (Shape shape : List.of(scalar, empty, ordinary, dynamic)) {
                Tensor input = tensor(dataType, shape, true);

                for (UnaryCall call : UNARY_CALLS) {
                    Tensor result = call.apply(input);
                    assertAll(
                            () -> assertSame(dataType, result.descriptor().dataType()),
                            () -> assertSame(shape, result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertTrue(result.descriptor().requiresGrad()));
                }
            }
        }

        TensorDescriptor resolvedDescriptor = new TensorDescriptor(
                DataType.FLOAT32,
                ordinary,
                Optional.of(LayoutDescriptor.contiguous(ordinary)),
                false);
        Tensor resolvedInput = TensorFactory.create(resolvedDescriptor);
        Tensor resolvedResult = resolvedInput.abs();
        assertAll(
                () -> assertSame(ordinary, resolvedResult.descriptor().shape()),
                () -> assertTrue(resolvedResult.descriptor().layout().isEmpty()),
                () -> assertFalse(resolvedResult.descriptor().requiresGrad()));
    }

    @Test
    void propagatesGradientEligibilityUnchangedForEveryKind() {
        for (boolean requiresGrad : List.of(false, true)) {
            Tensor input = tensor(DataType.FLOAT32, Shape.of(2), requiresGrad);
            for (UnaryCall call : UNARY_CALLS) {
                assertEquals(requiresGrad, call.apply(input).descriptor().requiresGrad());
            }
        }

        Tensor input = tensor(DataType.FLOAT32, Shape.scalar(), true);
        assertAll(
                () -> assertTrue(input.floor().descriptor().requiresGrad()),
                () -> assertTrue(input.ceil().descriptor().requiresGrad()),
                () -> assertTrue(input.sign().descriptor().requiresGrad()));
    }

    @Test
    void everyValidCallIsFreshUnlabeledStorageFreeAndNeverCanonicalized() {
        Tensor input = TensorFactory.scalar(1.0f, Optional.of("input"), true);
        Tensor firstNegation = input.neg();
        Tensor secondNegation = input.neg();
        Tensor doubleNegation = firstNegation.neg();
        Tensor doubleInverse = input.inv().inv();
        Tensor strictExp = input.exp();
        Tensor fastExp = input.fastExp();
        Tensor strictTanh = input.tanh();
        Tensor fastTanh = input.fastTanh();

        assertAll(
                () -> assertNotSame(input, firstNegation),
                () -> assertNotSame(firstNegation, secondNegation),
                () -> assertNotEquals(firstNegation.id(), secondNegation.id()),
                () -> assertNotSame(input, doubleNegation),
                () -> assertNotSame(input, doubleInverse),
                () -> assertSame(firstNegation,
                        doubleNegation.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(UnaryElementwiseKind.EXP,
                        strictExp.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.FAST_EXP,
                        fastExp.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.TANH,
                        strictTanh.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.FAST_TANH,
                        fastTanh.provenance().orElseThrow().operation().kind()),
                () -> assertTrue(firstNegation.label().isEmpty()),
                () -> assertTrue(firstNegation.hostStorage().isEmpty()),
                () -> assertTrue(firstNegation.descriptor().layout().isEmpty()));
    }

    @Test
    void validatesNullsAndEveryNonFloatingTypeBeforeAllocatingAnIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        long beforeFailures = nextId.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> TensorUnaryExpressions.apply(null, null));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorUnaryExpressions.apply(floating, null));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()));

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalid = tensor(dataType, Shape.scalar(), false);
            long beforeTypeFailure = nextId.get();
            for (UnaryCall call : UNARY_CALLS) {
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> call.apply(invalid));
                assertEquals(
                        "input must be a floating data type, but was " + dataType,
                        failure.getMessage());
            }
            assertEquals(beforeTypeFailure, nextId.get());
        }
        assertEquals(beforeFailures + 3, nextId.get());
    }

    @Test
    void doesNotInspectDomainsOrStoredValues() {
        Tensor negative = TensorFactory.scalar(-1.0f, Optional.of("negative"), false);
        Tensor zero = TensorFactory.scalar(0.0f, Optional.of("zero"), false);

        Tensor logarithm = negative.log();
        Tensor squareRoot = negative.sqrt();
        Tensor inverse = zero.inv();

        assertAll(
                () -> assertSame(UnaryElementwiseKind.LOG,
                        logarithm.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.SQRT,
                        squareRoot.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.INV,
                        inverse.provenance().orElseThrow().operation().kind()),
                () -> assertTrue(logarithm.hostStorage().isEmpty()),
                () -> assertTrue(squareRoot.hostStorage().isEmpty()),
                () -> assertTrue(inverse.hostStorage().isEmpty()));
    }

    @Test
    void preservesInputMetadataProvenanceStorageAssociationAndContents() {
        float[] values = {-1.0f, 0.0f, 2.0f};
        Shape shape = Shape.of(3);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor leaf = TensorFactory.create(
                descriptor, Optional.of("leaf"), Optional.of(storage));
        TensorProvenance inputProvenance = new TensorProvenance(
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE), List.of(leaf));
        Tensor input = TensorFactory.createDerived(
                descriptor, Optional.of("derived"), inputProvenance);
        input.replaceHostStorage(storage);

        Tensor result = input.sqrt();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(shape, input.descriptor().shape()),
                () -> assertEquals(Optional.of("derived"), input.label()),
                () -> assertSame(inputProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertArrayEquals(new float[] {-1.0f, 0.0f, 2.0f}, values));
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

    private record UnaryCall(
            String methodName,
            UnaryElementwiseKind kind,
            Function<Tensor, Tensor> function) {
        private Tensor apply(Tensor input) {
            return function.apply(input);
        }
    }
}
