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
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
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

class TensorFloatingClassificationTest {
    private static final List<ClassificationCall> CLASSIFICATION_CALLS = List.of(
            new ClassificationCall(
                    "isFinite", FloatingClassificationKind.IS_FINITE, Tensor::isFinite),
            new ClassificationCall("isNaN", FloatingClassificationKind.IS_NAN, Tensor::isNaN),
            new ClassificationCall("isInf", FloatingClassificationKind.IS_INF, Tensor::isInf));

    @Test
    void helperAndTensorMethodsHaveExactlyTheRequiredShape() throws ReflectiveOperationException {
        int classModifiers = TensorFloatingClassifications.class.getModifiers();
        var constructors = TensorFloatingClassifications.class.getDeclaredConstructors();
        var methods = TensorFloatingClassifications.class.getDeclaredMethods();

        assertAll(
                () -> assertTrue(Modifier.isFinal(classModifiers)),
                () -> assertFalse(Modifier.isPublic(classModifiers)),
                () -> assertFalse(Modifier.isProtected(classModifiers)),
                () -> assertFalse(TensorFloatingClassifications.class.isRecord()),
                () -> assertEquals(
                        Set.of(), Set.of(TensorFloatingClassifications.class.getInterfaces())),
                () -> assertEquals(0, TensorFloatingClassifications.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorFloatingClassifications.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length));

        Method apply = TensorFloatingClassifications.class.getDeclaredMethod(
                "apply", Tensor.class, FloatingClassificationKind.class);
        assertAll(
                () -> assertEquals(apply, methods[0]),
                () -> assertEquals(Tensor.class, apply.getReturnType()),
                () -> assertTrue(Modifier.isStatic(apply.getModifiers())),
                () -> assertFalse(Modifier.isPublic(apply.getModifiers())),
                () -> assertFalse(Modifier.isProtected(apply.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(apply.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(apply.getModifiers())));

        for (ClassificationCall call : CLASSIFICATION_CALLS) {
            Method method = Tensor.class.getDeclaredMethod(call.methodName());
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(0, method.getParameterCount()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        for (String alias : List.of(
                "isInfinite", "isNotFinite", "isNormal", "isSubnormal", "isZero")) {
            assertThrows(
                    NoSuchMethodException.class,
                    () -> Tensor.class.getDeclaredMethod(alias));
        }
    }

    @Test
    void mapsEveryPublicMethodToItsExactKindAndOneInputProvenance() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), true);

        for (ClassificationCall call : CLASSIFICATION_CALLS) {
            Tensor result = call.apply(input);
            TensorProvenance provenance = result.provenance().orElseThrow();
            Operation operation = provenance.operation();

            assertAll(
                    () -> assertSame(call.kind(), operation.kind()),
                    () -> assertSame(NoOperationAttrs.INSTANCE, operation.attrs()),
                    () -> assertEquals(1, provenance.inputs().size()),
                    () -> assertSame(input, provenance.inputs().getFirst()),
                    () -> assertEquals(1, provenance.producer().outputCount()),
                    () -> assertEquals(0, provenance.outputIndex()),
                    () -> assertSame(
                            result.descriptor(),
                            provenance.producer().outputDescriptors().getFirst()));
        }
    }

    @Test
    void acceptsEveryFloatingTypeAndReturnsFixedBoolMetadataAcrossShapeStates() {
        Shape scalar = Shape.scalar();
        Shape empty = Shape.of(2, 0, 4);
        Shape ordinary = Shape.of(2, 3);
        Shape dynamic = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));

        for (DataType dataType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            for (Shape shape : List.of(scalar, empty, ordinary, dynamic)) {
                Tensor input = tensor(dataType, shape, true);
                for (ClassificationCall call : CLASSIFICATION_CALLS) {
                    Tensor result = call.apply(input);
                    assertAll(
                            () -> assertSame(DataType.BOOL, result.descriptor().dataType()),
                            () -> assertSame(shape, result.descriptor().shape()),
                            () -> assertTrue(result.descriptor().layout().isEmpty()),
                            () -> assertFalse(result.descriptor().requiresGrad()));
                }
            }
        }

        TensorDescriptor resolvedDescriptor = new TensorDescriptor(
                DataType.FLOAT64,
                ordinary,
                Optional.of(LayoutDescriptor.contiguous(ordinary)),
                true);
        Tensor resolvedInput = TensorFactory.create(resolvedDescriptor);
        Tensor resolvedResult = resolvedInput.isFinite();
        assertAll(
                () -> assertSame(ordinary, resolvedResult.descriptor().shape()),
                () -> assertTrue(resolvedResult.descriptor().layout().isEmpty()),
                () -> assertFalse(resolvedResult.descriptor().requiresGrad()));
    }

    @Test
    void everyValidCallIsFreshUnlabeledStorageFreeAndNeverCanonicalized() {
        Tensor input = TensorFactory.scalar(Double.NaN, Optional.of("input"), true);
        Tensor first = input.isNaN();
        Tensor second = input.isNaN();
        Tensor finite = input.isFinite();
        Tensor infinite = input.isInf();

        assertAll(
                () -> assertNotSame(input, first),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertNotSame(
                        first.provenance().orElseThrow().producer(),
                        finite.provenance().orElseThrow().producer()),
                () -> assertNotSame(
                        finite.provenance().orElseThrow().producer(),
                        infinite.provenance().orElseThrow().producer()),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(first.descriptor().layout().isEmpty()),
                () -> assertFalse(first.descriptor().requiresGrad()));
    }

    @Test
    void validatesNullsAndEveryNonFloatingTypeBeforeAllocatingAnIdentity()
            throws ReflectiveOperationException {
        AtomicLong nextId = nextTensorIdState();
        Tensor floating = tensor(DataType.FLOAT32, Shape.scalar(), false);
        long beforeFailures = nextId.get();

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> TensorFloatingClassifications.apply(null, null));
        NullPointerException nullKind = assertThrows(
                NullPointerException.class,
                () -> TensorFloatingClassifications.apply(floating, null));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertEquals("kind", nullKind.getMessage()),
                () -> assertEquals(beforeFailures, nextId.get()));

        for (DataType dataType : List.of(DataType.INT32, DataType.INT64, DataType.BOOL)) {
            Tensor invalid = tensor(dataType, Shape.scalar(), false);
            long beforeTypeFailure = nextId.get();
            for (ClassificationCall call : CLASSIFICATION_CALLS) {
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
    void doesNotInspectNegativeZeroInfinityOrNanStoredValues() {
        float[] values = {-1.0f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN};
        Shape shape = Shape.of(values.length);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, values.length, MemorySegment.ofArray(values));
        Tensor input = TensorFactory.create(
                descriptor, Optional.of("special-values"), Optional.of(storage));

        for (ClassificationCall call : CLASSIFICATION_CALLS) {
            Tensor result = call.apply(input);
            assertAll(
                    () -> assertTrue(result.hostStorage().isEmpty()),
                    () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                    () -> assertSame(DataType.BOOL, result.descriptor().dataType()));
        }

        assertAll(
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertArrayEquals(
                        new float[] {-1.0f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN},
                        values));
    }

    @Test
    void preservesAlreadyDerivedInputMetadataProvenanceAndStorageAssociation() {
        Shape shape = Shape.of(2);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        Tensor leaf = TensorFactory.create(descriptor);
        Tensor input = TensorFactory.createDerived(
                descriptor,
                Optional.of("derived"),
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(leaf));
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32, 2, MemorySegment.ofArray(new float[] {1.0f, 2.0f}));
        TensorProvenance inputProvenance = input.provenance().orElseThrow();
        input.replaceHostStorage(storage);

        Tensor result = input.isFinite();

        assertAll(
                () -> assertSame(descriptor, input.descriptor()),
                () -> assertSame(inputProvenance, input.provenance().orElseThrow()),
                () -> assertSame(storage, input.hostStorage().orElseThrow()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(input, result.provenance().orElseThrow().inputs().getFirst()),
                () -> assertFalse(result.descriptor().requiresGrad()));
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

    private record ClassificationCall(
            String methodName,
            FloatingClassificationKind kind,
            Function<Tensor, Tensor> function) {
        private Tensor apply(Tensor input) {
            return function.apply(input);
        }
    }
}
