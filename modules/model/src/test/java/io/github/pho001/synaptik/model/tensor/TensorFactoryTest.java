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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class TensorFactoryTest {
    @Test
    void hasExactlyTheRequiredStaticUtilityShape() throws ReflectiveOperationException {
        assertAll(
                () -> assertTrue(Modifier.isPublic(TensorFactory.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorFactory.class.getModifiers())),
                () -> assertFalse(TensorFactory.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorFactory.class.getInterfaces())));

        Field[] fields = TensorFactory.class.getDeclaredFields();
        assertEquals(
                List.of("NEXT_TENSOR_ID", "MAXIMUM_TENSOR_ID_CLAIMED"),
                Arrays.stream(fields).map(Field::getName).toList());
        assertAll(
                () -> assertEquals(AtomicLong.class, fields[0].getType()),
                () -> assertEquals(AtomicBoolean.class, fields[1].getType()),
                () -> assertTrue(Arrays.stream(fields).allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isStatic(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))));

        var constructors = TensorFactory.class.getDeclaredConstructors();
        assertEquals(1, constructors.length);
        assertAll(
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()));

        Method convenience = TensorFactory.class.getDeclaredMethod(
                "create", TensorDescriptor.class);
        Method complete = TensorFactory.class.getDeclaredMethod(
                "create", TensorDescriptor.class, Optional.class, Optional.class);
        Method allocateConvenience = TensorFactory.class.getDeclaredMethod(
                "allocate", TensorDescriptor.class);
        Method allocateComplete = TensorFactory.class.getDeclaredMethod(
                "allocate", TensorDescriptor.class, Optional.class);
        Method importFloat64 = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, double[].class);
        Method importFloat32 = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, float[].class);
        Method importBfloat16 = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, short[].class);
        Method importInt32 = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, int[].class);
        Method importInt64 = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, long[].class);
        Method importBool = TensorFactory.class.getDeclaredMethod(
                "fromFlatArray", TensorDescriptor.class, Optional.class, byte[].class);
        Method importNested = TensorFactory.class.getDeclaredMethod(
                "fromNestedArray", Object.class, Optional.class, boolean.class);
        Method scalarFloat64 = TensorFactory.class.getDeclaredMethod(
                "scalar", double.class, Optional.class, boolean.class);
        Method scalarFloat32 = TensorFactory.class.getDeclaredMethod(
                "scalar", float.class, Optional.class, boolean.class);
        Method scalarBfloat16 = TensorFactory.class.getDeclaredMethod(
                "scalarBFloat16", float.class, Optional.class, boolean.class);
        Method scalarInt32 = TensorFactory.class.getDeclaredMethod(
                "scalar", int.class, Optional.class, boolean.class);
        Method scalarInt64 = TensorFactory.class.getDeclaredMethod(
                "scalar", long.class, Optional.class, boolean.class);
        Method scalarBool = TensorFactory.class.getDeclaredMethod(
                "scalar", boolean.class, Optional.class, boolean.class);
        Method zeros = TensorFactory.class.getDeclaredMethod(
                "zeros", Shape.class, DataType.class, Optional.class, boolean.class);
        Method ones = TensorFactory.class.getDeclaredMethod(
                "ones", Shape.class, DataType.class, Optional.class, boolean.class);
        Method zerosLike = TensorFactory.class.getDeclaredMethod(
                "zerosLike", Tensor.class, Optional.class, boolean.class);
        Method onesLike = TensorFactory.class.getDeclaredMethod(
                "onesLike", Tensor.class, Optional.class, boolean.class);
        Method rangeInt32 = TensorFactory.class.getDeclaredMethod(
                "range", int.class, int.class, int.class, Optional.class);
        Method rangeInt64 = TensorFactory.class.getDeclaredMethod(
                "range", long.class, long.class, long.class, Optional.class);
        Method fullFloat64 = TensorFactory.class.getDeclaredMethod(
                "full", Shape.class, double.class, Optional.class, boolean.class);
        Method fullFloat32 = TensorFactory.class.getDeclaredMethod(
                "full", Shape.class, float.class, Optional.class, boolean.class);
        Method fullBfloat16 = TensorFactory.class.getDeclaredMethod(
                "fullBFloat16", Shape.class, float.class, Optional.class, boolean.class);
        Method fullInt32 = TensorFactory.class.getDeclaredMethod(
                "full", Shape.class, int.class, Optional.class, boolean.class);
        Method fullInt64 = TensorFactory.class.getDeclaredMethod(
                "full", Shape.class, long.class, Optional.class, boolean.class);
        Method fullBool = TensorFactory.class.getDeclaredMethod(
                "full", Shape.class, boolean.class, Optional.class, boolean.class);
        Method identityMatrix = TensorFactory.class.getDeclaredMethod(
                "identityMatrix",
                long.class,
                long.class,
                DataType.class,
                Optional.class,
                boolean.class);
        Method eye = TensorFactory.class.getDeclaredMethod(
                "eye",
                long.class,
                long.class,
                DataType.class,
                Optional.class,
                boolean.class);
        Method importFlat = TensorFactory.class.getDeclaredMethod(
                "importFlat",
                TensorDescriptor.class,
                Optional.class,
                DataType.class,
                int.class,
                MemorySegment.class);
        Method createDerived = TensorFactory.class.getDeclaredMethod(
                "createDerived",
                TensorDescriptor.class,
                Optional.class,
                Operation.class,
                List.class);
        Method createDerivedOutputs = TensorFactory.class.getDeclaredMethod(
                "createDerivedOutputs", Operation.class, List.class, List.class);
        Method allocator = TensorFactory.class.getDeclaredMethod("nextTensorId");
        assertEquals(35, TensorFactory.class.getDeclaredMethods().length);
        assertEquals(
                31,
                Arrays.stream(TensorFactory.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count());
        assertEquals(
                Set.of(
                        convenience,
                        complete,
                        allocateConvenience,
                        allocateComplete,
                        importFloat64,
                        importFloat32,
                        importBfloat16,
                        importInt32,
                        importInt64,
                        importBool,
                        importNested,
                        scalarFloat64,
                        scalarFloat32,
                        scalarBfloat16,
                        scalarInt32,
                        scalarInt64,
                        scalarBool,
                        zeros,
                        ones,
                        zerosLike,
                        onesLike,
                        rangeInt32,
                        rangeInt64,
                        fullFloat64,
                        fullFloat32,
                        fullBfloat16,
                        fullInt32,
                        fullInt64,
                        fullBool,
                        identityMatrix,
                        eye,
                        importFlat,
                        createDerived,
                        createDerivedOutputs,
                        allocator),
                Set.of(TensorFactory.class.getDeclaredMethods()));
        assertAll(
                () -> assertEquals(Tensor.class, convenience.getReturnType()),
                () -> assertEquals(Tensor.class, complete.getReturnType()),
                () -> assertEquals(Tensor.class, allocateConvenience.getReturnType()),
                () -> assertEquals(Tensor.class, allocateComplete.getReturnType()),
                () -> assertEquals(Tensor.class, importFloat64.getReturnType()),
                () -> assertEquals(Tensor.class, importFloat32.getReturnType()),
                () -> assertEquals(Tensor.class, importBfloat16.getReturnType()),
                () -> assertEquals(Tensor.class, importInt32.getReturnType()),
                () -> assertEquals(Tensor.class, importInt64.getReturnType()),
                () -> assertEquals(Tensor.class, importBool.getReturnType()),
                () -> assertEquals(Tensor.class, importNested.getReturnType()),
                () -> assertEquals(Tensor.class, scalarFloat64.getReturnType()),
                () -> assertEquals(Tensor.class, scalarFloat32.getReturnType()),
                () -> assertEquals(Tensor.class, scalarBfloat16.getReturnType()),
                () -> assertEquals(Tensor.class, scalarInt32.getReturnType()),
                () -> assertEquals(Tensor.class, scalarInt64.getReturnType()),
                () -> assertEquals(Tensor.class, scalarBool.getReturnType()),
                () -> assertEquals(Tensor.class, zeros.getReturnType()),
                () -> assertEquals(Tensor.class, ones.getReturnType()),
                () -> assertEquals(Tensor.class, zerosLike.getReturnType()),
                () -> assertEquals(Tensor.class, onesLike.getReturnType()),
                () -> assertEquals(Tensor.class, rangeInt32.getReturnType()),
                () -> assertEquals(Tensor.class, rangeInt64.getReturnType()),
                () -> assertTrue(List.of(
                                fullFloat64,
                                fullFloat32,
                                fullBfloat16,
                                fullInt32,
                                fullInt64,
                                fullBool,
                                identityMatrix,
                                eye)
                        .stream()
                        .allMatch(method -> method.getReturnType() == Tensor.class)),
                () -> assertEquals(Tensor.class, importFlat.getReturnType()),
                () -> assertEquals(Tensor.class, createDerived.getReturnType()),
                () -> assertEquals(List.class, createDerivedOutputs.getReturnType()),
                () -> assertEquals(TensorId.class, allocator.getReturnType()),
                () -> assertTrue(Modifier.isPublic(convenience.getModifiers())),
                () -> assertTrue(Modifier.isPublic(complete.getModifiers())),
                () -> assertTrue(Modifier.isPublic(allocateConvenience.getModifiers())),
                () -> assertTrue(Modifier.isPublic(allocateComplete.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importFloat64.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importFloat32.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importBfloat16.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importInt32.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importInt64.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importBool.getModifiers())),
                () -> assertTrue(Modifier.isPublic(importNested.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarFloat64.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarFloat32.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarBfloat16.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarInt32.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarInt64.getModifiers())),
                () -> assertTrue(Modifier.isPublic(scalarBool.getModifiers())),
                () -> assertTrue(Modifier.isPublic(zeros.getModifiers())),
                () -> assertTrue(Modifier.isPublic(ones.getModifiers())),
                () -> assertTrue(Modifier.isPublic(zerosLike.getModifiers())),
                () -> assertTrue(Modifier.isPublic(onesLike.getModifiers())),
                () -> assertTrue(Modifier.isPublic(rangeInt32.getModifiers())),
                () -> assertTrue(Modifier.isPublic(rangeInt64.getModifiers())),
                () -> assertTrue(List.of(
                                fullFloat64,
                                fullFloat32,
                                fullBfloat16,
                                fullInt32,
                                fullInt64,
                                fullBool,
                                identityMatrix,
                                eye)
                        .stream()
                        .allMatch(method -> Modifier.isPublic(method.getModifiers()))),
                () -> assertTrue(Modifier.isPrivate(importFlat.getModifiers())),
                () -> assertFalse(Modifier.isPublic(createDerived.getModifiers())),
                () -> assertFalse(Modifier.isProtected(createDerived.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(createDerived.getModifiers())),
                () -> assertFalse(Modifier.isPublic(createDerivedOutputs.getModifiers())),
                () -> assertFalse(Modifier.isProtected(createDerivedOutputs.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(createDerivedOutputs.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(allocator.getModifiers())),
                () -> assertTrue(Modifier.isStatic(convenience.getModifiers())),
                () -> assertTrue(Modifier.isStatic(complete.getModifiers())),
                () -> assertTrue(Modifier.isStatic(allocateConvenience.getModifiers())),
                () -> assertTrue(Modifier.isStatic(allocateComplete.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importFloat64.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importFloat32.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importBfloat16.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importInt32.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importInt64.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importBool.getModifiers())),
                () -> assertTrue(Modifier.isStatic(importNested.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarFloat64.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarFloat32.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarBfloat16.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarInt32.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarInt64.getModifiers())),
                () -> assertTrue(Modifier.isStatic(scalarBool.getModifiers())),
                () -> assertTrue(Modifier.isStatic(zeros.getModifiers())),
                () -> assertTrue(Modifier.isStatic(ones.getModifiers())),
                () -> assertTrue(Modifier.isStatic(zerosLike.getModifiers())),
                () -> assertTrue(Modifier.isStatic(onesLike.getModifiers())),
                () -> assertTrue(Modifier.isStatic(rangeInt32.getModifiers())),
                () -> assertTrue(Modifier.isStatic(rangeInt64.getModifiers())),
                () -> assertTrue(List.of(
                                fullFloat64,
                                fullFloat32,
                                fullBfloat16,
                                fullInt32,
                                fullInt64,
                                fullBool,
                                identityMatrix,
                                eye)
                        .stream()
                        .allMatch(method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(Modifier.isStatic(importFlat.getModifiers())),
                () -> assertTrue(Modifier.isStatic(createDerived.getModifiers())),
                () -> assertTrue(Modifier.isStatic(createDerivedOutputs.getModifiers())),
                () -> assertTrue(Modifier.isStatic(allocator.getModifiers())));
    }

    @Test
    void createsStorageFreeTensorsWithoutChangingResolvedOrUnresolvedDescriptors() {
        TensorDescriptor resolved = resolved(
                DataType.FLOAT32,
                Shape.of(2, 3),
                LayoutDescriptor.contiguous(Shape.of(2, 3)));
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        TensorDescriptor unresolved = unresolved(DataType.FLOAT32, dynamicShape);

        Tensor resolvedTensor = TensorFactory.create(resolved);
        Tensor unresolvedTensor = TensorFactory.create(unresolved);

        assertAll(
                () -> assertSame(resolved, resolvedTensor.descriptor()),
                () -> assertEquals(Optional.empty(), resolvedTensor.label()),
                () -> assertEquals(Optional.empty(), resolvedTensor.provenance()),
                () -> assertEquals(Optional.empty(), resolvedTensor.hostStorage()),
                () -> assertSame(unresolved, unresolvedTensor.descriptor()),
                () -> assertTrue(unresolvedTensor.descriptor().layout().isEmpty()),
                () -> assertEquals(Optional.empty(), unresolvedTensor.label()),
                () -> assertEquals(Optional.empty(), unresolvedTensor.provenance()),
                () -> assertEquals(Optional.empty(), unresolvedTensor.hostStorage()),
                () -> assertNotSame(resolvedTensor, unresolvedTensor),
                () -> assertNotEquals(resolvedTensor.id(), unresolvedTensor.id()));
    }

    @Test
    void createsFullTensorWithNormalizedLabelAndExactBorrowedStorageReference() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor descriptor = resolved(
                DataType.FLOAT32, shape, LayoutDescriptor.contiguous(shape));
        HostTensorStorage storage = new MemorySegmentStorage(
                DataType.FLOAT32,
                6,
                MemorySegment.ofArray(new byte[24]).asReadOnly());

        Tensor tensor = TensorFactory.create(
                descriptor, Optional.of("  weights\n"), Optional.of(storage));
        Tensor alias = TensorFactory.create(
                descriptor, Optional.empty(), Optional.of(storage));

        assertAll(
                () -> assertSame(descriptor, tensor.descriptor()),
                () -> assertEquals(Optional.of("weights"), tensor.label()),
                () -> assertEquals(Optional.empty(), tensor.provenance()),
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertTrue(storage.isAlive()),
                () -> assertTrue(storage.isReadOnly()),
                () -> assertSame(storage, alias.hostStorage().orElseThrow()),
                () -> assertNotSame(tensor, alias),
                () -> assertNotEquals(tensor.id(), alias.id()));
    }

    @Test
    void createsDerivedTensorWithExactProvenanceAndRequiredFailureEffects()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());
        Tensor input = TensorFactory.create(descriptor);
        Operation operation = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);
        List<Tensor> inputs = List.of(input);
        long afterInput = next.get();

        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerived(null, null, null, null));
        NullPointerException nullLabel = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerived(descriptor, null, null, null));
        NullPointerException nullOperation = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerived(descriptor, Optional.empty(), null, null));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerived(descriptor, Optional.empty(), operation, null));
        IllegalArgumentException wrongInputs = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.createDerived(
                        descriptor, Optional.empty(), operation, List.of()));
        assertEquals(afterInput, next.get());

        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.createDerived(
                        descriptor, Optional.of(" \t\n "), operation, inputs));
        assertEquals(afterInput + 1, next.get());

        Tensor derived = TensorFactory.createDerived(
                descriptor, Optional.of("  result  "), operation, inputs);
        TensorProvenance provenance = derived.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(before, input.id().value()),
                () -> assertEquals("descriptor", nullDescriptor.getMessage()),
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals("operation", nullOperation.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertTrue(wrongInputs.getMessage().contains("input count 0")),
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(afterInput + 1, derived.id().value()),
                () -> assertSame(descriptor, derived.descriptor()),
                () -> assertEquals(Optional.of("result"), derived.label()),
                () -> assertSame(provenance, derived.provenance().orElseThrow()),
                () -> assertSame(operation, provenance.operation()),
                () -> assertSame(input, provenance.inputs().getFirst()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(descriptor, provenance.outputDescriptor()),
                () -> assertTrue(derived.hostStorage().isEmpty()));
    }

    @Test
    void createsImmutableOrderedOutputsWithOneExactSharedProducer()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        TensorDescriptor firstDescriptor = unresolved(DataType.FLOAT32, Shape.of(2));
        TensorDescriptor secondDescriptor = unresolved(DataType.INT64, Shape.of(2));
        Tensor input = TensorFactory.create(firstDescriptor);
        Operation operation = new Operation(MultiKind.MULTI, NoOperationAttrs.INSTANCE);
        long beforeOutputs = next.get();

        List<Tensor> outputs = TensorFactory.createDerivedOutputs(
                operation,
                List.of(input),
                List.of(firstDescriptor, secondDescriptor));
        Tensor first = outputs.get(0);
        Tensor second = outputs.get(1);
        TensorProvenance firstProvenance = first.provenance().orElseThrow();
        TensorProvenance secondProvenance = second.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(2, outputs.size()),
                () -> assertThrows(UnsupportedOperationException.class, outputs::clear),
                () -> assertNotSame(first, second),
                () -> assertEquals(beforeOutputs, first.id().value()),
                () -> assertEquals(beforeOutputs + 1, second.id().value()),
                () -> assertSame(firstDescriptor, first.descriptor()),
                () -> assertSame(secondDescriptor, second.descriptor()),
                () -> assertTrue(first.label().isEmpty()),
                () -> assertTrue(second.label().isEmpty()),
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertTrue(second.hostStorage().isEmpty()),
                () -> assertSame(firstProvenance.producer(), secondProvenance.producer()),
                () -> assertSame(operation, firstProvenance.operation()),
                () -> assertSame(input, firstProvenance.inputs().getFirst()),
                () -> assertEquals(0, firstProvenance.outputIndex()),
                () -> assertEquals(1, secondProvenance.outputIndex()),
                () -> assertSame(firstDescriptor, firstProvenance.outputDescriptor()),
                () -> assertSame(secondDescriptor, secondProvenance.outputDescriptor()));
    }

    @Test
    void multiOutputPrevalidationFailuresConsumeNoIdentifiers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());
        Tensor input = TensorFactory.create(descriptor);
        Operation operation = new Operation(MultiKind.MULTI, NoOperationAttrs.INSTANCE);
        long before = next.get();

        NullPointerException nullOperation = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerivedOutputs(null, null, null));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerivedOutputs(operation, null, null));
        NullPointerException nullOutputs = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.createDerivedOutputs(operation, List.of(input), null));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.createDerivedOutputs(operation, List.of(input), List.of()));
        IllegalArgumentException wrongCount = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.createDerivedOutputs(
                        operation, List.of(input), List.of(descriptor)));

        assertAll(
                () -> assertEquals("operation", nullOperation.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertEquals("outputDescriptors", nullOutputs.getMessage()),
                () -> assertEquals("outputDescriptors must not be empty", empty.getMessage()),
                () -> assertTrue(wrongCount.getMessage().contains("output count 1")),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void rejectsNullArgumentsInOrderWithoutConsumingIdentifiers() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());

        NullPointerException convenience = assertThrows(
                NullPointerException.class, () -> TensorFactory.create(null));
        NullPointerException descriptorFailure = assertThrows(
                NullPointerException.class, () -> TensorFactory.create(null, null, null));
        NullPointerException labelFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.create(descriptor, null, null));
        NullPointerException storageFailure = assertThrows(
                NullPointerException.class,
                () -> TensorFactory.create(descriptor, Optional.empty(), null));

        assertAll(
                () -> assertEquals("descriptor", convenience.getMessage()),
                () -> assertEquals("descriptor", descriptorFailure.getMessage()),
                () -> assertEquals("label", labelFailure.getMessage()),
                () -> assertEquals("hostStorage", storageFailure.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void delegatedSemanticFailuresKeepTensorOrderAndConsumeIdentifiers()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        TensorDescriptor unresolved = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        Shape shape = Shape.of(2, 3);
        TensorDescriptor resolved = resolved(
                DataType.FLOAT32, shape, LayoutDescriptor.contiguous(shape));

        Arena arena = Arena.ofConfined();
        HostTensorStorage wrongDead = new MemorySegmentStorage(
                DataType.INT32, 6, arena.allocate(24, 1));
        HostTensorStorage smallDead = new MemorySegmentStorage(
                DataType.FLOAT32, 5, arena.allocate(20, 1));
        HostTensorStorage dead = new MemorySegmentStorage(
                DataType.FLOAT32, 0, arena.allocate(0, 1));
        arena.close();

        long beforeBlank = next.get();
        IllegalArgumentException blank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.create(
                        unresolved, Optional.of(" \t\n "), Optional.empty()));
        assertEquals(beforeBlank + 1, next.get());

        long beforeType = next.get();
        IllegalArgumentException wrongType = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.create(
                        resolved,
                        Optional.empty(),
                        Optional.of(wrongDead)));
        assertEquals(beforeType + 1, next.get());

        long beforeCapacity = next.get();
        IllegalArgumentException tooSmall = assertThrows(
                IllegalArgumentException.class,
                () -> TensorFactory.create(
                        resolved,
                        Optional.empty(),
                        Optional.of(smallDead)));
        assertEquals(beforeCapacity + 1, next.get());

        long beforeLiveness = next.get();
        IllegalStateException notAlive = assertThrows(
                IllegalStateException.class,
                () -> TensorFactory.create(
                        unresolved, Optional.empty(), Optional.of(dead)));

        assertAll(
                () -> assertEquals("label must not be blank", blank.getMessage()),
                () -> assertEquals(
                        "hostStorage data type must match descriptor data type: expected=FLOAT32, actual=INT32",
                        wrongType.getMessage()),
                () -> assertEquals(
                        "hostStorage element capacity is smaller than resolved layout span: required=6, actual=5",
                        tooSmall.getMessage()),
                () -> assertEquals("hostStorage must be alive when attached", notAlive.getMessage()),
                () -> assertEquals(beforeLiveness + 1, next.get()));

        Tensor success = TensorFactory.create(unresolved);
        assertEquals(beforeLiveness + 1, success.id().value());
    }

    @Test
    void concurrentOrdinaryCreationReturnsDistinctObjectsAndIdentifiers() throws Exception {
        int taskCount = 128;
        int workerCount = 8;
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        CountDownLatch ready = new CountDownLatch(workerCount);
        CountDownLatch start = new CountDownLatch(1);
        ExecutorService executor = Executors.newFixedThreadPool(workerCount);
        List<Future<Tensor>> futures = new ArrayList<>();
        try {
            for (int index = 0; index < taskCount; index++) {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    return TensorFactory.create(descriptor);
                }));
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS));
            start.countDown();

            Set<Tensor> tensors = Collections.newSetFromMap(new IdentityHashMap<>());
            Set<TensorId> ids = new HashSet<>();
            for (Future<Tensor> future : futures) {
                Tensor tensor = future.get(10, TimeUnit.SECONDS);
                tensors.add(tensor);
                ids.add(tensor.id());
            }
            assertAll(
                    () -> assertEquals(taskCount, tensors.size()),
                    () -> assertEquals(taskCount, ids.size()));
        } finally {
            start.countDown();
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    @Test
    void multiOutputExhaustionConsumesAllocatedIdsAndReturnsNoPartialList()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalMaximumClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());
            Tensor input = new Tensor(
                    new TensorId(0),
                    descriptor,
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty());
            Operation operation = new Operation(MultiKind.MULTI, NoOperationAttrs.INSTANCE);

            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.createDerivedOutputs(
                            operation,
                            List.of(input),
                            List.of(descriptor, descriptor)));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(originalNext);
            maximumClaimed.set(originalMaximumClaimed);
        }
    }

    @Test
    void finalCandidateIsClaimedOnceAndExhaustionIsPermanent() throws Exception {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalMaximumClaimed = maximumClaimed.get();
        int workerCount = 8;
        ExecutorService executor = Executors.newFixedThreadPool(workerCount);
        CountDownLatch ready = new CountDownLatch(workerCount);
        CountDownLatch start = new CountDownLatch(1);
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(false);
            TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());
            List<Future<Tensor>> futures = new ArrayList<>();
            for (int index = 0; index < 16; index++) {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();
                    return TensorFactory.create(descriptor);
                }));
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS));
            start.countDown();

            List<Tensor> successes = new ArrayList<>();
            List<Throwable> failures = new ArrayList<>();
            for (Future<Tensor> future : futures) {
                try {
                    successes.add(future.get(10, TimeUnit.SECONDS));
                } catch (ExecutionException exception) {
                    failures.add(exception.getCause());
                }
            }

            assertEquals(1, successes.size());
            assertEquals(Long.MAX_VALUE, successes.getFirst().id().value());
            assertEquals(15, failures.size());
            assertTrue(failures.stream().allMatch(failure ->
                    failure instanceof IllegalStateException
                            && "tensor identifier space exhausted".equals(failure.getMessage())));
            assertAll(
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));

            for (int attempt = 0; attempt < 3; attempt++) {
                IllegalStateException exhausted = assertThrows(
                        IllegalStateException.class, () -> TensorFactory.create(descriptor));
                assertEquals("tensor identifier space exhausted", exhausted.getMessage());
            }

            Operation operation = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);
            Tensor input = new Tensor(
                    new TensorId(0),
                    descriptor,
                    Optional.empty(),
                    Optional.empty(),
                    Optional.empty());
            IllegalStateException derivedExhausted = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.createDerived(
                            descriptor, Optional.of("derived"), operation, List.of(input)));

            NullPointerException nullWins = assertThrows(
                    NullPointerException.class,
                    () -> TensorFactory.create(null, Optional.of(" "), Optional.empty()));
            IllegalStateException exhaustionWins = assertThrows(
                    IllegalStateException.class,
                    () -> TensorFactory.create(
                            descriptor, Optional.of(" "), Optional.empty()));
            assertAll(
                    () -> assertEquals("descriptor", nullWins.getMessage()),
                    () -> assertEquals("tensor identifier space exhausted", exhaustionWins.getMessage()),
                    () -> assertEquals(
                            "tensor identifier space exhausted", derivedExhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            start.countDown();
            executor.shutdownNow();
            try {
                assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
            } finally {
                next.set(originalNext);
                maximumClaimed.set(originalMaximumClaimed);
            }
        }
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(
            DataType dataType, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }

    private static HostTensorStorage storage(DataType dataType, long capacity) {
        long byteSize = Math.multiplyExact(capacity, dataType.byteWidth());
        return new MemorySegmentStorage(
                dataType,
                capacity,
                MemorySegment.ofArray(new byte[Math.toIntExact(byteSize)]));
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private enum MultiKind implements OperationKind {
        MULTI;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 2));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
