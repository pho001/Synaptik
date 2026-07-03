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
        Method allocator = TensorFactory.class.getDeclaredMethod("nextTensorId");
        assertEquals(
                Set.of(convenience, complete, allocator),
                Set.of(TensorFactory.class.getDeclaredMethods()));
        assertAll(
                () -> assertEquals(Tensor.class, convenience.getReturnType()),
                () -> assertEquals(Tensor.class, complete.getReturnType()),
                () -> assertEquals(TensorId.class, allocator.getReturnType()),
                () -> assertTrue(Modifier.isPublic(convenience.getModifiers())),
                () -> assertTrue(Modifier.isPublic(complete.getModifiers())),
                () -> assertTrue(Modifier.isPrivate(allocator.getModifiers())),
                () -> assertTrue(Modifier.isStatic(convenience.getModifiers())),
                () -> assertTrue(Modifier.isStatic(complete.getModifiers())),
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
                () -> assertEquals(Optional.empty(), resolvedTensor.hostStorage()),
                () -> assertSame(unresolved, unresolvedTensor.descriptor()),
                () -> assertTrue(unresolvedTensor.descriptor().layout().isEmpty()),
                () -> assertEquals(Optional.empty(), unresolvedTensor.label()),
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
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertTrue(storage.isAlive()),
                () -> assertTrue(storage.isReadOnly()),
                () -> assertSame(storage, alias.hostStorage().orElseThrow()),
                () -> assertNotSame(tensor, alias),
                () -> assertNotEquals(tensor.id(), alias.id()));
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
}
