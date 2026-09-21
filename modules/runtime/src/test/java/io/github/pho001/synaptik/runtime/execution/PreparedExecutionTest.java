package io.github.pho001.synaptik.runtime.execution;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class PreparedExecutionTest {
    @Test
    void exposesExactPublicLifecycleOwnerSurface() {
        Class<PreparedExecution> type = PreparedExecution.class;
        List<List<Class<?>>> constructors = Arrays.stream(type.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(constructor -> List.of(constructor.getParameterTypes()))
                .sorted(java.util.Comparator.comparingInt(List::size))
                .toList();
        Class<?> lease = PreparedExecution.RunLease.class;

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.runtime.execution", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertEquals(Object.class, type.getSuperclass()),
                () -> assertArrayEquals(new Class<?>[] {AutoCloseable.class}, type.getInterfaces()),
                () -> assertEquals(
                        List.of(
                                List.of(PreparedMemoryPlan.class, PreparedSchedule.class),
                                List.of(PreparedMemoryPlan.class, PreparedSchedule.class, List.class)),
                        constructors),
                () -> assertEquals(
                        List.of("acquireRunLease", "close", "isClosed", "memoryPlan", "schedule"),
                        publicDeclaredMethodNames(type)),
                () -> assertTrue(Modifier.isPublic(lease.getModifiers())),
                () -> assertTrue(Modifier.isFinal(lease.getModifiers())),
                () -> assertTrue(Modifier.isStatic(lease.getModifiers())),
                () -> assertArrayEquals(new Class<?>[] {AutoCloseable.class}, lease.getInterfaces()),
                () -> assertTrue(Arrays.stream(lease.getDeclaredConstructors())
                        .noneMatch(constructor -> Modifier.isPublic(constructor.getModifiers()))),
                () -> assertEquals(List.of("close"), publicDeclaredMethodNames(lease)));
    }

    @Test
    void validatesConstructorInputsInExactOrderWithoutTakingOwnership() {
        PreparedMemoryPlan plan = plan(0);
        PreparedMemoryPlan equalPlan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        PreparedSchedule foreignSchedule = new PreparedSchedule(equalPlan, List.of());
        TestResource resource = new TestResource("resource", new ArrayList<>());

        assertAll(
                () -> assertFailure(NullPointerException.class, "memoryPlan", () ->
                        new PreparedExecution(null, null, null)),
                () -> assertFailure(NullPointerException.class, "schedule", () ->
                        new PreparedExecution(plan, null, null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "schedule memory plan does not match prepared execution memory plan",
                        () -> new PreparedExecution(plan, foreignSchedule, null)),
                () -> assertFailure(NullPointerException.class, "resources", () ->
                        new PreparedExecution(plan, schedule, null)),
                () -> assertFailure(NullPointerException.class, "resources[1]", () ->
                        new PreparedExecution(plan, schedule, Arrays.asList(resource, null))),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "resource is already owned by this prepared execution",
                        () -> new PreparedExecution(plan, schedule, List.of(resource, resource))),
                () -> assertEquals(0, resource.closeCount));
    }

    @Test
    void resourceUniquenessUsesIdentityAndTheSuppliedListIsSnapshotted() {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        List<String> order = new ArrayList<>();
        EqualResource first = new EqualResource("first", order);
        EqualResource second = new EqualResource("second", order);
        List<PreparedResource> resources = new ArrayList<>(List.of(first, second));

        PreparedExecution execution = new PreparedExecution(plan, schedule, resources);
        resources.clear();
        execution.close();

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertEquals(first, second),
                () -> assertEquals(List.of("second", "first"), order),
                () -> assertEquals(1, first.closeCount),
                () -> assertEquals(1, second.closeCount));
    }

    @Test
    void retainsExactRecipeAndUsesObjectIdentityRatherThanRecordEquality() {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        PreparedExecution first = new PreparedExecution(plan, schedule);
        PreparedExecution second = new PreparedExecution(plan, schedule);

        assertAll(
                () -> assertSame(plan, first.memoryPlan()),
                () -> assertSame(schedule, first.schedule()),
                () -> assertNotEquals(first, second),
                () -> assertFalse(first.toString().startsWith("PreparedExecution[")));
    }

    @Test
    void closeIsIdempotentReverseAttemptAllAndPreservesFailureIdentity() {
        PreparedMemoryPlan plan = plan(0);
        List<String> order = new ArrayList<>();
        RuntimeException primary = new RuntimeException("primary");
        RuntimeException later = new RuntimeException("later");
        TestResource first = new TestResource("first", order, later);
        TestResource second = new TestResource("second", order);
        TestResource third = new TestResource("third", order, primary);
        PreparedExecution execution = new PreparedExecution(
                plan, new PreparedSchedule(plan, List.of()), List.of(first, second, third));

        RuntimeException observed = assertThrows(RuntimeException.class, execution::close);
        execution.close();

        assertAll(
                () -> assertSame(primary, observed),
                () -> assertArrayEquals(new Throwable[] {later}, observed.getSuppressed()),
                () -> assertEquals(List.of("third", "second", "first"), order),
                () -> assertEquals(1, first.closeCount),
                () -> assertEquals(1, second.closeCount),
                () -> assertEquals(1, third.closeCount),
                () -> assertTrue(execution.isClosed()),
                () -> assertFailure(
                        IllegalStateException.class,
                        "prepared execution is closed",
                        execution::acquireRunLease));
    }

    @Test
    void repeatedPrimaryThrowableIsNotSelfSuppressedAndDoesNotStopCleanup() {
        PreparedMemoryPlan plan = plan(0);
        List<String> order = new ArrayList<>();
        Error primary = new AssertionError("primary");
        RuntimeException distinct = new RuntimeException("distinct");
        TestResource first = new TestResource("first", order);
        TestResource second = new TestResource("second", order, distinct);
        TestResource third = new TestResource("third", order, primary);
        TestResource fourth = new TestResource("fourth", order, primary);
        PreparedExecution execution = new PreparedExecution(
                plan,
                new PreparedSchedule(plan, List.of()),
                List.of(first, second, third, fourth));

        Error observed = assertThrows(Error.class, execution::close);

        assertAll(
                () -> assertSame(primary, observed),
                () -> assertArrayEquals(new Throwable[] {distinct}, observed.getSuppressed()),
                () -> assertEquals(List.of("fourth", "third", "second", "first"), order));
    }

    @Test
    void closeDoesNotWaitAndLastLeasePerformsDeferredCleanup() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        TestResource resource = new TestResource("resource", new ArrayList<>());
        PreparedExecution execution = new PreparedExecution(
                plan, new PreparedSchedule(plan, List.of()), List.of(resource));
        PreparedExecution.RunLease first = execution.acquireRunLease();
        PreparedExecution.RunLease second = execution.acquireRunLease();

        try (var pool = Executors.newSingleThreadExecutor()) {
            pool.submit(execution::close).get(5, TimeUnit.SECONDS);
        }
        first.close();
        assertAll(
                () -> assertTrue(execution.isClosed()),
                () -> assertEquals(0, resource.closeCount),
                () -> assertFailure(
                        IllegalStateException.class,
                        "prepared execution is closed",
                        execution::acquireRunLease));

        second.close();
        second.close();
        execution.close();
        assertEquals(1, resource.closeCount);
    }

    @Test
    void backendCloseRunsWithoutHoldingLifecycleMonitor() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        CountDownLatch callbackEntered = new CountDownLatch(1);
        CountDownLatch callbackRelease = new CountDownLatch(1);
        PreparedResource resource = () -> {
            callbackEntered.countDown();
            await(callbackRelease);
        };
        PreparedExecution execution = new PreparedExecution(
                plan, new PreparedSchedule(plan, List.of()), List.of(resource));

        try (var pool = Executors.newFixedThreadPool(2)) {
            var closing = pool.submit(execution::close);
            assertTrue(callbackEntered.await(5, TimeUnit.SECONDS));
            assertTrue(pool.submit(execution::isClosed).get(5, TimeUnit.SECONDS));
            ExecutionException failure = assertThrows(
                    ExecutionException.class,
                    () -> pool.submit(execution::acquireRunLease).get(5, TimeUnit.SECONDS));
            assertAll(
                    () -> assertTrue(failure.getCause() instanceof IllegalStateException),
                    () -> assertEquals("prepared execution is closed", failure.getCause().getMessage()));
            callbackRelease.countDown();
            closing.get(5, TimeUnit.SECONDS);
        }
    }

    @Test
    void concurrentCloseAndAdmissionSerializeAndCleanExactlyOnce() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        TestResource resource = new TestResource("resource", new ArrayList<>());
        PreparedExecution execution = new PreparedExecution(
                plan, new PreparedSchedule(plan, List.of()), List.of(resource));
        CountDownLatch ready = new CountDownLatch(2);
        CountDownLatch start = new CountDownLatch(1);
        AtomicReference<PreparedExecution.RunLease> admitted = new AtomicReference<>();
        AtomicReference<IllegalStateException> rejected = new AtomicReference<>();

        try (var pool = Executors.newFixedThreadPool(2)) {
            var closing = pool.submit(() -> {
                ready.countDown();
                await(start);
                execution.close();
            });
            var acquiring = pool.submit(() -> {
                ready.countDown();
                await(start);
                try {
                    admitted.set(execution.acquireRunLease());
                } catch (IllegalStateException failure) {
                    rejected.set(failure);
                }
            });
            assertTrue(ready.await(5, TimeUnit.SECONDS));
            start.countDown();
            closing.get(5, TimeUnit.SECONDS);
            acquiring.get(5, TimeUnit.SECONDS);
        }
        if (admitted.get() != null) {
            admitted.get().close();
        }

        assertAll(
                () -> assertTrue(execution.isClosed()),
                () -> assertTrue((admitted.get() == null) != (rejected.get() == null)),
                () -> assertEquals(1, resource.closeCount),
                () -> {
                    if (rejected.get() != null) {
                        assertEquals("prepared execution is closed", rejected.get().getMessage());
                    }
                });
    }

    @Test
    void immutableRecipeSupportsConcurrentReadersAndResourceFreeConstruction() throws Exception {
        PreparedMemoryPlan plan = plan(0);
        PreparedSchedule schedule = new PreparedSchedule(plan, List.of());
        PreparedExecution execution = new PreparedExecution(plan, schedule);

        try (var readers = Executors.newFixedThreadPool(2)) {
            var first = readers.submit(() -> read(execution, plan, schedule));
            var second = readers.submit(() -> read(execution, plan, schedule));
            assertAll(
                    () -> assertEquals(2_000, first.get()),
                    () -> assertEquals(2_000, second.get()),
                    () -> assertFalse(execution.isClosed()));
        }
        execution.close();
        assertTrue(execution.isClosed());
    }

    @Test
    void compiledContractContainsNoForbiddenUpstreamOrDynamicMechanismReferences()
            throws Exception {
        String compiled = classBytes(PreparedExecution.class);
        String resourceCompiled = classBytes(PreparedResource.class);

        assertAll(
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/prepare")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/planning")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/compiler")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/model")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/engine")),
                () -> assertFalse(compiled.contains("io/github/pho001/synaptik/config")),
                () -> assertFalse(compiled.contains("java/lang/reflect")),
                () -> assertFalse(compiled.contains("java/util/Map")),
                () -> assertFalse(compiled.contains("java/util/ServiceLoader")),
                () -> assertFalse(Arrays.stream(PreparedExecution.class.getDeclaredFields())
                        .anyMatch(field -> field.getType() == Object.class)),
                () -> assertEquals(List.of("close"), publicDeclaredMethodNames(PreparedResource.class)),
                () -> assertTrue(resourceCompiled.contains("java/lang/AutoCloseable")));
    }

    private static void await(CountDownLatch latch) {
        try {
            if (!latch.await(5, TimeUnit.SECONDS)) {
                throw new AssertionError("latch timed out");
            }
        } catch (InterruptedException failure) {
            Thread.currentThread().interrupt();
            throw new AssertionError(failure);
        }
    }

    private static int read(
            PreparedExecution execution,
            PreparedMemoryPlan plan,
            PreparedSchedule schedule) {
        int observed = 0;
        for (int iteration = 0; iteration < 1_000; iteration++) {
            assertSame(plan, execution.memoryPlan());
            assertSame(schedule, execution.schedule());
            observed += 2;
        }
        return observed;
    }

    private static List<String> publicDeclaredMethodNames(Class<?> type) {
        return Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .sorted()
                .toList();
    }

    private static String classBytes(Class<?> type) throws Exception {
        String resource = "/" + type.getName().replace('.', '/') + ".class";
        try (InputStream stream = type.getResourceAsStream(resource)) {
            return new String(stream.readAllBytes(), StandardCharsets.ISO_8859_1);
        }
    }

    private static PreparedMemoryPlan plan(int bufferCount) {
        var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        for (int index = 0; index < bufferCount; index++) {
            buffers.add(new PreparedMemoryPlan.BufferEntry(
                    new BufferSlot(100L + index), index, 1L));
        }
        return new PreparedMemoryPlan(buffers, List.of());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> failureType, String message, Runnable action) {
        T failure = assertThrows(failureType, action::run);
        assertEquals(message, failure.getMessage());
    }

    private static class TestResource implements PreparedResource {
        private final String name;
        private final List<String> order;
        private final Throwable failure;
        int closeCount;

        private TestResource(String name, List<String> order) {
            this(name, order, null);
        }

        private TestResource(String name, List<String> order, Throwable failure) {
            this.name = name;
            this.order = order;
            this.failure = failure;
        }

        @Override
        public void close() {
            closeCount++;
            order.add(name);
            if (failure instanceof RuntimeException runtimeFailure) {
                throw runtimeFailure;
            }
            if (failure instanceof Error error) {
                throw error;
            }
        }
    }

    private static final class EqualResource extends TestResource {
        private EqualResource(String name, List<String> order) {
            super(name, order);
        }

        @Override
        public boolean equals(Object ignored) {
            return true;
        }

        @Override
        public int hashCode() {
            return 1;
        }
    }
}
