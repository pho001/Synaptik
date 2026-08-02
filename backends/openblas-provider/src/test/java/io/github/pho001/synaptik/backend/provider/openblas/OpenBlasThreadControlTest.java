package io.github.pho001.synaptik.backend.provider.openblas;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.WrongMethodTypeException;
import java.lang.reflect.Modifier;
import java.lang.reflect.Method;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Verifies direct thread-count validation, invocation, failure, and shared-state behavior. */
final class OpenBlasThreadControlTest {
    private static final MethodHandle EMPTY = MethodHandles.empty(MethodType.methodType(void.class));

    /** Locks the helper as one package-private final field-free stateless boundary. */
    @Test
    void helperShapeIsExact() {
        assertTrue(Modifier.isFinal(OpenBlasThreadControl.class.getModifiers()));
        assertFalse(Modifier.isPublic(OpenBlasThreadControl.class.getModifiers()));
        assertEquals(0, OpenBlasThreadControl.class.getDeclaredFields().length);
        assertEquals(
                java.util.Set.of("threadCount(OpenBlasNativeBindings):int", "setThreadCount(OpenBlasNativeBindings,int):void"),
                Arrays.stream(OpenBlasThreadControl.class.getDeclaredMethods())
                        .filter(method -> !method.isSynthetic())
                        .map(method -> method.getName() + "(" + Arrays.stream(method.getParameterTypes())
                                .map(Class::getSimpleName).collect(java.util.stream.Collectors.joining(","))
                                + "):" + method.getReturnType().getSimpleName())
                        .collect(java.util.stream.Collectors.toUnmodifiableSet()));
    }

    /** Returns exact positive values and forwards positive setter boundaries unchanged. */
    @Test
    void returnsAndForwardsExactPositiveValues() {
        AtomicInteger cell = new AtomicInteger(1);
        try (OpenBlasLibrary library = library(cell)) {
            assertEquals(1, library.threadCount());
            library.setThreadCount(Integer.MAX_VALUE);
            assertEquals(Integer.MAX_VALUE, cell.get());
            assertEquals(Integer.MAX_VALUE, library.threadCount());
        }
    }

    /** Rejects every selected non-positive setter value without invoking the native handle. */
    @Test
    void rejectsNonPositiveSetterValuesBeforeInvocation() {
        AtomicInteger calls = new AtomicInteger();
        MethodHandle setter = findStatic("countSetter", MethodType.methodType(void.class, AtomicInteger.class, int.class))
                .bindTo(calls);
        try (OpenBlasLibrary library = library(MethodHandles.constant(int.class, 1), setter)) {
            for (int value : new int[] {Integer.MIN_VALUE, -1, 0}) {
                IllegalArgumentException failure = assertThrows(
                        IllegalArgumentException.class, () -> library.setThreadCount(value));
                assertEquals("threadCount must be positive: " + value, failure.getMessage());
            }
            assertEquals(0, calls.get());
        }
    }

    /** Rejects zero and negative native getter results after one exact invocation. */
    @Test
    void rejectsNonPositiveGetterResults() {
        for (int value : new int[] {0, -7}) {
            AtomicInteger calls = new AtomicInteger();
            MethodHandle getter = MethodHandles.insertArguments(
                    findStatic("countingGetter", MethodType.methodType(int.class, AtomicInteger.class, int.class))
                            .bindTo(calls),
                    0,
                    value);
            try (OpenBlasLibrary library = library(getter, MethodHandles.empty(MethodType.methodType(void.class, int.class)))) {
                IllegalStateException failure = assertThrows(IllegalStateException.class, library::threadCount);
                assertEquals("OpenBLAS returned non-positive thread count: " + value, failure.getMessage());
                assertEquals(1, calls.get());
            }
        }
    }

    /** Preserves closed-owner precedence over setter argument validation and performs no restore. */
    @Test
    void closedOwnerFailsFirstAndCloseDoesNotRestore() {
        AtomicInteger cell = new AtomicInteger(3);
        OpenBlasLibrary library = library(cell);
        library.setThreadCount(8);
        library.close();
        assertEquals(8, cell.get());
        assertEquals("OpenBLAS library is closed", assertThrows(
                IllegalStateException.class, () -> library.setThreadCount(0)).getMessage());
        assertEquals("OpenBLAS library is closed", assertThrows(
                IllegalStateException.class, library::threadCount).getMessage());
    }

    /** Wraps all non-Error invocation failures and preserves every Error unchanged. */
    @Test
    void translatesInvocationFailuresExactly() {
        for (Throwable cause : new Throwable[] {new Exception("checked"), new IllegalArgumentException("runtime")}) {
            try (OpenBlasLibrary library = library(throwing(int.class, cause), normalSetter())) {
                IllegalStateException failure = assertThrows(IllegalStateException.class, library::threadCount);
                assertEquals("OpenBLAS get thread count invocation failed", failure.getMessage());
                assertSame(cause, failure.getCause());
            }
            try (OpenBlasLibrary library = library(MethodHandles.constant(int.class, 1), throwing(void.class, cause, int.class))) {
                IllegalStateException failure = assertThrows(
                        IllegalStateException.class, () -> library.setThreadCount(2));
                assertEquals("OpenBLAS set thread count invocation failed", failure.getMessage());
                assertSame(cause, failure.getCause());
            }
        }

        AssertionError getterError = new AssertionError("getter");
        try (OpenBlasLibrary library = library(throwing(int.class, getterError), normalSetter())) {
            assertSame(getterError, assertThrows(AssertionError.class, library::threadCount));
        }
        AssertionError setterError = new AssertionError("setter");
        try (OpenBlasLibrary library = library(MethodHandles.constant(int.class, 1), throwing(void.class, setterError, int.class))) {
            assertSame(setterError, assertThrows(AssertionError.class, () -> library.setThreadCount(2)));
        }
    }

    /** Wrong handle types fail through the same stable invocation boundary. */
    @Test
    void wrongMethodTypesAreWrapped() {
        try (OpenBlasLibrary library = library(MethodHandles.constant(long.class, 1L), normalSetter())) {
            assertTrue(assertThrows(IllegalStateException.class, library::threadCount).getCause()
                    instanceof WrongMethodTypeException);
        }
        try (OpenBlasLibrary library = library(MethodHandles.constant(int.class, 1), EMPTY)) {
            assertTrue(assertThrows(IllegalStateException.class, () -> library.setThreadCount(2)).getCause()
                    instanceof WrongMethodTypeException);
        }
    }

    /** Two owners observe one fake global cell without caching per-owner state. */
    @Test
    void ownersObserveOneSharedCellSequentiallyAndConcurrently() throws Exception {
        AtomicInteger cell = new AtomicInteger(2);
        try (OpenBlasLibrary first = library(cell); OpenBlasLibrary second = library(cell)) {
            first.setThreadCount(5);
            assertEquals(5, second.threadCount());
            second.setThreadCount(7);
            assertEquals(7, first.threadCount());

            CountDownLatch start = new CountDownLatch(1);
            try (var executor = Executors.newFixedThreadPool(2)) {
                var one = executor.submit(() -> { start.await(); first.setThreadCount(11); return first.threadCount(); });
                var two = executor.submit(() -> { start.await(); second.setThreadCount(13); return second.threadCount(); });
                start.countDown();
                one.get(10, TimeUnit.SECONDS);
                two.get(10, TimeUnit.SECONDS);
            }
            assertTrue(cell.get() == 11 || cell.get() == 13);
            assertEquals(cell.get(), first.threadCount());
            assertEquals(cell.get(), second.threadCount());
        }
    }

    /** Validates checkpoint arguments before any native lookup is attempted. */
    @Test
    void checkpointRequiresExactlyOneAbsolutePath() {
        IllegalArgumentException countFailure = assertThrows(
                IllegalArgumentException.class, () -> OpenBlasNativeCheckpoint.main(new String[0]));
        assertEquals("expected exactly one absolute OpenBLAS library path argument", countFailure.getMessage());

        Path relative = Path.of("relative/libopenblas.so");
        IllegalArgumentException pathFailure = assertThrows(
                IllegalArgumentException.class,
                () -> OpenBlasNativeCheckpoint.main(new String[] {relative.toString()}));
        assertEquals("OpenBLAS library path must be absolute: " + relative, pathFailure.getMessage());
    }

    /** Preserves a primary failure, suppresses a distinct restoration failure, and avoids self-suppression. */
    @Test
    void checkpointSuppressionRetainsPrimaryIdentity() throws Exception {
        Method suppress = OpenBlasNativeCheckpoint.class.getDeclaredMethod(
                "suppressDistinct", Throwable.class, Throwable.class);
        suppress.setAccessible(true);
        RuntimeException primary = new RuntimeException("primary");
        RuntimeException restoration = new RuntimeException("restoration");
        suppress.invoke(null, primary, restoration);
        suppress.invoke(null, primary, primary);
        assertEquals(java.util.List.of(restoration), java.util.List.of(primary.getSuppressed()));
    }

    private static OpenBlasLibrary library(AtomicInteger cell) {
        MethodHandle getter = findVirtual(AtomicInteger.class, "get", MethodType.methodType(int.class)).bindTo(cell);
        MethodHandle setter = findVirtual(AtomicInteger.class, "set", MethodType.methodType(void.class, int.class)).bindTo(cell);
        return library(getter, setter);
    }

    private static OpenBlasLibrary library(MethodHandle getter, MethodHandle setter) {
        OpenBlasNativeBindings bindings = new OpenBlasNativeBindings(
                Arena.ofShared(), EMPTY, EMPTY, setter, getter);
        OpenBlasNativeAccess access = new OpenBlasNativeAccess() {
            @Override public OpenBlasNativeBindings open(String libraryName) { return bindings; }
            @Override public OpenBlasNativeBindings open(Path absoluteLibraryPath) { return bindings; }
        };
        return OpenBlasLibrary.open("fake", access);
    }

    private static MethodHandle normalSetter() {
        return MethodHandles.empty(MethodType.methodType(void.class, int.class));
    }

    private static MethodHandle throwing(Class<?> returnType, Throwable failure, Class<?>... parameters) {
        return MethodHandles.dropArguments(
                MethodHandles.throwException(returnType, Throwable.class).bindTo(failure), 0, parameters);
    }

    private static MethodHandle findStatic(String name, MethodType type) {
        try { return MethodHandles.lookup().findStatic(OpenBlasThreadControlTest.class, name, type); }
        catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    private static MethodHandle findVirtual(Class<?> owner, String name, MethodType type) {
        try { return MethodHandles.lookup().findVirtual(owner, name, type); }
        catch (ReflectiveOperationException failure) { throw new AssertionError(failure); }
    }

    @SuppressWarnings("unused")
    private static void countSetter(AtomicInteger calls, int value) { calls.incrementAndGet(); }

    @SuppressWarnings("unused")
    private static int countingGetter(AtomicInteger calls, int value) { calls.incrementAndGet(); return value; }
}
