package io.github.pho001.synaptik.backend.provider.openblas;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/** Verifies explicit input, failure, ownership, and concurrent lifecycle behavior. */
final class OpenBlasLibraryTest {
    /** Rejects invalid names before invoking native access. */
    @Test
    void validatesNameBeforeNativeAccess() {
        RecordingNativeAccess access = new RecordingNativeAccess();
        assertEquals("libraryName", assertThrows(
                NullPointerException.class, () -> OpenBlasLibrary.open((String) null, access)).getMessage());
        assertEquals("libraryName must not be blank", assertThrows(
                IllegalArgumentException.class, () -> OpenBlasLibrary.open(" \t", access)).getMessage());
        assertEquals(0, access.calls.get());
    }

    /** Rejects invalid paths before invoking native access. */
    @Test
    void validatesPathBeforeNativeAccess() {
        RecordingNativeAccess access = new RecordingNativeAccess();
        assertEquals("absoluteLibraryPath", assertThrows(
                NullPointerException.class, () -> OpenBlasLibrary.open((Path) null, access)).getMessage());
        assertEquals("absoluteLibraryPath must be absolute", assertThrows(
                IllegalArgumentException.class, () -> OpenBlasLibrary.open(Path.of("relative/libopenblas"), access))
                .getMessage());
        assertEquals(0, access.calls.get());
    }

    /** Passes a valid name or path unchanged and retains every fake binding reference. */
    @Test
    void successfulOpenRetainsExactBindings() {
        RecordingNativeAccess access = new RecordingNativeAccess();
        Path path = Path.of("/explicit/libopenblas.so");

        try (OpenBlasLibrary byName = OpenBlasLibrary.open("libopenblas.so", access);
                OpenBlasLibrary byPath = OpenBlasLibrary.open(path, access)) {
            assertEquals(List.of("libopenblas.so"), access.names);
            assertEquals(List.of(path), access.paths);
            assertSame(access.results.get(0), byName.bindings());
            assertSame(access.results.get(1), byPath.bindings());
            assertTrue(byName.isOpen());
            assertTrue(byPath.isOpen());
        }
    }

    /** Wraps native failures once while retaining the original cause and caller selection. */
    @Test
    void translatesNativeFailureWithOriginalCause() {
        IllegalCallerException failure = new IllegalCallerException("native access denied");
        OpenBlasNativeAccess access = failingAccess(failure);

        OpenBlasLoadException thrown = assertThrows(
                OpenBlasLoadException.class, () -> OpenBlasLibrary.open("chosen-openblas", access));
        assertTrue(thrown.getMessage().contains("chosen-openblas"));
        assertSame(failure, thrown.getCause());
    }

    /** Preserves ordered missing-symbol diagnostics and suppressed partial-cleanup failure. */
    @Test
    void preservesBindingAndCleanupFailureDetails() {
        IllegalStateException bindingFailure = new IllegalStateException(
                "Missing required OpenBLAS symbols: cblas_dgemm, openblas_get_num_threads");
        IllegalStateException cleanupFailure = new IllegalStateException("cleanup failed");
        bindingFailure.addSuppressed(cleanupFailure);

        OpenBlasLoadException thrown = assertThrows(
                OpenBlasLoadException.class,
                () -> OpenBlasLibrary.open(Path.of("/chosen/libopenblas.so"), failingAccess(bindingFailure)));
        assertSame(bindingFailure, thrown.getCause());
        assertEquals(List.of(cleanupFailure), List.of(thrown.getCause().getSuppressed()));
        assertTrue(thrown.getMessage().contains("/chosen/libopenblas.so"));
    }

    /** Rejects a broken seam result instead of returning a partial library handle. */
    @Test
    void nullBindingsBecomeTypedLoadFailure() {
        OpenBlasNativeAccess access = new OpenBlasNativeAccess() {
            @Override
            public OpenBlasNativeBindings open(String libraryName) {
                return null;
            }

            @Override
            public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
                return null;
            }
        };
        OpenBlasLoadException thrown = assertThrows(
                OpenBlasLoadException.class, () -> OpenBlasLibrary.open("broken", access));
        assertEquals("nativeAccess returned null bindings", thrown.getCause().getMessage());
    }

    /** Creates a fresh owner and native result for every open without global caching. */
    @Test
    void everyOpenIsFreshAndFailureDoesNotPoisonLaterOpen() {
        AtomicInteger attempt = new AtomicInteger();
        OpenBlasNativeAccess access = new OpenBlasNativeAccess() {
            @Override
            public OpenBlasNativeBindings open(String libraryName) {
                if (attempt.getAndIncrement() == 0) {
                    throw new IllegalStateException("first attempt");
                }
                return bindings();
            }

            @Override
            public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
                return bindings();
            }
        };

        assertThrows(OpenBlasLoadException.class, () -> OpenBlasLibrary.open("same", access));
        OpenBlasLibrary first = OpenBlasLibrary.open("same", access);
        OpenBlasLibrary second = OpenBlasLibrary.open("same", access);
        try (first; second) {
            assertNotSame(first, second);
            assertNotSame(first.bindings(), second.bindings());
        }
    }

    /** Closes the arena once and rejects package-local binding access afterward. */
    @Test
    void closeIsSequentiallyIdempotent() {
        OpenBlasNativeBindings bindings = bindings();
        OpenBlasLibrary library = OpenBlasLibrary.open("fake", returningAccess(bindings));
        assertTrue(bindings.arena().scope().isAlive());

        library.close();
        library.close();

        assertFalse(library.isOpen());
        assertFalse(bindings.arena().scope().isAlive());
        assertEquals("OpenBLAS library is closed", assertThrows(
                IllegalStateException.class, library::bindings).getMessage());
    }

    /** Allows concurrent close attempts while exactly one closes the shared arena. */
    @Test
    void closeIsConcurrentlyIdempotent() throws InterruptedException {
        OpenBlasNativeBindings bindings = bindings();
        OpenBlasLibrary library = OpenBlasLibrary.open("fake", returningAccess(bindings));
        CountDownLatch start = new CountDownLatch(1);
        try (var executor = Executors.newFixedThreadPool(8)) {
            for (int index = 0; index < 64; index++) {
                executor.submit(() -> {
                    start.await();
                    library.close();
                    return null;
                });
            }
            start.countDown();
            executor.shutdown();
            assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
        }
        assertFalse(library.isOpen());
        assertFalse(bindings.arena().scope().isAlive());
    }

    private static OpenBlasNativeAccess returningAccess(OpenBlasNativeBindings result) {
        return new OpenBlasNativeAccess() {
            @Override
            public OpenBlasNativeBindings open(String libraryName) {
                return result;
            }

            @Override
            public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
                return result;
            }
        };
    }

    private static OpenBlasNativeAccess failingAccess(RuntimeException failure) {
        return new OpenBlasNativeAccess() {
            @Override
            public OpenBlasNativeBindings open(String libraryName) {
                throw failure;
            }

            @Override
            public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
                throw failure;
            }
        };
    }

    private static OpenBlasNativeBindings bindings() {
        return new OpenBlasNativeBindings(
                Arena.ofShared(),
                emptyHandle(void.class),
                emptyHandle(void.class),
                emptyHandle(void.class),
                emptyHandle(int.class));
    }

    private static MethodHandle emptyHandle(Class<?> returnType) {
        if (returnType == void.class) {
            return MethodHandles.empty(MethodType.methodType(void.class));
        }
        return MethodHandles.constant(returnType, 0);
    }

    private static final class RecordingNativeAccess implements OpenBlasNativeAccess {
        private final AtomicInteger calls = new AtomicInteger();
        private final List<String> names = new ArrayList<>();
        private final List<Path> paths = new ArrayList<>();
        private final List<OpenBlasNativeBindings> results = new ArrayList<>();

        @Override
        public OpenBlasNativeBindings open(String libraryName) {
            calls.incrementAndGet();
            names.add(libraryName);
            OpenBlasNativeBindings result = bindings();
            results.add(result);
            return result;
        }

        @Override
        public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
            calls.incrementAndGet();
            paths.add(absoluteLibraryPath);
            OpenBlasNativeBindings result = bindings();
            results.add(result);
            return result;
        }
    }
}
