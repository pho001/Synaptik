package io.github.pho001.synaptik.backend.provider.openblas;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Locks the native symbol inventory, standard ABI descriptors, and provider leaf boundary. */
final class OpenBlasAbiContractTest {
    /** Verifies the exact required symbol order with no optional or discovery symbol. */
    @Test
    void requiredSymbolInventoryIsExactAndOrdered() throws ReflectiveOperationException {
        assertEquals(
                List.of(
                        "cblas_sgemm",
                        "cblas_dgemm",
                        "openblas_set_num_threads",
                        "openblas_get_num_threads"),
                staticField("REQUIRED_SYMBOLS"));
    }

    /** Verifies production resolution visits every symbol and reports all missing names in order. */
    @Test
    void requiredSymbolResolutionIsFailClosedAndOrdered() throws ReflectiveOperationException {
        List<String> requested = new java.util.ArrayList<>();
        java.lang.foreign.SymbolLookup lookup = symbol -> {
            requested.add(symbol);
            return switch (symbol) {
                case "cblas_sgemm", "openblas_set_num_threads" -> Optional.of(MemorySegment.NULL);
                default -> Optional.empty();
            };
        };
        Method resolver = FfmOpenBlasNativeAccess.class.getDeclaredMethod(
                "resolveRequiredSymbols", java.lang.foreign.SymbolLookup.class);
        resolver.setAccessible(true);

        InvocationTargetException reflectionFailure = assertThrows(
                InvocationTargetException.class, () -> resolver.invoke(null, lookup));
        assertEquals(
                "Missing required OpenBLAS symbols: cblas_dgemm, openblas_get_num_threads",
                reflectionFailure.getCause().getMessage());
        assertEquals(
                List.of(
                        "cblas_sgemm",
                        "cblas_dgemm",
                        "openblas_set_num_threads",
                        "openblas_get_num_threads"),
                requested);
    }

    /** Verifies a production partial lookup failure closes its shared arena before escaping. */
    @Test
    void partialLookupFailureClosesProductionArena() throws ReflectiveOperationException {
        AtomicReference<java.lang.foreign.Arena> capturedArena = new AtomicReference<>();
        Function<java.lang.foreign.Arena, java.lang.foreign.SymbolLookup> factory = arena -> {
            capturedArena.set(arena);
            return symbol -> Optional.empty();
        };
        Method opener = FfmOpenBlasNativeAccess.class.getDeclaredMethod("open", Function.class);
        opener.setAccessible(true);

        InvocationTargetException reflectionFailure = assertThrows(
                InvocationTargetException.class, () -> opener.invoke(null, factory));
        assertTrue(reflectionFailure.getCause() instanceof IllegalStateException);
        assertFalse(capturedArena.get().scope().isAlive());
    }

    /** Verifies FLOAT32 and FLOAT64 GEMM use the standard 32-bit-blasint C descriptors. */
    @Test
    void gemmDescriptorsAreExact() throws ReflectiveOperationException {
        assertDescriptor(
                descriptor("SGEMM_DESCRIPTOR"),
                null,
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_FLOAT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_FLOAT, ADDRESS, JAVA_INT);
        assertDescriptor(
                descriptor("DGEMM_DESCRIPTOR"),
                null,
                JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
                JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT);
    }

    /** Verifies thread control retains its task-0001 descriptors. */
    @Test
    void threadDescriptorsAreExact() throws ReflectiveOperationException {
        assertDescriptor(descriptor("SET_NUM_THREADS_DESCRIPTOR"), null, JAVA_INT);
        assertDescriptor(descriptor("GET_NUM_THREADS_DESCRIPTOR"), JAVA_INT);
    }

    /** Verifies the binding carrier retains all exact references in required-symbol order. */
    @Test
    void bindingCarrierRetainsExactReferences() {
        var arena = java.lang.foreign.Arena.ofShared();
        MethodHandle sgemm = MethodHandles.empty(MethodType.methodType(void.class));
        MethodHandle dgemm = MethodHandles.empty(MethodType.methodType(void.class));
        MethodHandle setter = MethodHandles.empty(MethodType.methodType(void.class));
        MethodHandle getter = MethodHandles.constant(int.class, 0);
        OpenBlasNativeBindings bindings =
                new OpenBlasNativeBindings(arena, sgemm, dgemm, setter, getter);
        try {
            assertSame(arena, bindings.arena());
            assertSame(sgemm, bindings.sgemm());
            assertSame(dgemm, bindings.dgemm());
            assertSame(setter, bindings.setNumThreads());
            assertSame(getter, bindings.getNumThreads());
        } finally {
            bindings.close();
        }
    }

    /** Verifies each binding carrier reference is mandatory in declaration order. */
    @Test
    void bindingCarrierRejectsEveryNullReference() {
        var arena = java.lang.foreign.Arena.ofShared();
        MethodHandle handle = MethodHandles.empty(MethodType.methodType(void.class));
        try (arena) {
            assertEquals("arena", assertThrows(NullPointerException.class,
                    () -> new OpenBlasNativeBindings(null, handle, handle, handle, handle)).getMessage());
            assertEquals("sgemm", assertThrows(NullPointerException.class,
                    () -> new OpenBlasNativeBindings(arena, null, handle, handle, handle)).getMessage());
            assertEquals("dgemm", assertThrows(NullPointerException.class,
                    () -> new OpenBlasNativeBindings(arena, handle, null, handle, handle)).getMessage());
            assertEquals("setNumThreads", assertThrows(NullPointerException.class,
                    () -> new OpenBlasNativeBindings(arena, handle, handle, null, handle)).getMessage());
            assertEquals("getNumThreads", assertThrows(NullPointerException.class,
                    () -> new OpenBlasNativeBindings(arena, handle, handle, handle, null)).getMessage());
        }
    }

    /** Verifies the package-private native seam contains only the two exact loading operations. */
    @Test
    void nativeAccessSeamIsNarrowAndPackagePrivate() {
        assertFalse(Modifier.isPublic(OpenBlasNativeAccess.class.getModifiers()));
        assertEquals(
                Set.of(
                        "open(java.lang.String)",
                        "open(java.nio.file.Path)"),
                Arrays.stream(OpenBlasNativeAccess.class.getDeclaredMethods())
                        .map(OpenBlasAbiContractTest::methodKey)
                        .collect(Collectors.toUnmodifiableSet()));
        assertTrue(Modifier.isFinal(FfmOpenBlasNativeAccess.class.getModifiers()));
    }

    /** Verifies source and build declarations preserve the JDK-only leaf and excluded mechanisms. */
    @Test
    void providerHasZeroProjectDependenciesAndNoExcludedVocabulary() throws Exception {
        Path root = repositoryRoot();
        String build = Files.readString(root.resolve("backends/openblas-provider/build.gradle.kts"));
        assertFalse(build.contains("project("));

        Path production = root.resolve("backends/openblas-provider/src/main/java");
        String source;
        try (var paths = Files.walk(production)) {
            source = paths.filter(path -> path.toString().endsWith(".java"))
                    .sorted()
                    .map(OpenBlasAbiContractTest::read)
                    .collect(Collectors.joining("\n"));
        }
        for (String forbidden : List.of(
                "System.getenv", "System.getProperty", "ServiceLoader", "loaderLookup()",
                "defaultLookup()", "Map<String, Object>", "Tensor", "PreparedExecution",
                "BackendId", "CompileArtifacts", "CpuBackend", "MemorySegment.copy",
                ".reinterpret(", "synchronized")) {
            assertFalse(source.contains(forbidden), () -> "provider source must exclude " + forbidden);
        }
        assertEquals(2, count(source, "invokeExact("));
        assertFalse(source.contains("invokeWithArguments("));
    }

    private static int count(String text, String fragment) {
        int count = 0;
        int offset = 0;
        while ((offset = text.indexOf(fragment, offset)) >= 0) {
            count++;
            offset += fragment.length();
        }
        return count;
    }

    private static void assertDescriptor(
            FunctionDescriptor descriptor, MemoryLayout returnLayout, MemoryLayout... arguments) {
        assertEquals(List.of(arguments), descriptor.argumentLayouts());
        if (returnLayout == null) {
            assertTrue(descriptor.returnLayout().isEmpty());
        } else {
            assertEquals(returnLayout, descriptor.returnLayout().orElseThrow());
        }
    }

    private static FunctionDescriptor descriptor(String name) throws ReflectiveOperationException {
        return staticField(name);
    }

    @SuppressWarnings("unchecked")
    private static <T> T staticField(String name) throws ReflectiveOperationException {
        Field field = FfmOpenBlasNativeAccess.class.getDeclaredField(name);
        field.setAccessible(true);
        return (T) field.get(null);
    }

    private static String methodKey(Method method) {
        return method.getName() + "(" + Arrays.stream(method.getParameterTypes())
                .map(Class::getName)
                .collect(Collectors.joining(",")) + ")";
    }

    private static String read(Path path) {
        try {
            return Files.readString(path);
        } catch (Exception failure) {
            throw new IllegalStateException("could not read " + path, failure);
        }
    }

    private static Path repositoryRoot() {
        Path directory = Path.of("").toAbsolutePath();
        while (directory != null && !Files.isRegularFile(directory.resolve("settings.gradle.kts"))) {
            directory = directory.getParent();
        }
        if (directory == null) {
            throw new IllegalStateException("could not locate repository root");
        }
        return directory;
    }
}
