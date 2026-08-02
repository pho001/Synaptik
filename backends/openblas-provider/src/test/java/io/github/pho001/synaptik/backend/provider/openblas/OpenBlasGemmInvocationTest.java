package io.github.pho001.synaptik.backend.provider.openblas;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.invoke.WrongMethodTypeException;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

/** Verifies GEMM validation, exact ABI forwarding, ownership, and invocation failures. */
final class OpenBlasGemmInvocationTest {
    private static final MethodType SGEMM_TYPE = MethodType.methodType(
            void.class,
            int.class, int.class, int.class, int.class, int.class, int.class,
            float.class, MemorySegment.class, int.class, MemorySegment.class, int.class,
            float.class, MemorySegment.class, int.class);
    private static final MethodType DGEMM_TYPE = MethodType.methodType(
            void.class,
            int.class, int.class, int.class, int.class, int.class, int.class,
            double.class, MemorySegment.class, int.class, MemorySegment.class, int.class,
            double.class, MemorySegment.class, int.class);

    /** Locks the package-private helper's final, stateless shape. */
    @Test
    void helperIsFinalPackagePrivateAndHasNoInstanceFields() {
        assertTrue(Modifier.isFinal(OpenBlasGemmInvocation.class.getModifiers()));
        assertFalse(Modifier.isPublic(OpenBlasGemmInvocation.class.getModifiers()));
        assertEquals(0, Arrays.stream(OpenBlasGemmInvocation.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .count());
        assertEquals(0, Arrays.stream(OpenBlasGemmInvocation.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count());
    }

    /** Forwards every ordered SGEMM argument and derives row-major leading dimensions. */
    @Test
    void sgemmForwardsExactArgumentsAndRawScalarBits() {
        FloatRecorder recorder = new FloatRecorder();
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(sgemmHandle(recorder), noopDgemm())) {
            MemorySegment a = memory.allocate(2L * 3L * Float.BYTES, Float.BYTES);
            MemorySegment b = memory.allocate(3L * 4L * Float.BYTES, Float.BYTES);
            MemorySegment c = memory.allocate(2L * 4L * Float.BYTES, Float.BYTES);
            float alpha = Float.intBitsToFloat(0x7fc01234);
            float beta = -0.0f;

            library.sgemm(2, 4, 3, alpha, a, b, beta, c);

            assertEquals(1, recorder.calls);
            assertEquals(List.of(101, 111, 111, 2, 4, 3, 3, 4, 4), recorder.integerArguments());
            assertEquals(Float.floatToRawIntBits(alpha), recorder.alphaBits);
            assertEquals(Float.floatToRawIntBits(beta), recorder.betaBits);
            assertSame(a, recorder.a);
            assertSame(b, recorder.b);
            assertSame(c, recorder.c);
        }
    }

    /** Forwards every ordered DGEMM argument and preserves infinity and signed zero. */
    @Test
    void dgemmForwardsExactArgumentsAndRawScalarBits() {
        DoubleRecorder recorder = new DoubleRecorder();
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), dgemmHandle(recorder))) {
            MemorySegment a = memory.allocate(6L * Double.BYTES, Double.BYTES);
            MemorySegment b = memory.allocate(12L * Double.BYTES, Double.BYTES);
            MemorySegment c = memory.allocate(8L * Double.BYTES, Double.BYTES);

            library.dgemm(2, 4, 3, Double.POSITIVE_INFINITY, a, b, -0.0d, c);

            assertEquals(1, recorder.calls);
            assertEquals(List.of(101, 111, 111, 2, 4, 3, 3, 4, 4), recorder.integerArguments());
            assertEquals(Double.doubleToRawLongBits(Double.POSITIVE_INFINITY), recorder.alphaBits);
            assertEquals(Double.doubleToRawLongBits(-0.0d), recorder.betaBits);
            assertSame(a, recorder.a);
            assertSame(b, recorder.b);
            assertSame(c, recorder.c);
        }
    }

    /** Performs the owner-open check before any argument validation. */
    @Test
    void closedOwnerFailsBeforeArguments() {
        OpenBlasLibrary library = library(noopSgemm(), noopDgemm());
        library.close();
        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> library.sgemm(-1, -1, -1, 1.0f, null, null, 0.0f, null));
        assertEquals("OpenBLAS library is closed", failure.getMessage());
    }

    /** Validates dimensions and null references in the exact recorded order. */
    @Test
    void validatesDimensionsThenNullReferencesInOrder() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment segment = memory.allocate(32, 8);
            assertFailure(IllegalArgumentException.class, "m must be non-negative: -1",
                    () -> library.sgemm(-1, -2, -3, 1, null, null, 0, null));
            assertFailure(IllegalArgumentException.class, "n must be non-negative: -2",
                    () -> library.sgemm(0, -2, -3, 1, null, null, 0, null));
            assertFailure(IllegalArgumentException.class, "k must be non-negative: -3",
                    () -> library.sgemm(0, 0, -3, 1, null, null, 0, null));
            assertFailure(NullPointerException.class, "a",
                    () -> library.sgemm(0, 0, 0, 1, null, null, 0, null));
            assertFailure(NullPointerException.class, "b",
                    () -> library.sgemm(0, 0, 0, 1, segment, null, 0, null));
            assertFailure(NullPointerException.class, "c",
                    () -> library.sgemm(0, 0, 0, 1, segment, segment, 0, null));
        }
    }

    /** Rejects heap segments in A, B, C order before later segment properties. */
    @Test
    void validatesNativeSegmentsInOrder() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment nativeSegment = memory.allocate(8, 8);
            MemorySegment heap = MemorySegment.ofArray(new byte[8]);
            assertFailure(IllegalArgumentException.class, "a must be a native memory segment",
                    () -> library.sgemm(0, 0, 0, 1, heap, heap, 0, heap));
            assertFailure(IllegalArgumentException.class, "b must be a native memory segment",
                    () -> library.sgemm(0, 0, 0, 1, nativeSegment, heap, 0, heap));
            assertFailure(IllegalArgumentException.class, "c must be a native memory segment",
                    () -> library.sgemm(0, 0, 0, 1, nativeSegment, nativeSegment, 0, heap));
        }
    }

    /** Rejects dead scopes in A, B, C order. */
    @Test
    void validatesLiveScopesInOrder() {
        Arena closedArena = Arena.ofShared();
        MemorySegment closed = closedArena.allocate(8, 8);
        closedArena.close();
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment live = memory.allocate(8, 8);
            assertFailure(IllegalStateException.class, "a scope is not alive",
                    () -> library.sgemm(0, 0, 0, 1, closed, closed, 0, closed));
            assertFailure(IllegalStateException.class, "b scope is not alive",
                    () -> library.sgemm(0, 0, 0, 1, live, closed, 0, closed));
            assertFailure(IllegalStateException.class, "c scope is not alive",
                    () -> library.sgemm(0, 0, 0, 1, live, live, 0, closed));
        }
    }

    /** Rejects thread-confined segments before mutability and spatial checks. */
    @Test
    void validatesCurrentThreadAccessibilityInOrder() throws Exception {
        try (Arena confined = Arena.ofConfined();
                Arena shared = Arena.ofShared();
                OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment inaccessible = confined.allocate(8, 8);
            MemorySegment accessible = shared.allocate(8, 8);
            try (var executor = Executors.newSingleThreadExecutor()) {
                var aResult = executor.submit(() -> assertThrows(
                        IllegalStateException.class,
                        () -> library.dgemm(0, 0, 0, 1, inaccessible, inaccessible, 0, inaccessible)));
                assertEquals("a is not accessible by the current thread", aResult.get().getMessage());
                var bResult = executor.submit(() -> assertThrows(
                        IllegalStateException.class,
                        () -> library.dgemm(0, 0, 0, 1, accessible, inaccessible, 0, inaccessible)));
                assertEquals("b is not accessible by the current thread", bResult.get().getMessage());
                var cResult = executor.submit(() -> assertThrows(
                        IllegalStateException.class,
                        () -> library.dgemm(0, 0, 0, 1, accessible, accessible, 0, inaccessible)));
                assertEquals("c is not accessible by the current thread", cResult.get().getMessage());
            }
        }
    }

    /** Preserves finite, infinite, signed-zero, and NaN carriers without scalar policy. */
    @Test
    void preservesEveryScalarValueFamily() {
        FloatRecorder floats = new FloatRecorder();
        DoubleRecorder doubles = new DoubleRecorder();
        try (Arena memory = Arena.ofShared();
                OpenBlasLibrary library = library(sgemmHandle(floats), dgemmHandle(doubles))) {
            MemorySegment fa = memory.allocate(4, 8);
            MemorySegment fb = memory.allocate(4, 8);
            MemorySegment fc = memory.allocate(4, 8);
            float floatNan = Float.intBitsToFloat(0x7fc05678);
            float[][] floatCases = {
                    {1.25f, Float.POSITIVE_INFINITY},
                    {Float.NEGATIVE_INFINITY, 0.0f},
                    {-0.0f, floatNan}
            };
            for (float[] values : floatCases) {
                library.sgemm(1, 1, 1, values[0], fa, fb, values[1], fc);
                assertEquals(Float.floatToRawIntBits(values[0]), floats.alphaBits);
                assertEquals(Float.floatToRawIntBits(values[1]), floats.betaBits);
            }

            MemorySegment da = memory.allocate(8, 8);
            MemorySegment db = memory.allocate(8, 8);
            MemorySegment dc = memory.allocate(8, 8);
            double doubleNan = Double.longBitsToDouble(0x7ff8000000005678L);
            double[][] doubleCases = {
                    {1.25d, Double.POSITIVE_INFINITY},
                    {Double.NEGATIVE_INFINITY, 0.0d},
                    {-0.0d, doubleNan}
            };
            for (double[] values : doubleCases) {
                library.dgemm(1, 1, 1, values[0], da, db, values[1], dc);
                assertEquals(Double.doubleToRawLongBits(values[0]), doubles.alphaBits);
                assertEquals(Double.doubleToRawLongBits(values[1]), doubles.betaBits);
            }
        }
        assertEquals(3, floats.calls);
        assertEquals(3, doubles.calls);
    }

    /** Accepts read-only inputs but rejects read-only output even for an output-empty call. */
    @Test
    void validatesOutputMutabilityAfterInputProperties() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment input = memory.allocate(8, 8).asReadOnly();
            MemorySegment output = memory.allocate(8, 8);
            library.dgemm(0, 0, 0, 1, input, input, 0, output);
            assertFailure(IllegalArgumentException.class, "c must be writable",
                    () -> library.dgemm(0, 0, 0, 1, input, input, 0, output.asReadOnly()));
        }
    }

    /** Validates base-address alignment in A, B, C order for both element widths. */
    @Test
    void validatesAlignmentInOrder() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment aligned = memory.allocate(32, 8);
            MemorySegment misaligned = aligned.asSlice(1);
            assertFailure(IllegalArgumentException.class, "a address must be aligned to 4 bytes",
                    () -> library.sgemm(0, 0, 0, 1, misaligned, misaligned, 0, misaligned));
            assertFailure(IllegalArgumentException.class, "b address must be aligned to 8 bytes",
                    () -> library.dgemm(0, 0, 0, 1, aligned, misaligned, 0, misaligned));
            assertFailure(IllegalArgumentException.class, "c address must be aligned to 8 bytes",
                    () -> library.dgemm(0, 0, 0, 1, aligned, aligned, 0, misaligned));
        }
    }

    /** Translates checked span overflow in A, B, and C order with the arithmetic cause. */
    @Test
    void validatesCheckedRequiredSpanOverflowInOrder() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment segment = memory.allocate(8, 8);
            assertOverflow("A", () -> library.dgemm(
                    Integer.MAX_VALUE, 0, Integer.MAX_VALUE, 1, segment, segment, 0, segment));
            assertOverflow("B", () -> library.dgemm(
                    0, Integer.MAX_VALUE, Integer.MAX_VALUE, 1, segment, segment, 0, segment));
            assertOverflow("C", () -> library.dgemm(
                    Integer.MAX_VALUE, Integer.MAX_VALUE, 0, 1, segment, segment, 0, segment));
        }
    }

    /** Checks exact row-major byte coverage in A, B, C order. */
    @Test
    void validatesRequiredCoverageInOrder() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment shortSegment = memory.allocate(23, 8);
            MemorySegment a = memory.allocate(24, 8);
            MemorySegment b = memory.allocate(48, 8);
            MemorySegment c = memory.allocate(32, 8);
            assertFailure(IllegalArgumentException.class, "A requires at least 24 bytes, but segment has 23",
                    () -> library.sgemm(2, 4, 3, 1, shortSegment, b, 0, c));
            assertFailure(IllegalArgumentException.class, "B requires at least 48 bytes, but segment has 23",
                    () -> library.sgemm(2, 4, 3, 1, a, shortSegment, 0, c));
            assertFailure(IllegalArgumentException.class, "C requires at least 32 bytes, but segment has 23",
                    () -> library.sgemm(2, 4, 3, 1, a, b, 0, shortSegment));
        }
    }

    /** Detects actual required C/input overlap in order and accepts A/B aliasing and disjoint slices. */
    @Test
    void validatesRequiredRegionOverlap() {
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(noopSgemm(), noopDgemm())) {
            MemorySegment shared = memory.allocate(64, 8);
            MemorySegment a = shared.asSlice(0, 16);
            MemorySegment b = shared.asSlice(0, 16);
            MemorySegment disjointC = shared.asSlice(32, 16);
            library.sgemm(2, 2, 2, 1, a, b, 0, disjointC);

            MemorySegment overlapA = shared.asSlice(8, 16);
            assertFailure(IllegalArgumentException.class, "c must not overlap a",
                    () -> library.sgemm(2, 2, 2, 1, a, b, 0, overlapA));

            MemorySegment separateA = memory.allocate(16, 8);
            MemorySegment overlapB = shared.asSlice(8, 16);
            assertFailure(IllegalArgumentException.class, "c must not overlap b",
                    () -> library.sgemm(2, 2, 2, 1, separateA, b, 0, overlapB));
        }
    }

    /** Fully validates zero-output calls without invoking and invokes positive-output k-zero calls. */
    @Test
    void appliesZeroDimensionBehaviorAfterValidation() {
        FloatRecorder recorder = new FloatRecorder();
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(sgemmHandle(recorder), noopDgemm())) {
            MemorySegment zero = memory.allocate(0, 4);
            MemorySegment c = memory.allocate(24, 4);
            library.sgemm(0, 3, 7, 1, zero, memory.allocate(84, 4), 0, zero);
            library.sgemm(2, 0, 7, 1, memory.allocate(56, 4), zero, 0, zero);
            library.sgemm(0, 0, 0, 1, zero, zero, 0, zero);
            assertEquals(0, recorder.calls);

            library.sgemm(2, 3, 0, -0.0f, zero, zero, Float.NEGATIVE_INFINITY, c);
            assertEquals(1, recorder.calls);
            assertEquals(List.of(101, 111, 111, 2, 3, 0, 1, 3, 3), recorder.integerArguments());
        }
    }

    /** Wraps every non-Error throwable with operation-specific text and original cause. */
    @Test
    void wrapsNonErrorInvocationFailures() {
        try (Arena memory = Arena.ofShared()) {
            MemorySegment segment = memory.allocate(32, 8);
            Exception checked = new Exception("checked");
            IllegalStateException checkedFailure = invokeFailure(
                    throwingSgemm(checked), noopDgemm(), segment, true);
            assertEquals("OpenBLAS sgemm invocation failed", checkedFailure.getMessage());
            assertSame(checked, checkedFailure.getCause());

            RuntimeException runtime = new RuntimeException("runtime");
            IllegalStateException runtimeFailure = invokeFailure(
                    noopSgemm(), throwingDgemm(runtime), segment, false);
            assertEquals("OpenBLAS dgemm invocation failed", runtimeFailure.getMessage());
            assertSame(runtime, runtimeFailure.getCause());

            IllegalStateException wrongTypeFailure = invokeFailure(
                    MethodHandles.empty(MethodType.methodType(void.class)), noopDgemm(), segment, true);
            assertTrue(wrongTypeFailure.getCause() instanceof WrongMethodTypeException);
        }
    }

    /** Rethrows every Error from the exact handle unchanged. */
    @Test
    void rethrowsInvocationErrorUnchanged() {
        AssertionError error = new AssertionError("fatal");
        try (Arena memory = Arena.ofShared(); OpenBlasLibrary library = library(throwingSgemm(error), noopDgemm())) {
            MemorySegment segment = memory.allocate(4, 4);
            AssertionError thrown = assertThrows(
                    AssertionError.class,
                    () -> library.sgemm(1, 1, 1, 1, segment, memory.allocate(4, 4), 0,
                            memory.allocate(4, 4)));
            assertSame(error, thrown);
        }
    }

    private static IllegalStateException invokeFailure(
            MethodHandle sgemm, MethodHandle dgemm, MemorySegment segment, boolean singlePrecision) {
        try (OpenBlasLibrary library = library(sgemm, dgemm)) {
            MemorySegment first = segment.asSlice(0, 8);
            MemorySegment second = segment.asSlice(8, 8);
            MemorySegment third = segment.asSlice(16, 8);
            if (singlePrecision) {
                return assertThrows(IllegalStateException.class,
                        () -> library.sgemm(1, 1, 1, 1, first, second, 0, third));
            }
            return assertThrows(IllegalStateException.class,
                    () -> library.dgemm(1, 1, 1, 1, first, second, 0, third));
        }
    }

    private static void assertOverflow(String role, Runnable call) {
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, call::run);
        assertEquals(role + " required byte span overflows long", failure.getMessage());
        assertTrue(failure.getCause() instanceof ArithmeticException);
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, Runnable call) {
        assertEquals(message, assertThrows(type, call::run).getMessage());
    }

    private static OpenBlasLibrary library(MethodHandle sgemm, MethodHandle dgemm) {
        OpenBlasNativeBindings bindings = new OpenBlasNativeBindings(
                Arena.ofShared(), sgemm, dgemm,
                MethodHandles.empty(MethodType.methodType(void.class, int.class)),
                MethodHandles.constant(int.class, 1));
        return OpenBlasLibrary.open("fake", new OpenBlasNativeAccess() {
            @Override
            public OpenBlasNativeBindings open(String libraryName) {
                return bindings;
            }

            @Override
            public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
                return bindings;
            }
        });
    }

    private static MethodHandle sgemmHandle(FloatRecorder recorder) {
        return findVirtual(FloatRecorder.class, "record", SGEMM_TYPE).bindTo(recorder);
    }

    private static MethodHandle dgemmHandle(DoubleRecorder recorder) {
        return findVirtual(DoubleRecorder.class, "record", DGEMM_TYPE).bindTo(recorder);
    }

    private static MethodHandle noopSgemm() {
        return MethodHandles.empty(SGEMM_TYPE);
    }

    private static MethodHandle noopDgemm() {
        return MethodHandles.empty(DGEMM_TYPE);
    }

    private static MethodHandle throwingSgemm(Throwable throwable) {
        return MethodHandles.dropArguments(
                MethodHandles.throwException(void.class, Throwable.class).bindTo(throwable),
                0,
                SGEMM_TYPE.parameterList());
    }

    private static MethodHandle throwingDgemm(Throwable throwable) {
        return MethodHandles.dropArguments(
                MethodHandles.throwException(void.class, Throwable.class).bindTo(throwable),
                0,
                DGEMM_TYPE.parameterList());
    }

    private static MethodHandle findVirtual(Class<?> owner, String name, MethodType type) {
        try {
            return MethodHandles.lookup().findVirtual(owner, name, type);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static final class FloatRecorder {
        private int calls;
        private int order;
        private int transA;
        private int transB;
        private int m;
        private int n;
        private int k;
        private int alphaBits;
        private MemorySegment a;
        private int lda;
        private MemorySegment b;
        private int ldb;
        private int betaBits;
        private MemorySegment c;
        private int ldc;

        @SuppressWarnings("unused")
        private void record(
                int order, int transA, int transB, int m, int n, int k,
                float alpha, MemorySegment a, int lda, MemorySegment b, int ldb,
                float beta, MemorySegment c, int ldc) {
            calls++;
            this.order = order;
            this.transA = transA;
            this.transB = transB;
            this.m = m;
            this.n = n;
            this.k = k;
            alphaBits = Float.floatToRawIntBits(alpha);
            this.a = a;
            this.lda = lda;
            this.b = b;
            this.ldb = ldb;
            betaBits = Float.floatToRawIntBits(beta);
            this.c = c;
            this.ldc = ldc;
        }

        private List<Integer> integerArguments() {
            return List.of(order, transA, transB, m, n, k, lda, ldb, ldc);
        }
    }

    private static final class DoubleRecorder {
        private int calls;
        private int order;
        private int transA;
        private int transB;
        private int m;
        private int n;
        private int k;
        private long alphaBits;
        private MemorySegment a;
        private int lda;
        private MemorySegment b;
        private int ldb;
        private long betaBits;
        private MemorySegment c;
        private int ldc;

        @SuppressWarnings("unused")
        private void record(
                int order, int transA, int transB, int m, int n, int k,
                double alpha, MemorySegment a, int lda, MemorySegment b, int ldb,
                double beta, MemorySegment c, int ldc) {
            calls++;
            this.order = order;
            this.transA = transA;
            this.transB = transB;
            this.m = m;
            this.n = n;
            this.k = k;
            alphaBits = Double.doubleToRawLongBits(alpha);
            this.a = a;
            this.lda = lda;
            this.b = b;
            this.ldb = ldb;
            betaBits = Double.doubleToRawLongBits(beta);
            this.c = c;
            this.ldc = ldc;
        }

        private List<Integer> integerArguments() {
            return List.of(order, transA, transB, m, n, k, lda, ldb, ldc);
        }
    }
}
