package io.github.pho001.synaptik.backend.provider.openblas;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns one explicitly loaded OpenBLAS library lookup and its required native bindings.
 *
 * <p>Each successful {@link #open(String)} or {@link #open(Path)} call creates a distinct Java
 * lifetime owner. The owner does not discover a library, choose a platform name, select a CPU
 * route, or provide fallback behavior. Closing it ends only this owner's Foreign Function and
 * Memory lookup lifetime; the operating system may retain the underlying library.
 *
 * <p>The immutable bindings use a shared arena so they may be consumed concurrently by later
 * package-colocated provider operations while this owner remains open. Callers must not race
 * {@link #close()} with such an operation. Lifecycle observation and closing are thread-safe;
 * concurrent GEMM calls are permitted when callers independently keep their segments alive,
 * accessible, and free of conflicting access.
 */
public final class OpenBlasLibrary implements AutoCloseable {
    private static final OpenBlasNativeAccess NATIVE_ACCESS = new FfmOpenBlasNativeAccess();

    private final OpenBlasNativeBindings bindings;
    private final AtomicBoolean open = new AtomicBoolean(true);

    /**
     * Creates one open lifetime owner around a complete binding set.
     *
     * @param bindings the complete native bindings owned by this handle; must not be {@code null}
     * @throws NullPointerException if {@code bindings} is {@code null}
     */
    private OpenBlasLibrary(OpenBlasNativeBindings bindings) {
        this.bindings = Objects.requireNonNull(bindings, "bindings");
    }

    /**
     * Loads exactly the caller-specified operating-system library name and binds all required
     * OpenBLAS symbols.
     *
     * @param libraryName the nonblank library name passed unchanged to the operating-system
     *                    library lookup; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if {@code libraryName} is {@code null}
     * @throws IllegalArgumentException if {@code libraryName} is blank
     * @throws OpenBlasLoadException if loading, symbol resolution, or native handle binding fails
     */
    public static OpenBlasLibrary open(String libraryName) {
        return open(libraryName, NATIVE_ACCESS);
    }

    /**
     * Loads exactly the caller-specified absolute library path and binds all required OpenBLAS
     * symbols.
     *
     * @param absoluteLibraryPath the absolute library path passed unchanged to the operating-
     *                            system library lookup; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if {@code absoluteLibraryPath} is not absolute
     * @throws OpenBlasLoadException if loading, symbol resolution, or native handle binding fails
     */
    public static OpenBlasLibrary open(Path absoluteLibraryPath) {
        return open(absoluteLibraryPath, NATIVE_ACCESS);
    }

    /**
     * Loads a validated library name through the deterministic package-private native seam.
     *
     * @param libraryName the nonblank caller-selected library name; must not be {@code null}
     * @param nativeAccess the exact native access implementation; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if {@code libraryName} is blank
     * @throws OpenBlasLoadException if native access fails or returns no bindings
     */
    static OpenBlasLibrary open(String libraryName, OpenBlasNativeAccess nativeAccess) {
        Objects.requireNonNull(libraryName, "libraryName");
        Objects.requireNonNull(nativeAccess, "nativeAccess");
        if (libraryName.isBlank()) {
            throw new IllegalArgumentException("libraryName must not be blank");
        }
        return load("name '" + libraryName + "'", () -> nativeAccess.open(libraryName));
    }

    /**
     * Loads a validated absolute path through the deterministic package-private native seam.
     *
     * @param absoluteLibraryPath the absolute caller-selected path; must not be {@code null}
     * @param nativeAccess the exact native access implementation; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if {@code absoluteLibraryPath} is not absolute
     * @throws OpenBlasLoadException if native access fails or returns no bindings
     */
    static OpenBlasLibrary open(Path absoluteLibraryPath, OpenBlasNativeAccess nativeAccess) {
        Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath");
        Objects.requireNonNull(nativeAccess, "nativeAccess");
        if (!absoluteLibraryPath.isAbsolute()) {
            throw new IllegalArgumentException("absoluteLibraryPath must be absolute");
        }
        return load("path '" + absoluteLibraryPath + "'", () -> nativeAccess.open(absoluteLibraryPath));
    }

    /**
     * Reports whether this Java lifetime owner has not yet been closed.
     *
     * <p>This local observation performs no symbol resolution or operating-system availability
     * check and cannot by itself coordinate a later native call with concurrent closure.
     *
     * @return {@code true} until the first close attempt claims this owner, otherwise {@code false}
     */
    public boolean isOpen() {
        return open.get();
    }

    /**
     * Invokes one dense row-major, non-transposed single-precision matrix multiplication.
     *
     * <p>The operation is {@code C = alpha * (A x B) + beta * C}, where {@code A} has shape
     * {@code [m,k]}, {@code B} has shape {@code [k,n]}, and {@code C} has shape {@code [m,n]}.
     * All segments remain caller-owned native memory. The call reads only the logical ranges of
     * {@code a} and {@code b}, mutates only the logical range of {@code c}, and retains no segment.
     * Matrices begin at byte offset zero; their row strides are derived as {@code k}, {@code n},
     * and {@code n} elements, with a minimum leading dimension of one. Inputs may overlap each
     * other, but the required output range must not overlap either required input range.
     *
     * <p>Validation is complete even when {@code m == 0 || n == 0}; after successful validation,
     * that output-empty case makes no native call. A positive-output call with {@code k == 0}
     * still invokes OpenBLAS so the supplied {@code beta} applies to {@code C}. Every scalar bit
     * is forwarded without a finiteness or special-value policy. This provider defines no result
     * accuracy, rounding, exceptional-value, determinism, or performance guarantee.
     *
     * <p>The caller must keep the library and all segment scopes alive for the complete call and
     * must not race {@link #close()} with this method. Concurrent calls are permitted only when
     * caller-managed segment access does not conflict; the provider performs no synchronization.
     *
     * @param m the non-negative number of rows in {@code A} and {@code C}
     * @param n the non-negative number of columns in {@code B} and {@code C}
     * @param k the non-negative contraction dimension shared by {@code A} and {@code B}
     * @param alpha the scalar multiplier for {@code A x B}; forwarded unchanged
     * @param a caller-owned native FLOAT32 storage for dense row-major {@code A}; not {@code null}
     * @param b caller-owned native FLOAT32 storage for dense row-major {@code B}; not {@code null}
     * @param beta the scalar multiplier for the existing {@code C}; forwarded unchanged
     * @param c caller-owned writable native FLOAT32 storage for dense row-major {@code C}; not
     *          {@code null} and not overlapping the required range of {@code a} or {@code b}
     * @throws IllegalStateException if this library is closed, a segment scope is not alive or
     *                               accessible by the current thread, or native invocation fails
     * @throws NullPointerException if {@code a}, {@code b}, or {@code c} is {@code null}
     * @throws IllegalArgumentException if a dimension, segment kind, output mutability, alignment,
     *                                  span, or overlap precondition is violated
     */
    public void sgemm(
            int m,
            int n,
            int k,
            float alpha,
            MemorySegment a,
            MemorySegment b,
            float beta,
            MemorySegment c) {
        OpenBlasGemmInvocation.sgemm(bindings(), m, n, k, alpha, a, b, beta, c);
    }

    /**
     * Invokes one dense row-major, non-transposed double-precision matrix multiplication.
     *
     * <p>The operation is {@code C = alpha * (A x B) + beta * C}, where {@code A} has shape
     * {@code [m,k]}, {@code B} has shape {@code [k,n]}, and {@code C} has shape {@code [m,n]}.
     * All segments remain caller-owned native memory. The call reads only the logical ranges of
     * {@code a} and {@code b}, mutates only the logical range of {@code c}, and retains no segment.
     * Matrices begin at byte offset zero; their row strides are derived as {@code k}, {@code n},
     * and {@code n} elements, with a minimum leading dimension of one. Inputs may overlap each
     * other, but the required output range must not overlap either required input range.
     *
     * <p>Validation is complete even when {@code m == 0 || n == 0}; after successful validation,
     * that output-empty case makes no native call. A positive-output call with {@code k == 0}
     * still invokes OpenBLAS so the supplied {@code beta} applies to {@code C}. Every scalar bit
     * is forwarded without a finiteness or special-value policy. This provider defines no result
     * accuracy, rounding, exceptional-value, determinism, or performance guarantee.
     *
     * <p>The caller must keep the library and all segment scopes alive for the complete call and
     * must not race {@link #close()} with this method. Concurrent calls are permitted only when
     * caller-managed segment access does not conflict; the provider performs no synchronization.
     *
     * @param m the non-negative number of rows in {@code A} and {@code C}
     * @param n the non-negative number of columns in {@code B} and {@code C}
     * @param k the non-negative contraction dimension shared by {@code A} and {@code B}
     * @param alpha the scalar multiplier for {@code A x B}; forwarded unchanged
     * @param a caller-owned native FLOAT64 storage for dense row-major {@code A}; not {@code null}
     * @param b caller-owned native FLOAT64 storage for dense row-major {@code B}; not {@code null}
     * @param beta the scalar multiplier for the existing {@code C}; forwarded unchanged
     * @param c caller-owned writable native FLOAT64 storage for dense row-major {@code C}; not
     *          {@code null} and not overlapping the required range of {@code a} or {@code b}
     * @throws IllegalStateException if this library is closed, a segment scope is not alive or
     *                               accessible by the current thread, or native invocation fails
     * @throws NullPointerException if {@code a}, {@code b}, or {@code c} is {@code null}
     * @throws IllegalArgumentException if a dimension, segment kind, output mutability, alignment,
     *                                  span, or overlap precondition is violated
     */
    public void dgemm(
            int m,
            int n,
            int k,
            double alpha,
            MemorySegment a,
            MemorySegment b,
            double beta,
            MemorySegment c) {
        OpenBlasGemmInvocation.dgemm(bindings(), m, n, k, alpha, a, b, beta, c);
    }

    /**
     * Ends this owner's native lookup lifetime once.
     *
     * <p>Repeated and concurrent calls are idempotent. If arena closure fails, this owner remains
     * observably closed and the unchecked cleanup failure is propagated to the winning caller.
     *
     * @throws IllegalStateException if the underlying shared arena cannot be closed
     */
    @Override
    public void close() {
        if (open.compareAndSet(true, false)) {
            bindings.close();
        }
    }

    /**
     * Returns the complete bindings for later package-colocated provider operations.
     *
     * @return the exact binding set retained by this owner; never {@code null}
     * @throws IllegalStateException if this owner has been closed
     */
    OpenBlasNativeBindings bindings() {
        if (!open.get()) {
            throw new IllegalStateException("OpenBLAS library is closed");
        }
        return bindings;
    }

    /**
     * Translates one already validated native loading attempt to the stable public failure type.
     *
     * @param description the caller selection included in failure diagnostics; must not be
     *                    {@code null}
     * @param loader the exact loading operation; must not be {@code null}
     * @return a new lifetime owner for the complete returned bindings; never {@code null}
     * @throws OpenBlasLoadException if loading fails or returns {@code null}
     */
    private static OpenBlasLibrary load(String description, BindingLoader loader) {
        Objects.requireNonNull(description, "description");
        Objects.requireNonNull(loader, "loader");
        try {
            OpenBlasNativeBindings bindings = loader.load();
            if (bindings == null) {
                throw new NullPointerException("nativeAccess returned null bindings");
            }
            return new OpenBlasLibrary(bindings);
        } catch (Throwable cause) {
            throw new OpenBlasLoadException("Failed to load OpenBLAS library " + description, cause);
        }
    }

    /** Loads one complete binding set for the shared public failure-translation boundary. */
    @FunctionalInterface
    private interface BindingLoader {
        /**
         * Loads one complete native binding set.
         *
         * @return the complete binding set; never {@code null}
         * @throws RuntimeException if native loading or binding fails
         */
        OpenBlasNativeBindings load();
    }
}
