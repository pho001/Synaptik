package io.github.pho001.synaptik.backend.provider.openblas;

import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandle;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Retains one shared arena and the complete ordered OpenBLAS method-handle set.
 *
 * <p>The order is single-precision GEMM, double-precision GEMM, thread-count setter, then
 * thread-count getter. The handles remain package-private so later provider operations can use
 * them without exposing Foreign Function and Memory types as public API.
 */
final class OpenBlasNativeBindings {
    private final Arena arena;
    private final MethodHandle sgemm;
    private final MethodHandle dgemm;
    private final MethodHandle setNumThreads;
    private final MethodHandle getNumThreads;
    private final AtomicBoolean open = new AtomicBoolean(true);

    /**
     * Creates a complete binding set and retains every reference exactly.
     *
     * @param arena the shared arena controlling the library lookup lifetime; must not be
     *              {@code null}
     * @param sgemm the {@code cblas_sgemm} downcall handle; must not be {@code null}
     * @param dgemm the {@code cblas_dgemm} downcall handle; must not be {@code null}
     * @param setNumThreads the {@code openblas_set_num_threads} downcall handle; must not be
     *                      {@code null}
     * @param getNumThreads the {@code openblas_get_num_threads} downcall handle; must not be
     *                      {@code null}
     * @throws NullPointerException if any argument is {@code null}
     */
    OpenBlasNativeBindings(
            Arena arena,
            MethodHandle sgemm,
            MethodHandle dgemm,
            MethodHandle setNumThreads,
            MethodHandle getNumThreads) {
        this.arena = Objects.requireNonNull(arena, "arena");
        this.sgemm = Objects.requireNonNull(sgemm, "sgemm");
        this.dgemm = Objects.requireNonNull(dgemm, "dgemm");
        this.setNumThreads = Objects.requireNonNull(setNumThreads, "setNumThreads");
        this.getNumThreads = Objects.requireNonNull(getNumThreads, "getNumThreads");
    }

    /**
     * Returns the arena that owns this binding set's lookup lifetime.
     *
     * @return the exact shared lifetime arena, which may already be closed; never {@code null}
     */
    Arena arena() {
        return arena;
    }

    /**
     * Returns the handle bound to {@code cblas_sgemm}.
     *
     * @return the exact single-precision GEMM handle; valid for native use only while the arena
     *         remains open and never {@code null}
     */
    MethodHandle sgemm() {
        return sgemm;
    }

    /**
     * Returns the handle bound to {@code cblas_dgemm}.
     *
     * @return the exact double-precision GEMM handle; valid for native use only while the arena
     *         remains open and never {@code null}
     */
    MethodHandle dgemm() {
        return dgemm;
    }

    /**
     * Returns the handle bound to {@code openblas_set_num_threads}.
     *
     * @return the exact thread-count setter handle; valid for native use only while the arena
     *         remains open and never {@code null}
     */
    MethodHandle setNumThreads() {
        return setNumThreads;
    }

    /**
     * Returns the handle bound to {@code openblas_get_num_threads}.
     *
     * @return the exact thread-count getter handle; valid for native use only while the arena
     *         remains open and never {@code null}
     */
    MethodHandle getNumThreads() {
        return getNumThreads;
    }

    /**
     * Closes the shared arena at most once.
     *
     * <p>A cleanup failure is propagated to the caller after the binding set becomes observably
     * closed; later close attempts do nothing.
     *
     * @throws IllegalStateException if the shared arena cannot be closed
     */
    void close() {
        if (open.compareAndSet(true, false)) {
            arena.close();
        }
    }
}
