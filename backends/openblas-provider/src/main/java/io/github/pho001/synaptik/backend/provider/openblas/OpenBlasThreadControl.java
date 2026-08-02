package io.github.pho001.synaptik.backend.provider.openblas;

import java.lang.invoke.MethodHandle;

/**
 * Performs exact low-level OpenBLAS thread-count invocations over already checked bindings.
 *
 * <p>This stateless helper neither chooses, caches, coordinates, nor restores a thread count.
 * The underlying value is conservatively treated as mutable library/process state. Owners of one
 * loaded binary may observe competing mutations, but this helper makes no shared-state guarantee
 * across independent binary copies, loader namespaces, or arbitrary native consumers. Calls are
 * not atomic with one another or with GEMM, and the caller must coordinate restoration and avoid
 * racing the owning library's closure.
 */
final class OpenBlasThreadControl {
    private OpenBlasThreadControl() {
    }

    /**
     * Invokes the exact thread-count getter and rejects an unusable native result.
     *
     * @param bindings the already open exact native bindings; not {@code null}
     * @return the exact positive count returned by OpenBLAS
     * @throws IllegalStateException if invocation fails or returns a non-positive count
     */
    static int threadCount(OpenBlasNativeBindings bindings) {
        MethodHandle getter = bindings.getNumThreads();
        int result;
        try {
            result = (int) getter.invokeExact();
        } catch (Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("OpenBLAS get thread count invocation failed", failure);
        }
        if (result <= 0) {
            throw new IllegalStateException("OpenBLAS returned non-positive thread count: " + result);
        }
        return result;
    }

    /**
     * Validates and passes one exact positive count to the thread-count setter.
     *
     * @param bindings the already open exact native bindings; not {@code null}
     * @param threadCount the positive count passed unchanged to OpenBLAS
     * @throws IllegalArgumentException if {@code threadCount} is not positive
     * @throws IllegalStateException if invocation fails
     */
    static void setThreadCount(OpenBlasNativeBindings bindings, int threadCount) {
        if (threadCount <= 0) {
            throw new IllegalArgumentException("threadCount must be positive: " + threadCount);
        }
        MethodHandle setter = bindings.setNumThreads();
        try {
            setter.invokeExact(threadCount);
        } catch (Error failure) {
            throw failure;
        } catch (Throwable failure) {
            throw new IllegalStateException("OpenBLAS set thread count invocation failed", failure);
        }
    }
}
