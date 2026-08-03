package io.github.pho001.synaptik.backend.cpu.execution;

/**
 * Reports CPU worker coordination, interruption, worker-group cancellation, or shutdown failure
 * when no original unchecked worker failure can be propagated directly.
 */
final class CpuParallelExecutionException extends RuntimeException {
    /**
     * Creates a coordination failure without a cause.
     * @param message non-null diagnostic message describing the lifecycle failure
     */
    CpuParallelExecutionException(String message) { super(message); }
    /**
     * Creates a coordination failure with its triggering cause.
     * @param message non-null diagnostic message describing the lifecycle failure
     * @param cause triggering failure, normally an {@link InterruptedException}
     */
    CpuParallelExecutionException(String message, Throwable cause) { super(message, cause); }
}
