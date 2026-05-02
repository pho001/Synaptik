package tuning.benchmark.report;

/**
 * v1.4 operation coverage truth status.
 *
 * <p>This is intentionally stricter than the source lowering matrix. A matrix row can be
 * supported before a target family is proven native-executable on a concrete backend.</p>
 */
public enum GpuTargetExecutionStatus {
    NATIVE_EXECUTABLE,
    MATRIX_SUPPORTED_ONLY,
    EXPLICIT_CPU_FALLBACK,
    UNSUPPORTED_REJECTION
}
