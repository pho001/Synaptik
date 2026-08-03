package io.github.pho001.synaptik.backend.cpu.execution;

import java.lang.classfile.CodeBuilder;
import java.util.Objects;

/**
 * Enumerates the complete portable CPU execution-mode vocabulary and owns the single structural
 * scalar-versus-Vector emission decision. Parallel modes describe generated entries that accept
 * an already-assigned half-open range; they do not create or manage workers.
 */
enum CpuPortableExecutionMode {
    /** Scalar code invoked once for the complete element count. */
    SCALAR_SINGLE_THREAD(false, false),
    /** Scalar code invoked for externally assigned parallel ranges. */
    SCALAR_PARALLEL(false, true),
    /** Vector API code invoked once for the complete element count. */
    VECTOR_API_SINGLE_THREAD(true, false),
    /** Vector API code invoked for externally assigned parallel ranges. */
    VECTOR_API_PARALLEL(true, true);

    private final boolean vectorized;
    private final boolean parallel;

    CpuPortableExecutionMode(boolean vectorized, boolean parallel) {
        this.vectorized = vectorized;
        this.parallel = parallel;
    }

    /** Reports whether this mode selects Vector emission.
     * @return {@code true} exactly when generated code uses the Java Vector API */
    boolean vectorized() { return vectorized; }

    /** Reports whether this mode receives externally assigned ranges.
     * @return {@code true} exactly when callers supply an already-assigned half-open range */
    boolean parallel() { return parallel; }

    /**
     * Constructs the low-level helpers for this already-selected mode and dispatches exactly one
     * matching family callback.
     *
     * <p>This is the sole structural scalar-versus-Vector dispatch point. It performs no family,
     * operation, carrier, route, worker, cache, or run-time selection.</p>
     *
     * @param code builder for the generated entry body; must not be {@code null}
     * @param specialization complete immutable specialization whose mode must be this value; must
     *     not be {@code null}
     * @param familyEmitter family-owned lowering callback; must not be {@code null}
     * @throws NullPointerException if any argument is {@code null}
     * @throws IllegalArgumentException if {@code specialization} selects another execution mode
     */
    void emit(CodeBuilder code, CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter) {
        Objects.requireNonNull(code, "code");
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(familyEmitter, "familyEmitter");
        if (specialization.executionMode() != this) {
            throw new IllegalArgumentException("specialization execution mode does not match dispatcher");
        }
        var carriers = new CpuCarrierEmitter(code, specialization);
        var loops = new CpuLoopEmitter(code, specialization);
        var reductions = new CpuReductionEmitter(code, specialization);
        if (vectorized) {
            familyEmitter.emitVector(
                    new CpuVectorEmitter(code, specialization), carriers, loops, reductions);
        } else {
            familyEmitter.emitScalar(
                    new CpuScalarEmitter(code, specialization), carriers, loops, reductions);
        }
    }
}
