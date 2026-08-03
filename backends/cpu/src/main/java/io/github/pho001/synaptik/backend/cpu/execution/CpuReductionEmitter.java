package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.util.Objects;

/**
 * Emits local-slot partial and combine structure without choosing a mathematical reduction,
 * accuracy policy, or parallel combine algorithm. Family callbacks provide the meaning of every
 * emitted update and combination.
 */
final class CpuReductionEmitter {
    /** Callback that emits one partial update from an index and accumulator local. */
    @FunctionalInterface interface PartialBody {
        /** Emits one family-owned partial update.
         * @param rangeIndexLocal long index local
         * @param partialValueLocal accumulator local */
        void emit(int rangeIndexLocal, int partialValueLocal);
    }
    /** Callback that emits one combination from two accumulator locals into a result local. */
    @FunctionalInterface interface CombineBody {
        /** Emits one family-owned combination.
         *
         * @param leftValueLocal first accumulator local
         * @param rightValueLocal second accumulator local
         * @param resultValueLocal result accumulator local
         */
        void emit(int leftValueLocal, int rightValueLocal, int resultValueLocal);
    }
    private final CodeBuilder code; private final CpuKernelSpecialization specialization;
    /**
     * Creates a structural reduction helper for one generated entry body.
     *
     * @param code method builder; must not be {@code null}
     * @param specialization complete immutable specialization; must not be {@code null}
     * @throws NullPointerException if either argument is {@code null}
     */
    CpuReductionEmitter(CodeBuilder code, CpuKernelSpecialization specialization) {
        this.code = Objects.requireNonNull(code, "code");
        this.specialization = Objects.requireNonNull(specialization, "specialization");
    }
    /** Returns the current entry-body builder.
     * @return the exact non-null code builder supplied at construction */ CodeBuilder code() { return code; }
    /** Returns the specialization governing emitted reduction structure.
     * @return the exact non-null immutable specialization supplied at construction */ CpuKernelSpecialization specialization() { return specialization; }
    /**
     * Emits one zero-initialized accumulator and invokes the callback inside the current range.
     * @param accumulatorType non-null primitive accumulator type
     * @param body non-null partial instruction callback
     * @throws NullPointerException if either argument is {@code null}
     */
    void emitPartials(DataType accumulatorType, PartialBody body) {
        Objects.requireNonNull(accumulatorType, "accumulatorType"); Objects.requireNonNull(body, "body");
        TypeKind kind = kind(accumulatorType); int partial = code.allocateLocal(kind);
        emitZero(kind); code.storeLocal(kind, partial);
        new CpuLoopEmitter(code, specialization).emitRange(index -> body.emit(index, partial));
    }
    /**
     * Allocates zero-initialized combine locals and emits one callback in selected operand order.
     * @param accumulatorType non-null primitive accumulator type
     * @param body non-null combine instruction callback
     * @throws NullPointerException if either argument is {@code null}
     */
    void emitCombine(DataType accumulatorType, CombineBody body) {
        Objects.requireNonNull(accumulatorType, "accumulatorType"); Objects.requireNonNull(body, "body");
        TypeKind kind = kind(accumulatorType); int left = code.allocateLocal(kind);
        int right = code.allocateLocal(kind); int result = code.allocateLocal(kind);
        emitZero(kind); code.storeLocal(kind, left); emitZero(kind); code.storeLocal(kind, right);
        emitZero(kind); code.storeLocal(kind, result);
        if (specialization.combineOrder() == CpuKernelSpecialization.CombineOrder.FIXED)
            body.emit(left, right, result); else body.emit(right, left, result);
    }
    private void emitZero(TypeKind kind) { switch (kind) {
        case DOUBLE -> code.loadConstant(0D); case FLOAT -> code.loadConstant(0F);
        case LONG -> code.loadConstant(0L); default -> code.loadConstant(0); } }
    private static TypeKind kind(DataType type) { return switch (type) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case INT64 -> TypeKind.LONG; default -> TypeKind.INT; }; }
}
