package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.TypeKind;
import java.util.Objects;

/**
 * Emits low-level primitive local-slot instructions for one scalar entry body. It retains the
 * generation call's builder and specialization but owns no operation or arithmetic semantics.
 */
final class CpuScalarEmitter {
    private final CodeBuilder code;
    private final CpuKernelSpecialization specialization;

    /**
     * Creates a helper for one generated method.
     * @param code non-null method code builder
     * @param specialization non-null complete specialization
     * @throws NullPointerException if either argument is {@code null}
     */
    CpuScalarEmitter(CodeBuilder code, CpuKernelSpecialization specialization) {
        this.code = Objects.requireNonNull(code, "code");
        this.specialization = Objects.requireNonNull(specialization, "specialization");
    }

    /** Returns the current entry-body builder.
     * @return the exact non-null code builder supplied at construction */
    CodeBuilder code() { return code; }
    /** Returns the specialization governing emitted scalar structure.
     * @return the exact non-null immutable specialization supplied at construction */
    CpuKernelSpecialization specialization() { return specialization; }
    /**
     * Maps a Model data type to its generated Java Virtual Machine (JVM) local-slot kind.
     *
     * @param dataType logical data type; must not be {@code null}
     * @return matching non-null JVM primitive kind
     * @throws NullPointerException if {@code dataType} is {@code null}
     */
    TypeKind typeKind(DataType dataType) { return switch (Objects.requireNonNull(dataType, "dataType")) {
        case FLOAT64 -> TypeKind.DOUBLE; case FLOAT32 -> TypeKind.FLOAT;
        case BFLOAT16, INT32, BOOL -> TypeKind.INT; case INT64 -> TypeKind.LONG; }; }
    /**
     * Allocates a local slot of the kind selected by {@code dataType}.
     *
     * @param dataType local value type; must not be {@code null}
     * @return newly allocated local-slot index
     * @throws NullPointerException if {@code dataType} is {@code null}
     */
    int allocateLocal(DataType dataType) { return code.allocateLocal(typeKind(dataType)); }
    /**
     * Emits a load from an existing compatible local slot.
     *
     * @param dataType local value type; must not be {@code null}
     * @param slot valid local-slot index containing that kind
     * @throws NullPointerException if {@code dataType} is {@code null}
     */
    void loadLocal(DataType dataType, int slot) { code.loadLocal(typeKind(dataType), slot); }
    /**
     * Emits a store into an existing compatible local slot.
     *
     * @param dataType local value type; must not be {@code null}
     * @param slot valid local-slot index for that kind
     * @throws NullPointerException if {@code dataType} is {@code null}
     */
    void storeLocal(DataType dataType, int slot) { code.storeLocal(typeKind(dataType), slot); }
}
