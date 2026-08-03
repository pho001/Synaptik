package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.CodeBuilder;
import java.lang.constant.ClassDesc;
import java.util.Objects;

/**
 * Emits class and species references for one exact Java Vector API specialization. It does not
 * select a species, implement family arithmetic, or promise support for arbitrary lane counts.
 */
final class CpuVectorEmitter {
    private static final ClassDesc VECTOR_SPECIES = ClassDesc.of("jdk.incubator.vector.VectorSpecies");
    private final CodeBuilder code;
    private final CpuKernelSpecialization specialization;

    /**
     * Creates an exact-species helper for a vector specialization.
     * @param code non-null generated-method builder
     * @param specialization non-null vector specialization
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if the specialization is scalar
     */
    CpuVectorEmitter(CodeBuilder code, CpuKernelSpecialization specialization) {
        this.code = Objects.requireNonNull(code, "code");
        this.specialization = Objects.requireNonNull(specialization, "specialization");
        if (!specialization.executionMode().vectorized()) {
            throw new IllegalArgumentException("vector emitter requires vector execution mode");
        }
    }
    /** Returns the current entry-body builder.
     * @return the exact non-null code builder supplied at construction */ CodeBuilder code() { return code; }
    /** Returns the specialization governing emitted Vector structure.
     * @return the exact non-null immutable specialization supplied at construction */ CpuKernelSpecialization specialization() { return specialization; }
    /** Returns the exact selected Vector species shape.
     * @return the exact non-null selected vector shape */ CpuKernelSpecialization.VectorShape vectorShape() {
        return specialization.vectorShape().orElseThrow(); }
    /** Returns the concrete Vector class descriptor for the selected lane type.
     * @return non-null concrete Vector API class descriptor for the selected lane type */ ClassDesc vectorClass() {
        return switch (vectorShape().laneType()) {
            case FLOAT64 -> ClassDesc.of("jdk.incubator.vector.DoubleVector");
            case FLOAT32 -> ClassDesc.of("jdk.incubator.vector.FloatVector");
            case INT32 -> ClassDesc.of("jdk.incubator.vector.IntVector");
            case INT64 -> ClassDesc.of("jdk.incubator.vector.LongVector");
            default -> throw new IllegalStateException("unsupported vector lane type");
        }; }
    /** Returns the class descriptor used by species constants.
     * @return the shared non-null {@code VectorSpecies} class descriptor */ ClassDesc speciesClass() {
        return VECTOR_SPECIES; }
    /** Emits a load of the exact fixed-width species constant selected by the specialization. */ void loadSpecies() {
        code.getstatic(vectorClass(), "SPECIES_" + vectorShape().vectorBitSize(), VECTOR_SPECIES);
    }
}
