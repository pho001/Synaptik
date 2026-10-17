package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one stable softmax or log-softmax kernel.
 *
 * @param kind exact first-class Model meaning
 * @param dataType identical floating input/output representation
 * @param axis normalized selected input axis
 * @param algorithmVersion CPU-private stable-algorithm version; version 2 binds FLOAT32 working
 *     sites for FLOAT32/FLOAT16 and retains admitted FLOAT32 shift overflow
 * @param passCount exact number of full selected-axis passes
 * @param inputAccess structural input read access
 * @param outputAccess structural output write access
 */
public record CpuSoftmaxIr(SoftmaxKind kind, DataType dataType, int axis,
        int algorithmVersion, int passCount, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /**
     * Validates a version-two, shape-preserving three-pass softmax identity before it can be
     * encoded for generated-artifact lookup. Version one is deliberately incompatible.
     *
     * @param kind non-null first-class softmax or log-softmax meaning
     * @param dataType non-null identical floating input/output representation
     * @param axis non-negative normalized axis within the input iteration rank
     * @param algorithmVersion exactly {@code 2}
     * @param passCount exactly three complete selected-axis passes
     * @param inputAccess non-null read plan for the input
     * @param outputAccess non-null write plan with the input's iteration rank
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if the dtype, axis, version, pass count, or access plans
     *     disagree with this structural identity
     */
    public CpuSoftmaxIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if ((dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32
                && dataType != DataType.BFLOAT16 && dataType != DataType.FLOAT16) || axis < 0
                || axis >= inputAccess.iterationRank()
                || inputAccess.iterationRank() != outputAccess.iterationRank()
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || algorithmVersion != 2 || passCount != 3) {
            throw new IllegalArgumentException("softmax structural facts disagree");
        }
    }

    /**
     * Returns the instruction-free cache-compatible representation.
     * @return a new canonical two-boundary generated-kernel identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess)),
                List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "softmax:" + kind + ":type=" + dataType + ":axis=" + axis
                        + ":algorithm=" + algorithmVersion + ":passes=" + passCount);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
