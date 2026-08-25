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
 * @param algorithmVersion CPU-private stable-algorithm version
 * @param passCount exact number of full selected-axis passes
 * @param inputAccess structural input read access
 * @param outputAccess structural output write access
 */
public record CpuSoftmaxIr(SoftmaxKind kind, DataType dataType, int axis,
        int algorithmVersion, int passCount, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess) implements CpuPortableKernelIr {
    /** Validates one shape-preserving three-pass structural identity. */
    public CpuSoftmaxIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if ((dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32
                && dataType != DataType.BFLOAT16) || axis < 0
                || axis >= inputAccess.iterationRank()
                || inputAccess.iterationRank() != outputAccess.iterationRank()
                || inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || algorithmVersion != 1 || passCount != 3) {
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
