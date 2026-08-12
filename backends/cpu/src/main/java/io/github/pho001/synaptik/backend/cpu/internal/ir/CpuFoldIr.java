package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one zero-initialized overlap fold.
 * Concrete extents, offsets, strides, axes, windows, ranges, and carrier instances remain cold.
 * The retained addition policy fixes canonical sequential represented-type addition and therefore
 * participates in generated-artifact compatibility.
 *
 * @param family non-null current fold coordinate family
 * @param dataType non-null supported represented input and output type; {@code BOOL} is rejected,
 *     and {@link Family#FOLD2D} accepts only floating types
 * @param inputAccess non-null structural read access for the sole input boundary
 * @param outputAccess non-null structural write access for the sole output boundary
 * @param additionPolicy stable represented-addition policy signature; currently must equal
 *     {@link #CANONICAL_SEQUENTIAL_ADDITION}
 */
public record CpuFoldIr(Family family, DataType dataType, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess, int additionPolicy) implements CpuPortableKernelIr {
    /** Distinguishes the two current Model fold coordinate mappings in artifact identity. */
    public enum Family {
        /** General-axis final-window-dimension overlap fold. */ FOLD_AXIS,
        /** Canonical rank-three columns to rank-four NCHW overlap fold. */ FOLD2D
    }

    /**
     * Identifies positive-zero initialization followed by canonical input-row-major sequential
     * represented-type addition.
     */
    public static final int CANONICAL_SEQUENTIAL_ADDITION = 1;

    /**
     * Creates the exact immutable structural fold form used by lowering and artifact identity.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if access, type, or addition-policy facts disagree
     */
    public CpuFoldIr {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(dataType, "dataType");
        Objects.requireNonNull(inputAccess, "inputAccess");
        Objects.requireNonNull(outputAccess, "outputAccess");
        if (inputAccess.accessKind() != CpuAccessPlan.AccessKind.READ
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || dataType == DataType.BOOL
                || family == Family.FOLD2D
                    && dataType != DataType.FLOAT64 && dataType != DataType.FLOAT32
                    && dataType != DataType.BFLOAT16
                || additionPolicy != CANONICAL_SEQUENTIAL_ADDITION) {
            throw new IllegalArgumentException("fold structural facts disagree");
        }
    }

    /**
     * Returns the instruction-free canonical generated form.
     *
     * @return a new non-null canonical kernel IR with one output store
     */
    public CpuKernelIr encodedKernelIr() {
        var values = List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "fold:" + family + ":addition=" + additionPolicy);
    }

    /**
     * Returns the deterministic canonical key for this structural fold form.
     *
     * @return a non-null stable key that excludes cold extents, layouts, ranges, and carriers
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
