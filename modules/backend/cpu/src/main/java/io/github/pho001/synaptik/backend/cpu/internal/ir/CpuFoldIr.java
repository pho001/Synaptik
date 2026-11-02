package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one zero-initialized overlap fold.
 * Concrete extents, offsets, strides, axes, windows, ranges, and carrier instances remain cold.
 * The addition policy fixes canonical sequential contributor order. FLOAT64/FLOAT32 keep their
 * represented precision; BFLOAT16/FLOAT16 use FLOAT32 accumulation and one final ties-to-even
 * narrowing; integer addition is modular. A low-only version discriminator makes pre-repair
 * per-add-narrowing artifacts incompatible without changing nonlow identities.
 *
 * @param family non-null current fold coordinate family
 * @param dataType non-null supported represented input and output type; {@code BOOL} is rejected,
 *     and {@link Family#FOLD2D} accepts only floating types
 * @param inputAccess non-null structural read access for the sole input boundary
 * @param outputAccess non-null structural write access for the sole output boundary
 * @param additionPolicy stable sequential-order policy signature; currently must equal
 *     {@link #CANONICAL_SEQUENTIAL_ADDITION}
 * @param kernelBroadcastInput whether F32/F64 FOLD2D reads physical dense [N,C,P]
 *     instead of logical repeated [N,C*K,P]; geometry remains cold
 */
public record CpuFoldIr(Family family, DataType dataType, CpuAccessPlan inputAccess,
        CpuAccessPlan outputAccess, int additionPolicy, boolean kernelBroadcastInput) implements CpuPortableKernelIr {
    /**
     * Creates the ordinary materialized-input form without compact-input admission. Either low
     * type still uses the versioned FLOAT32 accumulation policy described by this record.
     * @param family non-null fold family
     * @param dataType non-null represented type
     * @param inputAccess non-null read access
     * @param outputAccess non-null write access
     * @param additionPolicy must equal {@link #CANONICAL_SEQUENTIAL_ADDITION}
     * @throws NullPointerException if a reference argument is {@code null}
     * @throws IllegalArgumentException if access, type, or addition-policy facts disagree
     */
    public CpuFoldIr(Family family, DataType dataType, CpuAccessPlan inputAccess,
            CpuAccessPlan outputAccess, int additionPolicy) {
        this(family, dataType, inputAccess, outputAccess, additionPolicy, false);
    }
    /** Distinguishes the two current Model fold coordinate mappings in artifact identity. */
    public enum Family {
        /** General-axis final-window-dimension overlap fold. */ FOLD_AXIS,
        /** Canonical rank-three columns to rank-four NCHW overlap fold. */ FOLD2D
    }

    /**
     * Identifies positive-zero initialization followed by canonical input-row-major sequential
     * addition in the declared working precision: FLOAT32 for either low type, represented
     * precision for FLOAT32/FLOAT64, and modular integer arithmetic otherwise. Low results narrow
     * only after the last contributor; the low-only identity suffix versions that arithmetic.
     */
    public static final int CANONICAL_SEQUENTIAL_ADDITION = 1;

    /**
     * Creates the exact immutable structural fold form used by lowering and artifact identity.
     *
     * @param family non-null fold family
     * @param dataType non-null represented input/output type; compact mode is F32/F64 only
     * @param inputAccess non-null physical read access; compact mode requires dense rank three
     * @param outputAccess non-null physical write access; compact mode requires rank four
     * @param additionPolicy canonical sequential-order signature; low accumulation is FLOAT32
     * @param kernelBroadcastInput whether logical kernel repetition is read from compact storage
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
                    && dataType != DataType.BFLOAT16 && dataType != DataType.FLOAT16
                || additionPolicy != CANONICAL_SEQUENTIAL_ADDITION
                || kernelBroadcastInput && (family != Family.FOLD2D
                    || dataType != DataType.FLOAT32 && dataType != DataType.FLOAT64
                    || inputAccess.iterationRank() != 3 || outputAccess.iterationRank() != 4
                    || inputAccess.regime() != CpuAccessPlan.Regime.DENSE_LINEAR)) {
            throw new IllegalArgumentException("fold structural facts disagree");
        }
    }

    /**
     * Returns the instruction-free canonical generated form.
     *
     * @return a new non-null canonical kernel IR with one output store; compact F32/F64 mode
     *     appends its physical-input discriminator, while either ordinary low type appends
     *     {@code :low-fp32-accumulation-v1}. Both specialization identity streams inherit the
     *     resulting fingerprint; nonlow ordinary identities are unchanged
     */
    public CpuKernelIr encodedKernelIr() {
        var values = List.of(
                new CpuKernelIr.Value(0, dataType, CpuKernelIr.Value.Kind.INPUT, inputAccess),
                new CpuKernelIr.Value(1, dataType, CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(1, 0)),
                "fold:" + family + ":addition=" + additionPolicy
                        + (kernelBroadcastInput ? ":kernel-broadcast-input-v1" : "")
                        + (dataType == DataType.BFLOAT16 || dataType == DataType.FLOAT16
                                ? ":low-fp32-accumulation-v1" : ""));
    }

    /**
     * Returns the deterministic canonical key for this structural fold form.
     *
     * @return a non-null stable key that excludes cold extents, layouts, ranges, and carriers
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }
}
