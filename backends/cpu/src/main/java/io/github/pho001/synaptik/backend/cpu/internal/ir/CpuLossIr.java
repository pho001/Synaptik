package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private generated-code identity for one first-class loss occurrence.
 *
 * <p>The identity deliberately retains only emitted-code facts: family, ordered represented
 * operand/result types, reduction, index representation and ignore form, direct range form, and
 * the ordered semantic-role to unique-read-boundary mapping. Normalized class axis, ranks,
 * extents, layouts, addresses, actual ignore bits, and range bounds are invocation geometry and
 * therefore remain cold. The generated implementation must keep the loss atomic; this type does
 * not describe or permit recognition of a decomposed loss topology.</p>
 *
 * @param kind exact first-class Model loss family
 * @param predictionType represented prediction or logits type
 * @param targetType represented dense target or exact integral index type
 * @param resultType represented output type
 * @param reduction exact complete-domain reduction
 * @param indexIgnorePresent whether index categorical metadata includes an ignore value
 * @param roleBoundaryPositions ordered prediction/logits and target positions in the unique-read
 *     boundary prefix
 * @param boundaryTypes ordered unique-read then one write boundary types
 * @param boundaryAccesses ordered direct occurrence access plans retained for lowering, binding,
 *     and lifecycle validation; their rank, regime, axis roles, and layout shape do not enter the
 *     generated loss identity because the emitter consumes that geometry only from its cold
 *     payload
 * @param rangeForm direct independent-domain or single-complete-reduction range form
 * @param geometry non-null cold normalized rank, layout, stride, offset, base-packing, and
 *     ignore-value facts consumed only when the generated entry is invoked. Vector eligibility
 *     remains a CPU-analysis fact for same-typed contiguous FLOAT32/FLOAT64 MSE {@code NONE};
 *     this route-independent identity and its historical {@code DIRECT_SCALAR} token remain
 *     unchanged for schema-58 compatibility
 */
public record CpuLossIr(LossKind kind, DataType predictionType, DataType targetType,
        DataType resultType, LossReduction reduction, boolean indexIgnorePresent,
        List<Integer> roleBoundaryPositions, List<DataType> boundaryTypes,
        List<CpuAccessPlan> boundaryAccesses, RangeForm rangeForm,
        CpuLossLowering.Geometry geometry) implements CpuPortableKernelIr {

    /** Direct generated work-range form. */
    public enum RangeForm {
        /** Each range owns independent MSE elements or categorical samples. */
        INDEPENDENT_DOMAIN,
        /** One range owns the complete ordered reduced domain and scalar result. */
        COMPLETE_REDUCTION
    }

    /**
     * Validates a code-shaping loss identity.
     *
     * @throws NullPointerException if a required component is null
     * @throws IllegalArgumentException if boundary, type, reduction, or range facts do not define
     *     one supported direct loss form
     */
    public CpuLossIr {
        Objects.requireNonNull(kind, "kind");
        Objects.requireNonNull(predictionType, "predictionType");
        Objects.requireNonNull(targetType, "targetType");
        Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(reduction, "reduction");
        Objects.requireNonNull(roleBoundaryPositions, "roleBoundaryPositions");
        Objects.requireNonNull(boundaryTypes, "boundaryTypes");
        Objects.requireNonNull(boundaryAccesses, "boundaryAccesses");
        Objects.requireNonNull(rangeForm, "rangeForm");
        Objects.requireNonNull(geometry, "geometry");
        roleBoundaryPositions = List.copyOf(roleBoundaryPositions);
        boundaryTypes = List.copyOf(boundaryTypes);
        boundaryAccesses = List.copyOf(boundaryAccesses);
        int uniqueInputs = boundaryTypes.size() - 1;
        if (!floating(predictionType) || !floating(resultType)
                || roleBoundaryPositions.size() != 2 || uniqueInputs < 1
                || boundaryAccesses.size() != boundaryTypes.size()
                || roleBoundaryPositions.stream().anyMatch(position -> position == null
                        || position < 0 || position >= uniqueInputs)
                || boundaryAccesses.subList(0, uniqueInputs).stream().anyMatch(access ->
                        access.accessKind() != CpuAccessPlan.AccessKind.READ)
                || boundaryAccesses.getLast().accessKind() != CpuAccessPlan.AccessKind.WRITE
                || boundaryTypes.get(roleBoundaryPositions.getFirst()) != predictionType
                || boundaryTypes.get(roleBoundaryPositions.getLast()) != targetType
                || boundaryTypes.getLast() != resultType
                || (reduction == LossReduction.NONE) != (rangeForm == RangeForm.INDEPENDENT_DOMAIN)) {
            throw new IllegalArgumentException("loss IR facts disagree");
        }
        boolean index = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
        if (index != (targetType == DataType.INT32 || targetType == DataType.INT64)
                || !index && (!floating(targetType) || indexIgnorePresent)) {
            throw new IllegalArgumentException("loss operand facts disagree");
        }
        if (index ? resultType != predictionType
                : resultType != DataTypePromotion.promoteFloating(predictionType, targetType)) {
            throw new IllegalArgumentException("loss result type disagrees with operands");
        }
    }

    /**
     * Preserves the pre-lowering identity constructor for structural tests.
     *
     * @param kind exact loss family
     * @param predictionType represented prediction/logits type
     * @param targetType represented target type
     * @param resultType represented result type
     * @param reduction complete-domain reduction
     * @param indexIgnorePresent whether index metadata contains an ignore scalar
     * @param roleBoundaryPositions semantic input role mapping
     * @param boundaryTypes unique-read then output represented types
     * @param boundaryAccesses matching boundary access plans
     * @param rangeForm generated range ownership form
     */
    public CpuLossIr(LossKind kind, DataType predictionType, DataType targetType,
            DataType resultType, LossReduction reduction, boolean indexIgnorePresent,
            List<Integer> roleBoundaryPositions, List<DataType> boundaryTypes,
            List<CpuAccessPlan> boundaryAccesses, RangeForm rangeForm) {
        this(kind, predictionType, targetType, resultType, reduction, indexIgnorePresent,
                roleBoundaryPositions, boundaryTypes, boundaryAccesses, rangeForm,
                CpuLossLowering.Geometry.unavailable());
    }

    /**
     * Encodes this identity for the existing generated-class cache boundary.
     *
     * @return immutable instruction-free generated-kernel identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>(boundaryTypes.size());
        int uniqueInputs = boundaryTypes.size() - 1;
        for (int boundary = 0; boundary < boundaryTypes.size(); boundary++) {
            values.add(new CpuKernelIr.Value(boundary, boundaryTypes.get(boundary),
                    boundary < uniqueInputs ? CpuKernelIr.Value.Kind.INPUT
                            : CpuKernelIr.Value.Kind.OUTPUT,
                    generatedAccess(boundary < uniqueInputs)));
        }
        String identity = "loss:kind=" + kind + ":prediction=" + predictionType + ":target="
                + targetType + ":result=" + resultType + ":reduction=" + reduction
                + ":indexIgnore=" + indexIgnorePresent + ":roles=" + roleBoundaryPositions
                + ":acc=" + (resultType == DataType.FLOAT64 ? "F64" : "F32") + ":range="
                + rangeForm + ":workspace=NONE:realization=DIRECT_SCALAR";
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(uniqueInputs, 0)), identity);
    }

    /**
     * Returns the deterministic generated-code cache key.
     *
     * @return lowercase hexadecimal structural key; never {@code null}
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    /*
     * CpuKernelIr predates loss entries and fingerprints a Value's whole access plan.  A loss
     * entry instead has one fixed direct-carrier signature and reads all rank/layout/axis/address
     * facts from Geometry.pack at invocation.  Project only the read/write distinction that is
     * still meaningful to the generic cache validator; occurrence plans remain above this
     * projection for the cold binding and overlap checks.
     */
    private static CpuAccessPlan generatedAccess(boolean input) {
        return new CpuAccessPlan(input ? CpuAccessPlan.AccessKind.READ
                : CpuAccessPlan.AccessKind.WRITE, CpuAccessPlan.Regime.DENSE_LINEAR, 0,
                List.of(), 0);
    }

    private static boolean floating(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }
}
