package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private structural identity for one explicit trailing Layer or root-mean-square (RMS)
 * normalization body.
 *
 * <p>The record contains only immutable code-shaping facts. Logical slice ranges, carrier
 * instances, assigned slots, concrete addresses, workers, and run state remain cold invocation
 * facts. Repeated semantic inputs are represented once in {@code inputAccesses} and recovered
 * through {@code positionToBoundary}.</p>
 *
 * @param kind exact first-class Layer or RMS family
 * @param form exact one-, two-, or three-input semantic form
 * @param inputTypes non-null semantic input types in occurrence order; copied defensively
 * @param resultType exact promoted output and epsilon type
 * @param epsilonBits exact raw result-type epsilon bits
 * @param normalizedRank positive number of trailing normalized axes
 * @param algorithmVersion emitted numerical algorithm identity; currently {@code 1}
 * @param passCount direct pass count; three for Layer and two for RMS
 * @param normalizedCount non-negative logical element count in one normalized slice
 * @param stateLimbCount exact-sum state limb count, or zero when no exact state is used
 * @param scratchSliceBytes bytes in one Layer exact-state slice, or zero for RMS and empty work
 * @param positionToBoundary semantic-position to unique-input-boundary mapping in first-use order;
 *     copied defensively
 * @param inputAccesses unique read-only input access plans in boundary order; copied defensively
 * @param outputAccess distinct write-only output access plan
 */
public record CpuTrailingNormalizationIr(Kind kind, Form form, List<DataType> inputTypes,
        DataType resultType, long epsilonBits, int normalizedRank, int algorithmVersion,
        int passCount, long normalizedCount, int stateLimbCount, long scratchSliceBytes,
        List<Integer> positionToBoundary, List<CpuAccessPlan> inputAccesses,
        CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {
    /** Exact first-class normalization family. */
    public enum Kind { /** Population layer normalization. */ LAYER,
        /** Uncentered root-mean-square normalization. */ RMS }
    /** Exact visible operand form. */
    public enum Form { /** Layer input only. */ LAYER,
        /** Layer input, scale, and bias. */ LAYER_AFFINE,
        /** RMS input only. */ RMS, /** RMS input and scale. */ RMS_SCALED }

    /**
     * Validates and snapshots one emitted-body identity.
     *
     * @throws NullPointerException if a required reference or list element is {@code null}
     * @throws IllegalArgumentException if family, form, promotion, epsilon, pass, resource,
     *     boundary-map, or access facts disagree
     * @throws ArithmeticException if exact scratch geometry overflows
     */
    public CpuTrailingNormalizationIr {
        Objects.requireNonNull(kind, "kind"); Objects.requireNonNull(form, "form");
        inputTypes = List.copyOf(inputTypes); Objects.requireNonNull(resultType, "resultType");
        positionToBoundary = List.copyOf(positionToBoundary);
        inputAccesses = List.copyOf(inputAccesses); Objects.requireNonNull(outputAccess, "outputAccess");
        int expected = switch (form) { case LAYER, RMS -> 1; case RMS_SCALED -> 2;
            case LAYER_AFFINE -> 3; };
        boolean floating = supported(resultType);
        int uniqueBoundaryCount = positionToBoundary.stream()
                .mapToInt(Integer::intValue).max().orElse(-1) + 1;
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int index = 1; index < inputTypes.size(); index++) {
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
        }
        if (!floating || inputTypes.size() != expected || positionToBoundary.size() != expected
                || inputAccesses.isEmpty() || uniqueBoundaryCount != inputAccesses.size()
                || inputTypes.stream().anyMatch(type -> !supported(type)) || promoted != resultType
                || (kind == Kind.LAYER) != (form == Form.LAYER || form == Form.LAYER_AFFINE)
                || normalizedRank <= 0 || algorithmVersion != 1
                || passCount != (kind == Kind.LAYER ? 3 : 2) || normalizedCount < 0
                || stateLimbCount < 0 || scratchSliceBytes < 0
                || (kind == Kind.RMS && (stateLimbCount != 0 || scratchSliceBytes != 0))
                || (kind == Kind.LAYER
                    && ((scratchSliceBytes == 0) != (stateLimbCount == 0)))
                || (kind == Kind.LAYER && scratchSliceBytes > 0 && (stateLimbCount <= 0
                    || scratchSliceBytes != Math.addExact(Long.BYTES,
                        Math.multiplyExact(Long.BYTES, stateLimbCount))))
                || inputAccesses.stream().anyMatch(access -> access.accessKind()
                    != CpuAccessPlan.AccessKind.READ)
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE)
            throw new IllegalArgumentException("trailing-normalization structural facts disagree");
        int nextBoundary = 0;
        for (int position = 0; position < positionToBoundary.size(); position++) {
            int index = positionToBoundary.get(position);
            if (index < 0 || index >= inputAccesses.size() || index > nextBoundary)
                throw new IllegalArgumentException("trailing-normalization boundary map disagrees");
            if (index == nextBoundary) nextBoundary++;
            int first = positionToBoundary.indexOf(index);
            if (inputTypes.get(position) != inputTypes.get(first))
                throw new IllegalArgumentException(
                        "repeated trailing-normalization boundary type disagrees");
        }
        if (!positiveFiniteEpsilon(resultType, epsilonBits))
            throw new IllegalArgumentException("trailing-normalization epsilon bits disagree");
    }

    /**
     * Encodes the exact code-shaping facts used for generated-artifact compatibility.
     *
     * @return a new immutable instruction-free kernel identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new java.util.ArrayList<CpuKernelIr.Value>();
        for (int i = 0; i < inputAccesses.size(); i++) {
            int position = positionToBoundary.indexOf(i);
            values.add(new CpuKernelIr.Value(i, inputTypes.get(position),
                    CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(i)));
        }
        values.add(new CpuKernelIr.Value(values.size(), resultType,
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)),
                "trailing-normalization:" + kind + ":form=" + form + ":inputs=" + inputTypes
                    + ":result=" + resultType + ":epsilon=" + Long.toUnsignedString(epsilonBits)
                    + ":rank=" + normalizedRank + ":algorithm=" + algorithmVersion
                    + ":passes=" + passCount + ":domain=" + normalizedCount
                    + ":map=" + positionToBoundary + ":limbs=" + stateLimbCount
                    + ":slice=" + scratchSliceBytes);
    }

    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    private static boolean supported(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
    }

    private static boolean positiveFiniteEpsilon(DataType type, long bits) {
        return switch (type) {
            case FLOAT64 -> Double.isFinite(Double.longBitsToDouble(bits))
                    && Double.longBitsToDouble(bits) > 0.0;
            case FLOAT32 -> (bits & ~0xffff_ffffL) == 0
                    && Float.isFinite(Float.intBitsToFloat((int) bits))
                    && Float.intBitsToFloat((int) bits) > 0.0f;
            case BFLOAT16 -> (bits & ~0xffffL) == 0
                    && Float.isFinite(Float.intBitsToFloat((int) bits << 16))
                    && Float.intBitsToFloat((int) bits << 16) > 0.0f;
            default -> false;
        };
    }
}
