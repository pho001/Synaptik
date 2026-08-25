package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private generated identity for one first-class batch-normalization inference occurrence.
 *
 * <p>The identity preserves all five semantic input positions even when exact repeated logical
 * values share one boundary. The selected range form is a generation-time fact; extents,
 * strides, bases, range bounds, carrier instances, and workers remain cold invocation data.</p>
 *
 * @param inputTypes five floating semantic input types in occurrence order
 * @param resultType exact ordered-promotion result and epsilon type
 * @param epsilonBits exact raw result-type epsilon bits
 * @param inputRank input rank, at least two
 * @param channelAxis normalized channel axis
 * @param algorithmVersion arithmetic and traversal version, currently {@code 1}
 * @param rangeForm selected channel or flattened non-channel range ownership
 * @param positionToBoundary five-position map to unique input boundaries
 * @param inputAccesses unique read access plans in first-occurrence order
 * @param outputAccess sole write access plan
 */
public record CpuBatchNormInferenceIr(List<DataType> inputTypes, DataType resultType,
        long epsilonBits, int inputRank, int channelAxis, int algorithmVersion,
        RangeForm rangeForm, List<Integer> positionToBoundary,
        List<CpuAccessPlan> inputAccesses, CpuAccessPlan outputAccess)
        implements CpuPortableKernelIr {
    /** Generated range ownership form. */
    public enum RangeForm {
        /** Start and end are channel coordinates. */ CHANNEL_RANGE,
        /** Start and end are flattened non-channel coordinates. */ NON_CHANNEL_RANGE
    }

    /**
     * Validates and snapshots the complete code-shaping identity.
     *
     * @throws NullPointerException if a required component or list element is {@code null}
     * @throws IllegalArgumentException if types, promotion, epsilon, rank, axis, access plans,
     *     algorithm version, or first-occurrence boundary mapping disagree
     */
    public CpuBatchNormInferenceIr {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        Objects.requireNonNull(rangeForm, "rangeForm");
        positionToBoundary = List.copyOf(positionToBoundary);
        inputAccesses = List.copyOf(inputAccesses);
        Objects.requireNonNull(outputAccess, "outputAccess");
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int index = 1; index < inputTypes.size(); index++) {
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(index));
        }
        int unique = positionToBoundary.stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
        if (inputTypes.size() != 5 || positionToBoundary.size() != 5
                || inputRank < 2 || channelAxis < 0 || channelAxis >= inputRank
                || algorithmVersion != 1 || promoted != resultType || !supported(resultType)
                || inputTypes.stream().anyMatch(type -> !supported(type))
                || unique != inputAccesses.size() || inputAccesses.isEmpty()
                || inputAccesses.stream().anyMatch(access ->
                    access.accessKind() != CpuAccessPlan.AccessKind.READ)
                || outputAccess.accessKind() != CpuAccessPlan.AccessKind.WRITE
                || !positiveFiniteEpsilon(resultType, epsilonBits)) {
            throw new IllegalArgumentException("batch-normalization inference facts disagree");
        }
        int next = 0;
        for (int position = 0; position < positionToBoundary.size(); position++) {
            int boundary = positionToBoundary.get(position);
            if (boundary < 0 || boundary >= inputAccesses.size() || boundary > next) {
                throw new IllegalArgumentException("batch-normalization boundary map disagrees");
            }
            if (boundary == next) next++;
            int first = positionToBoundary.indexOf(boundary);
            if (inputTypes.get(position) != inputTypes.get(first)) {
                throw new IllegalArgumentException(
                        "repeated batch-normalization boundary type disagrees");
            }
        }
    }

    /**
     * Returns the same semantic identity with the preparation-selected range form.
     *
     * @param selected non-null selected form
     * @return a new immutable identity, or this identity when unchanged
     * @throws NullPointerException if {@code selected} is {@code null}
     */
    public CpuBatchNormInferenceIr withRangeForm(RangeForm selected) {
        Objects.requireNonNull(selected, "selected");
        return selected == rangeForm ? this : new CpuBatchNormInferenceIr(inputTypes, resultType,
                epsilonBits, inputRank, channelAxis, algorithmVersion, selected,
                positionToBoundary, inputAccesses, outputAccess);
    }

    /**
     * Encodes this instruction-free family at the generator/cache seam.
     *
     * @return a new immutable canonical kernel identity
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int boundary = 0; boundary < inputAccesses.size(); boundary++) {
            int position = positionToBoundary.indexOf(boundary);
            values.add(new CpuKernelIr.Value(boundary, inputTypes.get(position),
                    CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(boundary)));
        }
        values.add(new CpuKernelIr.Value(values.size(), resultType,
                CpuKernelIr.Value.Kind.OUTPUT, outputAccess));
        String family = "batch-normalization-inference:inputs=" + inputTypes
                + ":result=" + resultType + ":epsilon=" + Long.toUnsignedString(epsilonBits)
                + ":rank=" + inputRank + ":axis=" + channelAxis
                + ":algorithm=" + algorithmVersion + ":map=" + positionToBoundary
                + ":range=" + rangeForm;
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(values.size() - 1, 0)), family);
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
