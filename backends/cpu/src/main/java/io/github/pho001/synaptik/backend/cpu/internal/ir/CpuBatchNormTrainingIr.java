package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * CPU-private immutable generated identity for one first-class batch-normalization training
 * occurrence.
 *
 * <p>The identity preserves the five semantic input positions even when equal graph values share
 * one physical read boundary. It also records all facts that can change the generated three-pass,
 * five-output body or its exact-state requirement. Concrete carrier objects, offsets, selected
 * range count, and run identity remain cold invocation facts.</p>
 *
 * @param inputTypes five semantic input data types in input, scale, bias, running-mean, and
 *     running-variance order; snapshotted and never {@code null}
 * @param resultType promoted output and computation data type; never {@code null}
 * @param momentumBits raw result-type bits for momentum, interpreted as the new-batch weight
 * @param epsilonBits raw result-type bits for the finite positive epsilon
 * @param inputRank static input rank, at least two
 * @param channelAxis normalized channel axis in {@code [0, inputRank)}
 * @param algorithmVersion generated arithmetic version; currently exactly {@code 1}
 * @param passCount number of complete non-channel traversals; currently exactly {@code 3}
 * @param reductionCount number of non-channel coordinates in each channel domain
 * @param stateLimbCount exact-sum limb count, or zero only for empty channel work
 * @param scratchSliceBytes bytes in one range-private exact-state slice, or zero only for empty
 *     channel work
 * @param positionToBoundary immutable five-position map to first-occurrence unique input
 *     boundaries
 * @param inputAccesses immutable read plans in unique-boundary order
 * @param outputAccesses immutable write plans in normalized output, next running mean, next
 *     running variance, saved batch mean, and saved inverse-standard-deviation order
 */
public record CpuBatchNormTrainingIr(List<DataType> inputTypes, DataType resultType,
        long momentumBits, long epsilonBits, int inputRank, int channelAxis,
        int algorithmVersion, int passCount, long reductionCount, int stateLimbCount,
        long scratchSliceBytes, List<Integer> positionToBoundary,
        List<CpuAccessPlan> inputAccesses, List<CpuAccessPlan> outputAccesses)
        implements CpuPortableKernelIr {
    /**
     * Validates and snapshots the complete code-shaping identity.
     *
     * @throws NullPointerException if a required list, type, access plan, or mapping entry is
     *     {@code null}
     * @throws IllegalArgumentException if cardinality, promotion, scalar bits, axis, pass,
     *     boundary-map, access, or exact-state facts disagree
     */
    public CpuBatchNormTrainingIr {
        inputTypes = List.copyOf(inputTypes);
        Objects.requireNonNull(resultType, "resultType");
        positionToBoundary = List.copyOf(positionToBoundary);
        inputAccesses = List.copyOf(inputAccesses);
        outputAccesses = List.copyOf(outputAccesses);
        DataType promoted = inputTypes.isEmpty() ? null : inputTypes.getFirst();
        for (int i = 1; i < inputTypes.size(); i++)
            promoted = DataTypePromotion.promoteFloating(promoted, inputTypes.get(i));
        int unique = positionToBoundary.stream().mapToInt(Integer::intValue).max().orElse(-1) + 1;
        if (inputTypes.size() != 5 || positionToBoundary.size() != 5
                || inputAccesses.size() != unique || inputAccesses.isEmpty()
                || outputAccesses.size() != 5 || promoted != resultType || !supported(resultType)
                || inputTypes.stream().anyMatch(t -> !supported(t)) || inputRank < 2
                || channelAxis < 0 || channelAxis >= inputRank || algorithmVersion != 1
                || passCount != 3 || reductionCount < 0
                || (scratchSliceBytes == 0) != (stateLimbCount == 0)
                || scratchSliceBytes > 0 && (reductionCount < 2 || stateLimbCount <= 0
                    || scratchSliceBytes != Math.addExact(Long.BYTES,
                    Math.multiplyExact(Long.BYTES, stateLimbCount)))
                || inputAccesses.stream().anyMatch(a -> a.accessKind() != CpuAccessPlan.AccessKind.READ)
                || outputAccesses.stream().anyMatch(a -> a.accessKind() != CpuAccessPlan.AccessKind.WRITE)
                || !unitInterval(resultType, momentumBits)
                || !positiveFinite(resultType, epsilonBits))
            throw new IllegalArgumentException("batch-normalization training facts disagree");
        int next = 0;
        for (int position = 0; position < 5; position++) {
            int boundary = positionToBoundary.get(position);
            if (boundary < 0 || boundary >= unique || boundary > next)
                throw new IllegalArgumentException("batch-normalization training boundary map disagrees");
            if (boundary == next) next++;
            int first = positionToBoundary.indexOf(boundary);
            if (inputTypes.get(position) != inputTypes.get(first))
                throw new IllegalArgumentException("repeated training boundary type disagrees");
        }
    }

    /**
     * Encodes this instruction-free family at the generator and artifact-cache seam.
     *
     * @return a new immutable kernel IR containing unique inputs, five ordered outputs, one
     *     channel range loop, and the complete structural family identity; never {@code null}
     */
    public CpuKernelIr encodedKernelIr() {
        var values = new ArrayList<CpuKernelIr.Value>();
        for (int boundary = 0; boundary < inputAccesses.size(); boundary++) {
            int position = positionToBoundary.indexOf(boundary);
            values.add(new CpuKernelIr.Value(boundary, inputTypes.get(position),
                    CpuKernelIr.Value.Kind.INPUT, inputAccesses.get(boundary)));
        }
        for (CpuAccessPlan access : outputAccesses) values.add(new CpuKernelIr.Value(values.size(),
                resultType, CpuKernelIr.Value.Kind.OUTPUT, access));
        var stores = new ArrayList<CpuKernelIr.Store>();
        for (int i = values.size() - 5; i < values.size(); i++) stores.add(new CpuKernelIr.Store(i, 0));
        String family = "batch-normalization-training:inputs=" + inputTypes + ":result="
                + resultType + ":momentum=" + Long.toUnsignedString(momentumBits)
                + ":epsilon=" + Long.toUnsignedString(epsilonBits) + ":rank=" + inputRank
                + ":axis=" + channelAxis + ":algorithm=" + algorithmVersion + ":passes="
                + passCount + ":domain=" + reductionCount + ":map=" + positionToBoundary
                + ":limbs=" + stateLimbCount + ":slice=" + scratchSliceBytes;
        return new CpuKernelIr(values, List.of(), new CpuKernelIr.Loop("start", "end"), stores, family);
    }

    /**
     * Returns the deterministic structural identity used by specialization and artifact caching.
     *
     * @return the encoded kernel IR structural key; never {@code null}
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    private static boolean supported(DataType t) {
        return t == DataType.BFLOAT16 || t == DataType.FLOAT32 || t == DataType.FLOAT64;
    }
    private static double value(DataType t, long bits) { return switch (t) {
        case FLOAT64 -> Double.longBitsToDouble(bits); case FLOAT32 -> Float.intBitsToFloat((int) bits);
        case BFLOAT16 -> Float.intBitsToFloat((int) bits << 16); default -> Double.NaN; }; }
    private static boolean bitsFit(DataType t, long bits) { return t == DataType.FLOAT64
            || t == DataType.FLOAT32 && (bits & ~0xffff_ffffL) == 0
            || t == DataType.BFLOAT16 && (bits & ~0xffffL) == 0; }
    private static boolean unitInterval(DataType t, long bits) { double v = value(t, bits);
        return bitsFit(t, bits) && Double.isFinite(v) && v >= 0 && v <= 1; }
    private static boolean positiveFinite(DataType t, long bits) { double v = value(t, bits);
        return bitsFit(t, bits) && Double.isFinite(v) && v > 0; }
}
