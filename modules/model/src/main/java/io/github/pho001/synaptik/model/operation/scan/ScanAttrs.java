package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationAttrs;
import java.util.List;
import java.util.OptionalInt;

/**
 * Scalar header for a pure fixed-count scan over axis zero; contains no body or runtime state.
 * Emissions and residual histories use original time coordinates even in reverse traversal.
 * The header validates intrinsic counts only. Producer/node attachment validation checks its
 * agreement with actual positions, formal declarations and canonical result groups; this value
 * neither contains a body nor proves backend capability. Explicit length mode adds one final
 * lengths actual with no body formal, independently of the number of ROW carries.
 * Record equality includes the ordered roles; diagnostic text is not a wire/cache encoding.
 *
 * @param timeCount non-negative static number of steps
 * @param carryCount non-negative number of carried positions; carry plus emission count is positive
 * @param sequenceCount positive number of axis-zero sequences
 * @param invariantCount non-negative number of invariant positions
 * @param emissionCount non-negative number of per-step emissions
 * @param residualCount non-negative number of latent retained-value ports
 * @param reverse whether traversal is decreasing rather than increasing in original time
 * @param lengthAware whether one final lengths actual controls row activity
 * @param rowCarryCount number of leading ROW(0) carries, zero for all-valid mode
 * @param sequenceRowAxes immutable ordered optional step axes; empty for all-valid, otherwise
 *     one non-null entry per sequence, absent for SHARED or a nonnegative ROW axis whose
 *     rank and exact batch coordinate are checked against its declaration
 */
public record ScanAttrs(long timeCount, int carryCount, int sequenceCount, int invariantCount,
        int emissionCount, int residualCount, boolean reverse, boolean lengthAware, int rowCarryCount,
        List<OptionalInt> sequenceRowAxes) implements OperationAttrs {
    /**
     * Preserves the all-valid constructor and its original counts.
     * @param timeCount non-negative static step count
     * @param carryCount non-negative carry count; carry plus emission count is positive
     * @param sequenceCount positive sequence count
     * @param invariantCount non-negative invariant count
     * @param emissionCount non-negative emission count
     * @param residualCount non-negative residual count
     * @param reverse decreasing traversal when true
     * @throws IllegalArgumentException if counts are outside their domains
     * @throws ArithmeticException if total cardinalities overflow
     */
    public ScanAttrs(long timeCount, int carryCount, int sequenceCount, int invariantCount,
            int emissionCount, int residualCount, boolean reverse) {
        this(timeCount, carryCount, sequenceCount, invariantCount, emissionCount, residualCount,
                reverse, false, 0, List.of());
    }
    /**
     * Validates scalar counts, including checked total input and output cardinalities.
     *
     * @param timeCount non-negative static step count
     * @param carryCount non-negative carry count; carry plus emission count is positive
     * @param sequenceCount positive sequence count
     * @param invariantCount non-negative invariant count
     * @param emissionCount non-negative emission count
     * @param residualCount non-negative residual count
     * @param reverse decreasing traversal when true
     * @param lengthAware whether lengths is an additional final actual, without a body formal
     * @param rowCarryCount in {@code [0, carryCount]} for length mode, otherwise zero
     * @param sequenceRowAxes non-null ordered role snapshot; empty for all-valid,
     *     otherwise one non-null optional nonnegative step axis per sequence; declaration
     *     validation checks each present axis against its rank and exact batch Dimension
     * @throws NullPointerException if the role list or an entry is null
     * @throws IllegalArgumentException if a count is outside its declared domain
     * @throws ArithmeticException if a total cardinality overflows an int
     */
    public ScanAttrs {
        sequenceRowAxes = List.copyOf(sequenceRowAxes);
        if (timeCount < 0 || carryCount < 0 || sequenceCount < 1 || invariantCount < 0
                || emissionCount < 0 || residualCount < 0
                || Math.addExact(carryCount, emissionCount) == 0) {
            throw new IllegalArgumentException("invalid scan counts");
        }
        if (rowCarryCount < 0 || rowCarryCount > carryCount
                || (lengthAware ? sequenceRowAxes.size() != sequenceCount
                        : rowCarryCount != 0 || !sequenceRowAxes.isEmpty())
                || sequenceRowAxes.stream().anyMatch(a -> a.isPresent() && a.getAsInt() < 0)) {
            throw new IllegalArgumentException("invalid scan row roles");
        }
        Math.addExact(Math.addExact(Math.addExact(carryCount, sequenceCount), invariantCount),
                lengthAware ? 1 : 0);
        Math.addExact(Math.addExact(carryCount, emissionCount), residualCount);
    }

    /**
     * Counts carry, sequence, invariant and optional final lengths positions without deduplication.
     * @return the checked number of ordered actual input positions
     */
    public int inputCount() {
        return Math.addExact(Math.addExact(Math.addExact(carryCount, sequenceCount), invariantCount),
                lengthAware ? 1 : 0);
    }

    /**
     * Counts final-carry, emission and latent residual ports in producer order.
     * @return the checked number of canonical outer output positions
     */
    public int outputCount() {
        return Math.addExact(Math.addExact(carryCount, emissionCount), residualCount);
    }
}
