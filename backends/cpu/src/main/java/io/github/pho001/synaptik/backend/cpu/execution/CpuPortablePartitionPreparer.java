package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Deterministically validates and selects the first target-eligible portable CPU candidate.
 *
 * <p>Planning has already selected CPU ownership. This analyzer checks that ownership, validates
 * each candidate against the projected values, hard-filters target eligibility, and selects the
 * first eligible candidate in source order. Backend-declared byte geometry remains opaque here;
 * shared Prepare validates declaration-to-assignment geometry during finalization. Analysis
 * performs no artifact access, generation, allocation, binding, execution, or measurement.</p>
 */
final class CpuPortablePartitionPreparer implements BackendPartitionPreparer<
        CpuPortableAnalysisInputs, CpuPortablePreparationPlan> {
    private final CpuPortableCandidateSource candidateSource;

    /**
     * Creates a preparer with one direct candidate collaboration.
     *
     * @param candidateSource non-null direct family-owned candidate collaboration retained by
     *     exact reference
     * @throws NullPointerException if {@code candidateSource} is {@code null}
     */
    CpuPortablePartitionPreparer(CpuPortableCandidateSource candidateSource) {
        this.candidateSource = Objects.requireNonNull(candidateSource, "candidateSource");
    }

    /**
     * Validates CPU ownership and candidate/context agreement before selecting in source order.
     *
     * @param context non-null complete immutable CPU analysis context
     * @return immutable analysis retaining the exact partition and selected declarations
     * @throws NullPointerException if the context, candidate list, or an entry is null
     * @throws IllegalArgumentException if ownership, candidate facts, or eligibility fail closed
     */
    @Override
    public BackendPartitionAnalysis<CpuPortablePreparationPlan> analyze(
            PrepareContext<CpuPortableAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (!context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        List<CpuPortablePartitionCandidate> candidates =
                Objects.requireNonNull(candidateSource.candidates(context), "candidates");
        var values = new HashMap<ValueId, GraphValue>();
        for (var value : context.values()) values.put(value.id(), value);
        for (int index = 0; index < candidates.size(); index++) {
            CpuPortablePartitionCandidate candidate =
                    Objects.requireNonNull(candidates.get(index), "candidates[" + index + "]");
            validateCandidate(index, candidate, values);
            if (!eligible(candidate, context.backendInputs())) continue;
            var plan = new CpuPortablePreparationPlan(
                    candidate, context.backendInputs().parallelConfiguration());
            return new BackendPartitionAnalysis<>(
                    context.partition(), plan, candidate.requirements());
        }
        throw new IllegalArgumentException("no supported CPU portable candidate");
    }

    private static void validateCandidate(
            int candidateIndex,
            CpuPortablePartitionCandidate candidate,
            Map<ValueId, GraphValue> values) {
        for (int index = 0; index < candidate.requirements().size(); index++) {
            PreparationResourceRequirement requirement = candidate.requirements().get(index);
            if (requirement instanceof PreparationResourceRequirement.Buffer buffer
                    && !values.containsKey(buffer.valueId())) {
                throw new IllegalArgumentException(
                        "candidates[" + candidateIndex + "].requirements[" + index
                                + "] value is not projected: " + buffer.valueId());
            }
        }
        for (int kernelIndex = 0; kernelIndex < candidate.kernels().size(); kernelIndex++) {
            var kernel = candidate.kernels().get(kernelIndex);
            for (int index = 0; index < kernel.bufferUses().size(); index++) {
                var use = kernel.bufferUses().get(index);
                var value = values.get(use.requirement().valueId());
                var argument = kernel.specialization().arguments().get(index);
                if (value == null) {
                    throw new IllegalArgumentException(
                            "candidates[" + candidateIndex + "].kernels[" + kernelIndex
                                    + "].bufferUses[" + index + "] value is not projected: "
                                    + use.requirement().valueId());
                }
                if (value.descriptor().dataType() != argument.dataType()) {
                    String location = candidate.kernels().size() == 1
                            ? "candidates[" + candidateIndex + "].bufferUses[" + index + "]"
                            : "candidates[" + candidateIndex + "].kernels[" + kernelIndex
                                    + "].bufferUses[" + index + "]";
                    throw new IllegalArgumentException(
                            location + " data type does not match specialization argument");
                }
            }
        }
    }

    private static boolean eligible(
            CpuPortablePartitionCandidate candidate, CpuPortableAnalysisInputs inputs) {
        for (var kernel : candidate.kernels()) {
            var specialization = kernel.specialization();
            if (specialization.executionMode().vectorized()
                    && !inputs.supportedVectorShapes().contains(
                            specialization.vectorShape().orElseThrow())) return false;
            if (specialization.executionMode().parallel()
                    && inputs.parallelConfiguration().workerCount() <= 0) return false;
        }
        return true;
    }
}
