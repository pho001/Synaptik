package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;
import java.util.Optional;

/**
 * Cold, fail-closed admission for the CPU-private modular partial-reduction route.
 *
 * <p>This owner deliberately accepts only facts already proved by ordinary aggregate lowering and
 * preparation.  In particular, an evidence failure is not approximated or inferred from another
 * form: it returns an empty result so the established complete-output-cell route remains selected.
 * It performs no carrier inspection, worker submission, allocation, or runtime route choice.</p>
 */
public final class CpuPartialReductionLowering {
    /** Creates a stateless cold admission owner. */
    public CpuPartialReductionLowering() { }

    /**
     * Returns partial-reduction facts only when every narrow route proof is present.
     *
     * @param inputs non-null immutable cold admission facts
     * @return an admitted partial identity, or empty when the complete-cell route must remain
     * @throws NullPointerException if {@code inputs} is {@code null}
     */
    public Optional<CpuPartialReductionIr> admit(AdmissionInputs inputs) {
        Objects.requireNonNull(inputs, "inputs");
        if (!inputs.densePrimitiveArrays || !inputs.injectiveDenseOutput || !inputs.workerAccessible
                || !inputs.performanceEvidencePassed || inputs.availableWorkers < inputs.partialCount
                || inputs.domainCount < inputs.partialCount
                || inputs.domainCount / inputs.partialCount < inputs.minimumElementsPerWorker
                || inputs.outputCount <= 0 || inputs.domainCount <= 0
                || inputs.form == CpuAggregateIr.Form.SUM_TO_SHAPE
                || (inputs.dataType != DataType.INT32 && inputs.dataType != DataType.INT64)
                || (inputs.kind != CpuAggregateIr.Kind.SUM && inputs.kind != CpuAggregateIr.Kind.PROD)) {
            return Optional.empty();
        }
        var kind = inputs.kind == CpuAggregateIr.Kind.SUM ? CpuPartialReductionIr.Kind.SUM
                : CpuPartialReductionIr.Kind.PROD;
        try {
            return Optional.of(new CpuPartialReductionIr(kind, inputs.dataType, inputs.form,
                    inputs.outputCount, inputs.domainCount, inputs.partialCount));
        } catch (IllegalArgumentException | ArithmeticException rejected) {
            return Optional.empty();
        }
    }

    /**
     * Immutable prepared-fact input to one cold admission decision.
     *
     * @param kind non-null ordinary aggregate kind
     * @param dataType non-null represented type
     * @param form non-null aggregate form
     * @param outputCount non-negative output-cell count
     * @param domainCount non-negative selected-domain count per cell
     * @param partialCount requested fixed partial count, two or four
     * @param minimumElementsPerWorker positive prepared lower bound per partial
     * @param availableWorkers non-negative available worker capacity
     * @param densePrimitiveArrays whether both boundaries are primitive arrays
     * @param injectiveDenseOutput whether the output is dense and injective
     * @param workerAccessible whether the prepared worker group is available
     * @param performanceEvidencePassed whether trusted evidence admits this exact row
     */
    public record AdmissionInputs(CpuAggregateIr.Kind kind, DataType dataType,
            CpuAggregateIr.Form form, long outputCount, long domainCount, int partialCount,
            long minimumElementsPerWorker, int availableWorkers, boolean densePrimitiveArrays,
            boolean injectiveDenseOutput, boolean workerAccessible,
            boolean performanceEvidencePassed) {
        /**
         * Validates facts before the route selector observes them.
         *
         * @throws NullPointerException if a semantic component is {@code null}
         * @throws IllegalArgumentException if a count is negative or partial count is not 2/4
         */
        public AdmissionInputs {
            Objects.requireNonNull(kind, "kind");
            Objects.requireNonNull(dataType, "dataType");
            Objects.requireNonNull(form, "form");
            if (outputCount < 0 || domainCount < 0 || minimumElementsPerWorker <= 0
                    || availableWorkers < 0 || (partialCount != 2 && partialCount != 4)) {
                throw new IllegalArgumentException("partial-reduction admission facts disagree");
            }
        }
    }
}
