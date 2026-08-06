package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.Objects;

/**
 * Immutable route-independent selection of one input copy into contiguous FLOAT64 workspace.
 *
 * <p>The source remains an ordinary read-only boundary buffer. The consumer binding describes
 * the canonical dense view presented to the generated kernel after the copy. The analysis-local
 * workspace identity and cost evidence are cold facts: they neither create a graph value nor
 * participate in generated-class identity.
 *
 * @param sourceBoundaryIndex zero-based copied input position; currently {@code 0}, {@code 1}, or
 *     {@code 2}
 * @param sourceValue non-null original graph-value identity retained for diagnostics only
 * @param sourceBinding non-null normalized read geometry used by the copy
 * @param consumerBinding non-null canonical dense, zero-offset read geometry used by the consumer
 * @param elementCount non-negative number of logical FLOAT64 elements copied per invocation
 * @param byteCount exact {@code elementCount * Double.BYTES} workspace size
 * @param workspaceRequirementId analysis-local workspace identity; exactly {@code 0}
 * @param byteAlignment workspace alignment in bytes; exactly {@link Double#BYTES}
 * @param useCount positive use count derived from the lowered unit
 * @param expectedRunCount positive cold-policy repeated-run estimate
 * @param directCost non-negative estimated total direct-access cost
 * @param copyCost non-negative estimated cost of one copy for one run
 * @param contiguousCost non-negative estimated contiguous-consumer cost for one use and run
 * @param copiedTotalCost non-negative estimated total copy-plus-consumer cost
 * @param netBenefit positive difference between direct and copied total cost
 * @param benefitBasisPoints selected benefit from {@code 0} through {@code 10_000} basis points
 * @param selectionReason non-null cold diagnostic explanation
 */
public record CpuMaterializationPlan(int sourceBoundaryIndex, ValueId sourceValue,
        CpuAccessPlan.Binding sourceBinding, CpuAccessPlan.Binding consumerBinding,
        long elementCount, long byteCount, long workspaceRequirementId, long byteAlignment,
        long useCount, long expectedRunCount, long directCost, long copyCost,
        long contiguousCost, long copiedTotalCost, long netBenefit,
        int benefitBasisPoints, String selectionReason) {
    /**
     * Validates one selected copy fact and its canonical consumer geometry.
     *
     * @throws NullPointerException if a reference component is {@code null}
     * @throws IllegalArgumentException if an identity, geometry, cost, or benefit invariant is
     *     outside the current one-input FLOAT64 materialization contract
     * @throws ArithmeticException if checked byte geometry overflows {@code long}
     */
    public CpuMaterializationPlan {
        if (sourceBoundaryIndex < 0 || sourceBoundaryIndex > 2 || elementCount < 0
                || byteCount != Math.multiplyExact(elementCount, Double.BYTES)
                || workspaceRequirementId != 0 || byteAlignment != Double.BYTES
                || useCount <= 0 || expectedRunCount <= 0 || directCost < 0 || copyCost < 0
                || contiguousCost < 0 || copiedTotalCost < 0 || netBenefit <= 0
                || benefitBasisPoints < 0 || benefitBasisPoints > 10_000) {
            throw new IllegalArgumentException("invalid CPU materialization plan");
        }
        Objects.requireNonNull(sourceValue, "sourceValue");
        Objects.requireNonNull(sourceBinding, "sourceBinding");
        Objects.requireNonNull(consumerBinding, "consumerBinding");
        Objects.requireNonNull(selectionReason, "selectionReason");
        if (sourceBinding.plan().accessKind() != CpuAccessPlan.AccessKind.READ
                || consumerBinding.plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                || consumerBinding.baseElementOffset() != 0
                || consumerBinding.referencedElementSpan() != elementCount) {
            throw new IllegalArgumentException("materialized consumer must be canonical dense read");
        }
    }
}
