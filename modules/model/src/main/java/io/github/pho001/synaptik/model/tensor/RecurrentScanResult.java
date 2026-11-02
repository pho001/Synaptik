package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Dense outputs and final hidden state from one RNN or GRU scan occurrence.
 *
 * <p>Component names denote semantic roles, not a universal producer order. Fixed RNN/GRU
 * construction supplies outputs at port zero and final hidden at port one; a single-carry
 * general RNN Scan supplies its emission at port one and final carry at port zero. This
 * shallowly immutable carrier retains exact references and performs no independent descriptor,
 * provenance, compiler, storage, or execution validation.</p>
 *
 * @param outputs non-null dense original-time-aligned output Tensor
 * @param finalHidden non-null explicit final hidden-state Tensor
 */
public record RecurrentScanResult(Tensor outputs, Tensor finalHidden) {
    /**
     * Retains the exact result wrappers after declaration-order null checks.
     *
     * @param outputs non-null dense output wrapper retained unchanged
     * @param finalHidden non-null final-hidden wrapper retained unchanged
     * @throws NullPointerException if a component is null, checked in declaration order
     */
    public RecurrentScanResult {
        outputs = Objects.requireNonNull(outputs, "outputs");
        finalHidden = Objects.requireNonNull(finalHidden, "finalHidden");
    }
}
