package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Dense outputs and final hidden state from one RNN or GRU scan occurrence.
 *
 * <p>Tensor construction supplies canonical producer slots zero and one. This shallowly immutable
 * carrier retains the exact references and performs no independent descriptor, provenance,
 * compiler, storage, or execution validation.</p>
 *
 * @param outputs non-null dense original-time-aligned output Tensor at producer slot zero
 * @param finalHidden non-null explicit final hidden-state Tensor at producer slot one
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
