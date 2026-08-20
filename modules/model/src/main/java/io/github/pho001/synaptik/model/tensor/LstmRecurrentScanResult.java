package io.github.pho001.synaptik.model.tensor;

import java.util.Objects;

/**
 * Dense outputs and explicit final hidden and cell states from one LSTM scan occurrence.
 *
 * <p>Tensor construction supplies canonical producer slots zero through two. This shallowly
 * immutable carrier retains the exact references and performs no independent descriptor,
 * provenance, compiler, storage, or execution validation.</p>
 *
 * @param outputs non-null dense original-time-aligned output Tensor at producer slot zero
 * @param finalHidden non-null explicit final hidden-state Tensor at producer slot one
 * @param finalCell non-null explicit final cell-state Tensor at producer slot two
 */
public record LstmRecurrentScanResult(Tensor outputs, Tensor finalHidden, Tensor finalCell) {
    /**
     * Retains the exact result wrappers after declaration-order null checks.
     *
     * @param outputs non-null dense output wrapper retained unchanged
     * @param finalHidden non-null final-hidden wrapper retained unchanged
     * @param finalCell non-null final-cell wrapper retained unchanged
     * @throws NullPointerException if a component is null, checked in declaration order
     */
    public LstmRecurrentScanResult {
        outputs = Objects.requireNonNull(outputs, "outputs");
        finalHidden = Objects.requireNonNull(finalHidden, "finalHidden");
        finalCell = Objects.requireNonNull(finalCell, "finalCell");
    }
}
