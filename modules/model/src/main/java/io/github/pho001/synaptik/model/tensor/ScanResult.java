package io.github.pho001.synaptik.model.tensor;

import java.util.List;
import java.util.Objects;

/**
 * Immutable ordered groups of scan ports; descriptors describe histories but allocate no storage.
 * Scan construction returns exact canonical wrappers in final-carry, emission, residual order.
 * This carrier alone checks containers, not provenance agreement of independently supplied values.
 * It permits empty groups and repeated references when constructed independently. Record equality
 * and hashing include all ordered group contents; the lists retain exact Tensor identities and
 * their mutable host-storage associations are not frozen by the carrier.
 *
 * @param finalCarry non-null immutable snapshot of non-null ordered final carried values
 * @param outputs non-null immutable snapshot of non-null ordered axis-zero stacked emissions
 * @param residuals non-null immutable snapshot of non-null ordered latent history ports, with
 *     previous carry denoting state before each step rather than allocated saved storage
 */
public record ScanResult(List<Tensor> finalCarry, List<Tensor> outputs, List<Tensor> residuals) {
    /**
     * Snapshots groups without deduplicating or reconstructing their wrappers.
     *
     * @param finalCarry non-null final-carry list with non-null elements
     * @param outputs non-null emission list with non-null elements
     * @param residuals non-null residual list with non-null elements
     * @throws NullPointerException if any list or element is null
     */
    public ScanResult {
        finalCarry = List.copyOf(Objects.requireNonNull(finalCarry, "finalCarry"));
        outputs = List.copyOf(Objects.requireNonNull(outputs, "outputs"));
        residuals = List.copyOf(Objects.requireNonNull(residuals, "residuals"));
    }
}
