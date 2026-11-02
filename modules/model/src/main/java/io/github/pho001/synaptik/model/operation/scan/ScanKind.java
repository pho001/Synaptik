package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Pure fixed-count recurrence with explicit carries, axis-zero sequences and invariants.
 * Each step binds prior carry and original-time sequence slices, computes next carry and emissions,
 * then advances. At zero steps final carry preserves initial represented values, and histories
 * are empty. This describes source/IR meaning, not execution, differentiation or backend support.
 */
public enum ScanKind implements OperationKind {
    /** One contained pure ordinary-Tensor body, with distinct canonical outer result ports. */
    SCAN;

    private static final List<OperationSignature> SIGNATURES = List.of(new OperationSignature(
            ScanAttrs.class, 1, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

    /**
     * Describes the scalar-header variant; attachment seams validate exact per-occurrence counts.
     * @return the immutable scalar-header structural signature, without capability claims
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
