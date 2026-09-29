package io.github.pho001.synaptik.trace.certificate;

import java.util.Objects;

/** Immutable schema-versioned certificate bundle with deterministic fail-closed eligibility. */
public record LowPrecisionCertificate(
        int schemaVersion,
        LowPrecisionCertificateKey key,
        LowPrecisionAccuracy accuracy,
        LowPrecisionDeterminism determinism) {
    /**
     * Constructs an explicitly schema-1 certificate bundle.
     *
     * @param key non-null complete certificate key
     * @param accuracy non-null accuracy qualification
     * @param determinism non-null separate determinism metadata
     * @return immutable schema-1 bundle
     */
    public static LowPrecisionCertificate schema1(
            LowPrecisionCertificateKey key,
            LowPrecisionAccuracy accuracy,
            LowPrecisionDeterminism determinism) {
        return new LowPrecisionCertificate(1, key, accuracy, determinism);
    }

    public LowPrecisionCertificate {
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(accuracy, "accuracy");
        Objects.requireNonNull(determinism, "determinism");
    }

    /**
     * Returns the accuracy eligibility for an exact expected route/environment identity.
     *
     * @param expected complete expected key, or {@code null} for no matching expectation
     * @return {@link Status#CERTIFIED} only for schema one, exact key equality, and accuracy PASS
     */
    public Status statusFor(LowPrecisionCertificateKey expected) {
        if (schemaVersion != LowPrecisionCertificateKey.SCHEMA_VERSION
                || expected == null || !key.equals(expected)
                || accuracy.verdict() != LowPrecisionAccuracy.Verdict.PASS) {
            return Status.NOT_CERTIFIED;
        }
        return Status.CERTIFIED;
    }

    /**
     * Checks determinism metadata separately; it never changes {@link #statusFor}.
     *
     * @return whether the determinism record names this certificate key
     */
    public DeterminismStatus determinismStatus() {
        return determinism.certificateKeyHash().equals(key.digest())
                ? DeterminismStatus.ASSOCIATED : DeterminismStatus.MISMATCHED;
    }


    /** Eligibility result used by preparation and route selection; no fallback is implied. */
    public enum Status { CERTIFIED, NOT_CERTIFIED }
    /** Independent association status for determinism metadata. */
    public enum DeterminismStatus { ASSOCIATED, MISMATCHED }
}
