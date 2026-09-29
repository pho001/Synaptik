package io.github.pho001.synaptik.trace.certificate;

import java.util.Objects;

/** Reproducibility metadata independent of accuracy qualification. */
public record LowPrecisionDeterminism(
        String certificateKeyHash,
        String determinismContractId,
        String evidenceDigest,
        Verdict verdict) {
    public LowPrecisionDeterminism {
        certificateKeyHash = LowPrecisionAccuracy.digest(certificateKeyHash, "certificateKeyHash");
        determinismContractId = LowPrecisionAccuracy.required(determinismContractId,
                "determinismContractId");
        evidenceDigest = LowPrecisionAccuracy.digest(evidenceDigest, "evidenceDigest");
        Objects.requireNonNull(verdict, "verdict");
    }

    /** Determinism metadata verdict; it never changes the accuracy verdict. */
    public enum Verdict { PASS, FAIL, UNKNOWN }
}
