package io.github.pho001.synaptik.trace.certificate;

import java.util.Objects;

/** Accuracy qualification kept separate from certificate identity and determinism evidence. */
public record LowPrecisionAccuracy(
        String envelopeId,
        QualificationMethod qualificationMethod,
        String evidenceDigest,
        Verdict verdict) {
    public LowPrecisionAccuracy {
        envelopeId = required(envelopeId, "envelopeId");
        Objects.requireNonNull(qualificationMethod, "qualificationMethod");
        evidenceDigest = digest(evidenceDigest, "evidenceDigest");
        Objects.requireNonNull(verdict, "verdict");
    }

    /** Allowed Model-owned qualification methods. */
    public enum QualificationMethod { DEDUCTIVE_PROOF, CERTIFIED_ENVELOPE }
    /** Accuracy verdict; only PASS can qualify a route. */
    public enum Verdict { PASS, FAIL }

    static String required(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isEmpty()) {
            throw new IllegalArgumentException(name + " must be a non-empty canonical field");
        }
        for (int index = 0; index < value.length(); index++) {
            if (Character.isISOControl(value.charAt(index))) {
                throw new IllegalArgumentException(name + " must be a non-empty canonical field");
            }
        }
        return value;
    }

    static String digest(String value, String name) {
        value = required(value, name);
        if (!value.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException(name + " must be a SHA-256 hexadecimal digest");
        }
        return value.toLowerCase(java.util.Locale.ROOT);
    }
}
