package io.github.pho001.synaptik.backend.cpu.execution;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.Objects;

/**
 * Immutable content identity of one family-owned canonical typed lowering. The fixed-width
 * SHA-256 value supports deterministic compatibility comparisons; it is not authentication and
 * does not by itself describe an operation, specialization, or cache entry.
 */
final class CpuLoweringFingerprint {
    private final byte[] bytes;

    private CpuLoweringFingerprint(byte[] ownedDigest) {
        this.bytes = ownedDigest;
    }

    /**
     * Hashes one canonical family encoding after validating it.
     *
     * @param canonicalFamilyBytes non-null, non-empty canonical family bytes; copied immediately
     * @return an immutable content fingerprint; never {@code null}
     * @throws NullPointerException if {@code canonicalFamilyBytes} is {@code null}
     * @throws IllegalArgumentException if it is empty
     */
    static CpuLoweringFingerprint of(byte[] canonicalFamilyBytes) {
        Objects.requireNonNull(canonicalFamilyBytes, "canonicalFamilyBytes");
        if (canonicalFamilyBytes.length == 0) {
            throw new IllegalArgumentException("canonicalFamilyBytes must not be empty");
        }
        return new CpuLoweringFingerprint(sha256(canonicalFamilyBytes.clone()));
    }

    /**
     * Retains a trusted, already-computed SHA-256 digest without hashing it again.
     *
     * @param digest non-null byte array containing exactly 32 digest bytes; copied once
     * @return an immutable content fingerprint retaining the supplied digest value
     * @throws NullPointerException if {@code digest} is {@code null}, with message {@code digest}
     * @throws IllegalArgumentException if {@code digest} does not contain exactly 32 bytes
     */
    static CpuLoweringFingerprint fromDigest(byte[] digest) {
        Objects.requireNonNull(digest, "digest");
        if (digest.length != CpuGeneratorSchema.FINGERPRINT_BYTE_COUNT) {
            throw new IllegalArgumentException("digest must contain exactly 32 bytes");
        }
        return new CpuLoweringFingerprint(digest.clone());
    }

    /** Returns the digest without exposing retained mutable state.
     * @return a new defensive copy of the exact 32 digest bytes; never {@code null} */
    byte[] bytes() { return bytes.clone(); }

    @Override public boolean equals(Object other) {
        return this == other || other instanceof CpuLoweringFingerprint that
                && Arrays.equals(bytes, that.bytes);
    }

    @Override public int hashCode() { return Arrays.hashCode(bytes); }

    /** @return fixed-width lowercase hexadecimal diagnostic form; never {@code null} */
    @Override public String toString() { return HexFormat.of().formatHex(bytes); }

    private static byte[] sha256(byte[] value) {
        try {
            return MessageDigest.getInstance("SHA-256").digest(value);
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("SHA-256 is required by the Java platform", impossible);
        }
    }
}
