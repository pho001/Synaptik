package io.github.pho001.synaptik.nn.checkpoint;

import java.util.Arrays;
import java.util.Objects;

/**
 * Application-verified input-artifact identity, represented by exactly 32 SHA-256 bytes.
 * This immutable value neither verifies an input artifact nor authenticates checkpoint bytes.
 * Presence is significant: absent identity differs from a present all-zero digest.
 *
 * <p>The typed {@link ModelCheckpoint} overloads require a non-null fingerprint. Loading checks
 * stored presence and exact expected bytes; overloads without this type require absence instead.
 * Never obtain the expected identity solely by trusting the checkpoint being loaded. This value
 * owns only its defensive byte snapshot, with no external artifact or resource lifetime.</p>
 */
public final class InputArtifactFingerprint {
    private final byte[] bytes;

    /**
     * Snapshots an identity already verified by the application.
     * @param bytes non-null exact 32-byte SHA-256 digest, defensively copied
     * @throws NullPointerException if bytes is null
     * @throws IllegalArgumentException if the digest is not 32 bytes
     */
    public InputArtifactFingerprint(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        if (bytes.length != 32) throw new IllegalArgumentException("SHA-256 fingerprint must be 32 bytes");
        this.bytes = bytes.clone();
    }

    /**
     * Exposes a detached copy without allowing mutation of this identity.
     * @return a fresh non-null 32-byte digest
     */
    public byte[] bytes() { return bytes.clone(); }

    /**
     * Compares all digest bytes; presence is represented by the enclosing API, not this value.
     * @param other possible fingerprint, including null
     * @return true exactly for another fingerprint with the same 32 bytes
     */
    @Override public boolean equals(Object other) {
        return this == other || other instanceof InputArtifactFingerprint that
                && Arrays.equals(bytes, that.bytes);
    }

    /**
     * Hashes the immutable digest consistently with byte equality.
     * @return hash of all 32 bytes
     */
    @Override public int hashCode() { return Arrays.hashCode(bytes); }
}
