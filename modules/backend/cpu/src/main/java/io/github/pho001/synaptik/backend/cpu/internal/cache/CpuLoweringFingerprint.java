package io.github.pho001.synaptik.backend.cpu.internal.cache;

import java.util.HexFormat;
import java.util.Objects;

/** Immutable 256-bit canonical lowering fingerprint. */
public final class CpuLoweringFingerprint {
    private final byte[] bytes;
    private CpuLoweringFingerprint(byte[] bytes) { this.bytes = bytes; }

    /**
     * Constructs a fingerprint from exactly 32 bytes.
     * @param bytes non-null 32-byte digest; copied defensively
     * @return a new immutable fingerprint; never {@code null}
     * @throws NullPointerException if {@code bytes} is {@code null}
     * @throws IllegalArgumentException if {@code bytes} does not contain exactly 32 bytes
     */
    public static CpuLoweringFingerprint of(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        if (bytes.length != 32) throw new IllegalArgumentException("fingerprint must contain 32 bytes");
        return new CpuLoweringFingerprint(bytes.clone());
    }

    /**
     * Parses a lowercase or uppercase hexadecimal digest.
     * @param hex 64 hexadecimal characters
     * @return a new immutable fingerprint; never {@code null}
     * @throws NullPointerException if {@code hex} is {@code null}
     * @throws IllegalArgumentException if {@code hex} is not exactly a 32-byte hexadecimal digest
     */
    public static CpuLoweringFingerprint fromHex(String hex) {
        Objects.requireNonNull(hex, "hex");
        try { return of(HexFormat.of().parseHex(hex)); }
        catch (IllegalArgumentException failure) { throw new IllegalArgumentException(
                "fingerprint must be 64 hexadecimal characters", failure); }
    }

    /** Returns digest bytes.
     * @return a new defensive copy of the 32 bytes */
    public byte[] bytes() { return bytes.clone(); }
    /** Returns text form.
     * @return canonical lowercase hexadecimal text; never {@code null} */
    public String hex() { return HexFormat.of().formatHex(bytes); }
    @Override public boolean equals(Object other) {
        return other instanceof CpuLoweringFingerprint that
                && java.util.Arrays.equals(bytes, that.bytes);
    }
    @Override public int hashCode() { return java.util.Arrays.hashCode(bytes); }
    @Override public String toString() { return hex(); }
}
