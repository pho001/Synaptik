package io.github.pho001.synaptik.backend.metal;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/**
 * Immutable prepare-time identity of the exact Metal route-certification environment.
 *
 * <p>The native bridge supplies the device, operating-system, SDK/framework, compiler, and graph
 * option fields from the context and binary that will execute the prepared route. Java hashes the
 * caller-selected dylib immediately before and after native loading and accepts the identity only
 * when both digests agree. The capability hash is the checked-in provider-derived ledger identity,
 * not a runtime capability claim.</p>
 */
record MetalCertificationEnvironment(
        String gpuFamily,
        String osBuild,
        String sdkVersion,
        String compilerVersion,
        String binaryDigest,
        String flagsOptions,
        String capabilityManifestHash,
        boolean certifiable) {
    static final String CAPABILITY_MANIFEST_HASH =
            "4a7871f0b4f11e44c7b3727acae45ecccbf0722dcb8f4c6460d453dbe620d61e";
    static final String NATIVE_RECORD_MAGIC = "SYNAPTIK_METAL_CERTIFICATION_ENVIRONMENT_V1";
    private static final String ZERO_DIGEST = "0".repeat(64);

    MetalCertificationEnvironment {
        gpuFamily = field(gpuFamily, "gpuFamily");
        osBuild = field(osBuild, "osBuild");
        sdkVersion = field(sdkVersion, "sdkVersion");
        compilerVersion = field(compilerVersion, "compilerVersion");
        binaryDigest = digest(binaryDigest, "binaryDigest");
        flagsOptions = field(flagsOptions, "flagsOptions");
        capabilityManifestHash = digest(capabilityManifestHash, "capabilityManifestHash");
    }

    /**
     * Parses the exact native record and binds it to a stable dylib byte digest.
     *
     * @param nativeRecord non-null six-field LF-separated native identity record
     * @param beforeDigest non-null SHA-256 observed before native loading
     * @param afterDigest non-null SHA-256 observed after context creation
     * @return a certifiable immutable environment identity
     * @throws IllegalArgumentException if the record or either digest is malformed or the dylib
     *     changed while opening
     */
    static MetalCertificationEnvironment parse(
            String nativeRecord, String beforeDigest, String afterDigest) {
        Objects.requireNonNull(nativeRecord, "nativeRecord");
        beforeDigest = digest(beforeDigest, "beforeDigest");
        afterDigest = digest(afterDigest, "afterDigest");
        if (!beforeDigest.equals(afterDigest)) {
            throw new IllegalArgumentException("Metal native library changed while opening");
        }
        String[] fields = nativeRecord.split("\\n", -1);
        if (fields.length != 6 || !fields[0].equals(NATIVE_RECORD_MAGIC)) {
            throw new IllegalArgumentException("Metal certification environment record is malformed");
        }
        return new MetalCertificationEnvironment(
                fields[1], fields[2], fields[3], fields[4], beforeDigest, fields[5],
                CAPABILITY_MANIFEST_HASH, true);
    }

    /**
     * Computes the SHA-256 of one exact regular dylib path.
     *
     * @param path non-null absolute native library path
     * @return lowercase 64-digit SHA-256
     * @throws IllegalArgumentException if the path cannot be read as one regular file
     */
    static String sha256(Path path) {
        Objects.requireNonNull(path, "path");
        if (!path.isAbsolute() || !Files.isRegularFile(path)) {
            throw new IllegalArgumentException("Metal native library must be an absolute regular file");
        }
        MessageDigest digest = sha256();
        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) >= 0) {
                if (read != 0) {
                    digest.update(buffer, 0, read);
                }
            }
        } catch (IOException failure) {
            throw new IllegalArgumentException("Metal native library cannot be hashed", failure);
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    /** Returns a deliberately non-certifiable identity for injected unit-test native seams. */
    static MetalCertificationEnvironment unavailable() {
        return new MetalCertificationEnvironment(
                "UNAVAILABLE", "UNAVAILABLE", "UNAVAILABLE", "UNAVAILABLE", ZERO_DIGEST,
                "UNAVAILABLE", CAPABILITY_MANIFEST_HASH, false);
    }

    private static String field(String value, String name) {
        Objects.requireNonNull(value, name);
        if (value.isEmpty()) {
            throw new IllegalArgumentException(name + " must be a non-empty canonical field");
        }
        for (int index = 0; index < value.length(); index++) {
            if (Character.isISOControl(value.charAt(index))) {
                throw new IllegalArgumentException(
                        name + " must be a non-empty canonical field");
            }
        }
        return value;
    }

    private static String digest(String value, String name) {
        value = field(value, name).toLowerCase(java.util.Locale.ROOT);
        if (!value.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(name + " must be a SHA-256 hexadecimal digest");
        }
        return value;
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
