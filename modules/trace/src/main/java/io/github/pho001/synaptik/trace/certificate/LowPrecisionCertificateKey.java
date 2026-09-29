package io.github.pho001.synaptik.trace.certificate;

import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;

/** Immutable schema-1 identity for one qualified low-precision route environment. */
public record LowPrecisionCertificateKey(
        TraceNumericalProfile profile,
        String operationFamily,
        List<String> dtypeTuple,
        String accumulatorDtype,
        String shapeLayoutDomain,
        String route,
        String gpuFamily,
        String osBuild,
        String sdkVersion,
        String compilerVersion,
        String binaryDigest,
        String shaderDigest,
        String flagsOptions,
        String capabilityManifestHash) {
    /** Frozen certificate field schema version. */
    public static final int SCHEMA_VERSION = 1;
    /** Canonical schema value for an exact non-arithmetic route with no accumulator. */
    public static final String NO_ACCUMULATOR = "NONE";
    /** Canonical schema value when a route has no shader or graph program digest. */
    public static final String NO_SHADER = "NONE";


    public LowPrecisionCertificateKey {
        Objects.requireNonNull(profile, "profile");
        operationFamily = required(operationFamily, "operationFamily");
        Objects.requireNonNull(dtypeTuple, "dtypeTuple");
        dtypeTuple = List.copyOf(dtypeTuple);
        if (dtypeTuple.isEmpty() || dtypeTuple.stream().anyMatch(value -> required(value, "dtypeTuple") == null)) {
            throw new IllegalArgumentException("dtypeTuple must contain non-empty values");
        }
        accumulatorDtype = accumulator(accumulatorDtype);
        shapeLayoutDomain = required(shapeLayoutDomain, "shapeLayoutDomain");
        route = required(route, "route");
        gpuFamily = required(gpuFamily, "gpuFamily");
        osBuild = required(osBuild, "osBuild");
        sdkVersion = required(sdkVersion, "sdkVersion");
        compilerVersion = required(compilerVersion, "compilerVersion");
        binaryDigest = digest(binaryDigest, "binaryDigest");
        shaderDigest = shaderDigest(shaderDigest);
        flagsOptions = required(flagsOptions, "flagsOptions");
        capabilityManifestHash = digest(capabilityManifestHash, "capabilityManifestHash");
    }

    /**
     * Returns collision-free canonical UTF-8 key bytes including every manifest field.
     *
     * @return newly allocated canonical bytes
     */
    public byte[] canonicalBytes() {
        StringBuilder encoded = new StringBuilder();
        appendField(encoded, Integer.toString(SCHEMA_VERSION));
        appendField(encoded, profile.name());
        appendField(encoded, operationFamily);
        appendField(encoded, Integer.toString(dtypeTuple.size()));
        dtypeTuple.forEach(value -> appendField(encoded, value));
        appendField(encoded, accumulatorDtype);
        appendField(encoded, shapeLayoutDomain);
        appendField(encoded, route);
        appendField(encoded, gpuFamily);
        appendField(encoded, osBuild);
        appendField(encoded, sdkVersion);
        appendField(encoded, compilerVersion);
        appendField(encoded, binaryDigest);
        appendField(encoded, shaderDigest);
        appendField(encoded, flagsOptions);
        appendField(encoded, capabilityManifestHash);
        return encoded.toString().getBytes(StandardCharsets.UTF_8);
    }

    private static void appendField(StringBuilder encoded, String value) {
        encoded.append(value.getBytes(StandardCharsets.UTF_8).length).append(':').append(value);
    }

    /**
     * Returns the SHA-256 digest of {@link #canonicalBytes()} as lowercase hexadecimal.
     *
     * @return 64-digit lowercase digest
     */
    public String digest() {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(canonicalBytes()));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static String required(String value, String name) {
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

    private static String accumulator(String value) {
        value = required(value, "accumulatorDtype");
        if (value.equals(NO_ACCUMULATOR)) {
            return value;
        }
        return switch (value) {
            case "BFLOAT16", "FLOAT16", "FLOAT32", "FLOAT64", "INT32", "INT64" -> value;
            default -> throw new IllegalArgumentException(
                    "accumulatorDtype must be a numeric dtype or canonical NONE");
        };
    }

    private static String shaderDigest(String value) {
        value = required(value, "shaderDigest");
        return value.equals(NO_SHADER) ? value : digest(value, "shaderDigest");
    }

    private static String digest(String value, String name) {
        value = required(value, name);
        if (!value.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalArgumentException(name + " must be a SHA-256 hexadecimal digest");
        }
        return value.toLowerCase(java.util.Locale.ROOT);
    }
}
