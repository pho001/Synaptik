package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.certificate.LowPrecisionAccuracy;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificate;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionDeterminism;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable schema-one Metal low-precision certificate store.
 *
 * <p>The bundled TSV is parsed once while the Metal context opens. Any missing resource,
 * non-canonical byte, malformed field, unknown schema/family/route/enum, duplicate key, or detached
 * determinism record rejects the complete store. A rejected store is empty for route eligibility,
 * preserving the custom baseline. Lookup is exact-key only and never performs prefix, version,
 * environment, or shape fallback.</p>
 */
final class MetalLowPrecisionCertificateStore {
    static final String RESOURCE =
            "/io/github/pho001/synaptik/backend/metal/low-precision-certificates-v1.tsv";
    static final String FAMILY = "RAW_PRESERVING_AFFINE_V1";
    static final String ROUTE = "MPSGRAPH_CERTIFIED_RAW_V1";
    static final String DOMAIN_PREFIX = "SM19_EXACT_STATIC_LAYOUT_V1/";
    static final String ENVELOPE = "RAW_BITS_EXACT_V1";
    static final String DETERMINISM_CONTRACT = "REPEAT_3_IDENTICAL_V1";
    private static final int MAX_BYTES = 1 << 20;
    private static final String HEADER = String.join("\t",
            "schema", "profile", "operation-family", "dtype-tuple", "accumulator-dtype",
            "shape-layout-domain", "route", "gpu-family", "os-build", "sdk-version",
            "compiler-version", "binary-digest", "shader-digest", "flags-options",
            "capability-manifest-hash", "envelope-id", "qualification-method",
            "accuracy-evidence-digest", "accuracy-verdict", "determinism-key-hash",
            "determinism-contract-id", "determinism-evidence-digest", "determinism-verdict");

    private final Map<LowPrecisionCertificateKey, LowPrecisionCertificate> certificates;
    private final Optional<String> rejectionReason;

    private MetalLowPrecisionCertificateStore(
            Map<LowPrecisionCertificateKey, LowPrecisionCertificate> certificates,
            Optional<String> rejectionReason) {
        this.certificates = Map.copyOf(certificates);
        this.rejectionReason = Objects.requireNonNull(rejectionReason, "rejectionReason");
    }

    /**
     * Loads the one bundled canonical store, rejecting the complete store on the first defect.
     *
     * @return immutable usable or rejected store; never {@code null}
     */
    static MetalLowPrecisionCertificateStore loadBundled() {
        try (InputStream input = MetalLowPrecisionCertificateStore.class.getResourceAsStream(RESOURCE)) {
            if (input == null) {
                return rejected("certificate resource is missing");
            }
            byte[] bytes = input.readNBytes(MAX_BYTES + 1);
            if (bytes.length > MAX_BYTES || input.read() != -1) {
                return rejected("certificate resource exceeds the size limit");
            }
            return parse(bytes);
        } catch (IOException failure) {
            return rejected("certificate resource cannot be read");
        }
    }

    /**
     * Parses canonical UTF-8 TSV bytes for focused contract tests and the bundled loader.
     *
     * @param bytes non-null complete resource bytes
     * @return immutable usable or rejected store
     */
    static MetalLowPrecisionCertificateStore parse(byte[] bytes) {
        Objects.requireNonNull(bytes, "bytes");
        try {
            if (bytes.length == 0 || bytes[bytes.length - 1] != '\n') {
                throw new IllegalArgumentException("certificate resource must end with LF");
            }
            for (byte value : bytes) {
                if (value == 0 || value == '\r') {
                    throw new IllegalArgumentException("certificate resource contains a forbidden byte");
                }
            }
            String text = new String(bytes, StandardCharsets.UTF_8);
            if (!java.util.Arrays.equals(bytes, text.getBytes(StandardCharsets.UTF_8))) {
                throw new IllegalArgumentException("certificate resource is not canonical UTF-8");
            }
            String[] lines = text.substring(0, text.length() - 1).split("\n", -1);
            if (lines.length == 0 || !lines[0].equals(HEADER)) {
                throw new IllegalArgumentException("certificate header is unknown");
            }
            Map<LowPrecisionCertificateKey, LowPrecisionCertificate> parsed = new HashMap<>();
            for (int line = 1; line < lines.length; line++) {
                if (lines[line].isEmpty()) {
                    throw new IllegalArgumentException("certificate row must not be empty");
                }
                String[] fields = lines[line].split("\t", -1);
                if (fields.length != 23) {
                    throw new IllegalArgumentException("certificate row field count is invalid");
                }
                if (!fields[0].equals("1")) {
                    throw new IllegalArgumentException("certificate schema is unknown");
                }
                TraceNumericalProfile profile = TraceNumericalProfile.valueOf(fields[1]);
                if (!fields[2].equals(FAMILY) || !fields[6].equals(ROUTE)) {
                    throw new IllegalArgumentException("certificate family or route is unknown");
                }
                List<String> dtypes = List.of(fields[3].split(",", -1));
                if (dtypes.isEmpty()
                        || dtypes.stream().anyMatch(value ->
                                !value.equals("BFLOAT16") && !value.equals("FLOAT16"))
                        || dtypes.stream().anyMatch(value -> !value.equals(dtypes.getFirst()))) {
                    throw new IllegalArgumentException(
                            "raw certificate dtype tuple is malformed or heterogeneous");
                }
                if (!fields[4].equals(LowPrecisionCertificateKey.NO_ACCUMULATOR)) {
                    throw new IllegalArgumentException("raw route accumulator must be canonical NONE");
                }
                if (!fields[5].startsWith(DOMAIN_PREFIX)
                        || fields[5].length() != DOMAIN_PREFIX.length() + 64) {
                    throw new IllegalArgumentException("certificate shape/layout domain is unknown");
                }
                if (!fields[12].matches("[0-9a-f]{64}")
                        || !fields[5].equals(DOMAIN_PREFIX + fields[12])) {
                    throw new IllegalArgumentException(
                            "certificate program digest is malformed or detached");
                }
                for (int digestField : new int[] {11, 14, 17, 19, 21}) {
                    if (!fields[digestField].matches("[0-9a-f]{64}")) {
                        throw new IllegalArgumentException(
                                "certificate digest field is not canonical");
                    }
                }
                if (!fields[14].equals(
                        MetalCertificationEnvironment.CAPABILITY_MANIFEST_HASH)) {
                    throw new IllegalArgumentException(
                            "certificate capability manifest is stale");
                }
                if (!fields[15].equals(ENVELOPE)
                        || !fields[16].equals(
                                LowPrecisionAccuracy.QualificationMethod.CERTIFIED_ENVELOPE.name())
                        || !fields[20].equals(DETERMINISM_CONTRACT)) {
                    throw new IllegalArgumentException(
                            "certificate evidence contract is unknown");
                }
                LowPrecisionCertificateKey key = new LowPrecisionCertificateKey(
                        profile, fields[2], dtypes, fields[4], fields[5], fields[6], fields[7],
                        fields[8], fields[9], fields[10], fields[11], fields[12], fields[13],
                        fields[14]);
                LowPrecisionAccuracy accuracy = new LowPrecisionAccuracy(
                        fields[15],
                        LowPrecisionAccuracy.QualificationMethod.valueOf(fields[16]),
                        fields[17], LowPrecisionAccuracy.Verdict.valueOf(fields[18]));
                LowPrecisionDeterminism determinism = new LowPrecisionDeterminism(
                        fields[19], fields[20], fields[21],
                        LowPrecisionDeterminism.Verdict.valueOf(fields[22]));
                LowPrecisionCertificate certificate =
                        LowPrecisionCertificate.schema1(key, accuracy, determinism);
                if (certificate.determinismStatus()
                        != LowPrecisionCertificate.DeterminismStatus.ASSOCIATED) {
                    throw new IllegalArgumentException("determinism metadata is detached");
                }
                if (parsed.putIfAbsent(key, certificate) != null) {
                    throw new IllegalArgumentException("certificate key is duplicated");
                }
            }
            return new MetalLowPrecisionCertificateStore(parsed, Optional.empty());
        } catch (IllegalArgumentException | NullPointerException failure) {
            return rejected(failure.getMessage() == null
                    ? "certificate resource is malformed" : failure.getMessage());
        }
    }

    /**
     * Finds one accuracy-qualified exact key.
     *
     * @param expected non-null complete expected identity
     * @return exact certificate only when the store is usable and its accuracy verdict qualifies;
     *     stale and mismatched keys return empty
     */
    Optional<LowPrecisionCertificate> find(LowPrecisionCertificateKey expected) {
        Objects.requireNonNull(expected, "expected");
        if (rejectionReason.isPresent()) {
            return Optional.empty();
        }
        LowPrecisionCertificate certificate = certificates.get(expected);
        return certificate != null
                        && certificate.statusFor(expected) == LowPrecisionCertificate.Status.CERTIFIED
                ? Optional.of(certificate) : Optional.empty();
    }

    /** @return whether the complete resource parsed without rejection */
    boolean usable() {
        return rejectionReason.isEmpty();
    }

    /** @return non-empty diagnostic reason when the complete store was rejected */
    Optional<String> rejectionReason() {
        return rejectionReason;
    }

    /** @return exact number of parsed certificates, or zero for a rejected store */
    int size() {
        return certificates.size();
    }

    static String header() {
        return HEADER;
    }

    static MetalLowPrecisionCertificateStore emptyForTesting() {
        return new MetalLowPrecisionCertificateStore(Map.of(), Optional.empty());
    }

    private static MetalLowPrecisionCertificateStore rejected(String reason) {
        return new MetalLowPrecisionCertificateStore(Map.of(), Optional.of(reason));
    }
}
