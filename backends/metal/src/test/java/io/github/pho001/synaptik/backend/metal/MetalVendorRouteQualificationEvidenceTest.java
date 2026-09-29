package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

class MetalVendorRouteQualificationEvidenceTest {
    private static final String HEADER = String.join("\t",
            "schema", "route-family", "operation", "dtype", "probe", "observed",
            "qualification", "rejection-reason", "candidate-options", "gpu-family",
            "os-build", "sdk-version", "compiler-version", "binary-digest",
            "flags-options", "capability-manifest-hash", "probe-digest", "generator-digest");

    @Test
    void qualifiedNegativeVendorEvidenceIsCanonicalAndNoCandidateOrRouteExists()
            throws IOException {
        Path root = repositoryRoot();
        byte[] bytes = Files.readAllBytes(root.resolve(
                "backends/metal/evidence/0072/vendor-route-qualification.tsv"));
        String matrix = new String(bytes, StandardCharsets.UTF_8);
        assertArrayEquals(bytes, matrix.getBytes(StandardCharsets.UTF_8));
        assertFalse(matrix.contains("\r"));
        assertTrue(matrix.endsWith("\n"));
        for (byte value : bytes) {
            int unsigned = Byte.toUnsignedInt(value);
            assertFalse(unsigned < 0x20 && unsigned != '\n' && unsigned != '\t');
        }

        List<String> lines = matrix.lines().toList();
        assertEquals(14, lines.size());
        assertEquals(HEADER, lines.getFirst());
        List<Row> rows = new ArrayList<>();
        Set<String> keys = new HashSet<>();
        for (String line : lines.subList(1, lines.size())) {
            String[] fields = line.split("\t", -1);
            assertEquals(18, fields.length);
            Row row = new Row(fields);
            assertEquals("1", row.schema());
            assertEquals("REJECTED", row.qualification());
            assertTrue(keys.add(row.key()));
            rows.add(row);
        }
        assertEquals(5, rows.stream().filter(row -> row.family().equals("CLASSIC_MPS")).count());
        assertEquals(8, rows.stream().filter(row -> row.family().equals("MPP")).count());
        assertEquals(List.of(
                "CLASSIC_MPS/ANY/BOTH/MPSSupportsMTLDevice",
                "CLASSIC_MPS/CONV2D/FLOAT16/MPSCNN_RUNTIME",
                "CLASSIC_MPS/CONV2D/FLOAT16/MPSCNN_ACCUMULATOR_OPTION",
                "CLASSIC_MPS/CONV2D/BFLOAT16/MPSCNN_RUNTIME",
                "CLASSIC_MPS/ANY/BFLOAT16/PUBLIC_ACCUMULATOR_GUARANTEE",
                "MPP/MATMUL2D/FLOAT16/COMPILE_PIPELINE",
                "MPP/MATMUL2D/FLOAT16/INTERNAL_ACCUMULATOR_GUARANTEE",
                "MPP/MATMUL2D/BFLOAT16/COMPILE_PIPELINE",
                "MPP/MATMUL2D/BFLOAT16/INTERNAL_ACCUMULATOR_GUARANTEE",
                "MPP/CONVOLUTION2D/FLOAT16/COMPILE_PIPELINE",
                "MPP/CONVOLUTION2D/FLOAT16/INTERNAL_ACCUMULATOR_GUARANTEE",
                "MPP/CONVOLUTION2D/BFLOAT16/COMPILE_PIPELINE",
                "MPP/CONVOLUTION2D/BFLOAT16/INTERNAL_ACCUMULATOR_GUARANTEE"),
                rows.stream().map(Row::key).toList());

        Map<String, Expected> expected = expectedRows();
        assertEquals(expected.keySet(), keys);
        for (Row row : rows) {
            Expected value = expected.get(row.key());
            assertEquals(value.observed(), row.observed());
            assertEquals(value.reason(), row.reason());
            assertEquals(value.options(), row.options());
        }

        String[] certificate = bundledCertificateRow();
        for (Row row : rows) {
            assertEquals(certificate[7], row.gpuFamily());
            assertEquals(certificate[8], row.osBuild());
            assertEquals(certificate[9], row.sdkVersion());
            assertEquals(certificate[10], row.compilerVersion());
            assertEquals(certificate[11], row.binaryDigest());
            assertEquals(certificate[13], row.flagsOptions());
            assertEquals(certificate[14], row.capabilityManifestHash());
        }
        assertTrue(bundledCertificateRoutes().stream()
                .allMatch(MetalLowPrecisionCertificateStore.ROUTE::equals));

        String mpsProbeDigest = sha256(root.resolve(
                "native/metal-macos-arm64/probes/low_precision_vendor_probe.m"));
        String mppKernelDigest = sha256(root.resolve(
                "native/metal-macos-arm64/probes/low_precision_mpp_probe.metal"));
        String mppProbeDigest = sha256(("host-probe\t" + mpsProbeDigest
                + "\nmpp-probe\t" + mppKernelDigest + "\n").getBytes(StandardCharsets.UTF_8));
        String generatorDigest = sha256(root.resolve(
                "native/metal-macos-arm64/qualify-low-precision-vendor-routes.sh"));
        for (Row row : rows) {
            assertEquals(row.family().equals("CLASSIC_MPS")
                    ? mpsProbeDigest : mppProbeDigest, row.probeDigest());
            assertEquals(generatorDigest, row.generatorDigest());
        }

        assertArrayEquals(
                new MetalPreparedRoute[] {
                    MetalPreparedRoute.CUSTOM_SINGLE_NEG,
                    MetalPreparedRoute.MPSGRAPH,
                    MetalPreparedRoute.CUSTOM_PROGRAM
                },
                MetalPreparedRoute.values());
        assertArrayEquals(
                new MetalNegTuningBatch.Candidate[] {
                    MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                    MetalNegTuningBatch.Candidate.MPSGRAPH,
                    MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM
                },
                MetalNegTuningBatch.Candidate.values());
        assertEquals(
                Set.of(MetalPreparedRoute.Family.CUSTOM_KERNEL,
                        MetalPreparedRoute.Family.MPSGRAPH),
                Set.of(MetalPreparedRoute.Family.values()));
    }

    private static Map<String, Expected> expectedRows() {
        Map<String, Expected> result = new HashMap<>();
        put(result, "CLASSIC_MPS", "ANY", "BOTH", "MPSSupportsMTLDevice", "PASS",
                "GATE_ONLY_NO_ADR0023_SEMANTIC_PROOF", "MPSSupportsMTLDevice=YES");
        put(result, "CLASSIC_MPS", "CONV2D", "FLOAT16", "MPSCNN_RUNTIME", "PASS",
                "NO_COMPLETE_EXACT_LAYOUT_DOMAIN_PROOF",
                "accumulator=FLOAT;mpsimage=FLOAT16;layout=MPSIMAGE_2X2X1");
        put(result, "CLASSIC_MPS", "CONV2D", "FLOAT16",
                "MPSCNN_ACCUMULATOR_OPTION", "FLOAT",
                "FP32_OPTION_WITHOUT_COMPLETE_ADR0023_FAMILY_PROOF",
                "accumulator=FLOAT;mpsimage=FLOAT16");
        put(result, "CLASSIC_MPS", "CONV2D", "BFLOAT16", "MPSCNN_RUNTIME",
                "REJECTED", "PUBLIC_DATASOURCE_CONTRACT_EXCLUDES_BFLOAT16",
                "datasource=MPSDataTypeBFloat16");
        put(result, "CLASSIC_MPS", "ANY", "BFLOAT16",
                "PUBLIC_ACCUMULATOR_GUARANTEE", "ABSENT",
                "NO_PUBLIC_BFLOAT16_ACCUMULATOR_CONTRACT",
                "accumulator=UNSPECIFIED");
        for (String operation : List.of("MATMUL2D", "CONVOLUTION2D")) {
            for (String dtype : List.of("FLOAT16", "BFLOAT16")) {
                put(result, "MPP", operation, dtype, "COMPILE_PIPELINE", "PASS",
                        "NO_PUBLIC_INTERNAL_ACCUMULATOR_CONTRACT",
                        "metal-language=4.0;deployment=26.2;__HAVE_TENSOR=1;"
                                + "relaxed-precision=false;destination=FLOAT32");
                put(result, "MPP", operation, dtype, "INTERNAL_ACCUMULATOR_GUARANTEE",
                        "DESTINATION_FLOAT32_ONLY",
                        "LOW_TO_F32_DESTINATION_DOES_NOT_ESTABLISH_F32_INTERNAL_ACCUMULATION",
                        "relaxed-precision=false;destination=FLOAT32;"
                                + "internal-accumulator=UNSPECIFIED");
            }
        }
        return Map.copyOf(result);
    }

    private static void put(
            Map<String, Expected> target,
            String family,
            String operation,
            String dtype,
            String probe,
            String observed,
            String reason,
            String options) {
        String key = String.join("/", family, operation, dtype, probe);
        assertFalse(target.containsKey(key));
        target.put(key, new Expected(observed, reason, options));
    }

    private static String[] bundledCertificateRow() throws IOException {
        try (InputStream input = MetalVendorRouteQualificationEvidenceTest.class
                .getResourceAsStream(MetalLowPrecisionCertificateStore.RESOURCE)) {
            if (input == null) throw new IOException("bundled certificate resource is absent");
            String[] lines = new String(input.readAllBytes(), StandardCharsets.UTF_8)
                    .split("\n", -1);
            return lines[1].split("\t", -1);
        }
    }

    private static List<String> bundledCertificateRoutes() throws IOException {
        try (InputStream input = MetalVendorRouteQualificationEvidenceTest.class
                .getResourceAsStream(MetalLowPrecisionCertificateStore.RESOURCE)) {
            if (input == null) throw new IOException("bundled certificate resource is absent");
            return new String(input.readAllBytes(), StandardCharsets.UTF_8).lines()
                    .skip(1)
                    .map(line -> line.split("\t", -1)[6])
                    .toList();
        }
    }

    private static Path repositoryRoot() {
        Path candidate = Path.of("").toAbsolutePath().normalize();
        while (candidate != null && !Files.isRegularFile(candidate.resolve("settings.gradle.kts"))) {
            candidate = candidate.getParent();
        }
        if (candidate == null) throw new IllegalStateException("repository root is unavailable");
        return candidate;
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = Files.newInputStream(path)) {
                byte[] buffer = new byte[8192];
                int read;
                while ((read = input.read(buffer)) >= 0) {
                    if (read > 0) digest.update(buffer, 0, read);
                }
            }
            return HexFormat.of().formatHex(digest.digest());
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }

    private record Expected(String observed, String reason, String options) { }

    private record Row(String[] fields) {
        Row {
            fields = fields.clone();
        }
        String schema() { return fields[0]; }
        String family() { return fields[1]; }
        String operation() { return fields[2]; }
        String dtype() { return fields[3]; }
        String probe() { return fields[4]; }
        String observed() { return fields[5]; }
        String qualification() { return fields[6]; }
        String reason() { return fields[7]; }
        String options() { return fields[8]; }
        String gpuFamily() { return fields[9]; }
        String osBuild() { return fields[10]; }
        String sdkVersion() { return fields[11]; }
        String compilerVersion() { return fields[12]; }
        String binaryDigest() { return fields[13]; }
        String flagsOptions() { return fields[14]; }
        String capabilityManifestHash() { return fields[15]; }
        String probeDigest() { return fields[16]; }
        String generatorDigest() { return fields[17]; }
        String key() { return String.join("/", family(), operation(), dtype(), probe()); }
    }
}
