package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Validates the frozen active version-one certificate field schema independently of route data. */
final class LowPrecisionCertificateSchemaContractTest {
    private static final String RESOURCE = "/low-precision-certificate-schema-v1.tsv";
    private static final String HEADER =
            "schema\trecord\tordinal\tfield\tkey_member\trequired\tencoding\tmeaning";

    @Test
    void certificateKeyAccuracyAndDeterminismSchemasRemainSeparateAndCanonical()
            throws Exception {
        String text = resource();
        assertTrue(text.endsWith("\n"));
        assertFalse(text.contains("\r"));
        String[] lines = text.split("\n", -1);
        assertEquals(HEADER, lines[0]);

        Map<String, List<String>> fieldsByRecord = new LinkedHashMap<>();
        String prior = null;
        for (int index = 1; index < lines.length - 1; index++) {
            String line = lines[index];
            String[] fields = line.split("\t", -1);
            assertEquals(8, fields.length, "certificate schema row " + index);
            assertEquals("1", fields[0]);
            List<String> recordFields =
                    fieldsByRecord.computeIfAbsent(fields[1], ignored -> new ArrayList<>());
            assertEquals(String.format("%02d", recordFields.size() + 1), fields[2]);
            assertFalse(recordFields.contains(fields[3]),
                    "duplicate certificate schema field " + fields[1] + "/" + fields[3]);
            recordFields.add(fields[3]);
            if (prior != null) {
                assertTrue(prior.compareTo(line) < 0,
                        "certificate schema rows must be strictly sorted at row " + index);
            }
            prior = line;
        }

        assertEquals(List.of("profile", "operation-family", "dtype-tuple", "accumulator-dtype",
                "shape-layout-domain", "route", "gpu-family", "os-build", "sdk-version",
                "compiler-version", "binary-digest", "shader-digest", "flags-options",
                "capability-manifest-hash"), fieldsByRecord.get("CERTIFICATE_KEY"));
        assertEquals(List.of("envelope-id", "qualification-method", "evidence-digest", "verdict"),
                fieldsByRecord.get("ACCURACY"));
        assertEquals(List.of("certificate-key-hash", "determinism-contract-id", "evidence-digest",
                "verdict"), fieldsByRecord.get("DETERMINISM_METADATA"));
        assertTrue(linesContaining(text, "\tCERTIFICATE_KEY\t").stream()
                .allMatch(line -> line.split("\t", -1)[4].equals("true")));
        assertTrue(linesContaining(text, "\tACCURACY\t").stream()
                .allMatch(line -> line.split("\t", -1)[4].equals("false")));
        assertTrue(linesContaining(text, "\tDETERMINISM_METADATA\t").stream()
                .allMatch(line -> line.split("\t", -1)[4].equals("false")));
        String accumulatorRow = linesContaining(text, "\taccumulator-dtype\t").getFirst();
        assertEquals("dtype-or-NONE", accumulatorRow.split("\t", -1)[6]);
    }
    @Test
    void canonicalManifestKeyFieldOrderMatchesTraceCertificateIdentity() {
        List<String> codeFields = java.util.Arrays.stream(
                        io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey.class
                                .getRecordComponents())
                .map(component -> switch (component.getName()) {
                    case "profile" -> "profile";
                    case "operationFamily" -> "operation-family";
                    case "dtypeTuple" -> "dtype-tuple";
                    case "accumulatorDtype" -> "accumulator-dtype";
                    case "shapeLayoutDomain" -> "shape-layout-domain";
                    case "route" -> "route";
                    case "gpuFamily" -> "gpu-family";
                    case "osBuild" -> "os-build";
                    case "sdkVersion" -> "sdk-version";
                    case "compilerVersion" -> "compiler-version";
                    case "binaryDigest" -> "binary-digest";
                    case "shaderDigest" -> "shader-digest";
                    case "flagsOptions" -> "flags-options";
                    case "capabilityManifestHash" -> "capability-manifest-hash";
                    default -> throw new AssertionError(component.getName());
                }).toList();
        assertEquals(List.of("profile", "operation-family", "dtype-tuple", "accumulator-dtype",
                "shape-layout-domain", "route", "gpu-family", "os-build", "sdk-version",
                "compiler-version", "binary-digest", "shader-digest", "flags-options",
                "capability-manifest-hash"), codeFields);
    }

    private static List<String> linesContaining(String text, String marker) {
        return text.lines().filter(line -> line.contains(marker)).toList();
    }

    private static String resource() throws Exception {
        try (InputStream input = LowPrecisionCertificateSchemaContractTest.class
                .getResourceAsStream(RESOURCE)) {
            if (input == null) throw new AssertionError("missing resource " + RESOURCE);
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
