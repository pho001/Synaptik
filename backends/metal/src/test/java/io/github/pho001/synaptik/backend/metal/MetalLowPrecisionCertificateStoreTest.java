package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalLowPrecisionCertificateStoreTest {
    @Test
    void bundledStoreLoadsEveryEnvironmentBoundCertificate() {
        MetalLowPrecisionCertificateStore store =
                MetalLowPrecisionCertificateStore.loadBundled();
        assertTrue(store.usable(), () -> store.rejectionReason().orElse("unknown rejection"));
        assertEquals(24, store.size());
        LowPrecisionCertificateKey key = firstKey(resource());
        assertTrue(store.find(key).isPresent());

        LowPrecisionCertificateKey staleBinary = new LowPrecisionCertificateKey(
                key.profile(), key.operationFamily(), key.dtypeTuple(), key.accumulatorDtype(),
                key.shapeLayoutDomain(), key.route(), key.gpuFamily(), key.osBuild(),
                key.sdkVersion(), key.compilerVersion(), "0".repeat(64), key.shaderDigest(),
                key.flagsOptions(), key.capabilityManifestHash());
        assertTrue(store.find(staleBinary).isEmpty());
        LowPrecisionCertificateKey foreignGpu = new LowPrecisionCertificateKey(
                key.profile(), key.operationFamily(), key.dtypeTuple(), key.accumulatorDtype(),
                key.shapeLayoutDomain(), key.route(), "APPLE_1", key.osBuild(), key.sdkVersion(),
                key.compilerVersion(), key.binaryDigest(), key.shaderDigest(), key.flagsOptions(),
                key.capabilityManifestHash());
        assertTrue(store.find(foreignGpu).isEmpty());
    }

    @Test
    void parserRejectsTheCompleteStoreForMalformedUnknownDuplicateOrDetachedRows() {
        String resource = resource();
        assertRejected(resource.substring(0, resource.length() - 1));
        assertRejected(resource.replaceFirst("(?m)^1\\t", "2\t"));
        assertRejected(resource.replaceFirst(
                MetalLowPrecisionCertificateStore.FAMILY, "UNKNOWN_FAMILY"));
        String[] lines = resource.split("\n");
        assertRejected(resource + lines[1] + '\n');

        String[] fields = lines[1].split("\t", -1);
        fields[19] = "0".repeat(64);
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');

        fields = lines[1].split("\t", -1);
        fields[3] = "FLOAT16,BFLOAT16";
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[4] = "FLOAT32";
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[5] = MetalLowPrecisionCertificateStore.DOMAIN_PREFIX + "0".repeat(64);
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[15] = "raw-envelope\u0007";
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[11] = fields[11].toUpperCase(java.util.Locale.ROOT);
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[14] = "0".repeat(64);
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[15] = "UNKNOWN_ENVELOPE";
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
        fields = lines[1].split("\t", -1);
        fields[20] = "UNKNOWN_DETERMINISM";
        assertRejected(lines[0] + '\n' + String.join("\t", fields) + '\n');
    }

    @Test
    void accuracyFailureRemainsAUsableButNonqualifyingExactRecord() {
        String resource = resource();
        String[] lines = resource.split("\n");
        String[] fields = lines[1].split("\t", -1);
        fields[18] = "FAIL";
        MetalLowPrecisionCertificateStore store = MetalLowPrecisionCertificateStore.parse(
                (lines[0] + '\n' + String.join("\t", fields) + '\n')
                        .getBytes(StandardCharsets.UTF_8));
        assertTrue(store.usable());
        assertEquals(1, store.size());
        assertTrue(store.find(firstKey(resource)).isEmpty());
    }

    @Test
    void emptyCanonicalStoreIsUsableButNeverQualifies() {
        MetalLowPrecisionCertificateStore store = MetalLowPrecisionCertificateStore.parse(
                (MetalLowPrecisionCertificateStore.header() + '\n')
                        .getBytes(StandardCharsets.UTF_8));
        assertTrue(store.usable());
        assertEquals(0, store.size());
        assertTrue(store.find(firstKey(resource())).isEmpty());
    }

    private static void assertRejected(String text) {
        MetalLowPrecisionCertificateStore store = MetalLowPrecisionCertificateStore.parse(
                text.getBytes(StandardCharsets.UTF_8));
        assertFalse(store.usable());
        assertEquals(0, store.size());
        assertTrue(store.rejectionReason().isPresent());
    }

    private static LowPrecisionCertificateKey firstKey(String resource) {
        String[] fields = resource.split("\n", 3)[1].split("\t", -1);
        return new LowPrecisionCertificateKey(
                TraceNumericalProfile.valueOf(fields[1]), fields[2],
                List.of(fields[3].split(",", -1)), fields[4], fields[5], fields[6], fields[7],
                fields[8], fields[9], fields[10], fields[11], fields[12], fields[13], fields[14]);
    }

    private static String resource() {
        try (InputStream input = MetalLowPrecisionCertificateStoreTest.class
                .getResourceAsStream(MetalLowPrecisionCertificateStore.RESOURCE)) {
            if (input == null) throw new AssertionError("missing bundled certificate store");
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        } catch (IOException failure) {
            throw new AssertionError(failure);
        }
    }
}
