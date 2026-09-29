package io.github.pho001.synaptik.trace;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.trace.certificate.LowPrecisionAccuracy;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificate;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionDeterminism;
import io.github.pho001.synaptik.trace.payload.LowPrecisionTraceMetadata;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class LowPrecisionCertificateTest {
    private static final String DIGEST = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    @Test
    void keyDigestCoversEveryFieldAndCertificateRequiresExactIdentity() {
        LowPrecisionCertificateKey key = key("route-a");
        LowPrecisionCertificate certificate = LowPrecisionCertificate.schema1(key,
                new LowPrecisionAccuracy("matmul-envelope-v1",
                        LowPrecisionAccuracy.QualificationMethod.CERTIFIED_ENVELOPE, DIGEST,
                        LowPrecisionAccuracy.Verdict.PASS),
                new LowPrecisionDeterminism(key.digest(), "stable-order-v1", DIGEST,
                        LowPrecisionDeterminism.Verdict.PASS));
        assertEquals(LowPrecisionCertificate.Status.CERTIFIED, certificate.statusFor(key));
        assertEquals(LowPrecisionCertificate.Status.NOT_CERTIFIED,
                certificate.statusFor(key("route-b")));
        assertNotEquals(key.digest(), key("route-b").digest());
    }
    @Test
    void determinismMismatchDoesNotChangeAccuracyAndUnknownSchemaFailsClosed() {
        LowPrecisionCertificateKey key = key("route-a");
        LowPrecisionCertificate mismatched = new LowPrecisionCertificate(1, key,
                new LowPrecisionAccuracy("envelope", LowPrecisionAccuracy.QualificationMethod.DEDUCTIVE_PROOF,
                        DIGEST, LowPrecisionAccuracy.Verdict.PASS),
                new LowPrecisionDeterminism(DIGEST, "contract", DIGEST,
                        LowPrecisionDeterminism.Verdict.UNKNOWN));
        assertEquals(LowPrecisionCertificate.Status.CERTIFIED, mismatched.statusFor(key));
        assertEquals(LowPrecisionCertificate.DeterminismStatus.MISMATCHED,
                mismatched.determinismStatus());
        LowPrecisionCertificate unknown = new LowPrecisionCertificate(2, key, mismatched.accuracy(),
                mismatched.determinism());
        assertEquals(LowPrecisionCertificate.Status.NOT_CERTIFIED, unknown.statusFor(key));
    }

    @Test
    void lengthPrefixedEncodingSeparatesCommaBearingDtypeStructures() {
        LowPrecisionCertificateKey one = key("route-a", List.of("A,B"));
        LowPrecisionCertificateKey two = key("route-a", List.of("A", "B"));
        assertFalse(java.util.Arrays.equals(one.canonicalBytes(), two.canonicalBytes()));
        assertNotEquals(one.digest(), two.digest());
    }

    @Test
    void accuracyFailureIsNotCertifiedRegardlessOfDeterminism() {
        LowPrecisionCertificateKey key = key("route-a");
        LowPrecisionCertificate failed = LowPrecisionCertificate.schema1(key,
                new LowPrecisionAccuracy("envelope", LowPrecisionAccuracy.QualificationMethod.DEDUCTIVE_PROOF,
                        DIGEST, LowPrecisionAccuracy.Verdict.FAIL),
                new LowPrecisionDeterminism(key.digest(), "contract", DIGEST,
                        LowPrecisionDeterminism.Verdict.PASS));
        assertEquals(LowPrecisionCertificate.Status.NOT_CERTIFIED, failed.statusFor(key));
    }

    @Test
    void nonArithmeticKeysUseCanonicalNoneAndRejectNonNumericAccumulators() {
        LowPrecisionCertificateKey raw = new LowPrecisionCertificateKey(
                TraceNumericalProfile.STRICT_IEEE,
                "RAW_PRESERVING_AFFINE_V1",
                List.of("FLOAT16", "FLOAT16"),
                LowPrecisionCertificateKey.NO_ACCUMULATOR,
                "exact-program",
                "MPSGRAPH_CERTIFIED_RAW_V1",
                "APPLE_9", "25G83", "sdk", "compiler", DIGEST, DIGEST, "flags", DIGEST);
        assertEquals("NONE", raw.accumulatorDtype());
        assertEquals(LowPrecisionCertificateKey.NO_SHADER, key("route-a").shaderDigest());
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionCertificateKey(
                raw.profile(), raw.operationFamily(), raw.dtypeTuple(), raw.accumulatorDtype(),
                raw.shapeLayoutDomain(), raw.route(), raw.gpuFamily(), raw.osBuild(),
                raw.sdkVersion(), raw.compilerVersion(), raw.binaryDigest(), "not-a-digest",
                raw.flagsOptions(), raw.capabilityManifestHash()));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionCertificateKey(
                raw.profile(), raw.operationFamily(), raw.dtypeTuple(), "BOOL",
                raw.shapeLayoutDomain(), raw.route(), raw.gpuFamily(), raw.osBuild(),
                raw.sdkVersion(), raw.compilerVersion(), raw.binaryDigest(), raw.shaderDigest(),
                raw.flagsOptions(), raw.capabilityManifestHash()));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionCertificateKey(
                raw.profile(), raw.operationFamily() + '\t', raw.dtypeTuple(),
                raw.accumulatorDtype(), raw.shapeLayoutDomain(), raw.route(), raw.gpuFamily(),
                raw.osBuild(), raw.sdkVersion(), raw.compilerVersion(), raw.binaryDigest(),
                raw.shaderDigest(), raw.flagsOptions(), raw.capabilityManifestHash()));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionAccuracy(
                "raw\u0007", LowPrecisionAccuracy.QualificationMethod.CERTIFIED_ENVELOPE,
                DIGEST, LowPrecisionAccuracy.Verdict.PASS));
    }

    private static LowPrecisionCertificateKey key(String route) {
        return key(route, List.of("FLOAT16", "FLOAT16", "FLOAT16"));
    }

    private static LowPrecisionCertificateKey key(String route, List<String> dtypes) {
        return new LowPrecisionCertificateKey(TraceNumericalProfile.ACCELERATOR,
                "MATMUL", dtypes, "FLOAT32", "rank2-dense",
                route, "gpu-family-a", "os-build-a", "sdk-a", "compiler-a", DIGEST, "NONE",
                "strict-math", DIGEST);
    }
    @Test
    void traceKeepsSelectedRouteAccuracyAndDeterminismIdentitySeparate() {
        LowPrecisionCertificateKey key = key(
                "route-a", List.of("FLOAT16", "FLOAT16"));
        LowPrecisionAccuracy accuracy = new LowPrecisionAccuracy(
                "envelope-v1",
                LowPrecisionAccuracy.QualificationMethod.CERTIFIED_ENVELOPE,
                DIGEST,
                LowPrecisionAccuracy.Verdict.PASS);
        LowPrecisionDeterminism determinism = new LowPrecisionDeterminism(
                key.digest(), "contract-v1", DIGEST, LowPrecisionDeterminism.Verdict.FAIL);
        LowPrecisionTraceMetadata certified = new LowPrecisionTraceMetadata(
                TraceRouteKind.GRAPH_EXECUTABLE,
                key.dtypeTuple(),
                key.accumulatorDtype(),
                key.profile(),
                LowPrecisionCertificateKey.SCHEMA_VERSION,
                Optional.of(key),
                LowPrecisionCertificate.Status.CERTIFIED,
                Optional.of(accuracy),
                Optional.of(determinism));
        TraceEvent<LowPrecisionTraceMetadata> event = new TraceEvent<>(
                new TraceEventId(7), TracePhase.PREPARE, TraceLevel.INFO, 11L, certified);
        assertSame(certified, event.payload());
        assertEquals(key, certified.certificateKey().orElseThrow());
        assertEquals(accuracy, certified.accuracy().orElseThrow());
        assertEquals(determinism, certified.determinism().orElseThrow());
        assertThrows(UnsupportedOperationException.class,
                () -> certified.logicalDtypeTuple().add("FLOAT32"));

        LowPrecisionTraceMetadata custom = new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                key.dtypeTuple(),
                key.accumulatorDtype(),
                key.profile(),
                LowPrecisionCertificateKey.SCHEMA_VERSION,
                Optional.empty(),
                LowPrecisionCertificate.Status.NOT_CERTIFIED,
                Optional.empty(),
                Optional.empty());
        assertTrue(custom.certificateKey().isEmpty());
        assertTrue(custom.accuracy().isEmpty());
        assertTrue(custom.determinism().isEmpty());
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.GRAPH_EXECUTABLE,
                key.dtypeTuple(),
                key.accumulatorDtype(),
                key.profile(),
                LowPrecisionCertificateKey.SCHEMA_VERSION,
                Optional.empty(),
                LowPrecisionCertificate.Status.NOT_CERTIFIED,
                Optional.empty(),
                Optional.empty()));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                key.dtypeTuple(),
                key.accumulatorDtype(),
                key.profile(),
                LowPrecisionCertificateKey.SCHEMA_VERSION,
                Optional.of(key),
                LowPrecisionCertificate.Status.CERTIFIED,
                Optional.of(accuracy),
                Optional.of(determinism)));
        assertThrows(IllegalArgumentException.class, () -> new LowPrecisionTraceMetadata(
                TraceRouteKind.CUSTOM_KERNEL,
                key.dtypeTuple(),
                key.accumulatorDtype(),
                key.profile(),
                LowPrecisionCertificateKey.SCHEMA_VERSION,
                Optional.of(key),
                LowPrecisionCertificate.Status.NOT_CERTIFIED,
                Optional.of(accuracy),
                Optional.of(determinism)));
    }
}

