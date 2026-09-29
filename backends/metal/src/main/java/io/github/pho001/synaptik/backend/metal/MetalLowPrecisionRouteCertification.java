package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificate;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.payload.TraceNumericalProfile;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** Exact prepare-time qualification filter for the certified raw-preserving MPSGraph family. */
final class MetalLowPrecisionRouteCertification {
    private static final Set<MetalMpsGraphProgram.NodeKind> CERTIFIED_NODE_KINDS = Set.of(
            MetalMpsGraphProgram.NodeKind.RESHAPE,
            MetalMpsGraphProgram.NodeKind.PERMUTE,
            MetalMpsGraphProgram.NodeKind.CONTIGUOUS,
            MetalMpsGraphProgram.NodeKind.SLICE,
            MetalMpsGraphProgram.NodeKind.CONCAT,
            MetalMpsGraphProgram.NodeKind.TILE);

    private MetalLowPrecisionRouteCertification() {
    }

    /** Complete expected identity before certificate lookup. */
    record Expected(LowPrecisionCertificateKey key, String programDigest) {
        Expected {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(programDigest, "programDigest");
        }
    }

    /** Exact usable certificate retained by the immutable context store. */
    record Qualification(Expected expected, LowPrecisionCertificate certificate) {
        Qualification {
            Objects.requireNonNull(expected, "expected");
            Objects.requireNonNull(certificate, "certificate");
        }
    }

    /**
     * Returns the complete expected identity only for the deliberately narrow certified family.
     * External storage offsets, holes, zero-element tensors, gradients, arithmetic, mixed dtypes,
     * and unavailable environment identities remain custom-only.
     */
    static Optional<Expected> expected(MetalNegPreparationPlan plan) {
        Objects.requireNonNull(plan, "plan");
        MetalCertificationEnvironment environment = plan.context().certificationEnvironment();
        if (!environment.certifiable()
                || plan.graphProgram().nodes().isEmpty()
                || plan.graphProgram().nodes().stream()
                        .anyMatch(node -> !CERTIFIED_NODE_KINDS.contains(node.kind()))) {
            return Optional.empty();
        }
        DataType dataType = plan.descriptors().getFirst().dataType();
        if ((dataType != DataType.BFLOAT16 && dataType != DataType.FLOAT16)
                || plan.descriptors().stream().anyMatch(descriptor ->
                        descriptor.dataType() != dataType || descriptor.requiresGrad())) {
            return Optional.empty();
        }
        int[] feeds = plan.feedValueIndices();
        int[] targets = plan.targetValueIndices();
        if (!canonicalNonEmptyBoundaries(plan, feeds)
                || !canonicalNonEmptyBoundaries(plan, targets)) {
            return Optional.empty();
        }

        byte[] image = plan.graphProgram().encodedProgramImage(
                plan.numericalProfile(), plan.programValueDescriptors(), feeds, targets,
                MetalPreparedRoute.MPSGRAPH);
        String programDigest = sha256(image);
        List<String> dtypeTuple = new ArrayList<>(feeds.length + targets.length);
        for (int value : feeds) dtypeTuple.add(plan.descriptors().get(value).dataType().name());
        for (int value : targets) dtypeTuple.add(plan.descriptors().get(value).dataType().name());
        LowPrecisionCertificateKey key = new LowPrecisionCertificateKey(
                TraceNumericalProfile.valueOf(plan.numericalProfile().name()),
                MetalLowPrecisionCertificateStore.FAMILY,
                dtypeTuple,
                LowPrecisionCertificateKey.NO_ACCUMULATOR,
                MetalLowPrecisionCertificateStore.DOMAIN_PREFIX + programDigest,
                MetalLowPrecisionCertificateStore.ROUTE,
                environment.gpuFamily(),
                environment.osBuild(),
                environment.sdkVersion(),
                environment.compilerVersion(),
                environment.binaryDigest(),
                programDigest,
                environment.flagsOptions(),
                environment.capabilityManifestHash());
        return Optional.of(new Expected(key, programDigest));
    }

    /** Returns an exact qualified match; there is no relaxed or stale fallback. */
    static Optional<Qualification> find(MetalNegPreparationPlan plan) {
        return expected(plan).flatMap(expected -> plan.context().certificateStore()
                .find(expected.key())
                .map(certificate -> new Qualification(expected, certificate)));
    }

    private static boolean canonicalNonEmptyBoundaries(
            MetalNegPreparationPlan plan, int[] valueIndices) {
        if (valueIndices.length == 0) return false;
        for (int value : valueIndices) {
            if (!isCanonicalNonEmptyBoundary(plan.physicalLayouts().get(value))) {
                return false;
            }
        }
        return true;
    }

    /**
     * Reports whether one physical boundary lies inside the certified dense non-empty domain.
     *
     * @param physical non-null resolved physical layout
     * @return {@code true} only for zero-offset contiguous storage with a positive span
     */
    static boolean isCanonicalNonEmptyBoundary(LayoutDescriptor physical) {
        Objects.requireNonNull(physical, "physical");
        return physical.isContiguous()
                && !physical.hasStorageOffset()
                && physical.referencedElementSpan() > 0L;
    }

    private static String sha256(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(
                    MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
