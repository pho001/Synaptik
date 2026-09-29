package io.github.pho001.synaptik.trace.payload;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionAccuracy;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificate;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionDeterminism;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Trace-ready immutable low-precision route facts; it does not enable or select a route.
 *
 * <p>A graph event is certified and carries the complete certificate key, accuracy evidence, and
 * independently associated determinism evidence. A custom-kernel event is uncertified and carries
 * none of those values; in particular, it cannot borrow evidence belonging to an unselected
 * candidate route.</p>
 */
public record LowPrecisionTraceMetadata(
        TraceRouteKind selectedRoute,
        List<String> logicalDtypeTuple,
        String accumulatorDtype,
        TraceNumericalProfile numericalProfile,
        int certificateSchema,
        Optional<LowPrecisionCertificateKey> certificateKey,
        LowPrecisionCertificate.Status certificateStatus,
        Optional<LowPrecisionAccuracy> accuracy,
        Optional<LowPrecisionDeterminism> determinism) implements TracePayload {
    public LowPrecisionTraceMetadata {
        Objects.requireNonNull(selectedRoute, "selectedRoute");
        Objects.requireNonNull(logicalDtypeTuple, "logicalDtypeTuple");
        logicalDtypeTuple = List.copyOf(logicalDtypeTuple);
        if (logicalDtypeTuple.isEmpty()) {
            throw new IllegalArgumentException("logicalDtypeTuple must not be empty");
        }
        logicalDtypeTuple.forEach(value -> {
            Objects.requireNonNull(value, "logicalDtypeTuple element");
            if (value.isEmpty()) {
                throw new IllegalArgumentException("logicalDtypeTuple element must not be empty");
            }
        });
        Objects.requireNonNull(accumulatorDtype, "accumulatorDtype");
        Objects.requireNonNull(numericalProfile, "numericalProfile");
        if (certificateSchema != LowPrecisionCertificateKey.SCHEMA_VERSION) {
            throw new IllegalArgumentException("certificateSchema must be the active schema");
        }
        Objects.requireNonNull(certificateKey, "certificateKey");
        Objects.requireNonNull(certificateStatus, "certificateStatus");
        Objects.requireNonNull(accuracy, "accuracy");
        Objects.requireNonNull(determinism, "determinism");
        if ((selectedRoute == TraceRouteKind.GRAPH_EXECUTABLE)
                != (certificateStatus == LowPrecisionCertificate.Status.CERTIFIED)) {
            throw new IllegalArgumentException(
                    "low-precision graph routes must be certified and custom routes uncertified");
        }
        if (certificateStatus == LowPrecisionCertificate.Status.CERTIFIED) {
            LowPrecisionCertificateKey key = certificateKey.orElseThrow(
                    () -> new IllegalArgumentException(
                            "certified trace metadata requires a certificate key"));
            LowPrecisionAccuracy accuracyValue = accuracy.orElseThrow(
                    () -> new IllegalArgumentException(
                            "certified trace metadata requires accuracy evidence"));
            LowPrecisionDeterminism determinismValue = determinism.orElseThrow(
                    () -> new IllegalArgumentException(
                            "certified trace metadata requires determinism evidence"));
            if (accuracyValue.verdict() != LowPrecisionAccuracy.Verdict.PASS
                    || !determinismValue.certificateKeyHash().equals(key.digest())
                    || key.profile() != numericalProfile
                    || !key.dtypeTuple().equals(logicalDtypeTuple)
                    || !key.accumulatorDtype().equals(accumulatorDtype)) {
                throw new IllegalArgumentException(
                        "certified trace metadata has inconsistent certificate evidence");
            }
        } else if (certificateKey.isPresent() || accuracy.isPresent() || determinism.isPresent()) {
            throw new IllegalArgumentException(
                    "uncertified trace metadata must not carry certificate evidence");
        }
    }
}
