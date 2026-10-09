package io.github.pho001.synaptik.trace.payload;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.id.TraceBackendId;
import io.github.pho001.synaptik.trace.id.TraceDeviceId;
import io.github.pho001.synaptik.trace.id.TraceInvocationId;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import java.util.Objects;
import java.util.Optional;

/**
 * Describes one invocation outcome for a backend prepared unit in trace-owned terms.
 *
 * <p>A successful outcome always carries native success. A failed outcome may omit native status
 * when failure occurred before native code returned a status; when present, that status must be
 * non-success. The prepared-unit and invocation correlations are distinct nominal domains. All
 * components are immutable and retained unchanged.</p>
 *
 * @param backendId non-null trace-local backend correlation
 * @param deviceId non-null trace-local device correlation
 * @param preparedUnitId non-null trace-local prepared-unit correlation
 * @param invocationId non-null trace-local invocation correlation
 * @param status non-null invocation outcome
 * @param route non-null neutral execution mechanism selected for the prepared unit
 * @param nativeStatus non-null optional exact native status; present success is required for
 *     {@code SUCCEEDED}, while {@code FAILED} permits empty or present non-success
 */
public record BackendInvocationOutcome(
        TraceBackendId backendId,
        TraceDeviceId deviceId,
        TracePreparedUnitId preparedUnitId,
        TraceInvocationId invocationId,
        TraceOutcomeStatus status,
        TraceRouteKind route,
        Optional<TraceNativeStatus> nativeStatus) implements TracePayload {
    /**
     * Validates component nullability and the outcome/native-status relationship.
     *
     * @param backendId non-null trace-local backend correlation
     * @param deviceId non-null trace-local device correlation
     * @param preparedUnitId non-null trace-local prepared-unit correlation
     * @param invocationId non-null trace-local invocation correlation
     * @param status non-null invocation outcome
     * @param route non-null neutral route kind
     * @param nativeStatus non-null optional native status
     * @throws NullPointerException if a reference component is null; validation follows component
     *     declaration order and the message names the component
     * @throws IllegalArgumentException if success lacks native success or failure carries native
     *     success
     */
    public BackendInvocationOutcome {
        Objects.requireNonNull(backendId, "backendId");
        Objects.requireNonNull(deviceId, "deviceId");
        Objects.requireNonNull(preparedUnitId, "preparedUnitId");
        Objects.requireNonNull(invocationId, "invocationId");
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(nativeStatus, "nativeStatus");
        validateStatus(status, nativeStatus);
    }

    private static void validateStatus(
            TraceOutcomeStatus status, Optional<TraceNativeStatus> nativeStatus) {
        boolean nativeSuccess = nativeStatus.isPresent()
                && nativeStatus.orElseThrow().kind() == TraceNativeStatusKind.SUCCESS;
        if (status == TraceOutcomeStatus.SUCCEEDED && !nativeSuccess) {
            throw new IllegalArgumentException("SUCCEEDED requires SUCCESS nativeStatus");
        }
        if (status == TraceOutcomeStatus.FAILED && nativeSuccess) {
            throw new IllegalArgumentException("FAILED forbids SUCCESS nativeStatus");
        }
    }
}
