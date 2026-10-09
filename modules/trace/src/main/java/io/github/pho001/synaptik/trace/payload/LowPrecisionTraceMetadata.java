package io.github.pho001.synaptik.trace.payload;

import io.github.pho001.synaptik.trace.TracePayload;
import java.util.List;
import java.util.Objects;

/**
 * Trace-ready immutable facts for one selected low-precision custom route.
 *
 * <p>The ordered logical dtype tuple contains boundary feeds followed by boundary targets.
 * Arithmetic and raw-preserving programs report the same three execution facts; working and
 * accumulator types are Model semantics, not trace route facts. This payload owns no
 * route-selection, certificate, backend, or runtime state.</p>
 *
 * @param selectedRoute the selected custom-kernel route, never a graph route or {@code null}
 * @param logicalDtypeTuple nonempty ordered feed-then-target logical dtype names; elements must
 *        be non-null and nonempty, and the list is copied on construction
 */
public record LowPrecisionTraceMetadata(
        TraceRouteKind selectedRoute,
        List<String> logicalDtypeTuple) implements TracePayload {
    /**
     * Validates and snapshots the selected-route boundary facts without inferring arithmetic
     * working types from the prepared program.
     *
     * @param selectedRoute the selected custom-kernel route, never a graph route or {@code null}
     * @param logicalDtypeTuple nonempty ordered feed-then-target logical dtype names; elements
     *        must be non-null and nonempty, and the list is copied
     * @throws NullPointerException if a required argument or tuple element is {@code null}
     * @throws IllegalArgumentException if the route is not custom or the tuple is empty or
     *         contains an empty dtype name
     */
    public LowPrecisionTraceMetadata {
        Objects.requireNonNull(selectedRoute, "selectedRoute");
        if (selectedRoute != TraceRouteKind.CUSTOM_KERNEL) {
            throw new IllegalArgumentException(
                    "low-precision trace metadata requires the custom-kernel route");
        }
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
    }
}
