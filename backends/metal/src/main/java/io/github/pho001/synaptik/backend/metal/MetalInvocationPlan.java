package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.id.TraceInvocationId;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.util.List;
import java.util.Objects;

/** Bounded binding and plan facts emitted immediately before one Metal native run call. */
public record MetalInvocationPlan(
        TracePreparedUnitId preparedUnitId,
        TraceInvocationId invocationId,
        TraceRouteKind route,
        int feedSlotCount,
        int targetSlotCount,
        int internalSlotCount,
        long aggregateInputBytes,
        long aggregateOutputBytes,
        long aggregateInternalBytes,
        long aggregateSplatBytes,
        long aggregateWorkspaceBytes,
        long aggregateRequiredBytes,
        int splatCount,
        int workspaceCount,
        List<MetalPreparationStructure.CustomStepSummary> plannedCustomSteps,
        boolean plannedCustomStepsTruncated) implements TracePayload {
    public MetalInvocationPlan {
        Objects.requireNonNull(preparedUnitId, "preparedUnitId");
        Objects.requireNonNull(invocationId, "invocationId");
        Objects.requireNonNull(route, "route");
        Objects.requireNonNull(plannedCustomSteps, "plannedCustomSteps");
        if (feedSlotCount < 0 || targetSlotCount < 0 || internalSlotCount < 0
                || aggregateInputBytes < 0L || aggregateOutputBytes < 0L
                || aggregateInternalBytes < 0L || aggregateSplatBytes < 0L
                || aggregateWorkspaceBytes < 0L || aggregateRequiredBytes < 0L
                || splatCount < 0 || workspaceCount < 0) {
            throw new IllegalArgumentException("binding facts must be non-negative");
        }
        long expectedRequired = Math.addExact(
                Math.addExact(aggregateInputBytes, aggregateOutputBytes),
                Math.addExact(aggregateInternalBytes, aggregateWorkspaceBytes));
        if (aggregateRequiredBytes != expectedRequired
                || aggregateSplatBytes > aggregateInputBytes) {
            throw new IllegalArgumentException("aggregate byte facts disagree");
        }
        if (plannedCustomSteps.size() > MetalPreparationStructure.MAX_PLANNED_CUSTOM_STEPS) {
            throw new IllegalArgumentException("too many plannedCustomSteps");
        }
        plannedCustomSteps = List.copyOf(plannedCustomSteps);
    }
}
