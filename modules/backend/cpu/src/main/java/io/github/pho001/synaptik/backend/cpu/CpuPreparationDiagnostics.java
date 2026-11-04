package io.github.pho001.synaptik.backend.cpu;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;

/**
 * Detached facts from one cold CPU partition preparation, not a native-selection promise.
 *
 * <p>The preparation identity correlates selection with finalization, including across threads.
 * Unit positions refer to the selected plan; rejection positions refer to the ordinary-contraction
 * candidate and are not necessarily positions in the selected baseline. Each list retains only its
 * first {@link #MAX_SUMMARIES} facts; truncation never affects selection. No values, graph objects,
 * storage, native handles, implementation pointers, or exception objects are retained.</p>
 *
 * @param preparationId non-null integration-local preparation identity
 * @param stage non-null observed lifecycle stage
 * @param configuredApi non-null configured oneDNN policy, not the selected route
 * @param selectedRoute non-null actual selected partition route, or UNKNOWN on analysis failure
 * @param units non-null bounded selected-unit summaries, copied on construction
 * @param unitsTruncated whether selected units exceeded the summary bound
 * @param rejections non-null bounded cold comparison reasons, copied on construction
 * @param rejectionsTruncated whether comparison reasons exceeded the summary bound
 * @param failureType non-null exception class name on failure, otherwise empty; never its message
 * @param nativeStatus non-null exact primary native failure status when known, otherwise empty
 */
public record CpuPreparationDiagnostics(
        TracePreparedUnitId preparationId, Stage stage, Api configuredApi, Route selectedRoute,
        List<Unit> units, boolean unitsTruncated, List<Rejection> rejections,
        boolean rejectionsTruncated, String failureType, OptionalInt nativeStatus) implements TracePayload {
    /** Maximum retained selected units and, independently, rejection summaries per event. */
    public static final int MAX_SUMMARIES = 32;

    /**
     * Validates and snapshots one detached preparation observation.
     * @param preparationId non-null integration-local identity
     * @param stage non-null observed stage
     * @param configuredApi non-null immutable configuration
     * @param selectedRoute non-null actual route, UNKNOWN only before successful analysis
     * @param units non-null selected facts, at most MAX_SUMMARIES; copied
     * @param unitsTruncated whether more selected facts existed
     * @param rejections non-null observed reasons, at most MAX_SUMMARIES; copied
     * @param rejectionsTruncated whether more observed reasons existed
     * @param failureType non-null primary exception class name or empty
     * @param nativeStatus non-null exact failing native status, or empty when unknown
     * @throws NullPointerException if a component or list element is null
     * @throws IllegalArgumentException if either summary list exceeds MAX_SUMMARIES
     */
    public CpuPreparationDiagnostics {
        Objects.requireNonNull(preparationId, "preparationId");
        Objects.requireNonNull(stage, "stage");
        Objects.requireNonNull(configuredApi, "configuredApi");
        Objects.requireNonNull(selectedRoute, "selectedRoute");
        units = List.copyOf(units);
        rejections = List.copyOf(rejections);
        Objects.requireNonNull(failureType, "failureType");
        Objects.requireNonNull(nativeStatus, "nativeStatus");
        if (units.size() > MAX_SUMMARIES || rejections.size() > MAX_SUMMARIES) {
            throw new IllegalArgumentException("CPU diagnostic summary bound exceeded");
        }
    }

    /** The actual cold operation outcome. */
    public enum Stage { SELECTED, ANALYSIS_FAILED, FINALIZED, FINALIZATION_FAILED }
    /** Configured or selected oneDNN API; NONE means no oneDNN API was used/configured. */
    public enum Api { NONE, AUTO, PRIMITIVES, GRAPH }
    /** Actual selected partition/unit realization; ONEDNN partitions may retain portable units. */
    public enum Route { UNKNOWN, PORTABLE, OPENBLAS, ONEDNN }
    /** Bounded structural family, independent of data values and implementation class names. */
    public enum Family { CONTRACTION, POINTWISE, REDUCTION, MOVEMENT, PIPELINE, OTHER }
    /** Why a native candidate did not displace the ordinary CPU plan. */
    public enum Reason {
        NOT_CONFIGURED, COMPARISON_SKIPPED, UNQUALIFIED, UNSUPPORTED_REALIZATION,
        NATIVE_UNSUPPORTED, UNIT_COST, PLAN_COST, COST_OVERFLOW
    }

    /**
     * One actual selected unit, never a hypothetical native candidate.
     * @param position non-negative selected-plan unit position
     * @param family non-null structural family
     * @param route non-null actual unit route
     * @param nativeApi non-null actual oneDNN API, or NONE for portable/OpenBLAS execution
     */
    public record Unit(int position, Family family, Route route, Api nativeApi) {
        /**
         * Validates one selected-unit fact.
         * @param position non-negative selected-plan position
         * @param family non-null structural family
         * @param route non-null actual realization
         * @param nativeApi non-null actual oneDNN API or NONE
         * @throws NullPointerException if a component is null
         * @throws IllegalArgumentException if position is negative
         */
        public Unit {
            if (position < 0) throw new IllegalArgumentException("negative unit position");
            Objects.requireNonNull(family, "family");
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(nativeApi, "nativeApi");
        }
    }

    /**
     * One observed cold rejection; a later candidate may still select native execution.
     * @param candidatePosition ordinary-contraction candidate position, or -1 for a whole-plan reason
     * @param family non-null candidate family, OTHER for a whole-plan reason
     * @param nativeApi non-null rejected oneDNN API, or NONE when no native query occurred
     * @param reason non-null rejection category
     * @param detail non-null bounded producer explanation, at most 240 characters; no data values
     * @param nativeStatus non-null exact returned native status when known, otherwise empty
     */
    public record Rejection(int candidatePosition, Family family, Api nativeApi, Reason reason,
            String detail, OptionalInt nativeStatus) {
        /**
         * Validates one candidate rejection without interpreting or selecting a route.
         * @param candidatePosition candidate-plan position, or -1 for the whole plan
         * @param family non-null candidate family
         * @param nativeApi non-null queried oneDNN API or NONE
         * @param reason non-null observed category
         * @param detail non-null producer explanation, at most 240 characters
         * @param nativeStatus non-null exact returned status or empty when unknown
         * @throws NullPointerException if a component is null
         * @throws IllegalArgumentException if position is below -1 or detail exceeds 240 characters
         */
        public Rejection {
            if (candidatePosition < -1) throw new IllegalArgumentException("invalid candidate position");
            Objects.requireNonNull(family, "family");
            Objects.requireNonNull(nativeApi, "nativeApi");
            Objects.requireNonNull(reason, "reason");
            Objects.requireNonNull(detail, "detail");
            Objects.requireNonNull(nativeStatus, "nativeStatus");
            if (detail.length() > 240) throw new IllegalArgumentException("rejection detail too long");
        }
    }
}
