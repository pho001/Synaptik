package tuning.autotune;

import backend.runtime.ExecutionMode;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;
import tuning.candidate.graph.GraphAutotuneCandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.search.SearchPolicy;
import tuning.autotune.AutotuneProgressListener;
import tuning.autotune.AutotuneRequest;
import tuning.store.PersistencePolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.Objects;

/**
 * Request for graph-policy autotune on top of a fixed platform runtime profile.
 *
 * <p>Graph autotune is narrower than platform calibration: it creates
 * {@link GraphAutotuneCandidateSpace} candidates that alter graph execution
 * policy while preserving the supplied {@link #runtimeProfile()}. It is also
 * different from benchmarking, which only measures caller-supplied profiles and
 * never searches.</p>
 *
 * <p>Use {@link GraphAutotuneMode#STANDARD} for production-eligible comparison
 * against the current graph policy and {@link GraphAutotuneMode#RESEARCH} for
 * exploratory policy variants. Research candidates are metadata-marked as not
 * production eligible so persistence consumers can avoid promoting them blindly.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * GraphAutotuneRequest graphRequest = new GraphAutotuneRequest(
 *         workload,
 *         "graph-policy-tune",
 *         DataType.FLOAT32,
 *         ExecutionMode.FORWARD_BACKWARD,
 *         graphPolicy,
 *         calibratedRuntimeProfile,
 *         GraphAutotuneMode.STANDARD,
 *         MeasurementPolicy.defaults(),
 *         ValidationPolicy.defaults(),
 *         new SearchPolicy(16, 4, 1, false),
 *         PersistencePolicy.disabled(),
 *         AutotuneProgressListener.noop());
 * GraphAutotuneResult result = GraphAutotuneSession.create(graphRequest).run();
 * }</pre>
 *
 * @param workload workload to instantiate for each graph candidate; required
 * @param profileName profile namespace; blank values become {@code "graph-autotune"}
 * @param dataType dtype for assembled candidates; required
 * @param executionMode execution mode for assembled candidates; required
 * @param graphPolicy seed graph policy that graph mutators vary; required
 * @param runtimeProfile runtime profile held fixed during graph autotune; required
 * @param mode standard or research graph search mode; {@code null} means standard
 * @param measurement measurement controls; {@code null} means defaults
 * @param validation validation controls; {@code null} disables validation
 * @param search search controls; {@code null} selects a mode-aware default
 * @param persistence optional persistence policy; {@code null} disables persistence
 * @param progressListener optional progress sink; {@code null} becomes no-op
 */
public record GraphAutotuneRequest(
        WorkloadSpec workload,
        String profileName,
        DataType dataType,
        ExecutionMode executionMode,
        GraphExecutionPolicy graphPolicy,
        PlatformRuntimeProfile runtimeProfile,
        GraphAutotuneMode mode,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        SearchPolicy search,
        PersistencePolicy persistence,
        AutotuneProgressListener progressListener
) {
    public GraphAutotuneRequest {
        Objects.requireNonNull(workload, "workload cannot be null");
        profileName = profileName == null || profileName.isBlank() ? "graph-autotune" : profileName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        mode = mode == null ? GraphAutotuneMode.STANDARD : mode;
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        search = search == null ? defaultSearchPolicy(mode) : search;
        persistence = persistence == null ? PersistencePolicy.disabled() : persistence;
        progressListener = progressListener == null ? AutotuneProgressListener.noop() : progressListener;
    }

    private static SearchPolicy defaultSearchPolicy(GraphAutotuneMode mode) {
        return mode == GraphAutotuneMode.RESEARCH
                ? TuningDefaults.balancedSearchPolicy()
                : new SearchPolicy(16, 4, 1, false);
    }

    /**
     * Converts this graph-specific request to the generic autotune request used
     * by {@link AutotuneSession}.
     *
     * <p>The conversion is allocation-only and has no measurement or persistence
     * side effects. The generated candidate space preserves the runtime profile
     * and varies only graph policy candidates allowed by {@link #mode()}.</p>
     *
     * @return generic autotune request backed by a graph candidate space
     */
    public AutotuneRequest toAutotuneRequest() {
        return new AutotuneRequest(
                workload,
                profileName,
                dataType,
                executionMode,
                graphPolicy,
                runtimeProfile,
                new GraphAutotuneCandidateSpace(profileName, dataType, executionMode, runtimeProfile, graphPolicy, mode),
                measurement,
                validation,
                search,
                persistence,
                progressListener
        );
    }
}
