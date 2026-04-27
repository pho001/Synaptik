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
        search = search == null ? new SearchPolicy(1, 1, 1, false) : search;
        persistence = persistence == null ? PersistencePolicy.disabled() : persistence;
        progressListener = progressListener == null ? AutotuneProgressListener.noop() : progressListener;
    }

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
