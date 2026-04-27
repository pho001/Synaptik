package tuning.autotune;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;
import tuning.candidate.CandidateSpace;
import tuning.measure.MeasurementPolicy;
import tuning.search.SearchPolicy;
import tuning.store.PersistencePolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.WorkloadSpec;

import java.util.List;
import java.util.Objects;

public record AutotuneRequest(
        WorkloadSpec workload,
        String profileName,
        DataType dataType,
        ExecutionMode executionMode,
        GraphExecutionPolicy graphPolicy,
        PlatformRuntimeProfile runtimeProfile,
        CandidateSpace candidateSpace,
        MeasurementPolicy measurement,
        ValidationPolicy validation,
        SearchPolicy search,
        PersistencePolicy persistence,
        AutotuneProgressListener progressListener
) {
    public AutotuneRequest {
        Objects.requireNonNull(workload, "workload cannot be null");
        profileName = profileName == null || profileName.isBlank() ? "autotune" : profileName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        Objects.requireNonNull(candidateSpace, "candidateSpace cannot be null");
        measurement = measurement == null ? MeasurementPolicy.defaults() : measurement;
        validation = validation == null ? ValidationPolicy.disabled() : validation;
        search = search == null ? new SearchPolicy(64, 8, 4, true) : search;
        persistence = persistence == null ? PersistencePolicy.disabled() : persistence;
        progressListener = progressListener == null ? AutotuneProgressListener.noop() : progressListener;
    }

    public AutotuneRequest(
            WorkloadSpec workload,
            String profileName,
            DataType dataType,
            ExecutionMode executionMode,
            GraphExecutionPolicy graphPolicy,
            PlatformRuntimeProfile runtimeProfile,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence
    ) {
        this(workload, profileName, dataType, executionMode, graphPolicy, runtimeProfile, candidateSpace, measurement, validation, search, persistence, null);
    }

    /**
     * Legacy adapter for tests and non-graph autotune callers. New graph autotune paths must pass
     * explicit graph and runtime profiles through {@code tuning.autotune.GraphAutotuneRequest}.
     */
    @Deprecated
    public static AutotuneRequest fromSeedExecutionProfile(
            WorkloadSpec workload,
            ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence,
            AutotuneProgressListener progressListener
    ) {
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        return new AutotuneRequest(
                workload,
                seedProfile.profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        seedProfile.profileName(),
                        seedProfile.profileName(),
                        "AUTOTUNE_SEED",
                        seedProfile
                ),
                candidateSpace,
                measurement,
                validation,
                search,
                persistence,
                progressListener
        );
    }

    /**
     * Legacy seed-profile adapter. Prefer the constructor that accepts explicit graph and runtime profiles.
     */
    @Deprecated
    public AutotuneRequest(
            WorkloadSpec workload,
            ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence
    ) {
        this(
                workload,
                Objects.requireNonNull(seedProfile, "seedProfile cannot be null").profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        seedProfile.profileName(),
                        seedProfile.profileName(),
                        "AUTOTUNE_SEED",
                        seedProfile
                ),
                candidateSpace,
                measurement,
                validation,
                search,
                persistence,
                null
        );
    }

    /**
     * Legacy seed-profile adapter. Prefer the constructor that accepts explicit graph and runtime profiles.
     */
    @Deprecated
    public AutotuneRequest(
            WorkloadSpec workload,
            ExecutionProfile seedProfile,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence,
            AutotuneProgressListener progressListener
    ) {
        this(
                workload,
                Objects.requireNonNull(seedProfile, "seedProfile cannot be null").profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        seedProfile.profileName(),
                        seedProfile.profileName(),
                        "AUTOTUNE_SEED",
                        seedProfile
                ),
                candidateSpace,
                measurement,
                validation,
                search,
                persistence,
                progressListener
        );
    }

    /**
     * Legacy candidate-space seed inference. Graph autotune must use {@code GraphAutotuneRequest}.
     */
    @Deprecated
    public AutotuneRequest(
            WorkloadSpec workload,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence
    ) {
        this(
                workload,
                defaultSeedProfile(candidateSpace).profileName(),
                defaultSeedProfile(candidateSpace).dataType(),
                defaultSeedProfile(candidateSpace).mode(),
                GraphExecutionPolicy.fromExecutionProfile(defaultSeedProfile(candidateSpace)),
                PlatformRuntimeProfile.fromExecutionProfile(
                        defaultSeedProfile(candidateSpace).profileName(),
                        defaultSeedProfile(candidateSpace).profileName(),
                        "AUTOTUNE_SEED",
                        defaultSeedProfile(candidateSpace)
                ),
                candidateSpace,
                measurement,
                validation,
                search,
                persistence,
                null
        );
    }

    /**
     * Legacy candidate-space seed inference. Graph autotune must use {@code GraphAutotuneRequest}.
     */
    @Deprecated
    public AutotuneRequest(
            WorkloadSpec workload,
            CandidateSpace candidateSpace,
            MeasurementPolicy measurement,
            ValidationPolicy validation,
            SearchPolicy search,
            PersistencePolicy persistence,
            AutotuneProgressListener progressListener
    ) {
        this(
                workload,
                defaultSeedProfile(candidateSpace).profileName(),
                defaultSeedProfile(candidateSpace).dataType(),
                defaultSeedProfile(candidateSpace).mode(),
                GraphExecutionPolicy.fromExecutionProfile(defaultSeedProfile(candidateSpace)),
                PlatformRuntimeProfile.fromExecutionProfile(
                        defaultSeedProfile(candidateSpace).profileName(),
                        defaultSeedProfile(candidateSpace).profileName(),
                        "AUTOTUNE_SEED",
                        defaultSeedProfile(candidateSpace)
                ),
                candidateSpace,
                measurement,
                validation,
                search,
                persistence,
                progressListener
        );
    }

    public ExecutionProfile assembleCurrentProfile(String candidateName) {
        return ExecutionProfileAssembler.assemble(
                profileName,
                candidateName,
                dataType,
                executionMode,
                runtimeProfile,
                graphPolicy
        );
    }

    private static ExecutionProfile defaultSeedProfile(CandidateSpace candidateSpace) {
        Objects.requireNonNull(candidateSpace, "candidateSpace cannot be null");
        if (candidateSpace instanceof tuning.candidate.ProfileGridCandidateSpace grid) {
            return grid.baseProfile();
        }
        List<tuning.candidate.Candidate> generated = candidateSpace.generate(new tuning.workload.TensorRootWorkloadSpec(
                "autotune_seed",
                tuning.workload.WorkloadKind.GENERIC,
                environment -> tensor.Tensor.scalar(1.0)
        ));
        if (generated.isEmpty()) {
            throw new IllegalArgumentException("candidateSpace must provide at least one candidate to infer autotune seed profile");
        }
        return generated.getFirst().profile();
    }
}
