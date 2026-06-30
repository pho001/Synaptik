package tuning.autotune;

import runtime.contract.ExecutionMode;
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

/**
 * Immutable request for one autotune run over a concrete {@link WorkloadSpec}.
 *
 * <p>This is the generic autotune contract used by {@link AutotuneSession}. It
 * evaluates candidates supplied by {@link #candidateSpace()} and selects the
 * fastest valid profile according to {@link #measurement()}, {@link #validation()},
 * and {@link #search()}. Benchmark sessions do not create candidates; graph
 * autotune adapts this request through {@link GraphAutotuneRequest}; platform
 * calibration uses {@code tuning.calibration.PlatformCalibrationRequest} because
 * it mutates runtime-platform knobs across ordered calibration families.</p>
 *
 * <p>The request is thread-safe after construction if the supplied candidate
 * space, policies, workload, and listener are themselves safe to share. The
 * default file-backed persistence used by sessions performs side effects only
 * when {@link #persistence()} enables paths.</p>
 *
 * <p>Example:</p>
 * <pre>{@code
 * AutotuneRequest request = new AutotuneRequest(
 *         workload,
 *         "matmul-tune",
 *         DataType.FLOAT32,
 *         ExecutionMode.CPU,
 *         graphPolicy,
 *         runtimeProfile,
 *         candidateSpace,
 *         MeasurementPolicy.defaults(),
 *         ValidationPolicy.balancedDTypeAware(false),
 *         new SearchPolicy(64, 8, 4, true),
 *         PersistencePolicy.disabled(),
 *         AutotuneProgressListener.noop());
 * TuningResult result = AutotuneSession.create(request).run();
 * }</pre>
 *
 * @param workload workload factory to instantiate for each candidate; required
 * @param profileName logical profile namespace; blank values become {@code "autotune"}
 * @param dataType tensor dtype used by assembled candidate profiles; required
 * @param executionMode execution mode used by assembled candidate profiles; required
 * @param graphPolicy graph optimizer/execution policy shared by generated profiles; required
 * @param runtimeProfile platform runtime profile shared by generated profiles; required
 * @param candidateSpace candidate generator searched by the session; required
 * @param measurement measurement controls; {@code null} means {@link MeasurementPolicy#defaults()}
 * @param validation validation controls; {@code null} means validation is disabled
 * @param search search limits and pruning controls; {@code null} uses a balanced default
 * @param persistence optional best-profile/history persistence; {@code null} disables persistence
 * @param progressListener optional progress sink; {@code null} becomes a no-op listener
 */
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

    /**
     * Assembles an {@link ExecutionProfile} from the request's current graph and
     * runtime policies without consulting the candidate space.
     *
     * @param candidateName candidate label to embed in the assembled profile
     * @return execution profile carrying this request's dtype, mode, graph policy,
     * and runtime profile
     */
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
