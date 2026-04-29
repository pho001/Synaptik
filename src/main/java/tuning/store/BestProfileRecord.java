package tuning.store;

import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

import java.time.OffsetDateTime;
import java.util.Objects;

/**
 * Persisted best-profile selection from autotune.
 *
 * <p>The record combines hardware/workload fingerprints with candidate metadata
 * so consumers can decide whether a saved profile is applicable and safe to
 * promote. For graph-autotune records, the embedded {@link #profile()} is the
 * measured graph-policy winner plus a runtime snapshot from the measurement
 * run. Consumers that execute the graph later should treat {@link #graphPolicy()}
 * as the graph-specific payload and rebase it onto the current calibrated
 * {@link PlatformRuntimeProfile} with {@link #rebaseOnRuntime(PlatformRuntimeProfile)}.</p>
 *
 * @param hardware hardware fingerprint captured at selection time
 * @param workload workload fingerprint used for applicability checks
 * @param profile selected execution profile
 * @param score primary latency score, usually median milliseconds
 * @param updatedAt save time
 * @param autotuneKind source autotune flow, such as graph or legacy
 * @param graphAutotuneMode graph autotune mode, when applicable
 * @param candidateKind selected candidate category
 * @param candidateMetadata selected candidate provenance metadata
 * @param runtimeProfileId runtime profile id associated with the selection
 * @param productionEligible whether automated promotion is allowed
 */
public record BestProfileRecord(
        HardwareFingerprint hardware,
        WorkloadFingerprint workload,
        ExecutionProfile profile,
        double score,
        OffsetDateTime updatedAt,
        String autotuneKind,
        String graphAutotuneMode,
        CandidateKind candidateKind,
        CandidateMetadata candidateMetadata,
        String runtimeProfileId,
        boolean productionEligible
) {
    public BestProfileRecord {
        if (hardware == null) {
            hardware = HardwareFingerprint.capture();
        }
        if (workload == null) {
            throw new IllegalArgumentException("workload cannot be null");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        updatedAt = updatedAt == null ? OffsetDateTime.now() : updatedAt;
        autotuneKind = autotuneKind == null || autotuneKind.isBlank() ? "unknown" : autotuneKind;
        graphAutotuneMode = graphAutotuneMode == null ? "" : graphAutotuneMode;
        candidateKind = candidateKind == null ? CandidateKind.GENERIC : candidateKind;
        candidateMetadata = candidateMetadata == null ? CandidateMetadata.generic() : candidateMetadata;
        runtimeProfileId = runtimeProfileId == null ? "" : runtimeProfileId;
    }

    public BestProfileRecord(
            HardwareFingerprint hardware,
            WorkloadFingerprint workload,
            ExecutionProfile profile,
            double score,
            OffsetDateTime updatedAt
    ) {
        this(
                hardware,
                workload,
                profile,
                score,
                updatedAt,
                "legacy",
                "",
                CandidateKind.GENERIC,
                CandidateMetadata.generic(),
                "",
                true
        );
    }

    /**
     * Extracts the graph-side optimizer policy from the stored execution profile.
     *
     * <p>This is the stable payload for graph autotune winners. Runtime/backend
     * fields inside {@link #profile()} are measurement context, not the
     * authoritative platform calibration for future executions.</p>
     *
     * @return graph policy stored in the winning profile
     */
    public GraphExecutionPolicy graphPolicy() {
        return GraphExecutionPolicy.fromExecutionProfile(profile);
    }

    /**
     * Reassembles the stored graph policy with a current calibrated runtime profile.
     *
     * <p>This keeps graph autotune graph/workload-scoped while allowing platform
     * calibration to remain platform/dtype/mode-scoped. The returned profile keeps
     * the stored profile and candidate names, dtype, execution mode, optimizer
     * policy, and workload descriptor, but replaces runtime/backend settings with
     * {@code runtimeProfile.toRuntimeConfig()}.</p>
     *
     * @param runtimeProfile current calibrated runtime profile for the target platform, dtype, and mode
     * @return executable profile using the stored graph policy and supplied runtime profile
     * @throws NullPointerException if {@code runtimeProfile} is {@code null}
     * @throws IllegalArgumentException if the runtime profile targets a different dtype or execution mode
     */
    public ExecutionProfile rebaseOnRuntime(PlatformRuntimeProfile runtimeProfile) {
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        if (runtimeProfile.metadata().dataType() != profile.dataType()) {
            throw new IllegalArgumentException("Runtime profile dtype " + runtimeProfile.metadata().dataType()
                    + " does not match graph winner dtype " + profile.dataType());
        }
        if (runtimeProfile.metadata().executionMode() != profile.mode()) {
            throw new IllegalArgumentException("Runtime profile execution mode "
                    + runtimeProfile.metadata().executionMode()
                    + " does not match graph winner execution mode " + profile.mode());
        }
        return ExecutionProfileAssembler.assemble(
                profile.profileName(),
                profile.candidateName(),
                profile.dataType(),
                profile.mode(),
                runtimeProfile,
                graphPolicy(),
                profile.workload()
        );
    }
}
