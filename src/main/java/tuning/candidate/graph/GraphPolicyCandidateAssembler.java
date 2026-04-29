package tuning.candidate.graph;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import tensor.DataType;
import tuning.candidate.RuntimeConfigOverride;

import java.util.Objects;

public final class GraphPolicyCandidateAssembler {
    private GraphPolicyCandidateAssembler() {
    }

    public static ExecutionProfile assemble(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy
    ) {
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        return ExecutionProfileAssembler.assemble(
                profileName,
                candidateName,
                dataType,
                executionMode,
                runtimeProfile,
                graphPolicy,
                WorkloadProfile.none()
        );
    }

    public static ExecutionProfile assemble(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy,
            RuntimeConfigOverride runtimeOverride
    ) {
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        var runtime = runtimeProfile.toRuntimeConfig();
        var overridden = (runtimeOverride == null ? RuntimeConfigOverride.identity() : runtimeOverride).apply(runtime);
        return new ExecutionProfile(
                profileName,
                candidateName,
                dataType,
                executionMode,
                graphPolicy.optimizer(),
                overridden,
                WorkloadProfile.none()
        );
    }
}
