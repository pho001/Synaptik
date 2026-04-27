package tuning.candidate.graph;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileAssembler;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import config.profile.WorkloadProfile;
import tensor.DataType;

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
}
