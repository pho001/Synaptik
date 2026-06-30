package config.profile;

import runtime.contract.ExecutionMode;
import tensor.DataType;

import java.util.Objects;

public final class ExecutionProfileAssembler {
    private ExecutionProfileAssembler() {
    }

    public static ExecutionProfile assemble(
            String profileName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy
    ) {
        return assemble(profileName, profileName, dataType, executionMode, runtimeProfile, graphPolicy, WorkloadProfile.none());
    }

    public static ExecutionProfile assemble(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy
    ) {
        return assemble(profileName, candidateName, dataType, executionMode, runtimeProfile, graphPolicy, WorkloadProfile.none());
    }

    public static ExecutionProfile assemble(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode executionMode,
            PlatformRuntimeProfile runtimeProfile,
            GraphExecutionPolicy graphPolicy,
            WorkloadProfile workload
    ) {
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
        Objects.requireNonNull(runtimeProfile, "runtimeProfile cannot be null");
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        return new ExecutionProfile(
                profileName,
                candidateName,
                dataType,
                executionMode,
                graphPolicy.compile(),
                runtimeProfile.toRuntimeConfig(),
                workload
        );
    }
}
