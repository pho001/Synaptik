package config.profile;

import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import backend.runtime.ExecutionMode;
import tensor.DataType;

import java.util.Objects;

public record ExecutionProfile(
        String profileName,
        String candidateName,
        DataType dataType,
        ExecutionMode mode,
        OptimizerConfig optimizer,
        RuntimeConfig runtime
) {
    public ExecutionProfile {
        profileName = Objects.requireNonNullElse(profileName, "default");
        candidateName = candidateName == null || candidateName.isBlank() ? profileName : candidateName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        Objects.requireNonNull(optimizer, "optimizer cannot be null");
        Objects.requireNonNull(runtime, "runtime cannot be null");
    }
}
