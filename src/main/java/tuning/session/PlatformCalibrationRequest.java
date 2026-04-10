package tuning.session;

import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.profile.GraphExecutionPolicy;
import config.profile.PlatformRuntimeProfile;
import tensor.DataType;

import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

public record PlatformCalibrationRequest(
        String platformId,
        String profileName,
        DataType dataType,
        ExecutionMode executionMode,
        GraphExecutionPolicy graphPolicy,
        PlatformRuntimeProfile seedRuntimeProfile,
        List<PlatformCalibrationStep> steps,
        Path outputProfilePath,
        PlatformCalibrationProgressListener progressListener
) {
    public PlatformCalibrationRequest {
        platformId = (platformId == null || platformId.isBlank()) ? "platform" : platformId;
        profileName = profileName == null || profileName.isBlank() ? platformId : profileName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(executionMode, "executionMode cannot be null");
        Objects.requireNonNull(graphPolicy, "graphPolicy cannot be null");
        Objects.requireNonNull(seedRuntimeProfile, "seedRuntimeProfile cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
        progressListener = progressListener == null ? PlatformCalibrationProgressListener.noop() : progressListener;
    }

    public static PlatformCalibrationRequest fromSeedExecutionProfile(
            String platformId,
            ExecutionProfile seedProfile,
            List<PlatformCalibrationStep> steps,
            Path outputProfilePath
    ) {
        Objects.requireNonNull(seedProfile, "seedProfile cannot be null");
        return new PlatformCalibrationRequest(
                platformId,
                seedProfile.profileName(),
                seedProfile.dataType(),
                seedProfile.mode(),
                GraphExecutionPolicy.fromExecutionProfile(seedProfile),
                PlatformRuntimeProfile.fromExecutionProfile(
                        platformId,
                        platformId,
                        "UNSPECIFIED",
                        seedProfile
                ),
                steps,
                outputProfilePath,
                null
        );
    }
}
