package config.profile;

import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import backend.runtime.ExecutionMode;
import tensor.DataType;

import java.util.Objects;

/**
 * Complete executable profile for one measured or runnable graph candidate.
 *
 * <p>An execution profile combines the graph-side optimizer policy with the runtime/backend policy
 * that will be used when a workload is compiled, prepared, and executed. Autotune and benchmark flows
 * compare multiple {@code ExecutionProfile} instances by assigning each candidate a stable
 * {@code candidateName} while preserving a broader {@code profileName} namespace.</p>
 *
 * <p>The record is immutable. Component objects such as {@link OptimizerConfig} and
 * {@link RuntimeConfig} are expected to be treated as immutable value objects as well.</p>
 *
 * <pre>{@code
 * ExecutionProfile baseline = new ExecutionProfile(
 *         "abc-baseline-no-opt-f64",
 *         "baseline-no-opt",
 *         DataType.FLOAT64,
 *         ExecutionMode.FORWARD_BACKWARD,
 *         OptimizerConfig.noOptimization(),
 *         RuntimeConfig.noOptNoVecNoPar());
 * // profileName = "abc-baseline-no-opt-f64"
 * // candidateName = "baseline-no-opt"
 * // dtype = FLOAT64, mode = FORWARD_BACKWARD
 * }</pre>
 *
 * @param profileName human-readable profile namespace; {@code null} becomes {@code "default"}
 * @param candidateName benchmark/autotune candidate name; blank or {@code null} falls back to
 *                      {@code profileName}
 * @param dataType tensor dtype this profile is intended to execute
 * @param mode execution mode, usually forward-only inference or forward/backward training
 * @param optimizer graph optimization policy
 * @param runtime backend/runtime policy
 * @param workload optional workload descriptor used by specialized tuning decisions; {@code null}
 *                 becomes {@link WorkloadProfile#none()}
 */
public record ExecutionProfile(
        String profileName,
        String candidateName,
        DataType dataType,
        ExecutionMode mode,
        OptimizerConfig optimizer,
        RuntimeConfig runtime,
        WorkloadProfile workload
) {
    public ExecutionProfile {
        profileName = Objects.requireNonNullElse(profileName, "default");
        candidateName = candidateName == null || candidateName.isBlank() ? profileName : candidateName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        Objects.requireNonNull(optimizer, "optimizer cannot be null");
        Objects.requireNonNull(runtime, "runtime cannot be null");
        workload = workload == null ? WorkloadProfile.none() : workload;
    }

    /**
     * Creates a profile without a specialized workload descriptor.
     *
     * @param profileName human-readable profile namespace
     * @param candidateName benchmark/autotune candidate name
     * @param dataType tensor dtype this profile is intended to execute
     * @param mode execution mode
     * @param optimizer graph optimization policy
     * @param runtime backend/runtime policy
     */
    public ExecutionProfile(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode mode,
            OptimizerConfig optimizer,
            RuntimeConfig runtime
    ) {
        this(profileName, candidateName, dataType, mode, optimizer, runtime, WorkloadProfile.none());
    }
}
