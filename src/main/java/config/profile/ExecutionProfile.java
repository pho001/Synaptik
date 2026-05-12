package config.profile;

import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import backend.runtime.ExecutionMode;
import tensor.DataType;

import java.util.Objects;

/**
 * Complete executable profile for one measured or runnable graph candidate.
 *
 * <p>An execution profile combines the compile policy with the runtime/backend policy
 * that will be used when a workload is compiled, prepared, and executed. Autotune and benchmark flows
 * compare multiple {@code ExecutionProfile} instances by assigning each candidate a stable
 * {@code candidateName} while preserving a broader {@code profileName} namespace.</p>
 *
 * <p>The record is immutable. Component objects such as {@link CompileConfig} and
 * {@link RuntimeConfig} are expected to be treated as immutable value objects as well.</p>
 *
 * <pre>{@code
 * ExecutionProfile baseline = new ExecutionProfile(
 *         "abc-baseline-no-opt-f64",
 *         "baseline-no-opt",
 *         DataType.FLOAT64,
 *         ExecutionMode.FORWARD_BACKWARD,
 *         CompileConfig.noGraphOptimizationBaseline(),
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
 * @param compile compile-time semantic, graph, backend, region, and memory planning policy
 * @param runtime backend/runtime policy
 * @param workload optional workload descriptor used by specialized tuning decisions; {@code null}
 *                 becomes {@link WorkloadProfile#none()}
 */
public record ExecutionProfile(
        String profileName,
        String candidateName,
        DataType dataType,
        ExecutionMode mode,
        CompileConfig compile,
        RuntimeConfig runtime,
        WorkloadProfile workload
) {
    public ExecutionProfile {
        profileName = Objects.requireNonNullElse(profileName, "default");
        candidateName = candidateName == null || candidateName.isBlank() ? profileName : candidateName;
        Objects.requireNonNull(dataType, "dataType cannot be null");
        Objects.requireNonNull(mode, "mode cannot be null");
        Objects.requireNonNull(compile, "compile cannot be null");
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
     * @param compile compile policy
     * @param runtime backend/runtime policy
     */
    public ExecutionProfile(
            String profileName,
            String candidateName,
            DataType dataType,
            ExecutionMode mode,
            CompileConfig compile,
            RuntimeConfig runtime
    ) {
        this(profileName, candidateName, dataType, mode, compile, runtime, WorkloadProfile.none());
    }

    public static ExecutionProfile trainingCpu() {
        return new ExecutionProfile(
                "training-cpu",
                "training-cpu",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.training(),
                RuntimeConfig.trainingDefaults()
        );
    }

    public static ExecutionProfile inferenceCpu() {
        return new ExecutionProfile(
                "inference-cpu",
                "inference-cpu",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference(),
                RuntimeConfig.inferenceDefaults()
        );
    }

    public static ExecutionProfile trainingAutoAccelerator() {
        return new ExecutionProfile(
                "training-auto-accelerator",
                "training-auto-accelerator",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.trainingAutoAccelerator(),
                RuntimeConfig.trainingDefaults()
        );
    }

    public static ExecutionProfile inferenceAutoAccelerator() {
        return new ExecutionProfile(
                "inference-auto-accelerator",
                "inference-auto-accelerator",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inferenceAutoAccelerator(),
                RuntimeConfig.inferenceDefaults()
        );
    }

    public static ExecutionProfile trainingExplicitAccelerator() {
        return new ExecutionProfile(
                "training-explicit-accelerator",
                "training-explicit-accelerator",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.trainingExplicitAccelerator(),
                RuntimeConfig.trainingDefaults()
        );
    }

    public static ExecutionProfile inferenceExplicitAccelerator() {
        return new ExecutionProfile(
                "inference-explicit-accelerator",
                "inference-explicit-accelerator",
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inferenceExplicitAccelerator(),
                RuntimeConfig.inferenceDefaults()
        );
    }

    public static ExecutionProfile requiredAcceleratorTest() {
        return new ExecutionProfile(
                "required-accelerator-test",
                "required-accelerator-test",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.requireExplicitAccelerator(),
                RuntimeConfig.trainingDefaults()
        );
    }

    public static ExecutionProfile noGraphOptimizationBaseline() {
        return new ExecutionProfile(
                "no-graph-optimization-baseline",
                "no-graph-optimization-baseline",
                DataType.FLOAT32,
                ExecutionMode.FORWARD_BACKWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                RuntimeConfig.noOptNoVecNoPar()
        );
    }
}
