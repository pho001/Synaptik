package tuning.benchmark;

import backend.runtime.ExecutionMode;
import config.compile.BackendPlanningConfig;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.DeviceTransferPolicy;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import tensor.DataType;
import tuning.measure.MeasurementPolicy;
import tuning.reporting.ReportPolicy;
import tuning.validate.ValidationPolicy;
import tuning.workload.StandardWorkloads;

import java.util.List;

/**
 * Factory for the Wave 7 native CPU to Metal bridge evidence benchmark.
 */
public final class NativeDeviceBridgeBenchmark {
    public static final String WORKLOAD_NAME = "native_device_bridge_f32_metal_dense";
    public static final String CPU_ARRAY_METAL = "cpu-array-metal";
    public static final String CPU_NATIVE_ARRAY_BRIDGE_METAL = "cpu-native-array-bridge-metal";
    public static final String CPU_NATIVE_DIRECT_METAL = "cpu-native-direct-metal";

    private static final int DEFAULT_M = 128;
    private static final int DEFAULT_K = 128;
    private static final int DEFAULT_N = 128;

    private NativeDeviceBridgeBenchmark() {
    }

    /**
     * Creates the default dense F32 Metal transfer benchmark request.
     *
     * @return benchmark request with CPU array, native array-bridge, and native direct candidates
     */
    public static BenchmarkRequest request() {
        return request(DEFAULT_M, DEFAULT_K, DEFAULT_N, measurementPolicy());
    }

    /**
     * Creates a dense F32 Metal transfer benchmark request with caller-selected shape and measurement policy.
     *
     * @param m left rows
     * @param k shared dimension
     * @param n right columns
     * @param measurement measurement policy; {@code null} uses the bridge benchmark default
     * @return benchmark request
     */
    public static BenchmarkRequest request(int m, int k, int n, MeasurementPolicy measurement) {
        return new BenchmarkRequest(
                StandardWorkloads.matmul(WORKLOAD_NAME, 1, m, k, n),
                entries(),
                measurement == null ? measurementPolicy() : measurement,
                ValidationPolicy.disabled(),
                ReportPolicy.defaults()
        );
    }

    /**
     * Returns the three candidate entries compared by the benchmark.
     *
     * @return benchmark entries
     */
    public static List<BenchmarkEntry> entries() {
        return List.of(
                BenchmarkEntry.baseline(CPU_ARRAY_METAL, profile(
                        CPU_ARRAY_METAL,
                        CpuStorageProfile.CPU_ARRAY,
                        NativeCpuFailurePolicy.FALLBACK_TO_ARRAY,
                        DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
                )),
                BenchmarkEntry.candidate(CPU_NATIVE_ARRAY_BRIDGE_METAL, profile(
                        CPU_NATIVE_ARRAY_BRIDGE_METAL,
                        CpuStorageProfile.CPU_NATIVE,
                        NativeCpuFailurePolicy.FALLBACK_TO_ARRAY,
                        DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
                )),
                BenchmarkEntry.candidate(CPU_NATIVE_DIRECT_METAL, profile(
                        CPU_NATIVE_DIRECT_METAL,
                        CpuStorageProfile.CPU_NATIVE,
                        NativeCpuFailurePolicy.REQUIRE_NATIVE,
                        DeviceTransferPolicy.REQUIRE_DIRECT
                ))
        );
    }

    /**
     * Uses a short steady-state loop while retaining cold-run transfer trace evidence.
     *
     * @return measurement policy for the bridge benchmark
     */
    public static MeasurementPolicy measurementPolicy() {
        return new MeasurementPolicy(2, 5, 3, true, true, true, true, true);
    }

    private static ExecutionProfile profile(
            String name,
            CpuStorageProfile cpuStorageProfile,
            NativeCpuFailurePolicy nativeFailurePolicy,
            DeviceTransferPolicy transferPolicy
    ) {
        return new ExecutionProfile(
                name,
                name,
                DataType.FLOAT32,
                ExecutionMode.FORWARD,
                CompileConfig.inference().withBackendPlanning(BackendPlanningConfig.requireAnyAcceleratorRegion()),
                RuntimeConfig.inferenceDefaults()
                        .withAccelerator(metalBufferRequired())
                        .withCpuStorageProfile(cpuStorageProfile)
                        .withNativeCpuFailurePolicy(nativeFailurePolicy)
                        .withDeviceTransferPolicy(transferPolicy),
                WorkloadProfile.none()
        );
    }

    private static AcceleratorConfig metalBufferRequired() {
        AcceleratorBufferConfig buffer = new AcceleratorBufferConfig(
                AcceleratorBufferBindingMode.REQUIRE,
                true,
                0L
        );
        AcceleratorBackendConfig metal = AcceleratorBackendConfig.defaults().withBuffer(buffer);
        return AcceleratorConfig.disabled().withMetal(metal);
    }
}
