import config.backend.KernelTuningConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BFloat16TrainingPolicy;
import config.runtime.BlasConfig;
import config.runtime.CpuExecutionPolicy;
import config.runtime.CpuStorageProfile;
import config.runtime.DeviceTransferPolicy;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.NativeCpuMemoryConfig;
import config.runtime.NativeMemoryPoolPolicy;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class RuntimeConfigTest {
    @Test
    void runtimeConfigDefaultsToArrayStorageAndArrayFallback() {
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults();

        assertEquals(CpuStorageProfile.CPU_ARRAY, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY, runtime.nativeCpuFailurePolicy());
        assertEquals(DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE, runtime.deviceTransferPolicy());
        assertEquals(NativeMemoryPoolPolicy.DISABLED, runtime.nativeCpuMemory().poolPolicy());
        assertEquals(BFloat16TrainingPolicy.ACTIVATIONS_ONLY, runtime.bfloat16TrainingPolicy());
        assertFalse(runtime.cpuExecutionPolicy().useCpu1Direct());
        assertTrue(runtime.cpuExecutionPolicy().allowCpu1DirectFallback());
    }

    @Test
    void runtimeConfigNormalizesNullStoragePolicyFields() {
        RuntimeConfig runtime = new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                null,
                null,
                null,
                null,
                null,
                null
        );

        assertEquals(CpuStorageProfile.CPU_ARRAY, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY, runtime.nativeCpuFailurePolicy());
        assertEquals(DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE, runtime.deviceTransferPolicy());
        assertEquals(NativeMemoryPoolPolicy.DISABLED, runtime.nativeCpuMemory().poolPolicy());
        assertEquals(BFloat16TrainingPolicy.ACTIVATIONS_ONLY, runtime.bfloat16TrainingPolicy());
        assertFalse(runtime.cpuExecutionPolicy().useCpu1Direct());
        assertTrue(runtime.cpuExecutionPolicy().allowCpu1DirectFallback());
    }

    @Test
    void runtimeConfigCopyMethodsPreserveUnchangedFields() {
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.AUTO)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.REQUIRE_NATIVE)
                .withDeviceTransferPolicy(DeviceTransferPolicy.REQUIRE_DIRECT)
                .withNativeCpuMemory(NativeCpuMemoryConfig.perExecution(4096L))
                .withBFloat16TrainingPolicy(BFloat16TrainingPolicy.PARAMS_BF16_EXPERIMENTAL)
                .withCpuExecutionPolicy(new CpuExecutionPolicy(true, false));

        assertEquals(new CpuExecutionPolicy(true, false), runtime.cpuExecutionPolicy());
        assertEquals(CpuStorageProfile.AUTO, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.REQUIRE_NATIVE, runtime.nativeCpuFailurePolicy());
        assertEquals(DeviceTransferPolicy.REQUIRE_DIRECT, runtime.deviceTransferPolicy());
        assertEquals(NativeMemoryPoolPolicy.PER_EXECUTION, runtime.nativeCpuMemory().poolPolicy());
        assertEquals(4096L, runtime.nativeCpuMemory().maxPoolBytes());
        assertEquals(BFloat16TrainingPolicy.PARAMS_BF16_EXPERIMENTAL, runtime.bfloat16TrainingPolicy());
        assertEquals(RuntimeConfig.inferenceDefaults().blas(), runtime.blas());
        assertEquals(RuntimeConfig.inferenceDefaults().accelerator(), runtime.accelerator());
    }
}
