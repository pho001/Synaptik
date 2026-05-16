import config.backend.KernelTuningConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class RuntimeConfigTest {
    @Test
    void runtimeConfigDefaultsToArrayStorageAndArrayFallback() {
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults();

        assertEquals(CpuStorageProfile.CPU_ARRAY, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY, runtime.nativeCpuFailurePolicy());
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
                null
        );

        assertEquals(CpuStorageProfile.CPU_ARRAY, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY, runtime.nativeCpuFailurePolicy());
    }

    @Test
    void runtimeConfigCopyMethodsPreserveUnchangedFields() {
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.AUTO)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.REQUIRE_NATIVE);

        assertEquals(CpuStorageProfile.AUTO, runtime.cpuStorageProfile());
        assertEquals(NativeCpuFailurePolicy.REQUIRE_NATIVE, runtime.nativeCpuFailurePolicy());
        assertEquals(RuntimeConfig.inferenceDefaults().blas(), runtime.blas());
        assertEquals(RuntimeConfig.inferenceDefaults().accelerator(), runtime.accelerator());
    }
}
