package backend.cpu1;

import backend.cpu1.prepare.Cpu1PrepareConfig;
import backend.cpu1.storage.Cpu1StorageKind;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class Cpu1PrepareConfigTest {
    @Test
    void automaticForRuntimeStorageMapsCpuStorageProfileToCpu1StorageKind() {
        assertRuntimeStorageMapping(CpuStorageProfile.CPU_ARRAY, Cpu1StorageKind.JAVA_ARRAY);
        assertRuntimeStorageMapping(CpuStorageProfile.CPU_NATIVE, Cpu1StorageKind.MEMORY_SEGMENT);
        assertRuntimeStorageMapping(CpuStorageProfile.AUTO, Cpu1StorageKind.JAVA_ARRAY);
    }

    private static void assertRuntimeStorageMapping(
            CpuStorageProfile cpuStorageProfile,
            Cpu1StorageKind expectedStorageKind
    ) {
        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(cpuStorageProfile);

        assertEquals(expectedStorageKind, Cpu1PrepareConfig.storageKindFor(runtimeConfig));
        assertEquals(
                expectedStorageKind,
                Cpu1PrepareConfig.automaticForRuntimeStorage(runtimeConfig, 4).storageKind()
        );
    }
}
