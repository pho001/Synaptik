package backend.cpu.nativecpu;

import backend.cpu.kernels.CpuKernelContext;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;

final class NativeCpuRuntimePolicy {
    private NativeCpuRuntimePolicy() {
    }

    static boolean nativeRequested(CpuKernelContext context) {
        if (context == null) {
            return false;
        }
        RuntimeConfig runtimeConfig = context.executionContext().runtimeConfig();
        if (runtimeConfig == null) {
            return false;
        }
        if (runtimeConfig.cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE) {
            return true;
        }
        if (runtimeConfig.cpuStorageProfile() != CpuStorageProfile.AUTO) {
            return false;
        }
        PreparedNativeCpuPlan plan = context.nodePlan().nativeCpuPlan();
        return plan != null && plan.allowsNativeInputs();
    }
}
