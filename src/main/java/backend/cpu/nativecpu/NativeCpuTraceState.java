package backend.cpu.nativecpu;

/**
 * Runtime trace evidence for a CPU step that considered native CPU storage.
 *
 * @param cpuStorageProfile runtime storage policy
 * @param nativeCpuFailurePolicy native fallback policy
 * @param requestedCpuStorage storage requested by runtime policy
 * @param actualCpuStorage storage actually used by this step
 * @param nativeCpuKernelStatus coverage/performance status used by the planner/executor
 * @param nativeCpuKernelFamily physical native CPU family
 * @param nativeCpuFallbackReason fallback reason, if native execution did not run
 */
public record NativeCpuTraceState(
        String cpuStorageProfile,
        String nativeCpuFailurePolicy,
        String requestedCpuStorage,
        String actualCpuStorage,
        String nativeCpuKernelStatus,
        String nativeCpuKernelFamily,
        String nativeCpuFallbackReason
) {
    public NativeCpuTraceState {
        cpuStorageProfile = cpuStorageProfile == null ? "" : cpuStorageProfile;
        nativeCpuFailurePolicy = nativeCpuFailurePolicy == null ? "" : nativeCpuFailurePolicy;
        requestedCpuStorage = requestedCpuStorage == null ? "" : requestedCpuStorage;
        actualCpuStorage = actualCpuStorage == null ? "" : actualCpuStorage;
        nativeCpuKernelStatus = nativeCpuKernelStatus == null ? "" : nativeCpuKernelStatus;
        nativeCpuKernelFamily = nativeCpuKernelFamily == null ? "" : nativeCpuKernelFamily;
        nativeCpuFallbackReason = nativeCpuFallbackReason == null ? "" : nativeCpuFallbackReason;
    }
}
