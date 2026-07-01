package config.runtime;

/**
 * Runtime policy for direct CPU execution routing.
 *
 * @param useCpu1Direct whether normal CPU nodes should prepare through the cpu1 direct path
 * @param allowCpu1DirectFallback whether unsupported cpu1 direct nodes may fall back to the legacy CPU preparer
 */
public record CpuExecutionPolicy(
        boolean useCpu1Direct,
        boolean allowCpu1DirectFallback
) {
    public static CpuExecutionPolicy defaults() {
        return new CpuExecutionPolicy(false, true);
    }

    public CpuExecutionPolicy withUseCpu1Direct(boolean value) {
        return new CpuExecutionPolicy(value, allowCpu1DirectFallback);
    }

    public CpuExecutionPolicy withAllowCpu1DirectFallback(boolean value) {
        return new CpuExecutionPolicy(useCpu1Direct, value);
    }
}
