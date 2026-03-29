package Backend.kernels.cpu;

import java.util.concurrent.ConcurrentHashMap;

public final class CpuSchedulerAdvisor {
    private static final ConcurrentHashMap<String, Ewma> EWMA_NS_PER_ELEM = new ConcurrentHashMap<>();
    private static final double ALPHA = 0.2d;

    private CpuSchedulerAdvisor() {}

    public static boolean shouldUseCommonPool(
            CpuKernelCostClass costClass,
            String key,
            int length,
            CpuExecutionConfig config
    ) {
        if (costClass != CpuKernelCostClass.LOW || key == null || config == null || length <= 0) {
            return false;
        }
        Ewma ewma = EWMA_NS_PER_ELEM.get(key);
        if (ewma == null || !ewma.initialized) {
            return true;
        }
        return ewma.value <= config.lowCostNsPerElementThreshold();
    }

    public static void recordSample(String key, int length, long elapsedNs) {
        if (key == null || key.isEmpty() || length <= 0 || elapsedNs <= 0L) {
            return;
        }
        double sample = (double) elapsedNs / (double) length;
        EWMA_NS_PER_ELEM.compute(key, (k, existing) -> {
            if (existing == null) {
                Ewma init = new Ewma();
                init.value = sample;
                init.initialized = true;
                return init;
            }
            existing.value = (ALPHA * sample) + ((1.0d - ALPHA) * existing.value);
            existing.initialized = true;
            return existing;
        });
    }

    public static void reset() {
        EWMA_NS_PER_ELEM.clear();
    }

    private static final class Ewma {
        private volatile double value;
        private volatile boolean initialized;
    }
}
