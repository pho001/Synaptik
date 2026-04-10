package backend.blas;

import backend.runtime.BlasConfig;

public final class BlasThreadController {
    private static volatile BlasProvider lastProvider;
    private static volatile int lastThreads = Integer.MIN_VALUE;

    private BlasThreadController() {
    }

    public static void apply(BlasConfig config) {
        if (config == null) {
            return;
        }
        BlasProvider provider = config.provider();
        int threads = config.threads();
        if (provider == lastProvider && threads == lastThreads) {
            return;
        }
        if (provider == BlasProvider.OPENBLAS_FFM && OpenBlasFfmBridge.isAvailable()) {
            OpenBlasFfmBridge.applyThreads(threads);
        }
        lastProvider = provider;
        lastThreads = threads;
    }
}
