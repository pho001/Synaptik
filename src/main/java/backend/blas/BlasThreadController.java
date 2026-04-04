package backend.blas;

import backend.runtime.BlasConfig;

public final class BlasThreadController {
    private static volatile BlasProvider lastProvider;
    private static volatile BlasThreadPolicy lastPolicy;
    private static volatile int lastThreads = Integer.MIN_VALUE;

    private BlasThreadController() {
    }

    public static void apply(BlasConfig config) {
        if (config == null) {
            return;
        }
        BlasProvider provider = config.provider();
        BlasThreadPolicy policy = config.threadPolicy();
        int threads = config.threads();
        if (provider == lastProvider && policy == lastPolicy && threads == lastThreads) {
            return;
        }
        if (provider == BlasProvider.OPENBLAS_FFM && OpenBlasFfmBridge.isAvailable()) {
            OpenBlasFfmBridge.applyThreadPolicy(policy, threads);
        }
        lastProvider = provider;
        lastPolicy = policy;
        lastThreads = threads;
    }
}
