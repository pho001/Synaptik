package io.github.pho001.synaptik.tools.cli;

/** Shared form and application defaults; optimizer-specific rates remain in the shared workflow. */
final class CliDefaults {
    static final int EPOCHS = 20;
    static final int BATCH_SIZE = 1;
    static final int OUTPUTS = 1;
    static final long SEED = 0L;
    static final long REFRESH_MILLIS = 500L;
    static final String BACKEND = "cpu";
    static final String OPTIMIZER = "sgd";
    static final String ONEDNN_API = "auto";

    private CliDefaults() { }
}
