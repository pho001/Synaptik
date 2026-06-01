package backend.cpu1.launch;

/**
 * Prepare-time launch policy inputs for cpu1 execution.
 *
 * @param workerCount number of workers requested for the unit
 * @param chunkSize resolved range chunk size, or 0 when the launch policy should split by worker count
 */
public record Cpu1LaunchConfig(int workerCount, int chunkSize) {
    public Cpu1LaunchConfig(int workerCount) {
        this(workerCount, 0);
    }

    public Cpu1LaunchConfig {
        if (workerCount < 1) {
            throw new IllegalArgumentException("workerCount must be >= 1");
        }
        if (chunkSize < 0) {
            throw new IllegalArgumentException("chunkSize must be >= 0");
        }
    }

    public static Cpu1LaunchConfig singleThread() {
        return new Cpu1LaunchConfig(1, 0);
    }

    public static Cpu1LaunchConfig parallel(int workerCount) {
        return new Cpu1LaunchConfig(workerCount, 0);
    }

    public static Cpu1LaunchConfig parallel(int workerCount, int chunkSize) {
        return new Cpu1LaunchConfig(workerCount, chunkSize);
    }

    public boolean hasResolvedChunkSize() {
        return chunkSize > 0;
    }
}
