package backend.cpu1.launch;

/**
 * Prepare-time launch policy inputs for cpu1 execution.
 *
 * @param workerCount number of workers requested for the unit
 */
public record Cpu1LaunchConfig(int workerCount) {
    public Cpu1LaunchConfig {
        if (workerCount < 1) {
            throw new IllegalArgumentException("workerCount must be >= 1");
        }
    }

    public static Cpu1LaunchConfig singleThread() {
        return new Cpu1LaunchConfig(1);
    }

    public static Cpu1LaunchConfig parallel(int workerCount) {
        return new Cpu1LaunchConfig(workerCount);
    }
}
