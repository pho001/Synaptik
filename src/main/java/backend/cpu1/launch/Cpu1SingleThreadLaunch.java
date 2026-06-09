package backend.cpu1.launch;

/**
 * Single-thread launch policy for initial cpu1 kernels.
 */
public final class Cpu1SingleThreadLaunch implements Cpu1LaunchPolicy {
    private final Cpu1LaunchConfig launchConfig;

    public Cpu1SingleThreadLaunch() {
        this(Cpu1LaunchConfig.singleThread());
    }

    public Cpu1SingleThreadLaunch(Cpu1LaunchConfig launchConfig) {
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        this.launchConfig = launchConfig;
        if (launchConfig.workerCount() != 1) {
            throw new IllegalArgumentException("Cpu1SingleThreadLaunch requires workerCount=1.");
        }
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    @Override
    public void launch(int elementCount, Cpu1RangeTask task) {
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount must be >= 0");
        }
        if (task == null) {
            throw new IllegalArgumentException("task cannot be null");
        }
        if (elementCount == 0) {
            return;
        }
        task.run(0, elementCount);
    }
}
