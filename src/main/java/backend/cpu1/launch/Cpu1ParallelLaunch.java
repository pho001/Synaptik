package backend.cpu1.launch;

/**
 * Range-splitting launch policy for prepared cpu1 units.
 */
public final class Cpu1ParallelLaunch implements Cpu1LaunchPolicy {
    private final Cpu1LaunchConfig launchConfig;

    public Cpu1ParallelLaunch(Cpu1LaunchConfig launchConfig) {
        if (launchConfig == null) {
            throw new IllegalArgumentException("launchConfig cannot be null");
        }
        this.launchConfig = launchConfig;
        if (launchConfig.workerCount() < 2) {
            throw new IllegalArgumentException("Cpu1ParallelLaunch requires workerCount >= 2.");
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
        Cpu1RangeLauncher.launch(elementCount, launchConfig, task::run);
    }
}
