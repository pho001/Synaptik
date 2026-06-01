package backend.cpu1.launch;

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.kernels.Cpu1KernelRangeRunner;

import java.util.Objects;

/**
 * Range-splitting launch policy for prepared cpu1 units.
 */
public final class Cpu1ParallelLaunch implements Cpu1LaunchPolicy {
    private final Cpu1LaunchConfig launchConfig;

    public Cpu1ParallelLaunch(Cpu1LaunchConfig launchConfig) {
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        if (launchConfig.workerCount() < 2) {
            throw new IllegalArgumentException("Cpu1ParallelLaunch requires workerCount >= 2.");
        }
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    @Override
    public void launch(Cpu1KernelRangeRunner kernelRunner, Cpu1KernelArgs args) {
        Objects.requireNonNull(kernelRunner, "kernelRunner cannot be null");
        Objects.requireNonNull(args, "args cannot be null");
        Cpu1RangeLauncher.launch(args.elementCount(), launchConfig,
                (rangeStart, rangeEnd) -> kernelRunner.computeRange(args, rangeStart, rangeEnd));
    }
}
