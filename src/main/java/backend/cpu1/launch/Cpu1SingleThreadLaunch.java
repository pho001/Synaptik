package backend.cpu1.launch;

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.kernels.Cpu1KernelRangeRunner;

import java.util.Objects;

/**
 * Single-thread launch policy for initial cpu1 kernels.
 */
public final class Cpu1SingleThreadLaunch implements Cpu1LaunchPolicy {
    private final Cpu1LaunchConfig launchConfig;

    public Cpu1SingleThreadLaunch() {
        this(Cpu1LaunchConfig.singleThread());
    }

    public Cpu1SingleThreadLaunch(Cpu1LaunchConfig launchConfig) {
        this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
        if (launchConfig.workerCount() != 1) {
            throw new IllegalArgumentException("Cpu1SingleThreadLaunch requires workerCount=1.");
        }
    }

    public Cpu1LaunchConfig launchConfig() {
        return launchConfig;
    }

    @Override
    public void launch(Cpu1KernelRangeRunner kernelRunner, Cpu1KernelArgs args) {
        Objects.requireNonNull(kernelRunner, "kernelRunner cannot be null");
        Objects.requireNonNull(args, "args cannot be null");
        kernelRunner.computeRange(args, 0, args.elementCount());
    }
}
