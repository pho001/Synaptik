package backend.cpu1.launch;

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.kernels.Cpu1KernelRangeRunner;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.RecursiveAction;

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
        int elementCount = args.elementCount();
        if (elementCount == 0) {
            return;
        }
        int taskCount = Math.min(launchConfig.workerCount(), elementCount);
        int chunk = (elementCount + taskCount - 1) / taskCount;
        List<RecursiveAction> tasks = new ArrayList<>(taskCount);
        for (int start = 0; start < elementCount; start += chunk) {
            int rangeStart = start;
            int rangeEnd = Math.min(elementCount, rangeStart + chunk);
            tasks.add(new RecursiveAction() {
                @Override
                protected void compute() {
                    kernelRunner.computeRange(args, rangeStart, rangeEnd);
                }
            });
        }
        ForkJoinTask.invokeAll(tasks);
    }
}
