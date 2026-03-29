package Benchmark.measure;

@FunctionalInterface
public interface NanoClock {
    NanoClock SYSTEM = System::nanoTime;

    long nanoTime();
}
