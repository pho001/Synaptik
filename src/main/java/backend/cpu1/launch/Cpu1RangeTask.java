package backend.cpu1.launch;

@FunctionalInterface
public interface Cpu1RangeTask {
    void run(int startInclusive, int endExclusive);
}
