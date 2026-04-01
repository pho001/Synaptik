package backend.kernels.cpu;

public record ResolvedMatMulHints(
        boolean useBlas,
        boolean parallel,
        int tileM,
        int tileN,
        int tileK,
        int plannedWorkers,
        long work
) {
    public ResolvedMatMulHints {
        tileM = Math.max(1, tileM);
        tileN = Math.max(1, tileN);
        tileK = Math.max(1, tileK);
        plannedWorkers = Math.max(1, plannedWorkers);
        work = Math.max(0L, work);
    }
}