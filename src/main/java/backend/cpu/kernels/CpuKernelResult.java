package backend.cpu.kernels;

public record CpuKernelResult(
        String route,
        boolean fallbackUsed,
        String fallbackReason
) {
    private static final String CPU_ARRAY_ROUTE = "CPU_ARRAY";
    private static final CpuKernelResult COMPLETED = new CpuKernelResult(CPU_ARRAY_ROUTE, false, "");

    public CpuKernelResult {
        route = route == null ? "" : route;
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
    }

    public static CpuKernelResult completed() {
        return COMPLETED;
    }

    public static CpuKernelResult route(String route) {
        return new CpuKernelResult(route, false, "");
    }

    public static CpuKernelResult fallback(String route, String fallbackReason) {
        return new CpuKernelResult(route, true, fallbackReason);
    }
}
