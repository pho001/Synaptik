package backend.cpu1.kernels.layout;

final class Cpu1AliasLayoutKernel {
    private Cpu1AliasLayoutKernel() {
    }

    static void runAlias(Cpu1LayoutKernelSupport support) {
        throw new UnsupportedOperationException("cpu1 layout alias kernels are handled by alias executable routes.");
    }
}
