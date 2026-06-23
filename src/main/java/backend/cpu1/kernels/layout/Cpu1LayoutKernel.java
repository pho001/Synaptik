package backend.cpu1.kernels.layout;

/**
 * Runtime kernel for one prepared cpu1 layout/view variant.
 */
public interface Cpu1LayoutKernel {
    void run(Cpu1LayoutKernelSupport support);
}
