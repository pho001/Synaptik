package backend.cpu.kernels.layout;

public class CpuPermuteKernel extends CpuAliasLayoutKernel {
    @Override
    protected boolean usesNativeViewAlias() {
        return false;
    }
}
