package backend.cpu.kernels.layout;

public final class CpuExpandKernel extends CpuAliasLayoutKernel {
    @Override
    protected boolean usesNativeViewAlias() {
        return false;
    }
}
