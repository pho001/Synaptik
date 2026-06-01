package backend.cpu1.kernels.layout;

import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;

final class Cpu1AliasLayoutKernel {
    private Cpu1AliasLayoutKernel() {
    }

    static void runAlias(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        new Cpu1LayoutKernelSupport(unit, context).aliasView();
    }
}
