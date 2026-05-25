package backend.cpu.kernels.linalg.matmul.exec;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.plan.linalg.matmul.MatMulExecutionRoute;
import tensor.Tensor;

public interface PreparedMatMulExecutable {
    void execute(Tensor a, Tensor b, Tensor node, CpuKernelContext context);

    default boolean acceptsNativeInputs() {
        return false;
    }

    default MatMulExecutionRoute lastExecutionRoute() {
        return null;
    }

    default long lastCopyInBytes() {
        return -1L;
    }

    default long lastCopyOutBytes() {
        return -1L;
    }

    default String lastFallbackReason() {
        return "";
    }

    default String lastBlasSymbol() {
        return "";
    }
}
