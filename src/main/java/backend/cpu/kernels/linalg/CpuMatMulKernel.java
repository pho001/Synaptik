package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.plan.CpuKernelCostClass;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public class CpuMatMulKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64, FLOAT32, BFLOAT16 -> runPrepared(call.inputTensors(), node, call.context());
            case INT32, INT64, BOOL -> unsupported(node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }

    private static void runPrepared(List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        PreparedMatMulExecutable executable = context.matMulExecutable();
        if (executable == null) {
            throw new IllegalStateException("Missing PreparedMatMulExecutable for matmul execution.");
        }
        executable.execute(inputs.get(0), inputs.get(1), node, context);
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuMatMulKernel does not support " + dtype);
    }
}
