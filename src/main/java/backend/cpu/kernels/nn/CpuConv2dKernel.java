package backend.cpu.kernels.nn;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.nn.conv.conv2d;
import tensor.DataType;

import java.util.List;

public final class CpuConv2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        CpuStorageView output = call.output();
        switch (output.dtype()) {
            case FLOAT64 -> {
                conv2d conv = require(call.operation());
                List<CpuStorageView> inputs = call.inputs();
                Conv2dDirectBackend.forwardF64(
                        conv, inputs.get(0), inputs.get(1), bias(inputs), output, call.outputTensor(), call.context());
            }
            case FLOAT32 -> {
                conv2d conv = require(call.operation());
                List<CpuStorageView> inputs = call.inputs();
                Conv2dDirectBackend.forwardF32(
                        conv, inputs.get(0), inputs.get(1), bias(inputs), output, call.outputTensor(), call.context());
            }
            case BFLOAT16 -> {
                conv2d conv = require(call.operation());
                List<CpuStorageView> inputs = call.inputs();
                Conv2dDirectBackend.forwardBF16(
                        conv, inputs.get(0), inputs.get(1), bias(inputs), output, call.outputTensor(), call.context());
            }
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static CpuStorageView bias(List<CpuStorageView> inputs) {
        return inputs.size() > 2 ? inputs.get(2) : null;
    }

    private static conv2d require(Operation op) {
        if (!(op instanceof conv2d conv)) {
            throw new IllegalArgumentException("CpuConv2dKernel requires conv2d operation");
        }
        return conv;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuConv2dKernel does not support " + dtype);
    }
}
