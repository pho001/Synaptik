package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.kernels.linalg.matmul.exec.PreparedMatMulExecutable;
import backend.cpu.plan.CpuKernelCostClass;
import backend.cpu.storage.CpuStorageView;
import config.runtime.CpuStorageProfile;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.List;

public class CpuMatMulKernel implements CpuStorageAwareKernel {
    private static final String CPU_NATIVE_ROUTE = "CPU_NATIVE";

    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        CpuStorageView output = requireOutput(call);
        List<CpuStorageView> inputs = requireInputs(call);
        switch (output.dtype()) {
            case FLOAT64, FLOAT32, BFLOAT16 -> {
                if (canUsePreparedPath(call, inputs, output)) {
                    runPrepared(call.inputTensors(), call.outputTensor(), call.context());
                    return CpuKernelResult.completed();
                }
                return runStorage(call, inputs.get(0), inputs.get(1), output);
            }
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        throw new IllegalStateException("Unhandled matmul dtype " + output.dtype());
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

    private static CpuKernelResult runStorage(
            CpuKernelCall call,
            CpuStorageView a,
            CpuStorageView b,
            CpuStorageView output
    ) {
        if (shouldAllocateNativeOutput(call, output)) {
            NativeTensorStorage nativeOutput = allocateNativeOutput(call, output);
            CpuStorageView nativeOutputView = CpuStorageView.segment(
                    output.dtype(),
                    nativeOutput.segment(),
                    output.shape(),
                    output.strides(),
                    output.storageOffset(),
                    output.logicalSize()
            );
            CpuMatMulStorageLoops.execute(a, b, nativeOutputView);
            nativeOutput.markModified();
            call.context().executionContext().attachNativeStorage(
                    call.context().nodeId(),
                    nativeOutput,
                    "matmul storage-view Java path wrote " + output.dtype() + " native output"
            );
            return CpuKernelResult.route(CPU_NATIVE_ROUTE);
        }

        CpuMatMulStorageLoops.execute(a, b, output);
        markExistingNativeOutputModified(call, output);
        return output.isMemorySegment()
                ? CpuKernelResult.route(CPU_NATIVE_ROUTE)
                : CpuKernelResult.completed();
    }

    private static CpuStorageView requireOutput(CpuKernelCall call) {
        CpuStorageView output = call.output();
        if (output == null) {
            throw new IllegalArgumentException("MATMUL requires an output storage view.");
        }
        return output;
    }

    private static List<CpuStorageView> requireInputs(CpuKernelCall call) {
        List<CpuStorageView> inputs = call.inputs();
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("MATMUL expects exactly two input storage views.");
        }
        return inputs;
    }

    private static boolean canUsePreparedPath(CpuKernelCall call, List<CpuStorageView> inputs, CpuStorageView output) {
        return canUsePreparedArrayPath(inputs, output) || canUsePreparedNativeSegmentPath(call, inputs, output);
    }

    private static boolean canUsePreparedArrayPath(List<CpuStorageView> inputs, CpuStorageView output) {
        return allDenseZeroOffset(inputs, output)
                && inputs.get(0).isArray()
                && inputs.get(1).isArray()
                && output.isArray();
    }

    private static boolean canUsePreparedNativeSegmentPath(
            CpuKernelCall call,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        PreparedMatMulExecutable executable = call.context().matMulExecutable();
        return executable != null
                && executable.acceptsNativeInputs()
                && allDenseZeroOffset(inputs, output)
                && inputs.get(0).isMemorySegment()
                && inputs.get(1).isMemorySegment();
    }

    private static boolean allDenseZeroOffset(List<CpuStorageView> inputs, CpuStorageView output) {
        return CpuMatMulStorageLoops.isDenseZeroOffset(inputs.get(0))
                && CpuMatMulStorageLoops.isDenseZeroOffset(inputs.get(1))
                && CpuMatMulStorageLoops.isDenseZeroOffset(output);
    }

    private static boolean shouldAllocateNativeOutput(CpuKernelCall call, CpuStorageView output) {
        return nativeRequested(call.context())
                && !output.isMemorySegment()
                && call.inputs().stream().allMatch(CpuStorageView::isMemorySegment);
    }

    private static boolean nativeRequested(CpuKernelContext context) {
        return context != null
                && context.executionContext().runtimeConfig() != null
                && context.executionContext().runtimeConfig().cpuStorageProfile() == CpuStorageProfile.CPU_NATIVE;
    }

    private static NativeTensorStorage allocateNativeOutput(CpuKernelCall call, CpuStorageView output) {
        return call.context().executionContext().allocateNativeStorage(
                output.dtype(),
                CpuMatMulStorageLoops.requiredElementCapacity(output),
                "node-" + call.context().nodeId() + ":" + call.outputTensor().getLabel() + ":matmul-storage-view"
        );
    }

    private static void markExistingNativeOutputModified(CpuKernelCall call, CpuStorageView output) {
        if (!output.isMemorySegment()) {
            return;
        }
        NativeTensorStorage nativeOutput = call.context().executionContext().nativeStorageForNodeId(call.context().nodeId());
        if (nativeOutput == null) {
            return;
        }
        nativeOutput.markModified();
        call.context().executionContext().attachNativeStorage(
                call.context().nodeId(),
                nativeOutput,
                "matmul storage-view Java path wrote existing native output"
        );
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuMatMulKernel does not support " + dtype);
    }
}
