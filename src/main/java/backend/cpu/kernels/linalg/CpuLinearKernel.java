package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import config.runtime.CpuStorageProfile;
import operations.Operation;
import operations.linalg.linear;
import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.List;

public final class CpuLinearKernel implements CpuStorageAwareKernel {
    private static final String CPU_NATIVE_ROUTE = "CPU_NATIVE";

    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        linear linear = require(call.operation());
        List<CpuStorageView> inputs = requireInputs(call, linear);
        CpuStorageView output = requireOutput(call);
        CpuKernelContext context = call.context();
        switch (output.dtype()) {
            case FLOAT64 -> {
                if (canUseDenseArrayHotPath(linear, inputs, output)) {
                    runDenseArrayHotPath(call, linear, output.dtype(), context);
                    return CpuKernelResult.completed();
                }
                return runStorage(call, linear, inputs, output);
            }
            case FLOAT32 -> {
                if (canUseDenseArrayHotPath(linear, inputs, output)) {
                    runDenseArrayHotPath(call, linear, output.dtype(), context);
                    return CpuKernelResult.completed();
                }
                return runStorage(call, linear, inputs, output);
            }
            case BFLOAT16 -> {
                if (canUseDenseArrayHotPath(linear, inputs, output)) {
                    runDenseArrayHotPath(call, linear, output.dtype(), context);
                    return CpuKernelResult.completed();
                }
                return runStorage(call, linear, inputs, output);
            }
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static linear require(Operation op) {
        if (!(op instanceof linear linear)) {
            throw new IllegalArgumentException("CpuLinearKernel requires linear operation");
        }
        return linear;
    }

    private static List<CpuStorageView> requireInputs(CpuKernelCall call, linear linear) {
        List<CpuStorageView> inputs = call.inputs();
        int expected = linear.hasBias() ? 3 : 2;
        if (inputs.size() != expected) {
            throw new IllegalArgumentException("LINEAR expects " + expected + " input storage views.");
        }
        return inputs;
    }

    private static CpuStorageView requireOutput(CpuKernelCall call) {
        CpuStorageView output = call.output();
        if (output == null) {
            throw new IllegalArgumentException("LINEAR requires an output storage view.");
        }
        return output;
    }

    private static boolean canUseDenseArrayHotPath(linear linear, List<CpuStorageView> inputs, CpuStorageView output) {
        return isDenseZeroOffsetArray(inputs.get(0))
                && isDenseZeroOffsetArray(inputs.get(1))
                && isDenseZeroOffsetArray(output)
                && (!linear.hasBias() || isDenseZeroOffsetArray(inputs.get(2)));
    }

    private static boolean isDenseZeroOffsetArray(CpuStorageView view) {
        return view.isArray() && CpuMatMulStorageLoops.isDenseZeroOffset(view);
    }

    private static void runDenseArrayHotPath(
            CpuKernelCall call,
            linear linear,
            DataType dtype,
            CpuKernelContext context
    ) {
        List<Tensor> tensors = call.inputTensors();
        Tensor node = call.outputTensor();
        switch (dtype) {
            case FLOAT64 -> LinearExecutor.forwardDenseArrayF64(linear, tensors.get(0), tensors.get(1),
                    linear.hasBias() ? tensors.get(2) : null, node, context);
            case FLOAT32 -> LinearExecutor.forwardDenseArrayF32(linear, tensors.get(0), tensors.get(1),
                    linear.hasBias() ? tensors.get(2) : null, node, context);
            case BFLOAT16 -> LinearExecutor.forwardDenseArrayBF16(linear, tensors.get(0), tensors.get(1),
                    linear.hasBias() ? tensors.get(2) : null, node, context);
            case INT32, INT64, BOOL -> unsupported(dtype);
        }
    }

    private static CpuKernelResult runStorage(
            CpuKernelCall call,
            linear linear,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        if (shouldAllocateNativeOutput(call, inputs, output)) {
            NativeTensorStorage nativeOutput = allocateNativeOutput(call, output);
            CpuStorageView nativeOutputView = CpuStorageView.segment(
                    output.dtype(),
                    nativeOutput.segment(),
                    output.shape(),
                    output.strides(),
                    output.storageOffset(),
                    output.logicalSize()
            );
            LinearExecutor.forwardStorage(linear, inputs.get(0), inputs.get(1),
                    linear.hasBias() ? inputs.get(2) : null, nativeOutputView);
            nativeOutput.markModified();
            call.context().executionContext().attachNativeStorage(
                    call.context().nodeId(),
                    nativeOutput,
                    "linear storage-view Java path wrote " + output.dtype() + " native output"
            );
            return CpuKernelResult.route(CPU_NATIVE_ROUTE);
        }

        LinearExecutor.forwardStorage(linear, inputs.get(0), inputs.get(1),
                linear.hasBias() ? inputs.get(2) : null, output);
        markExistingNativeOutputModified(call, output);
        return output.isMemorySegment()
                ? CpuKernelResult.route(CPU_NATIVE_ROUTE)
                : CpuKernelResult.completed();
    }

    private static boolean shouldAllocateNativeOutput(
            CpuKernelCall call,
            List<CpuStorageView> inputs,
            CpuStorageView output
    ) {
        return nativeRequested(call.context())
                && !output.isMemorySegment()
                && inputs.stream().allMatch(CpuStorageView::isMemorySegment);
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
                "node-" + call.context().nodeId() + ":" + call.outputTensor().getLabel() + ":linear-storage-view"
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
                "linear storage-view Java path wrote existing native output"
        );
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuLinearKernel does not support " + dtype);
    }
}
