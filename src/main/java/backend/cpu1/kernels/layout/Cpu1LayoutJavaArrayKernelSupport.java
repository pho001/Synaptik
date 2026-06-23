package backend.cpu1.kernels.layout;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorRemap;

import java.util.ArrayList;
import java.util.List;

/**
 * Java array storage services for materializing cpu1 layout kernels.
 */
public final class Cpu1LayoutJavaArrayKernelSupport extends Cpu1LayoutKernelSupport {
    public Cpu1LayoutJavaArrayKernelSupport(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        super(unit, context);
    }

    @Override
    public LayoutCall bindMaterializingCall() {
        List<Cpu1TensorView> inputs = new ArrayList<>(unit().inputNodeIds().size());
        for (int inputNodeId : unit().inputNodeIds()) {
            Tensor inputTensor = context().runtimeTensorForNodeId(inputNodeId);
            context().requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
            inputs.add(Cpu1TensorView.fromTensor(inputTensor));
        }
        Tensor outputTensor = context().runtimeTensorForNodeId(unit().nodeId());
        return new LayoutCall(inputs, Cpu1TensorView.fromTensor(outputTensor), null);
    }

    @Override
    public void markOutputWritten(LayoutCall call) {
        call.output().markStorageModified();
        context().markCpuCurrent(unit().nodeId(), "cpu1 " + unit().opType() + " materialized CPU array");
    }

    @Override
    public void fillOutputScalar(Cpu1TensorView output, double value) {
        int[] shape = output.shape();
        int[] dense = denseStrides(shape);
        launchRange(output.elementCount(), (start, end) -> {
            for (int logical = start; logical < end; logical++) {
                int offset = output.storageOffset() + logicalOffset(logical, shape, output.strides(), dense);
                writeElement(output, offset, value);
            }
        });
    }

    @Override
    public void fillOutputVector(Cpu1TensorView output, double value) {
        if (output.contiguous() && output.storageOffset() == 0) {
            Cpu1LayoutVectorLoops.fillDenseArray(output, 0, output.elementCount(), value, unit().dataType());
            return;
        }
        fillOutputScalar(output, value);
    }

    @Override
    public double readElement(Cpu1TensorView view, int elementOffset) {
        return switch (unit().dataType()) {
            case FLOAT32 -> view.float32Array()[elementOffset];
            case FLOAT64 -> view.float64Array()[elementOffset];
            case BFLOAT16 -> tensor.dtype.TensorDTypeOps.fromBFloat16Bits(view.bfloat16Array()[elementOffset]);
            case BOOL -> view.boolArray()[elementOffset] == 0 ? 0.0d : 1.0d;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit().dataType());
        };
    }

    @Override
    public void writeElement(Cpu1TensorView view, int elementOffset, double value) {
        switch (unit().dataType()) {
            case FLOAT32 -> view.float32Array()[elementOffset] = (float) value;
            case FLOAT64 -> view.float64Array()[elementOffset] = value;
            case BFLOAT16 -> view.bfloat16Array()[elementOffset] =
                    tensor.dtype.TensorDTypeOps.toBFloat16Bits((float) value);
            case BOOL -> view.boolArray()[elementOffset] = value == 0.0d ? (byte) 0 : (byte) 1;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit().dataType());
        }
    }

    @Override
    public void copyDenseBlockScalar(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        switch (unit().dataType()) {
            case FLOAT32 -> System.arraycopy(input.float32Array(), inputOffset, output.float32Array(), outputOffset, elements);
            case FLOAT64 -> System.arraycopy(input.float64Array(), inputOffset, output.float64Array(), outputOffset, elements);
            case BFLOAT16 -> System.arraycopy(input.bfloat16Array(), inputOffset, output.bfloat16Array(), outputOffset, elements);
            case BOOL -> System.arraycopy(input.boolArray(), inputOffset, output.boolArray(), outputOffset, elements);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit().dataType());
        }
    }

    @Override
    public void copyDenseBlockVector(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        Cpu1LayoutVectorLoops.copyDenseArray(input, inputOffset, output, outputOffset, elements, unit().dataType());
    }

    @Override
    public void copyLinearizedScalar() {
        Tensor input = context().runtimeTensorForNodeId(unit().inputNodeId());
        Tensor output = context().runtimeTensorForNodeId(unit().nodeId());
        context().requireCpuReadable(unit().inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.copyLinearized(input, output);
        context().markCpuCurrent(unit().nodeId(), "cpu1 RESHAPE materialized CPU array");
    }

    @Override
    public void copyContiguousScalar() {
        Tensor input = context().runtimeTensorForNodeId(unit().inputNodeId());
        Tensor output = context().runtimeTensorForNodeId(unit().nodeId());
        context().requireCpuReadable(unit().inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.apply(input, output, unit().materializeThreshold());
        context().markCpuCurrent(unit().nodeId(), "cpu1 CONTIGUOUS materialized CPU array");
    }

    @Override
    public void copyContiguousVector() {
        Tensor input = context().runtimeTensorForNodeId(unit().inputNodeId());
        Tensor output = context().runtimeTensorForNodeId(unit().nodeId());
        if (tryVectorDenseArrayCopy(input, output)) {
            return;
        }
        context().requireCpuReadable(unit().inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.apply(input, output, unit().materializeThreshold());
        context().markCpuCurrent(unit().nodeId(), "cpu1 CONTIGUOUS materialized CPU array");
    }

    private boolean tryVectorDenseArrayCopy(Tensor input, Tensor output) {
        if (!denseContiguousWithoutOffset(input)
                || !denseContiguousWithoutOffset(output)
                || input.getFlatDataSize() != output.getFlatDataSize()) {
            return false;
        }
        context().requireCpuReadable(unit().inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        int elements = output.getFlatDataSize();
        Cpu1LayoutVectorLoops.copyDenseArray(
                Cpu1TensorView.fromTensor(input),
                0,
                Cpu1TensorView.fromTensor(output),
                0,
                elements,
                unit().dataType()
        );
        TensorInternalAccess.markStorageModified(output);
        context().markCpuCurrent(unit().nodeId(), "cpu1 CONTIGUOUS vector bulk-copied CPU array");
        return true;
    }
}
