package backend.cpu1.kernels.layout;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import runtime.contract.CpuMaterializationReason;
import runtime.execution.ExecutionContext;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Native memory segment storage services for materializing cpu1 layout kernels.
 */
public final class Cpu1LayoutMemorySegmentKernelSupport extends Cpu1LayoutKernelSupport {
    public Cpu1LayoutMemorySegmentKernelSupport(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        super(unit, context);
    }

    @Override
    public LayoutCall bindMaterializingCall() {
        List<Cpu1TensorView> inputs = new ArrayList<>(unit().inputNodeIds().size());
        for (int inputNodeId : unit().inputNodeIds()) {
            Tensor inputTensor = context().runtimeTensorForNodeId(inputNodeId);
            NativeTensorStorage nativeInput = context().requireNativeReadable(
                    inputNodeId,
                    CpuMaterializationReason.CPU_CONSUMER
            );
            inputs.add(Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput));
        }
        Tensor outputTensor = context().runtimeTensorForNodeId(unit().nodeId());
        NativeTensorStorage nativeOutput = context().requireNativeOutputStorage(
                unit().nodeId(),
                unit().dataType(),
                outputTensor.getFlatDataSize(),
                "cpu1-layout-node-" + unit().nodeId()
        );
        return new LayoutCall(inputs, Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput), nativeOutput);
    }

    @Override
    public void markOutputWritten(LayoutCall call) {
        call.nativeOutput().markModified();
        context().attachNativeStorage(
                unit().nodeId(),
                call.nativeOutput(),
                "cpu1 " + unit().opType() + " materialized native segment"
        );
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
            Cpu1LayoutVectorLoops.fillDenseSegment(output, 0, output.elementCount(), value, unit().dataType());
            return;
        }
        fillOutputScalar(output, value);
    }

    @Override
    public double readElement(Cpu1TensorView view, int elementOffset) {
        return switch (unit().dataType()) {
            case FLOAT32 -> view.segment().get(JAVA_FLOAT, (long) elementOffset * Float.BYTES);
            case FLOAT64 -> view.segment().get(JAVA_DOUBLE, (long) elementOffset * Double.BYTES);
            case BFLOAT16 -> tensor.dtype.TensorDTypeOps.fromBFloat16Bits(
                    view.segment().get(JAVA_SHORT, (long) elementOffset * Short.BYTES)
            );
            case BOOL -> view.segment().get(JAVA_BYTE, elementOffset) == 0 ? 0.0d : 1.0d;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit().dataType());
        };
    }

    @Override
    public void writeElement(Cpu1TensorView view, int elementOffset, double value) {
        switch (unit().dataType()) {
            case FLOAT32 -> view.segment().set(JAVA_FLOAT, (long) elementOffset * Float.BYTES, (float) value);
            case FLOAT64 -> view.segment().set(JAVA_DOUBLE, (long) elementOffset * Double.BYTES, value);
            case BFLOAT16 -> view.segment().set(
                    JAVA_SHORT,
                    (long) elementOffset * Short.BYTES,
                    tensor.dtype.TensorDTypeOps.toBFloat16Bits((float) value)
            );
            case BOOL -> view.segment().set(JAVA_BYTE, elementOffset, value == 0.0d ? (byte) 0 : (byte) 1);
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
        int bytesPerElement = bytesPerElement();
        MemorySegment.copy(
                input.segment(),
                (long) inputOffset * bytesPerElement,
                output.segment(),
                (long) outputOffset * bytesPerElement,
                (long) elements * bytesPerElement
        );
    }

    @Override
    public void copyDenseBlockVector(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        Cpu1LayoutVectorLoops.copyDenseSegment(input, inputOffset, output, outputOffset, elements, unit().dataType());
    }

    @Override
    public void copyLinearizedScalar() {
        copyNative(true, false);
    }

    @Override
    public void copyContiguousScalar() {
        copyNative(false, false);
    }

    @Override
    public void copyContiguousVector() {
        copyNative(false, true);
    }

    private void copyNative(boolean linearized, boolean vectorized) {
        Tensor input = context().runtimeTensorForNodeId(unit().inputNodeId());
        Tensor outputTensor = context().runtimeTensorForNodeId(unit().nodeId());
        NativeTensorStorage inputStorage = context().requireNativeReadable(
                unit().inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage outputStorage = context().requireNativeOutputStorage(
                unit().nodeId(),
                unit().dataType(),
                outputTensor.getFlatDataSize(),
                "cpu1-layout-node-" + unit().nodeId()
        );
        copyNative(input, outputTensor, inputStorage, outputStorage, linearized, vectorized);
        outputStorage.markModified();
        context().attachNativeStorage(
                unit().nodeId(),
                outputStorage,
                "cpu1 " + unit().opType() + " materialized native segment"
        );
    }

    private void copyNative(
            Tensor input,
            Tensor output,
            NativeTensorStorage inputStorage,
            NativeTensorStorage outputStorage,
            boolean linearized,
            boolean vectorized
    ) {
        if (input.getDataType() != unit().dataType() || inputStorage.getType() != unit().dataType()
                || outputStorage.getType() != unit().dataType()) {
            throw new IllegalStateException("cpu1 layout native copy dtype mismatch for nodeId="
                    + unit().nodeId());
        }
        int size = input.getFlatDataSize();
        if (linearized || !denseContiguousWithoutOffset(input)) {
            copyNativeLogical(input, inputStorage.segment(), outputStorage.segment(), size);
            return;
        }
        if (vectorized) {
            Cpu1LayoutVectorLoops.copyDenseSegment(
                    Cpu1TensorView.fromNativeStorage(input, inputStorage),
                    0,
                    Cpu1TensorView.fromNativeStorage(output, outputStorage),
                    0,
                    size,
                    unit().dataType()
            );
            return;
        }
        switch (unit().dataType()) {
            case FLOAT32 -> MemorySegment.copy(inputStorage.segment(), JAVA_FLOAT, 0L,
                    outputStorage.segment(), JAVA_FLOAT, 0L, size);
            case FLOAT64 -> MemorySegment.copy(inputStorage.segment(), JAVA_DOUBLE, 0L,
                    outputStorage.segment(), JAVA_DOUBLE, 0L, size);
            case BFLOAT16 -> MemorySegment.copy(inputStorage.segment(), JAVA_SHORT, 0L,
                    outputStorage.segment(), JAVA_SHORT, 0L, size);
            case BOOL -> MemorySegment.copy(inputStorage.segment(), 0L,
                    outputStorage.segment(), 0L, size);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout native copy dtype="
                    + unit().dataType());
        }
    }

    private void copyNativeLogical(Tensor input, MemorySegment source, MemorySegment target, int size) {
        int[] shape = input.getShapeUnsafe();
        int[] sourceStrides = input.getStridesUnsafe();
        int[] denseStrides = denseStrides(shape);
        int sourceBaseOffset = input.getStorageOffsetUnsafe();
        for (int linear = 0; linear < size; linear++) {
            int sourceOffset = sourceBaseOffset + logicalOffset(linear, shape, sourceStrides, denseStrides);
            copyNativeElement(source, sourceOffset, target, linear);
        }
    }

    private void copyNativeElement(
            MemorySegment source,
            int sourceElementOffset,
            MemorySegment target,
            int targetElementOffset
    ) {
        switch (unit().dataType()) {
            case FLOAT32 -> target.set(
                    JAVA_FLOAT,
                    (long) targetElementOffset * Float.BYTES,
                    source.get(JAVA_FLOAT, (long) sourceElementOffset * Float.BYTES)
            );
            case FLOAT64 -> target.set(
                    JAVA_DOUBLE,
                    (long) targetElementOffset * Double.BYTES,
                    source.get(JAVA_DOUBLE, (long) sourceElementOffset * Double.BYTES)
            );
            case BFLOAT16 -> target.set(
                    JAVA_SHORT,
                    (long) targetElementOffset * Short.BYTES,
                    source.get(JAVA_SHORT, (long) sourceElementOffset * Short.BYTES)
            );
            case BOOL -> target.set(
                    JAVA_BYTE,
                    targetElementOffset,
                    source.get(JAVA_BYTE, sourceElementOffset)
            );
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout native copy dtype="
                    + unit().dataType());
        }
    }

    private int bytesPerElement() {
        return switch (unit().dataType()) {
            case FLOAT32 -> Float.BYTES;
            case FLOAT64 -> Double.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit().dataType());
        };
    }
}
