package backend.cpu1.kernels.layout;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.memory.TensorResidencyState;
import backend.runtime.ExecutionContext;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.layout.TensorRemap;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Shared runtime services used by concrete cpu1 layout kernels.
 */
public final class Cpu1LayoutKernelSupport {
    private final Cpu1PreparedLayoutUnit unit;
    private final ExecutionContext context;

    public Cpu1LayoutKernelSupport(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        this.unit = Objects.requireNonNull(unit, "unit cannot be null");
        this.context = Objects.requireNonNull(context, "context cannot be null");
    }

    public Cpu1PreparedLayoutUnit unit() {
        return unit;
    }

    public ExecutionContext context() {
        return context;
    }

    public void aliasView() {
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        TensorInternalAccess.aliasRuntimeFrom(output, input);

        TensorResidencyState inputResidency = context.residencyForNodeId(unit.inputNodeId());
        NativeTensorStorage nativeInput = context.nativeStorageForNodeId(unit.inputNodeId());
        if (inputResidency != null && inputResidency.nativeCurrent() && nativeInput != null) {
            context.aliasNativeStorage(
                    unit.nodeId(),
                    unit.inputNodeId(),
                    "cpu1 " + unit.opType() + " view aliases native storage"
            );
            return;
        }
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " view aliases CPU array");
    }

    public void copyLinearizedScalar() {
        if (canReadNative()) {
            copyNative(true, false);
            return;
        }
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.copyLinearized(input, output);
        context.markCpuCurrent(unit.nodeId(), "cpu1 RESHAPE materialized CPU array");
    }

    public void copyContiguousScalar() {
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        if (canReadNative()) {
            copyNative(false, false);
            return;
        }
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.apply(input, output, unit.materializeThreshold());
        context.markCpuCurrent(unit.nodeId(), "cpu1 CONTIGUOUS materialized CPU array");
    }

    public void copyContiguousVector() {
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        Tensor output = context.runtimeTensorForNodeId(unit.nodeId());
        if (tryVectorDenseArrayCopy(input, output)) {
            return;
        }
        if (canReadNative()) {
            copyNative(false, true);
            return;
        }
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        TensorRemap.apply(input, output, unit.materializeThreshold());
        context.markCpuCurrent(unit.nodeId(), "cpu1 CONTIGUOUS materialized CPU array");
    }

    public LayoutCall bindMaterializingCall() {
        boolean nativeInputs = canReadAllNative();
        List<Cpu1TensorView> inputs = new ArrayList<>(unit.inputNodeIds().size());
        for (int inputNodeId : unit.inputNodeIds()) {
            Tensor inputTensor = context.runtimeTensorForNodeId(inputNodeId);
            if (nativeInputs) {
                NativeTensorStorage nativeInput = context.requireNativeReadable(
                        inputNodeId,
                        CpuMaterializationReason.CPU_CONSUMER
                );
                inputs.add(Cpu1TensorView.fromNativeStorage(inputTensor, nativeInput));
            } else {
                context.requireCpuReadable(inputNodeId, CpuMaterializationReason.CPU_CONSUMER);
                inputs.add(Cpu1TensorView.fromTensor(inputTensor));
            }
        }
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        NativeTensorStorage nativeOutput = null;
        Cpu1TensorView output;
        if (nativeInputs) {
            nativeOutput = context.requireNativeOutputStorage(
                    unit.nodeId(),
                    unit.dataType(),
                    outputTensor.getFlatDataSize(),
                    "cpu1-layout-node-" + unit.nodeId()
            );
            output = Cpu1TensorView.fromNativeStorage(outputTensor, nativeOutput);
        } else {
            output = Cpu1TensorView.fromTensor(outputTensor);
        }
        return new LayoutCall(inputs, output, nativeOutput);
    }

    public void markOutputWritten(LayoutCall call) {
        if (call.nativeOutput() == null) {
            call.output().markStorageModified();
            context.markCpuCurrent(unit.nodeId(), "cpu1 " + unit.opType() + " materialized CPU array");
            return;
        }
        call.nativeOutput().markModified();
        context.attachNativeStorage(
                unit.nodeId(),
                call.nativeOutput(),
                "cpu1 " + unit.opType() + " materialized native segment"
        );
    }

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

    public void fillOutputVector(Cpu1TensorView output, double value) {
        if (tryVectorFillOutput(output, value)) {
            return;
        }
        fillOutputScalar(output, value);
    }

    public double readElement(Cpu1TensorView view, int elementOffset) {
        return switch (unit.dataType()) {
            case FLOAT32 -> view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                    ? view.float32Array()[elementOffset]
                    : view.segment().get(JAVA_FLOAT, (long) elementOffset * Float.BYTES);
            case FLOAT64 -> view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                    ? view.float64Array()[elementOffset]
                    : view.segment().get(JAVA_DOUBLE, (long) elementOffset * Double.BYTES);
            case BFLOAT16 -> tensor.dtype.TensorDTypeOps.fromBFloat16Bits(
                    view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                            ? view.bfloat16Array()[elementOffset]
                            : view.segment().get(JAVA_SHORT, (long) elementOffset * Short.BYTES)
            );
            case BOOL -> {
                byte value = view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                        ? view.boolArray()[elementOffset]
                        : view.segment().get(JAVA_BYTE, elementOffset);
                yield value == 0 ? 0.0d : 1.0d;
            }
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit.dataType());
        };
    }

    public void writeElement(Cpu1TensorView view, int elementOffset, double value) {
        switch (unit.dataType()) {
            case FLOAT32 -> {
                if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
                    view.float32Array()[elementOffset] = (float) value;
                } else {
                    view.segment().set(JAVA_FLOAT, (long) elementOffset * Float.BYTES, (float) value);
                }
            }
            case FLOAT64 -> {
                if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
                    view.float64Array()[elementOffset] = value;
                } else {
                    view.segment().set(JAVA_DOUBLE, (long) elementOffset * Double.BYTES, value);
                }
            }
            case BFLOAT16 -> {
                short bits = tensor.dtype.TensorDTypeOps.toBFloat16Bits((float) value);
                if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
                    view.bfloat16Array()[elementOffset] = bits;
                } else {
                    view.segment().set(JAVA_SHORT, (long) elementOffset * Short.BYTES, bits);
                }
            }
            case BOOL -> {
                byte normalized = value == 0.0d ? (byte) 0 : (byte) 1;
                if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
                    view.boolArray()[elementOffset] = normalized;
                } else {
                    view.segment().set(JAVA_BYTE, elementOffset, normalized);
                }
            }
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit.dataType());
        }
    }

    public void copyDenseBlockVector(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        if (elements <= 0) {
            return;
        }
        Cpu1LayoutVectorLoops.copyDense(input, inputOffset, output, outputOffset, elements, unit.dataType());
    }

    public double[] foldAccumulator(int elements, int slotCount) {
        Cpu1ScratchBuffer scratchBuffer = context.cpu1ScratchBufferForNodeId(unit.nodeId());
        if (scratchBuffer == null) {
            throw new IllegalStateException("cpu1 FOLD2D requires prepared F64 scratch buffer for nodeId="
                    + unit.nodeId());
        }
        return scratchBuffer.requireF64Array(Math.multiplyExact(elements, slotCount));
    }

    public void launchRange(int elementCount, Cpu1RangeLauncher.RangeBody body) {
        Cpu1RangeLauncher.launch(elementCount, unit.launchConfig(), body);
    }

    public void launchRangeWithSlot(int elementCount, Cpu1RangeLauncher.IndexedRangeBody body) {
        Cpu1RangeLauncher.launchIndexed(elementCount, unit.launchConfig(), body);
    }

    public int rangeSlotCount(int elementCount) {
        return Cpu1RangeLauncher.slotCount(elementCount, unit.launchConfig());
    }

    private boolean tryVectorDenseArrayCopy(Tensor input, Tensor output) {
        if (unit.storageKind() != Cpu1StorageKind.JAVA_ARRAY
                || !denseContiguousWithoutOffset(input)
                || !denseContiguousWithoutOffset(output)
                || input.getFlatDataSize() != output.getFlatDataSize()) {
            return false;
        }
        context.requireCpuReadable(unit.inputNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        int elements = output.getFlatDataSize();
        Cpu1LayoutVectorLoops.copyDense(
                Cpu1TensorView.fromTensor(input),
                0,
                Cpu1TensorView.fromTensor(output),
                0,
                elements,
                unit.dataType()
        );
        TensorInternalAccess.markStorageModified(output);
        context.markCpuCurrent(unit.nodeId(), "cpu1 CONTIGUOUS vector bulk-copied CPU array");
        return true;
    }

    private boolean canReadNative() {
        if (unit.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            return false;
        }
        TensorResidencyState inputResidency = context.residencyForNodeId(unit.inputNodeId());
        return inputResidency != null
                && inputResidency.nativeCurrent()
                && context.nativeStorageForNodeId(unit.inputNodeId()) != null;
    }

    private boolean canReadAllNative() {
        if (unit.storageKind() != Cpu1StorageKind.MEMORY_SEGMENT) {
            return false;
        }
        for (int inputNodeId : unit.inputNodeIds()) {
            TensorResidencyState inputResidency = context.residencyForNodeId(inputNodeId);
            if (inputResidency == null
                    || !inputResidency.nativeCurrent()
                    || context.nativeStorageForNodeId(inputNodeId) == null) {
                return false;
            }
        }
        return true;
    }

    private void copyNative(boolean linearized, boolean vectorized) {
        Tensor input = context.runtimeTensorForNodeId(unit.inputNodeId());
        Tensor outputTensor = context.runtimeTensorForNodeId(unit.nodeId());
        NativeTensorStorage inputStorage = context.requireNativeReadable(
                unit.inputNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        NativeTensorStorage outputStorage = context.requireNativeOutputStorage(
                unit.nodeId(),
                unit.dataType(),
                outputTensor.getFlatDataSize(),
                "cpu1-layout-node-" + unit.nodeId()
        );
        copyNative(input, outputTensor, inputStorage, outputStorage, linearized, vectorized);
        outputStorage.markModified();
        context.attachNativeStorage(
                unit.nodeId(),
                outputStorage,
                "cpu1 " + unit.opType() + " materialized native segment"
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
        if (input.getDataType() != unit.dataType() || inputStorage.getType() != unit.dataType()
                || outputStorage.getType() != unit.dataType()) {
            throw new IllegalStateException("cpu1 layout native copy dtype mismatch for nodeId="
                    + unit.nodeId());
        }
        int size = input.getFlatDataSize();
        if (linearized || !denseContiguousWithoutOffset(input)) {
            copyNativeLogical(input, inputStorage.segment(), outputStorage.segment(), size);
            return;
        }
        if (vectorized) {
            Cpu1LayoutVectorLoops.copyDense(
                    Cpu1TensorView.fromNativeStorage(input, inputStorage),
                    0,
                    Cpu1TensorView.fromNativeStorage(output, outputStorage),
                    0,
                    size,
                    unit.dataType()
            );
            return;
        }
        switch (unit.dataType()) {
            case FLOAT32 -> MemorySegment.copy(inputStorage.segment(), JAVA_FLOAT, 0L,
                    outputStorage.segment(), JAVA_FLOAT, 0L, size);
            case FLOAT64 -> MemorySegment.copy(inputStorage.segment(), JAVA_DOUBLE, 0L,
                    outputStorage.segment(), JAVA_DOUBLE, 0L, size);
            case BFLOAT16 -> MemorySegment.copy(inputStorage.segment(), JAVA_SHORT, 0L,
                    outputStorage.segment(), JAVA_SHORT, 0L, size);
            case BOOL -> MemorySegment.copy(inputStorage.segment(), 0L,
                    outputStorage.segment(), 0L, size);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout native copy dtype="
                    + unit.dataType());
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

    private void copyNativeElement(MemorySegment source, int sourceElementOffset, MemorySegment target, int targetElementOffset) {
        switch (unit.dataType()) {
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
                    + unit.dataType());
        }
    }

    private boolean tryVectorFillOutput(Cpu1TensorView output, double value) {
        if (!output.contiguous()
                || output.storageOffset() != 0) {
            return false;
        }
        Cpu1LayoutVectorLoops.fillDense(output, 0, output.elementCount(), value, unit.dataType());
        return true;
    }

    public void copyDenseBlockScalar(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        if (input.storageKind() != output.storageKind()) {
            for (int i = 0; i < elements; i++) {
                writeElement(output, outputOffset + i, readElement(input, inputOffset + i));
            }
            return;
        }
        if (input.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            copyDenseArrayBlock(input, inputOffset, output, outputOffset, elements);
            return;
        }
        copyDenseSegmentBlock(input, inputOffset, output, outputOffset, elements);
    }

    private void copyDenseArrayBlock(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        switch (unit.dataType()) {
            case FLOAT32 -> System.arraycopy(input.float32Array(), inputOffset, output.float32Array(), outputOffset, elements);
            case FLOAT64 -> System.arraycopy(input.float64Array(), inputOffset, output.float64Array(), outputOffset, elements);
            case BFLOAT16 -> System.arraycopy(input.bfloat16Array(), inputOffset, output.bfloat16Array(), outputOffset, elements);
            case BOOL -> System.arraycopy(input.boolArray(), inputOffset, output.boolArray(), outputOffset, elements);
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit.dataType());
        }
    }

    private void copyDenseSegmentBlock(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    ) {
        int bytesPerElement = switch (unit.dataType()) {
            case FLOAT32 -> Float.BYTES;
            case FLOAT64 -> Double.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32, INT64 -> throw new UnsupportedOperationException("cpu1 layout dtype="
                    + unit.dataType());
        };
        MemorySegment.copy(
                input.segment(),
                (long) inputOffset * bytesPerElement,
                output.segment(),
                (long) outputOffset * bytesPerElement,
                (long) elements * bytesPerElement
        );
    }

    static boolean denseContiguousWithoutOffset(Tensor tensor) {
        return tensor.isContiguous() && !tensor.hasStorageOffset();
    }

    public static int logicalOffset(int linear, int[] shape, int[] strides, int[] denseStrides) {
        int remaining = linear;
        int offset = 0;
        for (int dim = 0; dim < shape.length; dim++) {
            int coordinate = denseStrides[dim] == 0 ? 0 : remaining / denseStrides[dim];
            remaining = denseStrides[dim] == 0 ? remaining : remaining % denseStrides[dim];
            offset += coordinate * strides[dim];
        }
        return offset;
    }

    public static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = Math.multiplyExact(stride, shape[i]);
        }
        return strides;
    }

    public record LayoutCall(
            List<Cpu1TensorView> inputs,
            Cpu1TensorView output,
            NativeTensorStorage nativeOutput
    ) {
        public LayoutCall {
            inputs = List.copyOf(inputs);
        }
    }
}
