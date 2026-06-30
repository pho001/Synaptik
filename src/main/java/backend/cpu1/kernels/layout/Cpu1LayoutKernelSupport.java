package backend.cpu1.kernels.layout;

import backend.cpu1.exec.Cpu1ScratchBuffer;
import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.launch.Cpu1RangeLauncher;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import runtime.execution.ExecutionContext;
import tensor.storage.NativeTensorStorage;

import java.util.List;

/**
 * Shared runtime services used by concrete cpu1 layout kernels.
 */
public abstract class Cpu1LayoutKernelSupport {
    private final Cpu1PreparedLayoutUnit unit;
    private final ExecutionContext context;

    protected Cpu1LayoutKernelSupport(Cpu1PreparedLayoutUnit unit, ExecutionContext context) {
        if (unit == null) {
            throw new IllegalArgumentException("unit cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        this.unit = unit;
        this.context = context;
    }

    public Cpu1PreparedLayoutUnit unit() {
        return unit;
    }

    public ExecutionContext context() {
        return context;
    }

    public abstract LayoutCall bindMaterializingCall();

    public abstract void markOutputWritten(LayoutCall call);

    public abstract void fillOutputScalar(Cpu1TensorView output, double value);

    public abstract void fillOutputVector(Cpu1TensorView output, double value);

    public abstract double readElement(Cpu1TensorView view, int elementOffset);

    public abstract void writeElement(Cpu1TensorView view, int elementOffset, double value);

    public abstract void copyDenseBlockScalar(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    );

    public abstract void copyDenseBlockVector(
            Cpu1TensorView input,
            int inputOffset,
            Cpu1TensorView output,
            int outputOffset,
            int elements
    );

    public abstract void copyLinearizedScalar();

    public abstract void copyContiguousScalar();

    public abstract void copyContiguousVector();

    public double[] foldAccumulator(int elements, int slotCount) {
        Cpu1ScratchBuffer scratchBuffer = context.requireWorkspace(unit.nodeId(), Cpu1ScratchBuffer.class);
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

    static boolean denseContiguousWithoutOffset(tensor.Tensor tensor) {
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
