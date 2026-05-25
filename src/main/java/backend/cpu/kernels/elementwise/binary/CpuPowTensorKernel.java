package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import operations.Operation;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public final class CpuPowTensorKernel extends TypedCpuKernel implements BinaryElementwiseKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double left, double right) {
        return CpuPowSupport.applyF64(left, right);
    }

    @Override
    public float applyF32(float left, float right) {
        return CpuPowSupport.applyF32(left, right);
    }

    @Override
    public float applyBF16(float left, float right) {
        return CpuPowSupport.applyF32(left, right);
    }

    @Override
    public void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, CpuPowSupport.applyF64(left.get(JAVA_DOUBLE, offset), right.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    public void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, CpuPowSupport.applyF32(left.get(JAVA_FLOAT, offset), right.get(JAVA_FLOAT, offset)));
        }
    }
}
