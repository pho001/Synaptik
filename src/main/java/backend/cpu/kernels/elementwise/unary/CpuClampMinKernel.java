package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import tensor.dtype.TensorDTypeOps;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import operations.elementwise.unary.clampMin;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuClampMinKernel extends TypedCpuKernel implements ScalarUnaryElementwiseKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    public double applyF64(double value, double parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public float applyF32(float value, float parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public float applyBF16(float value, float parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value, DoubleVector parameter) {
        return value.max(parameter);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value, FloatVector parameter) {
        return value.max(parameter);
    }

    @Override
    public void runSegmentF64(MemorySegment in, double parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, Math.max(parameter, in.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    public void runSegmentF32(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, Math.max(parameter, in.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    public void runSegmentBF16(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(Math.max(parameter, value)));
        }
    }
}
