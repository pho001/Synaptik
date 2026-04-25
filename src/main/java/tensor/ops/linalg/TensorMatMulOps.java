package tensor.ops.linalg;

import backend.ComputeBackend;
import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

public final class TensorMatMulOps {
    private TensorMatMulOps() {
    }

    public static Tensor matmul(Tensor first, Tensor second) {
        MatMulSpec spec = MatMulSpec.resolve(first, second);
        Operation op = spec.operation();
        Tensor out = TensorPrimitiveBuilder.binary(first, second, spec.outShape(), op, "matmul", spec.outputType());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null) {
                return;
            }

            if (first.getRequiresGrad()) {
                Tensor gradRaw = outGrad.matmul(LinalgSupport.transposeLastTwoAxes(second));
                propagateAcceleratorBackend(out, gradRaw);
                Tensor gradForFirst = LinalgSupport.sumToShape(gradRaw, first.getShapeUnsafe());
                LinalgSupport.accumulateGradient(first, gradForFirst);
            }
            if (second.getRequiresGrad()) {
                Tensor gradRaw = LinalgSupport.transposeLastTwoAxes(first).matmul(outGrad);
                propagateAcceleratorBackend(out, gradRaw);
                Tensor gradForSecond = LinalgSupport.sumToShape(gradRaw, second.getShapeUnsafe());
                LinalgSupport.accumulateGradient(second, gradForSecond);
            }
        });
        return out;
    }

    private static void propagateAcceleratorBackend(Tensor source, Tensor target) {
        ComputeBackend backend = source == null ? null : source.resolveBackend();
        if (backend == ComputeBackend.GPU_METAL || backend == ComputeBackend.GPU_CUDA || backend == ComputeBackend.GPU_OPENCL) {
            TensorInternalAccess.setBackend(target, backend);
        }
    }
}
