package tensor.ops.pool;

import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;
import tensor.options.Pool2dOptions;

public final class TensorPoolOps {
    private TensorPoolOps() {
    }

    public static Tensor maxPool2d(Tensor input, Pool2dOptions options) {
        PoolSupport.validateInput(input, options, "maxPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = PoolSupport.inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = PoolSupport.inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        PoolSupport.validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        PoolSupport.validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{inputShape[0], inputShape[1], outH, outW},
                new maxPool2d(options),
                "maxPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = TensorPrimitiveBuilder.binaryNoGrad(
                    outGrad,
                    out,
                    inputShape.clone(),
                    new operations.nn.pool.maxPool2dBackwardInput(options, inputShape),
                    "maxPool2dBackwardInput",
                    input.getDataType()
            );
            PoolSupport.accumulateGradient(input, grad);
        });
        return out;
    }

    public static Tensor avgPool2d(Tensor input, Pool2dOptions options) {
        PoolSupport.validateInput(input, options, "avgPool2d");
        int[] inputShape = input.getShapeUnsafe();
        int outH = PoolSupport.inferOutputSize(inputShape[2], options.kernelH(), options.padH(), options.strideH(), "height");
        int outW = PoolSupport.inferOutputSize(inputShape[3], options.kernelW(), options.padW(), options.strideW(), "width");
        PoolSupport.validateWindowCoverage(inputShape[2], options.kernelH(), options.padH(), options.strideH(), outH, "height");
        PoolSupport.validateWindowCoverage(inputShape[3], options.kernelW(), options.padW(), options.strideW(), outW, "width");

        Tensor out = TensorPrimitiveBuilder.unary(
                input,
                new int[]{inputShape[0], inputShape[1], outH, outW},
                new avgPool2d(options),
                "avgPool2d",
                input.getDataType()
        );
        out.setRequiresGrad(input.getRequiresGrad());
        TensorInternalAccess.setBackwardFunction(out, () -> {
            Tensor outGrad = out.getGradient();
            if (outGrad == null || !input.getRequiresGrad()) {
                return;
            }
            Tensor grad = TensorPrimitiveBuilder.unaryNoGrad(
                    outGrad,
                    inputShape.clone(),
                    new operations.nn.pool.avgPool2dBackwardInput(options, inputShape),
                    "avgPool2dBackwardInput",
                    input.getDataType()
            );
            PoolSupport.accumulateGradient(input, grad);
        });
        return out;
    }
}
