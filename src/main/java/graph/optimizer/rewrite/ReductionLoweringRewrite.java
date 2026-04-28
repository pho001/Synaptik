package graph.optimizer.rewrite;

import graph.optimizer.intent.BackendIntentPropagator;
import operations.Operation;
import operations.reduction.logSoftmax;
import operations.reduction.logSoftmaxGrad;
import operations.reduction.reduceMaxGrad;
import operations.reduction.reduceMinGrad;
import operations.reduction.softmax;
import operations.reduction.softmaxGrad;
import operations.reduction.sum;
import tensor.Tensor;

import java.util.List;

/**
 * Lowers decomposed reduction backward patterns to specialized reduction-gradient operations.
 */
public final class ReductionLoweringRewrite extends AbstractRewriteRule {
    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        if (!tensor.isBackward()) {
            return tensor;
        }
        Operation op = tensor.getOperation();
        if (op == null) {
            return tensor;
        }
        return switch (op.opType()) {
            case MUL -> lowerSoftmaxGrad(tensor);
            case SUB -> lowerLogSoftmaxGrad(tensor);
            default -> tensor;
        };
    }

    private Tensor lowerSoftmaxGrad(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return tensor;
        }

        SoftmaxOutputMatch left = matchSoftmaxOutput(inputs.get(0));
        Tensor outGradLeft = matchSoftmaxGradSub(inputs.get(1), inputs.get(0), left == null ? -1 : left.dimension());
        if (left != null && outGradLeft != null) {
            return loweredSoftmaxGrad(tensor, inputs.get(0), outGradLeft, left.dimension());
        }

        SoftmaxOutputMatch right = matchSoftmaxOutput(inputs.get(1));
        Tensor outGradRight = matchSoftmaxGradSub(inputs.get(0), inputs.get(1), right == null ? -1 : right.dimension());
        if (right != null && outGradRight != null) {
            return loweredSoftmaxGrad(tensor, inputs.get(1), outGradRight, right.dimension());
        }
        return tensor;
    }

    private Tensor lowerLogSoftmaxGrad(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return tensor;
        }
        Tensor outGrad = inputs.get(0);
        LogSoftmaxGradMulMatch match = matchLogSoftmaxGradMul(inputs.get(1), outGrad);
        if (match == null) {
            return tensor;
        }
        return loweredLogSoftmaxGrad(tensor, match.logSoftmaxOut(), outGrad, match.dimension());
    }

    private Tensor loweredSoftmaxGrad(Tensor original, Tensor softmaxOut, Tensor outGrad, int dimension) {
        if (!sameShapeAndDType(softmaxOut, outGrad)) {
            return original;
        }
        Tensor lowered = new Tensor(
                original.getShape().clone(),
                List.of(softmaxOut, outGrad),
                new softmaxGrad(dimension),
                "softmaxGrad",
                original.getDataType()
        );
        lowered.setRequiresGrad(original.getRequiresGrad());
        BackendIntentPropagator.preserve(lowered, softmaxOut);
        return lowered;
    }

    private Tensor loweredLogSoftmaxGrad(Tensor original, Tensor logSoftmaxOut, Tensor outGrad, int dimension) {
        if (!sameShapeAndDType(logSoftmaxOut, outGrad)) {
            return original;
        }
        Tensor lowered = new Tensor(
                original.getShape().clone(),
                List.of(logSoftmaxOut, outGrad),
                new logSoftmaxGrad(dimension),
                "logSoftmaxGrad",
                original.getDataType()
        );
        lowered.setRequiresGrad(original.getRequiresGrad());
        BackendIntentPropagator.preserve(lowered, logSoftmaxOut);
        return lowered;
    }

    private SoftmaxOutputMatch matchSoftmaxOutput(Tensor tensor) {
        if (!(tensor != null && tensor.getOperation() instanceof softmax softmaxOp)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        return new SoftmaxOutputMatch(tensor, softmaxOp.getDimension());
    }

    private Tensor matchSoftmaxGradSub(Tensor tensor, Tensor softmaxOut, int dimension) {
        if (dimension < 0 || !isOp(tensor, Operation.OpType.SUB)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        Tensor outGrad = inputs.get(0);
        return matchSoftmaxGradDot(inputs.get(1), outGrad, softmaxOut, dimension) ? outGrad : null;
    }

    private boolean matchSoftmaxGradDot(Tensor tensor, Tensor outGrad, Tensor softmaxOut, int dimension) {
        if (!(tensor != null && tensor.getOperation() instanceof sum sumOp)) {
            return false;
        }
        if (sumOp.getDimension() != dimension || !sumOp.keepDims()) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return false;
        }
        return matchMulOf(inputs.get(0), outGrad, softmaxOut);
    }

    private LogSoftmaxGradMulMatch matchLogSoftmaxGradMul(Tensor tensor, Tensor outGrad) {
        if (!isOp(tensor, Operation.OpType.MUL)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }

        LogSoftmaxOutputMatch left = matchLogSoftmaxExp(inputs.get(0));
        if (left != null && matchLogSoftmaxGradSum(inputs.get(1), outGrad, left.dimension())) {
            return new LogSoftmaxGradMulMatch(left.logSoftmaxOut(), left.dimension());
        }

        LogSoftmaxOutputMatch right = matchLogSoftmaxExp(inputs.get(1));
        if (right != null && matchLogSoftmaxGradSum(inputs.get(0), outGrad, right.dimension())) {
            return new LogSoftmaxGradMulMatch(right.logSoftmaxOut(), right.dimension());
        }
        return null;
    }

    private LogSoftmaxOutputMatch matchLogSoftmaxExp(Tensor tensor) {
        if (!isOp(tensor, Operation.OpType.EXP)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        Tensor logSoftmaxOut = inputs.get(0);
        if (!(logSoftmaxOut.getOperation() instanceof logSoftmax logSoftmaxOp)) {
            return null;
        }
        return new LogSoftmaxOutputMatch(logSoftmaxOut, logSoftmaxOp.getDimension());
    }

    private boolean matchLogSoftmaxGradSum(Tensor tensor, Tensor outGrad, int dimension) {
        if (!(tensor != null && tensor.getOperation() instanceof sum sumOp)) {
            return false;
        }
        if (sumOp.getDimension() != dimension || !sumOp.keepDims()) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        return inputs != null && inputs.size() == 1 && inputs.get(0) == outGrad;
    }

    private boolean matchMulOf(Tensor tensor, Tensor leftTarget, Tensor rightTarget) {
        if (!isOp(tensor, Operation.OpType.MUL)) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return false;
        }
        return (inputs.get(0) == leftTarget && inputs.get(1) == rightTarget)
                || (inputs.get(0) == rightTarget && inputs.get(1) == leftTarget);
    }

    private boolean sameShapeAndDType(Tensor left, Tensor right) {
        return left != null
                && right != null
                && left.getDataType() == right.getDataType()
                && java.util.Arrays.equals(left.getShapeUnsafe(), right.getShapeUnsafe());
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    private record SoftmaxOutputMatch(Tensor softmaxOut, int dimension) {}

    private record LogSoftmaxOutputMatch(Tensor logSoftmaxOut, int dimension) {}

    private record LogSoftmaxGradMulMatch(Tensor logSoftmaxOut, int dimension) {}
}
