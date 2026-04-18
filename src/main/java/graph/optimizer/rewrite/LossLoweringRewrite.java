package graph.optimizer.rewrite;

import operations.Operation;
import operations.loss.crossEntropyLossIndicesGrad;
import operations.layout.expand;
import operations.layout.expandDims;
import operations.loss.crossEntropyLossIndices;
import operations.index.gather;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.index.scatterAdd;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.Tensor;
import tensor.loss.LossReduction;

import java.util.List;

public final class LossLoweringRewrite extends AbstractRewriteRule {
    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (op == null) {
            return tensor;
        }
        return switch (op.opType()) {
            case NEG -> lowerCrossEntropyFromIndices(tensor, LossReduction.NONE);
            case SUM -> lowerReducedCrossEntropyFromIndices(tensor, LossReduction.SUM);
            case MEAN -> lowerReducedCrossEntropyFromIndices(tensor, LossReduction.MEAN);
            case SUB -> lowerCrossEntropyFromIndicesGrad(tensor);
            default -> tensor;
        };
    }

    private Tensor lowerReducedCrossEntropyFromIndices(Tensor tensor, LossReduction reduction) {
        if (tensor.getPrevTensors() == null || tensor.getPrevTensors().size() != 1) {
            return tensor;
        }
        if (reduction == LossReduction.SUM && (!(tensor.getOperation() instanceof sum sumOp) || sumOp.getDimension() != -1)) {
            return tensor;
        }
        if (reduction == LossReduction.MEAN && (!(tensor.getOperation() instanceof mean meanOp) || meanOp.getDimension() != -1)) {
            return tensor;
        }
        Match match = matchPerSampleCrossEntropy(tensor.getPrevTensors().get(0));
        if (match == null) {
            return tensor;
        }
        return loweredTensor(tensor, match, reduction);
    }

    private Tensor lowerCrossEntropyFromIndices(Tensor tensor, LossReduction reduction) {
        Match match = matchPerSampleCrossEntropy(tensor);
        if (match == null) {
            return tensor;
        }
        return loweredTensor(tensor, match, reduction);
    }

    private Tensor loweredTensor(Tensor original, Match match, LossReduction reduction) {
        Tensor lowered = new Tensor(
                original.getShape().clone(),
                List.of(match.logits(), match.targetIndices()),
                new crossEntropyLossIndices(match.classDimension(), reduction, null),
                "crossEntropyLossFromIndices",
                original.getDataType()
        );
        lowered.setRequiresGrad(original.getRequiresGrad());
        return lowered;
    }

    private Match matchPerSampleCrossEntropy(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null || tensor.getOperation().opType() != Operation.OpType.NEG) {
            return null;
        }
        List<Tensor> negInputs = tensor.getPrevTensors();
        if (negInputs == null || negInputs.size() != 1) {
            return null;
        }
        Tensor gathered = negInputs.get(0);
        if (!(gathered.getOperation() instanceof gather gatherOp)) {
            return null;
        }
        List<Tensor> gatherInputs = gathered.getPrevTensors();
        if (gatherInputs == null || gatherInputs.size() != 2) {
            return null;
        }
        Tensor logSoftmaxTensor = gatherInputs.get(0);
        Tensor targetIndices = gatherInputs.get(1);
        if (!(logSoftmaxTensor.getOperation() instanceof logSoftmax logSoftmaxOp)) {
            return null;
        }
        List<Tensor> softmaxInputs = logSoftmaxTensor.getPrevTensors();
        if (softmaxInputs == null || softmaxInputs.size() != 1) {
            return null;
        }
        if (gatherOp.getDimension() != logSoftmaxOp.getDimension()) {
            return null;
        }
        return new Match(softmaxInputs.get(0), targetIndices, logSoftmaxOp.getDimension());
    }

    private Tensor lowerCrossEntropyFromIndicesGrad(Tensor tensor) {
        if (!tensor.isBackward()) {
            return tensor;
        }
        GradMatch match = matchCrossEntropyFromIndicesGrad(tensor);
        if (match == null) {
            return tensor;
        }
        Tensor lowered = new Tensor(
                tensor.getShape().clone(),
                List.of(match.logits(), match.targetIndices(), match.sampleScale()),
                new crossEntropyLossIndicesGrad(match.classDimension()),
                "crossEntropyLossFromIndicesGrad",
                tensor.getDataType()
        );
        lowered.setRequiresGrad(tensor.getRequiresGrad());
        return lowered;
    }

    private GradMatch matchCrossEntropyFromIndicesGrad(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null || tensor.getOperation().opType() != Operation.OpType.SUB) {
            return null;
        }
        List<Tensor> subInputs = tensor.getPrevTensors();
        if (subInputs == null || subInputs.size() != 2) {
            return null;
        }

        GradMulMatch scaledSoftmax = matchScaledSoftmax(subInputs.get(0));
        ScatterMatch scatter = matchScatter(subInputs.get(1));
        if (scaledSoftmax == null || scatter == null) {
            scaledSoftmax = matchScaledSoftmax(subInputs.get(1));
            scatter = matchScatter(subInputs.get(0));
        }
        if (scaledSoftmax == null || scatter == null) {
            return null;
        }
        if (scaledSoftmax.classDimension() != scatter.classDimension()) {
            return null;
        }
        if (!sameShapeAndDType(scatter.zeroBase(), scaledSoftmax.logits())) {
            return null;
        }
        if (scaledSoftmax.sampleScale() != scatter.sampleScale()) {
            return null;
        }
        return new GradMatch(scaledSoftmax.logits(), scatter.targetIndices(), scaledSoftmax.sampleScale(), scaledSoftmax.classDimension());
    }

    private GradMulMatch matchScaledSoftmax(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null || tensor.getOperation().opType() != Operation.OpType.MUL) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }

        SoftmaxInputMatch softmaxLeft = matchSoftmaxInput(inputs.get(0));
        Tensor sampleScaleLeft = matchBroadcastedSampleScale(inputs.get(1), softmaxLeft == null ? -1 : softmaxLeft.classDimension());
        if (softmaxLeft != null && sampleScaleLeft != null) {
            return new GradMulMatch(softmaxLeft.logits(), sampleScaleLeft, softmaxLeft.classDimension());
        }

        SoftmaxInputMatch softmaxRight = matchSoftmaxInput(inputs.get(1));
        Tensor sampleScaleRight = matchBroadcastedSampleScale(inputs.get(0), softmaxRight == null ? -1 : softmaxRight.classDimension());
        if (softmaxRight != null && sampleScaleRight != null) {
            return new GradMulMatch(softmaxRight.logits(), sampleScaleRight, softmaxRight.classDimension());
        }
        return null;
    }

    private SoftmaxInputMatch matchSoftmaxInput(Tensor tensor) {
        if (tensor == null || !(tensor.getOperation() instanceof softmax softmaxOp)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        return new SoftmaxInputMatch(inputs.get(0), softmaxOp.getDimension());
    }

    private Tensor matchBroadcastedSampleScale(Tensor tensor, int classDimension) {
        if (tensor == null || classDimension < 0) {
            return null;
        }
        if (tensor.getOperation() instanceof expandDims expandDimsOp) {
            if (expandDimsOp.getAxis() != classDimension) {
                return null;
            }
            List<Tensor> inputs = tensor.getPrevTensors();
            return (inputs != null && inputs.size() == 1) ? inputs.get(0) : null;
        }
        if (tensor.getOperation() instanceof expand expandOp) {
            List<Tensor> expandInputs = tensor.getPrevTensors();
            if (expandInputs == null || expandInputs.size() != 1) {
                return null;
            }
            Tensor expanded = expandInputs.get(0);
            if (!(expanded.getOperation() instanceof expandDims expandDimsOp) || expandDimsOp.getAxis() != classDimension) {
                return null;
            }
            List<Tensor> inputs = expanded.getPrevTensors();
            return (inputs != null && inputs.size() == 1) ? inputs.get(0) : null;
        }
        return null;
    }

    private ScatterMatch matchScatter(Tensor tensor) {
        if (tensor == null || !(tensor.getOperation() instanceof scatterAdd scatterAddOp)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 3) {
            return null;
        }
        Tensor zeroBase = inputs.get(0);
        Tensor targetIndices = inputs.get(1);
        Tensor sampleScale = inputs.get(2);
        if (!isZeroLikeLeaf(zeroBase)) {
            return null;
        }
        return new ScatterMatch(zeroBase, targetIndices, sampleScale, scatterAddOp.getDimension());
    }

    private boolean isZeroLikeLeaf(Tensor tensor) {
        if (tensor == null || tensor.getOperation() != null) {
            return false;
        }
        if (tensor.getPrevTensors() != null && !tensor.getPrevTensors().isEmpty()) {
            return false;
        }
        return "zeros_like".equals(tensor.getLabel());
    }

    private boolean sameShapeAndDType(Tensor left, Tensor right) {
        return left != null
                && right != null
                && left.getDataType() == right.getDataType()
                && java.util.Arrays.equals(left.getShapeUnsafe(), right.getShapeUnsafe());
    }

    private record Match(Tensor logits, Tensor targetIndices, int classDimension) {}

    private record SoftmaxInputMatch(Tensor logits, int classDimension) {}

    private record GradMulMatch(Tensor logits, Tensor sampleScale, int classDimension) {}

    private record ScatterMatch(Tensor zeroBase, Tensor targetIndices, Tensor sampleScale, int classDimension) {}

    private record GradMatch(Tensor logits, Tensor targetIndices, Tensor sampleScale, int classDimension) {}
}
