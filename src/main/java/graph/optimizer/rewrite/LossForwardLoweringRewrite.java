package graph.optimizer.rewrite;

import operations.Operation;
import operations.index.gather;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.reduction.sum;
import tensor.Tensor;
import tensor.loss.LossReduction;

import java.util.List;

public final class LossForwardLoweringRewrite extends AbstractRewriteRule {
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
        return match.logits().crossEntropyLossFromIndices(match.targetIndices(), match.classDimension(), reduction);
    }

    private Tensor lowerCrossEntropyFromIndices(Tensor tensor, LossReduction reduction) {
        Match match = matchPerSampleCrossEntropy(tensor);
        if (match == null) {
            return tensor;
        }
        return match.logits().crossEntropyLossFromIndices(match.targetIndices(), match.classDimension(), reduction);
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

    private record Match(Tensor logits, Tensor targetIndices, int classDimension) {}
}
