package graph.optimizer.rewrite;

import operations.Operation;
import operations.index.scatterAdd;
import operations.layout.expand;
import operations.layout.expandDims;
import operations.loss.crossEntropyLossIndicesGrad;
import operations.reduction.softmax;
import graph.optimizer.intent.BackendIntentPropagator;
import tensor.Tensor;

import java.util.List;

/**
 * Optional legacy specialization matcher for explicit index-loss backward shapes.
 *
 * <p>Public Tensor APIs build canonical primitive backward DAGs by default. This
 * matcher only recognizes graphs that already contain explicit legacy
 * {@code SOFTMAX} operation descriptors, so it does not collapse the canonical
 * decomposed {@code softmax} DAG back into {@code CROSS_ENTROPY_LOSS_INDICES_GRAD}.
 * If this becomes a CPU specialization again, keep it as an explicit
 * specialization decision rather than semantic graph lowering.</p>
 */
public final class LossBackwardLoweringRewrite extends AbstractRewriteRule {
    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        Operation op = tensor.getOperation();
        if (op == null || op.opType() != Operation.OpType.SUB) {
            return tensor;
        }
        return lowerCrossEntropyFromIndicesGrad(tensor);
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
        BackendIntentPropagator.preserve(lowered, tensor);
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

    private record SoftmaxInputMatch(Tensor logits, int classDimension) {}

    private record GradMulMatch(Tensor logits, Tensor sampleScale, int classDimension) {}

    private record ScatterMatch(Tensor zeroBase, Tensor targetIndices, Tensor sampleScale, int classDimension) {}

    private record GradMatch(Tensor logits, Tensor targetIndices, Tensor sampleScale, int classDimension) {}
}
