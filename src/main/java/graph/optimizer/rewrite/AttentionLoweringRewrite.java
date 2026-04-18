package graph.optimizer.rewrite;

import operations.Operation;
import operations.elementwise.unary.mulScalar;
import operations.layout.permute;
import operations.reduction.softmax;
import operations.elementwise.where.where;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.AttentionOptions;

import java.util.Arrays;
import java.util.List;

public final class AttentionLoweringRewrite extends AbstractRewriteRule {
    @Override
    protected Tensor rewriteTensor(Tensor tensor) {
        if (!isOp(tensor, Operation.OpType.MATMUL)) {
            return tensor;
        }
        AttentionMatch match = matchAttention(tensor);
        if (match == null) {
            return tensor;
        }

        AttentionOptions options = AttentionOptions.defaults().withScale(match.scale());
        Tensor lowered = match.mask() == null
                ? match.query().scaledDotProductAttention(match.key(), match.value(), options)
                : match.query().scaledDotProductAttention(match.key(), match.value(), match.mask(), options);
        lowered.setRequiresGrad(tensor.getRequiresGrad());
        return lowered;
    }

    private AttentionMatch matchAttention(Tensor tensor) {
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 2) {
            return null;
        }
        Tensor weights = inputs.get(0);
        Tensor value = inputs.get(1);
        if (!(weights.getOperation() instanceof softmax softmaxOp)) {
            return null;
        }
        if (softmaxOp.getDimension() != weights.getShapeUnsafe().length - 1) {
            return null;
        }
        List<Tensor> weightInputs = weights.getPrevTensors();
        if (weightInputs == null || weightInputs.size() != 1) {
            return null;
        }
        ScoreMatch scores = matchScores(weightInputs.get(0));
        if (scores == null) {
            return null;
        }
        if (!Arrays.equals(tensor.getShapeUnsafe(), expectedOutputShape(scores.query(), value))) {
            return null;
        }
        return new AttentionMatch(scores.query(), scores.key(), value, scores.mask(), scores.scale());
    }

    private ScoreMatch matchScores(Tensor tensor) {
        if (tensor != null && tensor.getOperation() instanceof where) {
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs == null || inputs.size() != 3) {
                return null;
            }
            Tensor mask = inputs.get(0);
            Tensor kept = inputs.get(1);
            Tensor fill = inputs.get(2);
            if (mask == null || mask.getDataType() != DataType.BOOL || !isMaskFillScalar(fill)) {
                return null;
            }
            ScoreMatch keptMatch = matchScaledQkMatMul(kept);
            if (keptMatch == null) {
                return null;
            }
            return new ScoreMatch(keptMatch.query(), keptMatch.key(), mask, keptMatch.scale());
        }
        return matchScaledQkMatMul(tensor);
    }

    private ScoreMatch matchScaledQkMatMul(Tensor tensor) {
        if (tensor == null) {
            return null;
        }
        double scale = 1.0d;
        Tensor candidate = tensor;
        if (tensor.getOperation() instanceof mulScalar mulScalarOp) {
            List<Tensor> inputs = tensor.getPrevTensors();
            if (inputs == null || inputs.size() != 1 || !(mulScalarOp.getScalar() > 0.0d)) {
                return null;
            }
            scale = mulScalarOp.getScalar();
            candidate = inputs.get(0);
        }
        if (!isOp(candidate, Operation.OpType.MATMUL)) {
            return null;
        }
        List<Tensor> matMulInputs = candidate.getPrevTensors();
        if (matMulInputs == null || matMulInputs.size() != 2) {
            return null;
        }
        Tensor query = matMulInputs.get(0);
        Tensor key = matchSwappedLastTwoAxes(matMulInputs.get(1));
        if (key == null) {
            return null;
        }
        return new ScoreMatch(query, key, null, scale);
    }

    private Tensor matchSwappedLastTwoAxes(Tensor tensor) {
        if (!(tensor != null && tensor.getOperation() instanceof permute permuteOp)) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        int[] axes = permuteOp.getAxes();
        if (!isLastTwoAxesSwap(axes)) {
            return null;
        }
        return inputs.get(0);
    }

    private boolean isLastTwoAxesSwap(int[] axes) {
        if (axes == null || axes.length < 2) {
            return false;
        }
        for (int i = 0; i < axes.length - 2; i++) {
            if (axes[i] != i) {
                return false;
            }
        }
        return axes[axes.length - 2] == axes.length - 1
                && axes[axes.length - 1] == axes.length - 2;
    }

    private boolean isMaskFillScalar(Tensor tensor) {
        if (tensor == null || tensor.getOperation() != null || tensor.getFlatDataSize() != 1) {
            return false;
        }
        double expected = maskFillValue(tensor.getDataType());
        double actual = tensor.scalarAsDouble();
        double tolerance = Math.max(1e-6d, Math.abs(expected) * 1e-6d);
        return Math.abs(actual - expected) <= tolerance;
    }

    private double maskFillValue(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> -1.0e30d;
            case FLOAT32 -> -1.0e9d;
            case BFLOAT16 -> -1.0e30d;
            case INT32, BOOL -> Double.NaN;
        };
    }

    private int[] expectedOutputShape(Tensor query, Tensor value) {
        int[] qShape = query.getShapeUnsafe();
        int[] vShape = value.getShapeUnsafe();
        int[] out = qShape.clone();
        out[out.length - 1] = vShape[vShape.length - 1];
        return out;
    }

    private static boolean isOp(Tensor tensor, Operation.OpType type) {
        return tensor != null && tensor.getOperation() != null && tensor.getOperation().opType() == type;
    }

    private record ScoreMatch(Tensor query, Tensor key, Tensor mask, double scale) {}

    private record AttentionMatch(Tensor query, Tensor key, Tensor value, Tensor mask, double scale) {}
}
