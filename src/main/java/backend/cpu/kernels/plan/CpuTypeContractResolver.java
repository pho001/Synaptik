package backend.cpu.kernels.plan;

import backend.cpu.fused.codegen.FusedExternalInputPlan;
import operations.Operation;
import backend.cpu.fused.plan.FusedOperation;
import operations.index.ScatterReduction;
import operations.index.scatterElements;
import operations.index.scatterNd;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

public final class CpuTypeContractResolver {
    private CpuTypeContractResolver() {
    }

    public static PreparedTypeContract resolve(Operation op, Tensor node, List<Tensor> inputs) {
        if (op == null || op.opType() == null) {
            DataType outputType = resolveTargetType(node, inputs);
            return uniformTypeContract(outputType, inputs == null ? 0 : inputs.size());
        }
        return switch (op.opType()) {
            case GT, GE, LT, LE, EQ, NE -> resolveCompareContract(inputs);
            case WHERE -> resolveWhereContract(inputs);
            case LOGICAL_AND, LOGICAL_OR -> resolveLogicalBinaryContract(inputs);
            case LOGICAL_NOT -> resolveLogicalUnaryContract(inputs);
            case REDUCE_ALL, REDUCE_ANY -> resolveBoolReductionContract(inputs);
            case ARGMAX -> resolveArgMaxContract(inputs);
            case CUMSUM -> resolveCumSumContract(inputs);
            case GATHER -> resolveGatherContract(inputs);
            case GATHER_GRAD -> resolveGatherGradContract(inputs);
            case GATHER_AXIS -> resolveGatherContract(inputs);
            case GATHER_AXIS_GRAD -> resolveGatherGradContract(inputs);
            case GATHER_ND -> resolveGatherContract(inputs);
            case GATHER_ND_GRAD -> resolveGatherGradContract(inputs);
            case TAKE_ALONG_AXIS -> resolveTakeAlongAxisContract(inputs);
            case TAKE_ALONG_AXIS_GRAD -> resolveTakeAlongAxisGradContract(inputs);
            case SCATTER_ADD -> resolveScatterAddContract(inputs);
            case SCATTER_ELEMENTS -> resolveScatterElementsContract(op, inputs);
            case SCATTER_ND -> resolveScatterNdContract(op, inputs);
            case SLICE_GRAD -> resolveSingleFloatingInputContract(inputs, "sliceGrad");
            case CAST -> resolveCastContract(node, inputs);
            case SCALED_DOT_PRODUCT_ATTENTION -> resolveAttentionContract(inputs);
            case SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> resolveSameFloatingBinaryContract(inputs, "scaledDotProductAttentionBackward");
            case SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS -> resolveSingleFloatingInputContract(inputs, "scaledDotProductAttentionWeights");
            case CROSS_ENTROPY_LOSS_INDICES -> resolveCrossEntropyLossIndicesContract(inputs);
            case CROSS_ENTROPY_LOSS_INDICES_GRAD -> resolveCrossEntropyLossIndicesGradContract(inputs);
            case SOFTMAX_GRAD -> resolveSameFloatingBinaryContract(inputs, "softmaxGrad");
            case LOG_SOFTMAX_GRAD -> resolveSameFloatingBinaryContract(inputs, "logSoftmaxGrad");
            case FUSED -> resolveFusedContract(op, inputs);
            default -> {
                DataType outputType = resolveTargetType(node, inputs);
                yield uniformTypeContract(outputType, inputs == null ? 0 : inputs.size());
            }
        };
    }

    private static PreparedTypeContract resolveFusedContract(Operation op, List<Tensor> inputs) {
        if (!(op instanceof FusedOperation fused)) {
            throw new IllegalArgumentException("FUSED contract resolution requires FusedOperation descriptor.");
        }
        List<FusedExternalInputPlan> inputPlans = fused.getPlan().inputs();
        if (inputs == null || inputs.size() != inputPlans.size()) {
            throw new IllegalArgumentException("FUSED input contract size mismatch.");
        }
        List<DataType> expectedInputTypes = new ArrayList<>(inputPlans.size());
        for (FusedExternalInputPlan inputPlan : inputPlans) {
            expectedInputTypes.add(inputPlan.dataType());
        }
        DataType outputType = fused.getPlan().outputNode().outputType();
        return new PreparedTypeContract(outputType, List.copyOf(expectedInputTypes));
    }

    private static PreparedTypeContract resolveCastContract(Tensor node, List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("cast expects exactly one input.");
        }
        return new PreparedTypeContract(node.getDataType(), List.of(inputs.getFirst().getDataType()));
    }

    private static PreparedTypeContract uniformTypeContract(DataType outputType, int arity) {
        List<DataType> inputTypes = new ArrayList<>(arity);
        for (int i = 0; i < arity; i++) {
            inputTypes.add(outputType);
        }
        return new PreparedTypeContract(outputType, List.copyOf(inputTypes));
    }

    private static PreparedTypeContract resolveCompareContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Comparison ops expect exactly two inputs.");
        }
        DataType left = inputs.get(0).getDataType();
        DataType right = inputs.get(1).getDataType();
        if (left == DataType.BOOL || right == DataType.BOOL) {
            throw new IllegalArgumentException("Comparison ops require numeric inputs.");
        }
        DataType numericType = promote(left, right);
        return new PreparedTypeContract(DataType.BOOL, List.of(numericType, numericType));
    }

    private static PreparedTypeContract resolveWhereContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("where expects exactly three inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("where condition must have BOOL dtype.");
        }
        DataType trueType = inputs.get(1).getDataType();
        DataType falseType = inputs.get(2).getDataType();
        if (trueType == DataType.BOOL || falseType == DataType.BOOL) {
            throw new IllegalArgumentException("where branches must have numeric dtypes.");
        }
        DataType branchType = promote(trueType, falseType);
        return new PreparedTypeContract(branchType, List.of(DataType.BOOL, branchType, branchType));
    }

    private static PreparedTypeContract resolveLogicalBinaryContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("logical binary ops expect exactly two inputs.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL || inputs.get(1).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logical binary ops require BOOL inputs.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL, DataType.BOOL));
    }

    private static PreparedTypeContract resolveLogicalUnaryContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("logicalNot expects exactly one input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("logicalNot requires BOOL input.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL));
    }

    private static PreparedTypeContract resolveBoolReductionContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("BOOL reductions expect exactly one input.");
        }
        if (inputs.get(0).getDataType() != DataType.BOOL) {
            throw new IllegalArgumentException("BOOL reductions require BOOL input.");
        }
        return new PreparedTypeContract(DataType.BOOL, List.of(DataType.BOOL));
    }

    private static PreparedTypeContract resolveArgMaxContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("argMax expects exactly one input.");
        }
        DataType inputType = inputs.getFirst().getDataType();
        if (inputType == DataType.BOOL) {
            throw new IllegalArgumentException("argMax requires numeric input.");
        }
        return new PreparedTypeContract(DataType.INT32, List.of(inputType));
    }

    private static PreparedTypeContract resolveCumSumContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("cumSum expects exactly one input.");
        }
        DataType inputType = inputs.getFirst().getDataType();
        if (inputType == DataType.BOOL) {
            throw new IllegalArgumentException("cumSum requires floating or INT32 input.");
        }
        return new PreparedTypeContract(inputType, List.of(inputType));
    }

    private static PreparedTypeContract resolveGatherContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gather expects exactly two inputs.");
        }
        DataType sourceType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("gather indices must be numeric integral values.");
        }
        return new PreparedTypeContract(sourceType, List.of(sourceType, indexType));
    }

    private static PreparedTypeContract resolveGatherGradContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherGrad expects exactly two inputs.");
        }
        DataType indexType = inputs.get(0).getDataType();
        DataType gradType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL || gradType == DataType.BOOL) {
            throw new IllegalArgumentException("gatherGrad requires numeric indices and numeric gradient input.");
        }
        return new PreparedTypeContract(gradType, List.of(indexType, gradType));
    }

    private static PreparedTypeContract resolveScatterAddContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAdd expects exactly three inputs.");
        }
        DataType baseType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        DataType srcType = inputs.get(2).getDataType();
        if (baseType == DataType.BOOL || srcType == DataType.BOOL || indexType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterAdd requires numeric tensors and numeric indices.");
        }
        if (baseType != srcType) {
            throw new IllegalArgumentException("scatterAdd requires base and src to have matching dtypes.");
        }
        return new PreparedTypeContract(baseType, List.of(baseType, indexType, srcType));
    }

    private static PreparedTypeContract resolveScatterElementsContract(Operation op, List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterElements expects exactly three inputs.");
        }
        DataType dataType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        DataType updatesType = inputs.get(2).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterElements indices must be numeric integral values.");
        }
        if (dataType != updatesType) {
            throw new IllegalArgumentException("scatterElements requires data and updates to have matching dtypes.");
        }
        if (op instanceof scatterElements scatterOp
                && scatterOp.getReduction() != ScatterReduction.NONE
                && dataType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterElements BOOL tensors support only NONE reduction.");
        }
        return new PreparedTypeContract(dataType, List.of(dataType, indexType, updatesType));
    }

    private static PreparedTypeContract resolveScatterNdContract(Operation op, List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterNd expects exactly three inputs.");
        }
        DataType dataType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        DataType updatesType = inputs.get(2).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterNd indices must be numeric integral values.");
        }
        if (dataType != updatesType) {
            throw new IllegalArgumentException("scatterNd requires data and updates to have matching dtypes.");
        }
        if (op instanceof scatterNd scatterOp
                && scatterOp.getReduction() != ScatterReduction.NONE
                && dataType == DataType.BOOL) {
            throw new IllegalArgumentException("scatterNd BOOL tensors support only NONE reduction.");
        }
        return new PreparedTypeContract(dataType, List.of(dataType, indexType, updatesType));
    }

    private static PreparedTypeContract resolveCrossEntropyLossIndicesContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices expects exactly two inputs.");
        }
        DataType logitsType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        if (logitsType == DataType.BOOL || logitsType == DataType.INT32) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices requires floating logits.");
        }
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLossFromIndices indices must be numeric integral values.");
        }
        return new PreparedTypeContract(logitsType, List.of(logitsType, indexType));
    }

    private static PreparedTypeContract resolveCrossEntropyLossIndicesGradContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("crossEntropyLossFromIndicesGrad expects exactly three inputs.");
        }
        DataType logitsType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        DataType scaleType = inputs.get(2).getDataType();
        if (logitsType == DataType.BOOL || logitsType == DataType.INT32 || scaleType == DataType.BOOL || scaleType == DataType.INT32) {
            throw new IllegalArgumentException("crossEntropyLossFromIndicesGrad requires floating logits and floating scale.");
        }
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("crossEntropyLossFromIndicesGrad indices must be numeric integral values.");
        }
        if (logitsType != scaleType) {
            throw new IllegalArgumentException("crossEntropyLossFromIndicesGrad requires matching logits and scale dtypes.");
        }
        return new PreparedTypeContract(logitsType, List.of(logitsType, indexType, scaleType));
    }

    private static PreparedTypeContract resolveAttentionContract(List<Tensor> inputs) {
        if (inputs == null || (inputs.size() != 3 && inputs.size() != 4)) {
            throw new IllegalArgumentException("scaledDotProductAttention expects three or four inputs.");
        }
        DataType queryType = inputs.get(0).getDataType();
        DataType keyType = inputs.get(1).getDataType();
        DataType valueType = inputs.get(2).getDataType();
        if (queryType == DataType.BOOL || queryType == DataType.INT32
                || keyType == DataType.BOOL || keyType == DataType.INT32
                || valueType == DataType.BOOL || valueType == DataType.INT32) {
            throw new IllegalArgumentException("scaledDotProductAttention requires floating q/k/v inputs.");
        }
        DataType promoted = promote(promote(queryType, keyType), valueType);
        if (inputs.size() == 4) {
            if (inputs.get(3).getDataType() != DataType.BOOL) {
                throw new IllegalArgumentException("scaledDotProductAttention mask must have BOOL dtype.");
            }
            return new PreparedTypeContract(promoted, List.of(promoted, promoted, promoted, DataType.BOOL));
        }
        return new PreparedTypeContract(promoted, List.of(promoted, promoted, promoted));
    }

    private static PreparedTypeContract resolveSameFloatingBinaryContract(List<Tensor> inputs, String opName) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException(opName + " expects exactly two inputs.");
        }
        DataType leftType = inputs.get(0).getDataType();
        DataType rightType = inputs.get(1).getDataType();
        if (leftType == DataType.BOOL || leftType == DataType.INT32
                || rightType == DataType.BOOL || rightType == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires floating inputs.");
        }
        if (leftType != rightType) {
            throw new IllegalArgumentException(opName + " requires matching floating dtypes.");
        }
        return new PreparedTypeContract(leftType, List.of(leftType, rightType));
    }

    private static PreparedTypeContract resolveSingleFloatingInputContract(List<Tensor> inputs, String opName) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException(opName + " expects exactly one input.");
        }
        DataType inputType = inputs.getFirst().getDataType();
        if (inputType == DataType.BOOL || inputType == DataType.INT32) {
            throw new IllegalArgumentException(opName + " requires a floating input.");
        }
        return new PreparedTypeContract(inputType, List.of(inputType));
    }

    private static PreparedTypeContract resolveTakeAlongAxisContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two inputs.");
        }
        DataType sourceType = inputs.get(0).getDataType();
        DataType indexType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxis indices must be numeric integral values.");
        }
        return new PreparedTypeContract(sourceType, List.of(sourceType, indexType));
    }

    private static PreparedTypeContract resolveTakeAlongAxisGradContract(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxisGrad expects exactly two inputs.");
        }
        DataType indexType = inputs.get(0).getDataType();
        DataType gradType = inputs.get(1).getDataType();
        if (indexType == DataType.BOOL || gradType == DataType.BOOL) {
            throw new IllegalArgumentException("takeAlongAxisGrad requires numeric indices and numeric gradient input.");
        }
        return new PreparedTypeContract(gradType, List.of(indexType, gradType));
    }

    private static DataType resolveTargetType(Tensor node, List<Tensor> inputs) {
        if (node != null && node.getDataType() != null) {
            return node.getDataType();
        }
        if (inputs == null || inputs.isEmpty()) {
            return DataType.FLOAT32;
        }

        DataType target = inputs.get(0).getDataType();
        for (int i = 1; i < inputs.size(); i++) {
            target = promote(target, inputs.get(i).getDataType());
        }
        return target == null ? DataType.FLOAT32 : target;
    }

    private static DataType promote(DataType left, DataType right) {
        if (left == DataType.INT32 || right == DataType.INT32) {
            throw new IllegalArgumentException("INT32 is not supported by generic floating numeric promotion. left=" + left + ", right=" + right);
        }
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        if (left == DataType.BFLOAT16 || right == DataType.BFLOAT16) {
            return DataType.BFLOAT16;
        }
        throw new IllegalArgumentException("BOOL is not supported by generic numeric promotion. left=" + left + ", right=" + right);
    }
}
