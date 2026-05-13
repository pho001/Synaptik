package onnx;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.reduction.mean;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;

final class OnnxExportPatternRegistry {
    private static final double EPS = 1.0e-6;

    private OnnxExportPatternRegistry() {
    }

    static Optional<OnnxExportPatternMatch> match(Tensor tensor, OnnxExportPatternContext context) {
        Optional<OnnxExportPatternMatch> batchNormalization = matchBatchNormalization(tensor, context);
        if (batchNormalization.isPresent()) {
            return batchNormalization;
        }
        Optional<OnnxExportPatternMatch> leakyRelu = matchLeakyRelu(tensor, context);
        if (leakyRelu.isPresent()) {
            return leakyRelu;
        }
        Optional<OnnxExportPatternMatch> elu = matchElu(tensor, context);
        if (elu.isPresent()) {
            return elu;
        }
        Optional<OnnxExportPatternMatch> hardSigmoid = matchHardSigmoid(tensor, context);
        if (hardSigmoid.isPresent()) {
            return hardSigmoid;
        }
        Optional<OnnxExportPatternMatch> softplus = matchSoftplus(tensor, context);
        if (softplus.isPresent()) {
            return softplus;
        }
        Optional<OnnxExportPatternMatch> reduceLogSumExp = matchReduceLogSumExp(tensor, context);
        if (reduceLogSumExp.isPresent()) {
            return reduceLogSumExp;
        }
        Optional<OnnxExportPatternMatch> reduceLogSum = matchReduceLogSum(tensor, context);
        if (reduceLogSum.isPresent()) {
            return reduceLogSum;
        }
        Optional<OnnxExportPatternMatch> reduceL2 = matchReduceL2(tensor, context);
        if (reduceL2.isPresent()) {
            return reduceL2;
        }
        Optional<OnnxExportPatternMatch> reduceL1 = matchReduceL1(tensor, context);
        if (reduceL1.isPresent()) {
            return reduceL1;
        }
        return matchGlobalAveragePool(tensor, context);
    }

    private static Optional<OnnxExportPatternMatch> matchBatchNormalization(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.ADD)) {
            return Optional.empty();
        }
        Tensor left = input(tensor, 0);
        Tensor right = input(tensor, 1);
        Tensor scaled;
        Tensor betaView;
        if (hasOp(left, Operation.OpType.MUL) && hasOp(right, Operation.OpType.RESHAPE)) {
            scaled = left;
            betaView = right;
        } else if (hasOp(right, Operation.OpType.MUL) && hasOp(left, Operation.OpType.RESHAPE)) {
            scaled = right;
            betaView = left;
        } else {
            return Optional.empty();
        }
        if (!context.canConsume(scaled) || !context.canConsume(betaView)) {
            return Optional.empty();
        }
        Tensor scaledLeft = input(scaled, 0);
        Tensor scaledRight = input(scaled, 1);
        Tensor normalized;
        Tensor gammaView;
        if (hasOp(scaledLeft, Operation.OpType.DIV) && hasOp(scaledRight, Operation.OpType.RESHAPE)) {
            normalized = scaledLeft;
            gammaView = scaledRight;
        } else if (hasOp(scaledRight, Operation.OpType.DIV) && hasOp(scaledLeft, Operation.OpType.RESHAPE)) {
            normalized = scaledRight;
            gammaView = scaledLeft;
        } else {
            return Optional.empty();
        }
        if (!context.canConsume(normalized) || !context.canConsume(gammaView)) {
            return Optional.empty();
        }
        Tensor centered = input(normalized, 0);
        Tensor denominator = input(normalized, 1);
        if (!hasOp(centered, Operation.OpType.SUB) || !hasOp(denominator, Operation.OpType.SQRT)
                || !context.canConsume(centered) || !context.canConsume(denominator)) {
            return Optional.empty();
        }
        Tensor source = input(centered, 0);
        Tensor meanView = input(centered, 1);
        int[] sourceShape = source.getShapeUnsafe();
        if (sourceShape.length < 2) {
            return Optional.empty();
        }
        Tensor variancePlusEpsilon = input(denominator, 0);
        if (!hasOp(variancePlusEpsilon, Operation.OpType.ADD) || !context.canConsume(variancePlusEpsilon)) {
            return Optional.empty();
        }
        Tensor addLeft = input(variancePlusEpsilon, 0);
        Tensor addRight = input(variancePlusEpsilon, 1);
        Tensor varianceView;
        Tensor epsilon;
        if (hasOp(addLeft, Operation.OpType.RESHAPE) && isScalarConstant(addRight)) {
            varianceView = addLeft;
            epsilon = addRight;
        } else if (hasOp(addRight, Operation.OpType.RESHAPE) && isScalarConstant(addLeft)) {
            varianceView = addRight;
            epsilon = addLeft;
        } else {
            return Optional.empty();
        }
        if (!context.canConsume(varianceView) || !isConsumableScalar(epsilon, epsilon.scalarAsDouble(), context)) {
            return Optional.empty();
        }
        Optional<Tensor> gamma = matchChannelReshape(gammaView, sourceShape, context);
        Optional<Tensor> beta = matchChannelReshape(betaView, sourceShape, context);
        Optional<Tensor> mean = matchChannelReshape(meanView, sourceShape, context);
        Optional<Tensor> variance = matchChannelReshape(varianceView, sourceShape, context);
        if (gamma.isEmpty() || beta.isEmpty() || mean.isEmpty() || variance.isEmpty()) {
            return Optional.empty();
        }

        Set<Tensor> consumed = identitySet();
        Collections.addAll(consumed, scaled, betaView, normalized, gammaView, centered, denominator,
                meanView, variancePlusEpsilon, varianceView, epsilon);
        OnnxProto.NodeProto node = OnnxProto.NodeProto.newBuilder()
                .setName("node_" + context.id(tensor))
                .setOpType("BatchNormalization")
                .addInput(context.name(source))
                .addInput(context.name(gamma.get()))
                .addInput(context.name(beta.get()))
                .addInput(context.name(mean.get()))
                .addInput(context.name(variance.get()))
                .addOutput(context.name(tensor))
                .addAttribute(floatAttr("epsilon", (float) epsilon.scalarAsDouble()))
                .build();
        return Optional.of(new OnnxExportPatternMatch(node, consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchLeakyRelu(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.WHERE)) {
            return Optional.empty();
        }
        Tensor condition = input(tensor, 0);
        Tensor positive = input(tensor, 1);
        Tensor negative = input(tensor, 2);
        if (!context.canConsume(condition) || !context.canConsume(negative)) {
            return Optional.empty();
        }
        Optional<Set<Tensor>> conditionConstants = matchesGreaterOrEqualZero(condition, positive, context);
        Optional<ScaleMatch> scale = matchScale(negative, positive, context);
        if (conditionConstants.isEmpty() || scale.isEmpty()) {
            return Optional.empty();
        }

        Set<Tensor> consumed = identitySet();
        consumed.add(condition);
        consumed.addAll(conditionConstants.get());
        consumed.addAll(scale.get().consumedTensors());
        return Optional.of(new OnnxExportPatternMatch(node("LeakyRelu", tensor, positive, context)
                .addAttribute(floatAttr("alpha", (float) scale.get().scalar()))
                .build(), consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchElu(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.WHERE)) {
            return Optional.empty();
        }
        Tensor condition = input(tensor, 0);
        Tensor positive = input(tensor, 1);
        Tensor negative = input(tensor, 2);
        if (!context.canConsume(condition) || !context.canConsume(negative)) {
            return Optional.empty();
        }
        Optional<Set<Tensor>> conditionConstants = matchesGreaterOrEqualZero(condition, positive, context);
        if (conditionConstants.isEmpty()) {
            return Optional.empty();
        }

        double alpha = 1.0d;
        Tensor base = negative;
        Set<Tensor> consumed = identitySet();
        consumed.add(condition);
        consumed.addAll(conditionConstants.get());
        Optional<ScaleMatch> scale = matchScale(negative, null, context);
        if (scale.isPresent()) {
            alpha = scale.get().scalar();
            base = scale.get().input();
            consumed.addAll(scale.get().consumedTensors());
        } else {
            consumed.add(negative);
        }
        if (!context.canConsume(base) || !hasOp(base, Operation.OpType.SUB)) {
            return Optional.empty();
        }
        Tensor exp = input(base, 0);
        Tensor one = input(base, 1);
        if (!context.canConsume(exp) || !hasOp(exp, Operation.OpType.EXP) || input(exp, 0) != positive
                || !isConsumableScalar(one, 1.0d, context)) {
            return Optional.empty();
        }
        consumed.add(base);
        consumed.add(exp);
        consumed.add(one);
        return Optional.of(new OnnxExportPatternMatch(node("Elu", tensor, positive, context)
                .addAttribute(floatAttr("alpha", (float) alpha))
                .build(), consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchHardSigmoid(Tensor tensor, OnnxExportPatternContext context) {
        if (!(tensor.getOperation() instanceof clampMin min) || !close(min.getMinValue(), 0.0d)) {
            return Optional.empty();
        }
        Tensor maxTensor = input(tensor, 0);
        if (!context.canConsume(maxTensor) || !(maxTensor.getOperation() instanceof clampMax max)
                || !close(max.getMaxValue(), 1.0d)) {
            return Optional.empty();
        }
        Tensor affine = input(maxTensor, 0);
        if (!context.canConsume(affine)) {
            return Optional.empty();
        }
        Optional<AddScalarMatch> add = matchAddScalar(affine, context);
        if (add.isEmpty()) {
            return Optional.empty();
        }
        Optional<ScaleMatch> scale = matchScale(add.get().nonScalarInput(), null, context);
        if (scale.isEmpty()) {
            return Optional.empty();
        }

        Set<Tensor> consumed = identitySet();
        consumed.add(maxTensor);
        consumed.add(affine);
        consumed.addAll(add.get().consumedTensors());
        consumed.addAll(scale.get().consumedTensors());
        return Optional.of(new OnnxExportPatternMatch(node("HardSigmoid", tensor, scale.get().input(), context)
                .addAttribute(floatAttr("alpha", (float) scale.get().scalar()))
                .addAttribute(floatAttr("beta", (float) add.get().scalar()))
                .build(), consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchSoftplus(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.LOG)) {
            return Optional.empty();
        }
        Tensor addTensor = input(tensor, 0);
        if (!context.canConsume(addTensor)) {
            return Optional.empty();
        }
        Optional<AddScalarMatch> add = matchAddScalar(addTensor, context);
        if (add.isEmpty() || !close(add.get().scalar(), 1.0d)) {
            return Optional.empty();
        }
        Tensor exp = add.get().nonScalarInput();
        if (!context.canConsume(exp) || !hasOp(exp, Operation.OpType.EXP)) {
            return Optional.empty();
        }

        Set<Tensor> consumed = identitySet();
        consumed.add(addTensor);
        consumed.addAll(add.get().consumedTensors());
        consumed.add(exp);
        return Optional.of(new OnnxExportPatternMatch(node("Softplus", tensor, input(exp, 0), context).build(), consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchReduceL1(Tensor tensor, OnnxExportPatternContext context) {
        Optional<ReductionMatch> reduce = matchSum(tensor);
        if (reduce.isEmpty()) {
            return Optional.empty();
        }
        Tensor abs = reduce.get().input();
        if (!context.canConsume(abs) || !hasOp(abs, Operation.OpType.ABS)) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(abs);
        return Optional.of(reduction("ReduceL1", tensor, input(abs, 0), reduce.get(), context, consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchReduceL2(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.SQRT)) {
            return Optional.empty();
        }
        Tensor reduceTensor = input(tensor, 0);
        if (!context.canConsume(reduceTensor)) {
            return Optional.empty();
        }
        Optional<ReductionMatch> reduce = matchSum(reduceTensor);
        if (reduce.isEmpty()) {
            return Optional.empty();
        }
        Tensor square = reduce.get().input();
        if (!context.canConsume(square) || !hasOp(square, Operation.OpType.MUL)) {
            return Optional.empty();
        }
        Tensor left = input(square, 0);
        Tensor right = input(square, 1);
        if (left != right) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(reduceTensor);
        consumed.add(square);
        return Optional.of(reduction("ReduceL2", tensor, left, reduce.get(), context, consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchReduceLogSum(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.LOG)) {
            return Optional.empty();
        }
        Tensor reduceTensor = input(tensor, 0);
        if (!context.canConsume(reduceTensor)) {
            return Optional.empty();
        }
        Optional<ReductionMatch> reduce = matchSum(reduceTensor);
        if (reduce.isEmpty()) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(reduceTensor);
        return Optional.of(reduction("ReduceLogSum", tensor, reduce.get().input(), reduce.get(), context, consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchReduceLogSumExp(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.LOG)) {
            return Optional.empty();
        }
        Tensor reduceTensor = input(tensor, 0);
        if (!context.canConsume(reduceTensor)) {
            return Optional.empty();
        }
        Optional<ReductionMatch> reduce = matchSum(reduceTensor);
        if (reduce.isEmpty()) {
            return Optional.empty();
        }
        Tensor exp = reduce.get().input();
        if (!context.canConsume(exp) || !hasOp(exp, Operation.OpType.EXP)) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(reduceTensor);
        consumed.add(exp);
        return Optional.of(reduction("ReduceLogSumExp", tensor, input(exp, 0), reduce.get(), context, consumed));
    }

    private static Optional<OnnxExportPatternMatch> matchGlobalAveragePool(Tensor tensor, OnnxExportPatternContext context) {
        Optional<MeanMatch> second = matchMean(tensor);
        if (second.isEmpty() || !second.get().keepDims()) {
            return Optional.empty();
        }
        Tensor firstTensor = second.get().input();
        if (!context.canConsume(firstTensor)) {
            return Optional.empty();
        }
        Optional<MeanMatch> first = matchMean(firstTensor);
        if (first.isEmpty() || !first.get().keepDims()) {
            return Optional.empty();
        }
        Tensor input = first.get().input();
        if (input.getShapeUnsafe().length != 4) {
            return Optional.empty();
        }
        int firstAxis = first.get().axis();
        int secondAxis = second.get().axis();
        if (!((firstAxis == 2 && secondAxis == 3) || (firstAxis == 3 && secondAxis == 2))) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(firstTensor);
        return Optional.of(new OnnxExportPatternMatch(node("GlobalAveragePool", tensor, input, context).build(), consumed));
    }

    private static Optional<Tensor> matchChannelReshape(Tensor tensor, int[] sourceShape, OnnxExportPatternContext context) {
        if (!(tensor.getOperation() instanceof operations.layout.reshape reshape) || !context.canConsume(tensor)) {
            return Optional.empty();
        }
        Tensor parameter = input(tensor, 0);
        int channels = sourceShape[1];
        int[] parameterShape = parameter.getShapeUnsafe();
        int[] targetShape = reshape.getTargetShape();
        if (parameterShape.length != 1 || parameterShape[0] != channels || targetShape.length != sourceShape.length) {
            return Optional.empty();
        }
        for (int i = 0; i < targetShape.length; i++) {
            int expected = i == 1 ? channels : 1;
            if (targetShape[i] != expected) {
                return Optional.empty();
            }
        }
        return Optional.of(parameter);
    }

    private static Optional<ReductionMatch> matchSum(Tensor tensor) {
        if (!(tensor.getOperation() instanceof sum reduce)) {
            return Optional.empty();
        }
        return Optional.of(new ReductionMatch(input(tensor, 0), reduce.getDimension(), reduce.keepDims()));
    }

    private static Optional<MeanMatch> matchMean(Tensor tensor) {
        if (!(tensor.getOperation() instanceof mean reduce)) {
            return Optional.empty();
        }
        return Optional.of(new MeanMatch(input(tensor, 0), reduce.getDimension(), reduce.keepDims()));
    }

    private static OnnxExportPatternMatch reduction(
            String opType,
            Tensor output,
            Tensor input,
            ReductionMatch reduction,
            OnnxExportPatternContext context,
            Set<Tensor> consumed
    ) {
        String axesName = context.auxiliary(context.name(output) + "_axes");
        OnnxProto.NodeProto node = node(opType, output, input, context)
                .addInput(axesName)
                .addAttribute(intAttr("keepdims", reduction.keepDims() ? 1 : 0))
                .build();
        return new OnnxExportPatternMatch(
                node,
                consumed,
                List.of(OnnxTensorProtoUtil.int64Initializer(axesName, new long[]{reduction.axis()}))
        );
    }

    private static Optional<ScaleMatch> matchScale(Tensor tensor, Tensor expectedInput, OnnxExportPatternContext context) {
        if (!context.canConsume(tensor)) {
            return Optional.empty();
        }
        if (tensor.getOperation() instanceof mulScalar scalar) {
            Tensor input = input(tensor, 0);
            if (expectedInput != null && input != expectedInput) {
                return Optional.empty();
            }
            Set<Tensor> consumed = identitySet();
            consumed.add(tensor);
            return Optional.of(new ScaleMatch(input, scalar.getScalar(), consumed));
        }
        if (!hasOp(tensor, Operation.OpType.MUL)) {
            return Optional.empty();
        }
        Tensor left = input(tensor, 0);
        Tensor right = input(tensor, 1);
        Tensor data;
        Tensor scalar;
        if (isScalarConstant(left)) {
            scalar = left;
            data = right;
        } else if (isScalarConstant(right)) {
            scalar = right;
            data = left;
        } else {
            return Optional.empty();
        }
        if (expectedInput != null && data != expectedInput) {
            return Optional.empty();
        }
        if (!context.canConsume(scalar)) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(tensor);
        consumed.add(scalar);
        return Optional.of(new ScaleMatch(data, scalar.scalarAsDouble(), consumed));
    }

    private static Optional<AddScalarMatch> matchAddScalar(Tensor tensor, OnnxExportPatternContext context) {
        if (!hasOp(tensor, Operation.OpType.ADD)) {
            return Optional.empty();
        }
        Tensor left = input(tensor, 0);
        Tensor right = input(tensor, 1);
        Tensor data;
        Tensor scalar;
        if (isScalarConstant(left)) {
            scalar = left;
            data = right;
        } else if (isScalarConstant(right)) {
            scalar = right;
            data = left;
        } else {
            return Optional.empty();
        }
        if (!context.canConsume(scalar)) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(scalar);
        return Optional.of(new AddScalarMatch(data, scalar.scalarAsDouble(), consumed));
    }

    private static Optional<Set<Tensor>> matchesGreaterOrEqualZero(
            Tensor condition,
            Tensor input,
            OnnxExportPatternContext context
    ) {
        if (!hasOp(condition, Operation.OpType.GE)) {
            return Optional.empty();
        }
        Tensor left = input(condition, 0);
        Tensor right = input(condition, 1);
        if (left != input || !isConsumableScalar(right, 0.0d, context)) {
            return Optional.empty();
        }
        Set<Tensor> consumed = identitySet();
        consumed.add(right);
        return Optional.of(consumed);
    }

    private static OnnxProto.NodeProto.Builder node(
            String opType,
            Tensor output,
            Tensor input,
            OnnxExportPatternContext context
    ) {
        return OnnxProto.NodeProto.newBuilder()
                .setName("node_" + context.id(output))
                .setOpType(opType)
                .addInput(context.name(input))
                .addOutput(context.name(output));
    }

    private static boolean hasOp(Tensor tensor, Operation.OpType opType) {
        Operation op = tensor.getOperation();
        return op != null && op.opType() == opType;
    }

    private static Tensor input(Tensor tensor, int index) {
        return tensor.getPrevTensors().get(index);
    }

    private static boolean isConsumableScalar(Tensor tensor, double expected, OnnxExportPatternContext context) {
        return isScalarConstant(tensor) && close(tensor.scalarAsDouble(), expected) && context.canConsume(tensor);
    }

    private static boolean isScalarConstant(Tensor tensor) {
        return tensor.getOperation() == null
                && !tensor.getRequiresGrad()
                && tensor.getFlatDataSize() == 1
                && tensor.getDataType() != DataType.BOOL;
    }

    private static boolean close(double left, double right) {
        return Math.abs(left - right) <= EPS;
    }

    private static Set<Tensor> identitySet() {
        return Collections.newSetFromMap(new IdentityHashMap<>());
    }

    private static OnnxProto.AttributeProto floatAttr(String name, float value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setF(value).build();
    }

    private static OnnxProto.AttributeProto intAttr(String name, long value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setI(value).build();
    }

    private record ScaleMatch(Tensor input, double scalar, Set<Tensor> consumedTensors) {
    }

    private record AddScalarMatch(Tensor nonScalarInput, double scalar, Set<Tensor> consumedTensors) {
    }

    private record ReductionMatch(Tensor input, int axis, boolean keepDims) {
    }

    private record MeanMatch(Tensor input, int axis, boolean keepDims) {
    }
}
