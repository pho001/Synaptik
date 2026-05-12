package onnx;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import tensor.DataType;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Optional;
import java.util.Set;

final class OnnxExportPatternRegistry {
    private static final double EPS = 1.0e-6;

    private OnnxExportPatternRegistry() {
    }

    static Optional<OnnxExportPatternMatch> match(Tensor tensor, OnnxExportPatternContext context) {
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
        return matchSoftplus(tensor, context);
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

    private record ScaleMatch(Tensor input, double scalar, Set<Tensor> consumedTensors) {
    }

    private record AddScalarMatch(Tensor nonScalarInput, double scalar, Set<Tensor> consumedTensors) {
    }
}
