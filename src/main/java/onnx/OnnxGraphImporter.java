package onnx;

import tensor.DataType;
import tensor.Tensor;
import tensor.TensorOps;

import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

final class OnnxGraphImporter {
    private static final Set<String> SUPPORTED_OPS = Set.of(
            "Add", "Sub", "Mul", "Div",
            "Min", "Max", "Pow",
            "Neg", "Abs", "Relu", "Tanh", "Sigmoid", "Exp", "Log", "Sqrt",
            "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
            "And", "Or", "Not",
            "Where", "Identity", "Clip",
            "MatMul", "Gemm",
            "Transpose", "Reshape", "Squeeze", "Unsqueeze",
            "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin",
            "Softmax", "LogSoftmax",
            "Constant"
    );

    private OnnxGraphImporter() {
    }

    static ImportedOnnxModel importModel(OnnxProto.ModelProto model, OnnxImportOptions options) {
        Objects.requireNonNull(model, "model cannot be null");
        options = options == null ? OnnxImportOptions.defaults() : options;
        validateOpset(model, options);
        if (!model.hasGraph()) {
            throw new OnnxUnsupportedException("ONNX model does not contain a graph.");
        }

        OnnxProto.GraphProto graph = model.getGraph();
        Map<String, Tensor> tensors = new LinkedHashMap<>();
        Map<String, long[]> int64Constants = new LinkedHashMap<>();
        Map<String, Tensor> inputs = new LinkedHashMap<>();
        Set<String> constantTensors = new HashSet<>();

        for (OnnxProto.TensorProto initializer : graph.getInitializerList()) {
            String name = initializer.getName();
            OnnxTensorProtoUtil.ImportedConstant constant = OnnxTensorProtoUtil.parseConstant(
                    initializer,
                    "initializer '" + name + "'"
            );
            if (constant.isTensor()) {
                tensors.put(name, constant.tensor());
                constantTensors.add(name);
            } else {
                int64Constants.put(name, constant.int64Values());
            }
        }

        for (OnnxProto.ValueInfoProto input : graph.getInputList()) {
            if (tensors.containsKey(input.getName()) || int64Constants.containsKey(input.getName())) {
                continue;
            }
            DataType dataType = OnnxTensorProtoUtil.valueInfoDataType(input);
            Tensor tensor = new Tensor(
                    OnnxTensorProtoUtil.staticShape(input),
                    null,
                    input.getName(),
                    dataType
            );
            tensors.put(input.getName(), tensor);
            inputs.put(input.getName(), tensor);
        }

        for (OnnxProto.NodeProto node : graph.getNodeList()) {
            importNode(node, tensors, int64Constants, constantTensors);
        }

        Map<String, Tensor> outputs = new LinkedHashMap<>();
        for (OnnxProto.ValueInfoProto output : graph.getOutputList()) {
            Tensor tensor = tensors.get(output.getName());
            if (tensor == null) {
                throw new OnnxUnsupportedException("Graph output '" + output.getName() + "' is not produced by a supported tensor node.");
            }
            outputs.put(output.getName(), tensor);
        }
        return new ImportedOnnxModel(model, inputs, outputs);
    }

    private static void validateOpset(OnnxProto.ModelProto model, OnnxImportOptions options) {
        int defaultDomainOpset = -1;
        for (OnnxProto.OperatorSetIdProto opset : model.getOpsetImportList()) {
            if (opset.getDomain().isEmpty()) {
                defaultDomainOpset = Math.toIntExact(opset.getVersion());
                break;
            }
        }
        if (defaultDomainOpset < 0) {
            throw new OnnxUnsupportedException("ONNX model has no default-domain opset import.");
        }
        if (defaultDomainOpset < options.minimumOpsetVersion() || defaultDomainOpset > options.maximumOpsetVersion()) {
            throw new OnnxUnsupportedException("ONNX opset " + defaultDomainOpset
                    + " is outside supported range [" + options.minimumOpsetVersion()
                    + ", " + options.maximumOpsetVersion() + "].");
        }
    }

    private static void importNode(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        if (!node.getDomain().isEmpty()) {
            throw unsupported(node, "custom domain '" + node.getDomain() + "' is unsupported");
        }
        if (!SUPPORTED_OPS.contains(node.getOpType())) {
            throw unsupported(node, "unsupported op type");
        }
        if (node.getOutputCount() != 1) {
            throw unsupported(node, "only single-output nodes are supported");
        }

        OnnxAttributeReader attrs = new OnnxAttributeReader(node);
        Tensor out = switch (node.getOpType()) {
            case "Add" -> binary(node, tensors, TensorOps::add);
            case "Sub" -> binary(node, tensors, TensorOps::sub);
            case "Mul" -> binary(node, tensors, TensorOps::mul);
            case "Div" -> binary(node, tensors, TensorOps::div);
            case "Min" -> binary(node, tensors, TensorOps::min);
            case "Max" -> binary(node, tensors, TensorOps::max);
            case "Pow" -> pow(node, tensors, int64Constants, constantTensors);
            case "Neg" -> unary(node, tensors, TensorOps::neg);
            case "Abs" -> unary(node, tensors, TensorOps::abs);
            case "Relu" -> unary(node, tensors, TensorOps::relu);
            case "Tanh" -> unary(node, tensors, TensorOps::tanh);
            case "Sigmoid" -> unary(node, tensors, TensorOps::sigmoid);
            case "Exp" -> unary(node, tensors, TensorOps::exp);
            case "Log" -> unary(node, tensors, TensorOps::log);
            case "Sqrt" -> unary(node, tensors, TensorOps::sqrt);
            case "Equal" -> binary(node, tensors, TensorOps::equalTo);
            case "Greater" -> binary(node, tensors, TensorOps::greaterThan);
            case "GreaterOrEqual" -> binary(node, tensors, TensorOps::greaterOrEqual);
            case "Less" -> binary(node, tensors, TensorOps::lessThan);
            case "LessOrEqual" -> binary(node, tensors, TensorOps::lessOrEqual);
            case "And" -> binary(node, tensors, TensorOps::logicalAnd);
            case "Or" -> binary(node, tensors, TensorOps::logicalOr);
            case "Not" -> unary(node, tensors, TensorOps::logicalNot);
            case "Where" -> ternary(node, tensors, TensorOps::where);
            case "Identity" -> identity(node, tensors);
            case "Clip" -> clip(node, tensors, int64Constants, attrs, constantTensors);
            case "MatMul" -> binary(node, tensors, TensorOps::matmul);
            case "Gemm" -> gemm(node, tensors, attrs);
            case "Transpose" -> transpose(node, tensors, attrs);
            case "Reshape" -> reshape(node, tensors, int64Constants);
            case "Squeeze" -> squeeze(node, tensors, int64Constants, attrs);
            case "Unsqueeze" -> unsqueeze(node, tensors, int64Constants, attrs);
            case "ReduceSum" -> reduce(node, tensors, int64Constants, attrs, ReductionKind.SUM);
            case "ReduceMean" -> reduce(node, tensors, int64Constants, attrs, ReductionKind.MEAN);
            case "ReduceMax" -> reduce(node, tensors, int64Constants, attrs, ReductionKind.MAX);
            case "ReduceMin" -> reduce(node, tensors, int64Constants, attrs, ReductionKind.MIN);
            case "Softmax" -> unaryAxis(node, tensors, attrs, TensorOps::softmax);
            case "LogSoftmax" -> unaryAxis(node, tensors, attrs, TensorOps::logSoftmax);
            case "Constant" -> constant(node, attrs, int64Constants, constantTensors);
            default -> throw unsupported(node, "unsupported op type");
        };

        if (out != null) {
            out.setLabel(node.getOutput(0));
            tensors.put(node.getOutput(0), out);
        }
    }

    private static Tensor constant(
            OnnxProto.NodeProto node,
            OnnxAttributeReader attrs,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        OnnxProto.TensorProto value = attrs.tensorAttribute("value");
        if (value == null) {
            throw unsupported(node, "Constant requires tensor attribute 'value'");
        }
        OnnxTensorProtoUtil.ImportedConstant constant = OnnxTensorProtoUtil.parseConstant(
                value,
                "Constant node '" + nodeName(node) + "'"
        );
        if (!constant.isTensor()) {
            int64Constants.put(node.getOutput(0), constant.int64Values());
            return null;
        }
        constantTensors.add(node.getOutput(0));
        return constant.tensor();
    }

    private static Tensor pow(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 2, 2);
        return TensorOps.pow(tensorInput(node, tensors, 0), scalarConstantInput(node, tensors, int64Constants, constantTensors, 1));
    }

    private static Tensor clip(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 1, 3);
        Tensor out = tensorInput(node, tensors, 0);
        Double minValue = null;
        Double maxValue = null;
        if (node.getInputCount() >= 2 && !node.getInput(1).isBlank()) {
            minValue = scalarConstantInput(node, tensors, int64Constants, constantTensors, 1);
        } else if (attrs.hasAttribute("min")) {
            minValue = (double) attrs.floatAttribute("min", 0.0f);
        }
        if (node.getInputCount() >= 3 && !node.getInput(2).isBlank()) {
            maxValue = scalarConstantInput(node, tensors, int64Constants, constantTensors, 2);
        } else if (attrs.hasAttribute("max")) {
            maxValue = (double) attrs.floatAttribute("max", 0.0f);
        }
        if (minValue != null && maxValue != null) {
            return TensorOps.clamp(out, minValue, maxValue);
        }
        if (minValue != null) {
            return TensorOps.clampMin(out, minValue);
        }
        if (maxValue != null) {
            return TensorOps.clampMax(out, maxValue);
        }
        return out;
    }

    private static Tensor identity(OnnxProto.NodeProto node, Map<String, Tensor> tensors) {
        requireInputCount(node, 1, 1);
        return tensorInput(node, tensors, 0);
    }

    private static Tensor gemm(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 2, 3);
        Tensor a = tensorInput(node, tensors, 0);
        Tensor b = tensorInput(node, tensors, 1);
        Tensor c = node.getInputCount() >= 3 && !node.getInput(2).isBlank() ? tensorInput(node, tensors, 2) : null;
        if (attrs.intAttribute("transA", 0) != 0) {
            a = requireRank2(a, node, "transA").transpose();
        }
        if (attrs.intAttribute("transB", 0) != 0) {
            b = requireRank2(b, node, "transB").transpose();
        }
        Tensor out = a.matmul(b);
        float alpha = attrs.floatAttribute("alpha", 1.0f);
        if (alpha != 1.0f) {
            out = out.mul(alpha);
        }
        if (c != null) {
            float beta = attrs.floatAttribute("beta", 1.0f);
            out = out.add(beta == 1.0f ? c : c.mul(beta));
        }
        return out;
    }

    private static Tensor transpose(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        int[] perm = attrs.intsAttribute("perm");
        if (perm == null) {
            int rank = input.getShapeUnsafe().length;
            perm = new int[rank];
            for (int i = 0; i < rank; i++) {
                perm[i] = rank - 1 - i;
            }
        }
        return input.permute(perm);
    }

    private static Tensor reshape(OnnxProto.NodeProto node, Map<String, Tensor> tensors, Map<String, long[]> int64Constants) {
        requireInputCount(node, 2, 2);
        return tensorInput(node, tensors, 0).reshape(toIntArray(intConstantInput(node, tensors, int64Constants, 1), node, "shape"));
    }

    private static Tensor squeeze(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 2);
        Tensor input = tensorInput(node, tensors, 0);
        int[] axes = axes(node, tensors, int64Constants, attrs);
        if (axes.length == 0) {
            axes = singletonAxes(input);
        } else {
            axes = normalizeAxes(axes, input.getShapeUnsafe().length, node, "axes");
        }
        Arrays.sort(axes);
        Tensor out = input;
        for (int i = axes.length - 1; i >= 0; i--) {
            out = out.squeeze(axes[i]);
        }
        return out;
    }

    private static Tensor unsqueeze(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 2);
        Tensor input = tensorInput(node, tensors, 0);
        int[] rawAxes = axes(node, tensors, int64Constants, attrs);
        if (rawAxes.length == 0) {
            throw unsupported(node, "Unsqueeze requires at least one axis");
        }
        int[] axes = normalizeAxes(rawAxes, input.getShapeUnsafe().length + rawAxes.length, node, "axes");
        Arrays.sort(axes);
        Tensor out = input;
        for (int axis : axes) {
            out = out.expandDims(axis);
        }
        return out;
    }

    private static Tensor reduce(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs,
            ReductionKind kind
    ) {
        requireInputCount(node, 1, 2);
        Tensor out = tensorInput(node, tensors, 0);
        boolean keepDims = attrs.intAttribute("keepdims", 1) != 0;
        int[] axes = node.getInputCount() >= 2 && !node.getInput(1).isBlank()
                ? toIntArray(intConstantInput(node, tensors, int64Constants, 1), node, "axes")
                : allAxes(out);
        axes = normalizeAxes(axes, out.getShapeUnsafe().length, node, "axes");
        Arrays.sort(axes);
        if (!keepDims) {
            for (int i = axes.length - 1; i >= 0; i--) {
                out = reduceOne(out, axes[i], false, kind);
            }
        } else {
            for (int axis : axes) {
                out = reduceOne(out, axis, true, kind);
            }
        }
        return out;
    }

    private static Tensor reduceOne(Tensor input, int axis, boolean keepDims, ReductionKind kind) {
        return switch (kind) {
            case SUM -> input.sum(axis, keepDims);
            case MEAN -> input.mean(axis, keepDims);
            case MAX -> input.max(axis, keepDims);
            case MIN -> input.min(axis, keepDims);
        };
    }

    private static Tensor unaryAxis(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            OnnxAttributeReader attrs,
            AxisOp op
    ) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        return op.apply(input, attrs.intAttribute("axis", -1));
    }

    private static Tensor unary(OnnxProto.NodeProto node, Map<String, Tensor> tensors, UnaryOp op) {
        requireInputCount(node, 1, 1);
        return op.apply(tensorInput(node, tensors, 0));
    }

    private static Tensor binary(OnnxProto.NodeProto node, Map<String, Tensor> tensors, BinaryOp op) {
        requireInputCount(node, 2, 2);
        return op.apply(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1));
    }

    private static Tensor ternary(OnnxProto.NodeProto node, Map<String, Tensor> tensors, TernaryOp op) {
        requireInputCount(node, 3, 3);
        return op.apply(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1), tensorInput(node, tensors, 2));
    }

    private static Tensor tensorInput(OnnxProto.NodeProto node, Map<String, Tensor> tensors, int index) {
        Tensor tensor = tensors.get(node.getInput(index));
        if (tensor == null) {
            throw unsupported(node, "input '" + node.getInput(index) + "' is missing or is not a tensor");
        }
        return tensor;
    }

    private static long[] intConstantInput(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            int index
    ) {
        String name = node.getInput(index);
        long[] int64 = int64Constants.get(name);
        if (int64 != null) {
            return int64;
        }
        Tensor tensor = tensors.get(name);
        if (tensor != null && tensor.getDataType() == DataType.INT32) {
            int[] values = tensor.getInt32Data();
            long[] out = new long[values.length];
            for (int i = 0; i < values.length; i++) {
                out[i] = values[i];
            }
            return out;
        }
        throw unsupported(node, "input '" + name + "' must be an INT64/INT32 constant");
    }

    private static double scalarConstantInput(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            int index
    ) {
        String name = node.getInput(index);
        long[] int64 = int64Constants.get(name);
        if (int64 != null) {
            if (int64.length != 1) {
                throw unsupported(node, "input '" + name + "' must be a scalar constant");
            }
            return int64[0];
        }
        if (!constantTensors.contains(name)) {
            throw unsupported(node, "input '" + name + "' must be a scalar initializer or Constant node");
        }
        Tensor tensor = tensors.get(name);
        if (tensor == null) {
            throw unsupported(node, "input '" + name + "' is missing or is not a tensor");
        }
        if (tensor.getFlatDataSize() != 1) {
            throw unsupported(node, "input '" + name + "' must be a scalar constant");
        }
        return tensor.scalarAsDouble();
    }

    private static int[] axes(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        if (node.getInputCount() >= 2 && !node.getInput(1).isBlank()) {
            return toIntArray(intConstantInput(node, tensors, int64Constants, 1), node, "axes");
        }
        int[] attrAxes = attrs.intsAttribute("axes");
        return attrAxes == null ? new int[0] : attrAxes;
    }

    private static int[] singletonAxes(Tensor input) {
        int count = 0;
        for (int dim : input.getShapeUnsafe()) {
            if (dim == 1) {
                count++;
            }
        }
        int[] axes = new int[count];
        for (int i = 0, j = 0; i < input.getShapeUnsafe().length; i++) {
            if (input.getShapeUnsafe()[i] == 1) {
                axes[j++] = i;
            }
        }
        return axes;
    }

    private static int[] allAxes(Tensor input) {
        int[] axes = new int[input.getShapeUnsafe().length];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i;
        }
        return axes;
    }

    private static int[] normalizeAxes(int[] axes, int rank, OnnxProto.NodeProto node, String field) {
        boolean[] seen = new boolean[rank];
        int[] out = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            int axis = axes[i] < 0 ? axes[i] + rank : axes[i];
            if (axis < 0 || axis >= rank) {
                throw unsupported(node, field + " contains out-of-range axis " + axes[i] + " for rank " + rank);
            }
            if (seen[axis]) {
                throw unsupported(node, field + " contains duplicate axis " + axis);
            }
            seen[axis] = true;
            out[i] = axis;
        }
        return out;
    }

    private static int[] toIntArray(long[] values, OnnxProto.NodeProto node, String field) {
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = Math.toIntExact(values[i]);
        }
        return out;
    }

    private static Tensor requireRank2(Tensor tensor, OnnxProto.NodeProto node, String attr) {
        if (tensor.getShapeUnsafe().length != 2) {
            throw unsupported(node, attr + " requires rank-2 tensor");
        }
        return tensor;
    }

    private static void requireInputCount(OnnxProto.NodeProto node, int min, int max) {
        int actual = node.getInputCount();
        if (actual < min || actual > max) {
            throw unsupported(node, "expected " + min + (min == max ? "" : ".." + max) + " inputs, got " + actual);
        }
    }

    private static OnnxUnsupportedException unsupported(OnnxProto.NodeProto node, String reason) {
        return new OnnxUnsupportedException("Unsupported ONNX node '" + nodeName(node)
                + "' (" + node.getOpType() + "): " + reason + ".");
    }

    private static String nodeName(OnnxProto.NodeProto node) {
        return node.getName().isBlank() ? node.getOutputList().toString() : node.getName();
    }

    private enum ReductionKind {
        SUM,
        MEAN,
        MAX,
        MIN
    }

    @FunctionalInterface
    private interface UnaryOp {
        Tensor apply(Tensor input);
    }

    @FunctionalInterface
    private interface BinaryOp {
        Tensor apply(Tensor left, Tensor right);
    }

    @FunctionalInterface
    private interface TernaryOp {
        Tensor apply(Tensor first, Tensor second, Tensor third);
    }

    @FunctionalInterface
    private interface AxisOp {
        Tensor apply(Tensor input, int axis);
    }
}
