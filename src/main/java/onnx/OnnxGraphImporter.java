package onnx;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import operations.reduction.ArgMaxTiePolicy;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorOps;
import operations.index.ScatterReduction;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

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
            "Neg", "Abs", "Relu", "LeakyRelu", "Elu", "HardSigmoid", "Softplus",
            "Tanh", "Sigmoid", "Exp", "Log", "Sqrt", "Reciprocal", "Erf", "Floor", "Ceil", "Sign",
            "Equal", "Greater", "GreaterOrEqual", "Less", "LessOrEqual",
            "And", "Or", "Not",
            "Where", "Identity", "Clip", "Cast",
            "MatMul", "Gemm",
            "Conv", "MaxPool", "AveragePool", "LayerNormalization", "BatchNormalization",
            "Transpose", "Reshape", "Flatten", "Expand", "Pad", "Tile", "ConstantOfShape", "Range", "Squeeze", "Unsqueeze", "Slice", "Concat", "Split", "Shape", "Size", "Gather", "GatherElements", "GatherND", "ScatterElements", "ScatterND",
            "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2", "ReduceLogSum", "ReduceLogSumExp", "ArgMax", "CumSum", "GlobalAveragePool",
            "Softmax", "LogSoftmax",
            "Constant"
    );

    private OnnxGraphImporter() {
    }

    static Set<String> supportedOps() {
        return SUPPORTED_OPS;
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
            }
            if (constant.int64Values() != null) {
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
        if (node.getOpType().equals("NonZero")) {
            throw unsupported(node, "runtime output shape depends on input values; NonZero requires dynamic-shape execution");
        }
        if (!SUPPORTED_OPS.contains(node.getOpType())) {
            throw unsupported(node, "unsupported op type");
        }
        OnnxAttributeReader attrs = new OnnxAttributeReader(node);
        if (node.getOpType().equals("Split")) {
            split(node, tensors, int64Constants, constantTensors, attrs);
            return;
        }
        if (node.getOutputCount() != 1) {
            throw unsupported(node, "only single-output nodes are supported");
        }

        Tensor out = switch (node.getOpType()) {
            case "Add" -> binary(node, tensors, TensorOps::add);
            case "Sub" -> binary(node, tensors, TensorOps::sub);
            case "Mul" -> binary(node, tensors, TensorOps::mul);
            case "Div" -> binary(node, tensors, TensorOps::div);
            case "Min" -> variadic(node, tensors, TensorOps::min);
            case "Max" -> variadic(node, tensors, TensorOps::max);
            case "Pow" -> pow(node, tensors, int64Constants, constantTensors);
            case "Neg" -> unary(node, tensors, TensorOps::neg);
            case "Abs" -> unary(node, tensors, TensorOps::abs);
            case "Relu" -> unary(node, tensors, TensorOps::relu);
            case "LeakyRelu" -> leakyRelu(node, tensors, attrs);
            case "Elu" -> elu(node, tensors, attrs);
            case "HardSigmoid" -> hardSigmoid(node, tensors, attrs);
            case "Softplus" -> softplus(node, tensors);
            case "Tanh" -> unary(node, tensors, TensorOps::tanh);
            case "Sigmoid" -> unary(node, tensors, TensorOps::sigmoid);
            case "Exp" -> unary(node, tensors, TensorOps::exp);
            case "Log" -> unary(node, tensors, TensorOps::log);
            case "Sqrt" -> unary(node, tensors, TensorOps::sqrt);
            case "Reciprocal" -> unary(node, tensors, TensorOps::inv);
            case "Erf" -> unary(node, tensors, TensorOps::erf);
            case "Floor" -> unary(node, tensors, TensorOps::floor);
            case "Ceil" -> unary(node, tensors, TensorOps::ceil);
            case "Sign" -> unary(node, tensors, TensorOps::sign);
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
            case "Cast" -> cast(node, tensors, int64Constants, attrs);
            case "MatMul" -> binary(node, tensors, TensorOps::matmul);
            case "Gemm" -> gemm(node, tensors, attrs);
            case "Conv" -> conv(node, tensors, attrs);
            case "MaxPool" -> maxPool(node, tensors, attrs);
            case "AveragePool" -> averagePool(node, tensors, attrs);
            case "LayerNormalization" -> layerNormalization(node, tensors, attrs);
            case "BatchNormalization" -> batchNormalization(node, tensors, attrs);
            case "Transpose" -> transpose(node, tensors, attrs);
            case "Reshape" -> reshape(node, tensors, int64Constants, constantTensors);
            case "Flatten" -> flatten(node, tensors, attrs);
            case "Expand" -> expand(node, tensors, int64Constants, constantTensors);
            case "Pad" -> pad(node, tensors, int64Constants, constantTensors, attrs);
            case "Tile" -> tile(node, tensors, int64Constants, constantTensors);
            case "ConstantOfShape" -> constantOfShape(node, tensors, int64Constants, constantTensors, attrs);
            case "Range" -> range(node, tensors, int64Constants, constantTensors);
            case "Squeeze" -> squeeze(node, tensors, int64Constants, constantTensors, attrs);
            case "Unsqueeze" -> unsqueeze(node, tensors, int64Constants, constantTensors, attrs);
            case "Slice" -> slice(node, tensors, int64Constants, constantTensors);
            case "Concat" -> concat(node, tensors, int64Constants, attrs);
            case "Shape" -> shape(node, tensors, int64Constants, attrs);
            case "Size" -> size(node, tensors, int64Constants);
            case "Gather" -> gather(node, tensors, int64Constants, constantTensors, attrs);
            case "GatherElements" -> gatherElements(node, tensors, int64Constants, attrs);
            case "GatherND" -> gatherNd(node, tensors, int64Constants, attrs);
            case "ScatterElements" -> scatterElements(node, tensors, int64Constants, attrs);
            case "ScatterND" -> scatterNd(node, tensors, int64Constants, attrs);
            case "ReduceSum" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.SUM);
            case "ReduceMean" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.MEAN);
            case "ReduceMax" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.MAX);
            case "ReduceMin" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.MIN);
            case "ReduceProd" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.PROD);
            case "ReduceL1" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.L1);
            case "ReduceL2" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.L2);
            case "ReduceLogSum" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.LOG_SUM);
            case "ReduceLogSumExp" -> reduce(node, tensors, int64Constants, constantTensors, attrs, ReductionKind.LOG_SUM_EXP);
            case "ArgMax" -> argMax(node, tensors, attrs);
            case "CumSum" -> cumSum(node, tensors, int64Constants, constantTensors, attrs);
            case "GlobalAveragePool" -> globalAveragePool(node, tensors);
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
        if (constant.int64Values() != null) {
            int64Constants.put(node.getOutput(0), constant.int64Values());
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
        Tensor base = tensorInput(node, tensors, 0);
        Tensor exponent = tensorInput(node, tensors, 1);
        if (constantTensors.contains(node.getInput(1)) && exponent.getFlatDataSize() == 1) {
            return TensorOps.pow(base, exponent.scalarAsDouble());
        }
        try {
            return TensorOps.pow(base, exponent);
        } catch (IllegalArgumentException e) {
            throw unsupported(node, e.getMessage());
        }
    }

    private static Tensor leakyRelu(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        double alpha = attrs.floatAttribute("alpha", 0.01f);
        Tensor zero = scalarLike(input, 0.0d);
        return TensorOps.where(
                TensorOps.greaterOrEqual(input, zero),
                input,
                TensorOps.mul(input, scalarLike(input, alpha))
        );
    }

    private static Tensor elu(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        double alpha = attrs.floatAttribute("alpha", 1.0f);
        Tensor zero = scalarLike(input, 0.0d);
        Tensor negative = TensorOps.mul(TensorOps.sub(TensorOps.exp(input), scalarLike(input, 1.0d)), scalarLike(input, alpha));
        return TensorOps.where(TensorOps.greaterOrEqual(input, zero), input, negative);
    }

    private static Tensor hardSigmoid(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        double alpha = attrs.floatAttribute("alpha", 0.2f);
        double beta = attrs.floatAttribute("beta", 0.5f);
        Tensor affine = TensorOps.add(TensorOps.mul(input, scalarLike(input, alpha)), scalarLike(input, beta));
        return TensorOps.clamp(affine, 0.0d, 1.0d);
    }

    private static Tensor softplus(OnnxProto.NodeProto node, Map<String, Tensor> tensors) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        return TensorOps.log(TensorOps.add(TensorOps.exp(input), scalarLike(input, 1.0d)));
    }

    private static Tensor scalarLike(Tensor input, double value) {
        return Tensor.scalar(value, input.getDataType());
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

    private static Tensor cast(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 1);
        int to = attrs.intAttribute("to", -1);
        if (to < 0) {
            throw unsupported(node, "Cast requires integer attribute 'to'");
        }
        String inputName = node.getInput(0);
        long[] shapeConstant = int64Constants.get(inputName);
        if (shapeConstant != null && !tensors.containsKey(inputName)) {
            if (!OnnxDataTypes.isInt64(to) && to != OnnxProto.TensorProto.DataType.INT32.getNumber()) {
                throw unsupported(node, "shape-only Cast supports only INT64/INT32 targets");
            }
            int64Constants.put(node.getOutput(0), shapeConstant.clone());
            return null;
        }
        return TensorOps.cast(tensorInput(node, tensors, 0), OnnxDataTypes.toSynaptik(to, "Cast node '" + nodeName(node) + "'"));
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

    private static Tensor conv(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 2, 3);
        rejectAutoPad(node, attrs);
        Tensor input = tensorInput(node, tensors, 0);
        Tensor weight = tensorInput(node, tensors, 1);
        Tensor bias = node.getInputCount() >= 3 && !node.getInput(2).isBlank() ? tensorInput(node, tensors, 2) : null;
        int[] weightShape = weight.getShapeUnsafe();
        if (weightShape.length != 4) {
            throw unsupported(node, "Conv currently requires rank-4 OIHW weights");
        }
        int[] kernel = pairAttribute(node, attrs, "kernel_shape", new int[]{weightShape[2], weightShape[3]});
        if (kernel[0] != weightShape[2] || kernel[1] != weightShape[3]) {
            throw unsupported(node, "Conv kernel_shape must match weight spatial dimensions");
        }
        int[] strides = pairAttribute(node, attrs, "strides", new int[]{1, 1});
        int[] dilations = pairAttribute(node, attrs, "dilations", new int[]{1, 1});
        int[] pads = pads4(node, attrs, "Conv");
        if (pads[0] != pads[2] || pads[1] != pads[3]) {
            input = input.pad(new int[]{0, 0, pads[0], pads[1]}, new int[]{0, 0, pads[2], pads[3]}, 0.0d);
            pads = new int[]{0, 0, 0, 0};
        }
        Conv2dOptions options = new Conv2dOptions(
                strides[0],
                strides[1],
                pads[0],
                pads[1],
                dilations[0],
                dilations[1],
                attrs.intAttribute("group", 1)
        );
        return bias == null ? input.conv2d(weight, options) : input.conv2d(weight, bias, options);
    }

    private static Tensor maxPool(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        return tensorInput(node, tensors, 0).maxPool2d(poolOptions(node, attrs, "MaxPool", false));
    }

    private static Tensor averagePool(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        return tensorInput(node, tensors, 0).avgPool2d(
                poolOptions(node, attrs, "AveragePool", attrs.intAttribute("count_include_pad", 0) != 0)
        );
    }

    private static Pool2dOptions poolOptions(
            OnnxProto.NodeProto node,
            OnnxAttributeReader attrs,
            String opName,
            boolean countIncludePad
    ) {
        rejectAutoPad(node, attrs);
        boolean ceilMode = attrs.intAttribute("ceil_mode", 0) != 0;
        if (attrs.intAttribute("storage_order", 0) != 0) {
            throw unsupported(node, opName + " storage_order must be 0");
        }
        int[] kernel = pairAttribute(node, attrs, "kernel_shape", null);
        if (kernel == null) {
            throw unsupported(node, opName + " requires kernel_shape");
        }
        int[] strides = pairAttribute(node, attrs, "strides", new int[]{1, 1});
        int[] pads = symmetricPads(node, attrs, opName);
        return new Pool2dOptions(kernel[0], kernel[1], strides[0], strides[1], pads[0], pads[1], countIncludePad, ceilMode);
    }

    private static Tensor layerNormalization(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 2, 3);
        Tensor input = tensorInput(node, tensors, 0);
        Tensor scale = tensorInput(node, tensors, 1);
        Tensor bias = node.getInputCount() >= 3 && !node.getInput(2).isBlank()
                ? tensorInput(node, tensors, 2)
                : Tensor.zerosLike(scale);
        int inputRank = input.getShapeUnsafe().length;
        int axis = attrs.intAttribute("axis", -1);
        int normalizedAxis = axis < 0 ? axis + inputRank : axis;
        if (normalizedAxis < 0 || normalizedAxis >= inputRank) {
            throw unsupported(node, "LayerNormalization axis out of range for rank " + inputRank + ": " + axis);
        }
        int normalizedRank = inputRank - normalizedAxis;
        int[] inputShape = input.getShapeUnsafe();
        int[] scaleShape = scale.getShapeUnsafe();
        if (scaleShape.length != normalizedRank) {
            throw unsupported(node, "LayerNormalization scale shape must match the normalized trailing axes");
        }
        for (int i = 0; i < normalizedRank; i++) {
            if (scaleShape[i] != inputShape[normalizedAxis + i]) {
                throw unsupported(node, "LayerNormalization scale shape must match the normalized trailing axes");
            }
        }
        if (!Arrays.equals(bias.getShapeUnsafe(), scaleShape)) {
            throw unsupported(node, "LayerNormalization bias shape must match scale shape");
        }
        return input.layerNorm(scale, bias, attrs.floatAttribute("epsilon", 1.0e-5f));
    }

    private static Tensor batchNormalization(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 5, 5);
        if (attrs.intAttribute("training_mode", 0) != 0) {
            throw unsupported(node, "BatchNormalization training_mode=1 is not supported");
        }
        return tensorInput(node, tensors, 0).batchNorm(
                tensorInput(node, tensors, 1),
                tensorInput(node, tensors, 2),
                tensorInput(node, tensors, 3),
                tensorInput(node, tensors, 4),
                1,
                attrs.floatAttribute("epsilon", 1.0e-5f)
        );
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

    private static Tensor reshape(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 2, 2);
        return tensorInput(node, tensors, 0).reshape(toIntArray(intConstantInput(node, tensors, int64Constants, constantTensors, 1), node, "shape"));
    }

    private static Tensor flatten(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        Tensor input = tensorInput(node, tensors, 0);
        int[] shape = input.getShapeUnsafe();
        int axis = attrs.intAttribute("axis", 1);
        if (axis < 0) {
            axis += shape.length;
        }
        if (axis < 0 || axis > shape.length) {
            throw unsupported(node, "Flatten axis out of range for rank " + shape.length + ": " + attrs.intAttribute("axis", 1));
        }
        int left = 1;
        for (int i = 0; i < axis; i++) {
            left = Math.multiplyExact(left, shape[i]);
        }
        int right = 1;
        for (int i = axis; i < shape.length; i++) {
            right = Math.multiplyExact(right, shape[i]);
        }
        return input.reshape(new int[]{left, right});
    }

    private static Tensor expand(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 2, 2);
        return tensorInput(node, tensors, 0).expand(toIntArray(intConstantInput(node, tensors, int64Constants, constantTensors, 1), node, "shape"));
    }

    private static Tensor pad(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 2, 3);
        String mode = attrs.stringAttribute("mode", "constant");
        if (!mode.equals("constant")) {
            throw unsupported(node, "Pad supports only mode='constant'");
        }
        Tensor input = tensorInput(node, tensors, 0);
        int rank = input.getShapeUnsafe().length;
        long[] padsRaw = intConstantInput(node, tensors, int64Constants, constantTensors, 1);
        if (padsRaw.length != rank * 2) {
            throw unsupported(node, "Pad pads length must be 2 * input rank");
        }
        int[] before = new int[rank];
        int[] after = new int[rank];
        for (int i = 0; i < rank; i++) {
            before[i] = Math.toIntExact(padsRaw[i]);
            after[i] = Math.toIntExact(padsRaw[i + rank]);
            if (before[i] < 0 || after[i] < 0) {
                throw unsupported(node, "Pad supports non-negative pads only");
            }
        }
        double value = 0.0d;
        if (node.getInputCount() >= 3 && !node.getInput(2).isBlank()) {
            value = scalarConstantInput(node, tensors, int64Constants, constantTensors, 2);
        }
        return TensorOps.pad(input, before, after, value);
    }

    private static Tensor tile(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 2, 2);
        Tensor input = tensorInput(node, tensors, 0);
        int[] repeats = staticIntVectorInput(node, tensors, int64Constants, constantTensors, 1, "repeats");
        if (repeats.length != input.getShapeUnsafe().length) {
            throw unsupported(node, "Tile repeats length must match input rank");
        }
        for (int repeat : repeats) {
            if (repeat <= 0) {
                throw unsupported(node, "Tile repeats must be positive");
            }
        }
        return TensorOps.tile(input, repeats);
    }

    private static Tensor constantOfShape(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 1);
        int[] shape = toIntArray(intConstantInput(node, tensors, int64Constants, constantTensors, 0), node, "shape");
        for (int dim : shape) {
            if (dim < 0) {
                throw unsupported(node, "ConstantOfShape shape dimensions must be non-negative");
            }
        }
        OnnxProto.TensorProto value = attrs.tensorAttribute("value");
        Tensor out = value == null
                ? filledConstant(shape, DataType.FLOAT32, 0.0d, node.getOutput(0))
                : constantOfShapeValue(node, value, shape);
        constantTensors.add(node.getOutput(0));
        return out;
    }

    private static Tensor constantOfShapeValue(OnnxProto.NodeProto node, OnnxProto.TensorProto value, int[] shape) {
        OnnxTensorProtoUtil.ImportedConstant constant = OnnxTensorProtoUtil.parseConstant(
                value,
                "ConstantOfShape node '" + nodeName(node) + "' value"
        );
        Tensor scalar = constant.tensor();
        if (scalar.getFlatDataSize() != 1) {
            throw unsupported(node, "ConstantOfShape value attribute must be scalar");
        }
        return filledConstant(shape, scalar.getDataType(), scalarConstantValue(scalar), node.getOutput(0));
    }

    private static double scalarConstantValue(Tensor scalar) {
        if (scalar.getDataType() == DataType.BOOL) {
            return scalar.toBooleanArrayCopy()[0] ? 1.0d : 0.0d;
        }
        return scalar.scalarAsDouble();
    }

    private static Tensor filledConstant(int[] shape, DataType dataType, double value, String label) {
        int count = 1;
        for (int dim : shape) {
            count = Math.multiplyExact(count, dim);
        }
        return switch (dataType) {
            case FLOAT64 -> {
                double[] data = new double[count];
                Arrays.fill(data, value);
                yield new Tensor(data, shape, null, label, DataType.FLOAT64);
            }
            case FLOAT32 -> {
                float[] data = new float[count];
                Arrays.fill(data, (float) value);
                yield new Tensor(data, shape, null, label, DataType.FLOAT32);
            }
            case BFLOAT16 -> {
                short[] data = new short[count];
                Arrays.fill(data, CpuDTypeOps.toBFloat16Bits((float) value));
                yield new Tensor(data, shape, null, label, DataType.BFLOAT16);
            }
            case INT32 -> {
                int[] data = new int[count];
                Arrays.fill(data, (int) value);
                yield new Tensor(data, shape, null, label, DataType.INT32);
            }
            case INT64 -> {
                long[] data = new long[count];
                Arrays.fill(data, (long) value);
                yield new Tensor(data, shape, null, label, DataType.INT64);
            }
            case BOOL -> {
                byte[] data = new byte[count];
                Arrays.fill(data, value == 0.0d ? (byte) 0 : (byte) 1);
                yield new Tensor(data, shape, null, label, DataType.BOOL);
            }
        };
    }

    private static Tensor range(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 3, 3);
        if (int64Constants.containsKey(node.getInput(0))
                && int64Constants.containsKey(node.getInput(1))
                && int64Constants.containsKey(node.getInput(2))) {
            long start = singleIntConstant(node, int64Constants.get(node.getInput(0)), "start");
            long limit = singleIntConstant(node, int64Constants.get(node.getInput(1)), "limit");
            long delta = singleIntConstant(node, int64Constants.get(node.getInput(2)), "delta");
            long[] values = longRange(node, start, limit, delta);
            int64Constants.put(node.getOutput(0), values);
            constantTensors.add(node.getOutput(0));
            return new Tensor(values, new int[]{values.length}, null, node.getOutput(0), DataType.INT64);
        }
        Tensor start = scalarTensorConstant(node, tensors, constantTensors, 0, "start");
        Tensor limit = scalarTensorConstant(node, tensors, constantTensors, 1, "limit");
        Tensor delta = scalarTensorConstant(node, tensors, constantTensors, 2, "delta");
        if (start.getDataType() != limit.getDataType() || start.getDataType() != delta.getDataType()) {
            throw unsupported(node, "Range tensor inputs must have matching dtypes");
        }
        if (start.getDataType() == DataType.BOOL) {
            throw unsupported(node, "Range BOOL inputs are unsupported");
        }
        Tensor out = tensorRange(node, start.scalarAsDouble(), limit.scalarAsDouble(), delta.scalarAsDouble(), start.getDataType());
        constantTensors.add(node.getOutput(0));
        return out;
    }

    private static long singleIntConstant(OnnxProto.NodeProto node, long[] values, String name) {
        if (values.length != 1) {
            throw unsupported(node, "Range " + name + " must be scalar");
        }
        return values[0];
    }

    private static long[] longRange(OnnxProto.NodeProto node, long start, long limit, long delta) {
        if (delta == 0) {
            throw unsupported(node, "Range delta cannot be zero");
        }
        java.util.ArrayList<Long> values = new java.util.ArrayList<>();
        if (delta > 0) {
            for (long value = start; value < limit; value += delta) {
                values.add(value);
                guardStaticRangeSize(node, values.size());
            }
        } else {
            for (long value = start; value > limit; value += delta) {
                values.add(value);
                guardStaticRangeSize(node, values.size());
            }
        }
        long[] out = new long[values.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = values.get(i);
        }
        return out;
    }

    private static Tensor tensorRange(OnnxProto.NodeProto node, double start, double limit, double delta, DataType dataType) {
        if (delta == 0.0d) {
            throw unsupported(node, "Range delta cannot be zero");
        }
        java.util.ArrayList<Double> values = new java.util.ArrayList<>();
        if (delta > 0.0d) {
            for (double value = start; value < limit; value += delta) {
                values.add(value);
                guardStaticRangeSize(node, values.size());
            }
        } else {
            for (double value = start; value > limit; value += delta) {
                values.add(value);
                guardStaticRangeSize(node, values.size());
            }
        }
        int[] shape = new int[]{values.size()};
        return switch (dataType) {
            case FLOAT64 -> {
                double[] data = new double[values.size()];
                for (int i = 0; i < data.length; i++) data[i] = values.get(i);
                yield new Tensor(data, shape, null, node.getOutput(0), DataType.FLOAT64);
            }
            case FLOAT32 -> {
                float[] data = new float[values.size()];
                for (int i = 0; i < data.length; i++) data[i] = values.get(i).floatValue();
                yield new Tensor(data, shape, null, node.getOutput(0), DataType.FLOAT32);
            }
            case BFLOAT16 -> {
                short[] data = new short[values.size()];
                for (int i = 0; i < data.length; i++) data[i] = CpuDTypeOps.toBFloat16Bits(values.get(i).floatValue());
                yield new Tensor(data, shape, null, node.getOutput(0), DataType.BFLOAT16);
            }
            case INT32 -> {
                int[] data = new int[values.size()];
                for (int i = 0; i < data.length; i++) data[i] = values.get(i).intValue();
                yield new Tensor(data, shape, null, node.getOutput(0), DataType.INT32);
            }
            case INT64 -> {
                long[] data = new long[values.size()];
                for (int i = 0; i < data.length; i++) data[i] = values.get(i).longValue();
                yield new Tensor(data, shape, null, node.getOutput(0), DataType.INT64);
            }
            case BOOL -> throw unsupported(node, "Range BOOL output is unsupported");
        };
    }

    private static void guardStaticRangeSize(OnnxProto.NodeProto node, int size) {
        if (size > 10_000_000) {
            throw unsupported(node, "Range static output is too large");
        }
    }

    private static Tensor scalarTensorConstant(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Set<String> constantTensors,
            int index,
            String name
    ) {
        String inputName = node.getInput(index);
        if (!constantTensors.contains(inputName)) {
            throw unsupported(node, "Range " + name + " must be a scalar initializer or Constant node");
        }
        Tensor tensor = tensors.get(inputName);
        if (tensor == null) {
            throw unsupported(node, "Range " + name + " is missing or is not a tensor");
        }
        if (tensor.getFlatDataSize() != 1) {
            throw unsupported(node, "Range " + name + " must be scalar");
        }
        return tensor;
    }

    private static Tensor squeeze(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 2);
        long[] shapeConstant = int64Constants.get(node.getInput(0));
        if (shapeConstant != null) {
            int64Constants.put(node.getOutput(0), shapeConstant.clone());
            return null;
        }
        Tensor input = tensorInput(node, tensors, 0);
        int[] axes = axes(node, tensors, int64Constants, constantTensors, attrs);
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
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 2);
        long[] shapeConstant = int64Constants.get(node.getInput(0));
        if (shapeConstant != null) {
            int64Constants.put(node.getOutput(0), shapeConstant.clone());
            return null;
        }
        Tensor input = tensorInput(node, tensors, 0);
        int[] rawAxes = axes(node, tensors, int64Constants, constantTensors, attrs);
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

    private static Tensor slice(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors
    ) {
        requireInputCount(node, 3, 5);
        long[] startsRaw = intConstantInput(node, tensors, int64Constants, constantTensors, 1);
        long[] endsRaw = intConstantInput(node, tensors, int64Constants, constantTensors, 2);
        long[] axesRaw = node.getInputCount() >= 4 && !node.getInput(3).isBlank()
                ? intConstantInput(node, tensors, int64Constants, constantTensors, 3)
                : new long[0];
        long[] stepsRaw = node.getInputCount() >= 5 && !node.getInput(4).isBlank()
                ? intConstantInput(node, tensors, int64Constants, constantTensors, 4)
                : new long[0];
        long[] shapeConstant = int64Constants.get(node.getInput(0));
        if (shapeConstant != null) {
            int64Constants.put(node.getOutput(0), sliceShapeConstant(node, shapeConstant, startsRaw, endsRaw, axesRaw, stepsRaw));
            return null;
        }
        Tensor input = tensorInput(node, tensors, 0);
        int[] starts = toSliceIntArray(startsRaw);
        int[] ends = toSliceIntArray(endsRaw);
        int[] axes = node.getInputCount() >= 4 && !node.getInput(3).isBlank()
                ? toSliceIntArray(axesRaw)
                : new int[0];
        int[] steps = node.getInputCount() >= 5 && !node.getInput(4).isBlank()
                ? toSliceIntArray(stepsRaw)
                : new int[0];
        try {
            return TensorOps.slice(input, starts, ends, axes, steps);
        } catch (IllegalArgumentException e) {
            throw unsupported(node, "Slice " + e.getMessage());
        }
    }

    private static Tensor concat(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, Integer.MAX_VALUE);
        int axis = attrs.intAttribute("axis", Integer.MIN_VALUE);
        if (axis == Integer.MIN_VALUE) {
            throw unsupported(node, "Concat requires axis attribute");
        }
        boolean allInt64 = node.getInputList().stream().allMatch(int64Constants::containsKey);
        boolean allTensor = node.getInputList().stream().allMatch(tensors::containsKey);
        if (allInt64) {
            if (axis != 0 && axis != -1) {
                throw unsupported(node, "shape-only Concat supports only axis 0");
            }
            int total = 0;
            for (String input : node.getInputList()) {
                total += int64Constants.get(input).length;
            }
            long[] out = new long[total];
            int p = 0;
            for (String input : node.getInputList()) {
                long[] values = int64Constants.get(input);
                System.arraycopy(values, 0, out, p, values.length);
                p += values.length;
            }
            int64Constants.put(node.getOutput(0), out);
            return null;
        }
        if (!allTensor) {
            throw unsupported(node, "Concat cannot mix shape constants and runtime tensors");
        }
        java.util.List<Tensor> inputs = new java.util.ArrayList<>(node.getInputCount());
        for (String input : node.getInputList()) {
            inputs.add(tensors.get(input));
        }
        return TensorOps.concat(axis, inputs);
    }

    private static void split(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 2);
        if (node.getOutputCount() < 1) {
            throw unsupported(node, "Split requires at least one output");
        }
        Tensor input = tensorInput(node, tensors, 0);
        int rank = input.getShapeUnsafe().length;
        int axis = attrs.intAttribute("axis", 0);
        axis = axis < 0 ? axis + rank : axis;
        if (axis < 0 || axis >= rank) {
            throw unsupported(node, "Split axis out of range for rank " + rank + ": " + attrs.intAttribute("axis", 0));
        }
        int[] splitSizes = splitSizes(node, tensors, int64Constants, constantTensors, attrs, input.getShapeUnsafe()[axis]);
        if (splitSizes.length != node.getOutputCount()) {
            throw unsupported(node, "Split size count must match output count");
        }
        int total = 0;
        for (int size : splitSizes) {
            if (size <= 0) {
                throw unsupported(node, "Split sizes must be positive");
            }
            total += size;
        }
        if (total != input.getShapeUnsafe()[axis]) {
            throw unsupported(node, "Split sizes must sum to the selected input dimension");
        }
        int offset = 0;
        for (int i = 0; i < node.getOutputCount(); i++) {
            int[] starts = new int[rank];
            int[] ends = input.getShape();
            int[] axes = allAxes(rank);
            int[] steps = ones(rank);
            starts[axis] = offset;
            ends[axis] = offset + splitSizes[i];
            Tensor part = TensorOps.slice(input, starts, ends, axes, steps);
            part.setLabel(node.getOutput(i));
            tensors.put(node.getOutput(i), part);
            offset += splitSizes[i];
        }
    }

    private static int[] splitSizes(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs,
            int axisSize
    ) {
        if (node.getInputCount() >= 2 && !node.getInput(1).isBlank()) {
            return toIntArray(intConstantInput(node, tensors, int64Constants, constantTensors, 1), node, "split");
        }
        int[] attrSplit = attrs.intsAttribute("split");
        if (attrSplit != null) {
            return attrSplit;
        }
        if (axisSize % node.getOutputCount() != 0) {
            throw unsupported(node, "Split without explicit sizes requires equal divisibility");
        }
        int[] out = new int[node.getOutputCount()];
        Arrays.fill(out, axisSize / node.getOutputCount());
        return out;
    }

    private static Tensor shape(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 1, 1);
        int[] shape = tensorInput(node, tensors, 0).getShapeUnsafe();
        int start = attrs.intAttribute("start", 0);
        int end = attrs.hasAttribute("end") ? attrs.intAttribute("end", shape.length) : shape.length;
        int[] range = normalizeShapeRange(start, end, shape.length);
        if (range[0] >= range[1]) {
            throw unsupported(node, "Shape start/end cannot produce an empty shape vector");
        }
        long[] out = new long[range[1] - range[0]];
        for (int i = 0; i < out.length; i++) {
            out[i] = shape[range[0] + i];
        }
        int64Constants.put(node.getOutput(0), out);
        return null;
    }

    private static Tensor size(OnnxProto.NodeProto node, Map<String, Tensor> tensors, Map<String, long[]> int64Constants) {
        requireInputCount(node, 1, 1);
        int64Constants.put(node.getOutput(0), new long[]{tensorInput(node, tensors, 0).getFlatDataSize()});
        return null;
    }

    private static Tensor gather(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 2, 2);
        long[] data = int64Constants.get(node.getInput(0));
        if (data != null) {
            if (attrs.intAttribute("axis", 0) != 0) {
                throw unsupported(node, "shape-only Gather supports only axis 0");
            }
            long[] indices = intConstantInput(node, Map.of(), int64Constants, constantTensors, 1);
            long[] out = new long[indices.length];
            for (int i = 0; i < indices.length; i++) {
                int index = Math.toIntExact(indices[i]);
                if (index < 0) {
                    index += data.length;
                }
                if (index < 0 || index >= data.length) {
                    throw unsupported(node, "Gather index out of bounds: " + indices[i]);
                }
                out[i] = data[index];
            }
            int64Constants.put(node.getOutput(0), out);
            return null;
        }
        return TensorOps.gatherAxis(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1), attrs.intAttribute("axis", 0));
    }

    private static Tensor gatherElements(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 2, 2);
        return TensorOps.takeAlongAxis(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1), attrs.intAttribute("axis", 0));
    }

    private static Tensor gatherNd(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 2, 2);
        int batchDims = attrs.intAttribute("batch_dims", 0);
        return TensorOps.gatherNd(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1), batchDims);
    }

    private static Tensor scatterElements(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 3, 3);
        return TensorOps.scatterElements(
                tensorInput(node, tensors, 0),
                tensorInput(node, tensors, 1),
                tensorInput(node, tensors, 2),
                attrs.intAttribute("axis", 0),
                scatterReduction(node, attrs.stringAttribute("reduction", "none"))
        );
    }

    private static Tensor scatterNd(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 3, 3);
        return TensorOps.scatterNd(
                tensorInput(node, tensors, 0),
                tensorInput(node, tensors, 1),
                tensorInput(node, tensors, 2),
                scatterReduction(node, attrs.stringAttribute("reduction", "none"))
        );
    }

    private static ScatterReduction scatterReduction(OnnxProto.NodeProto node, String value) {
        return switch (value) {
            case "none" -> ScatterReduction.NONE;
            case "add" -> ScatterReduction.ADD;
            case "mul" -> ScatterReduction.MUL;
            case "max" -> ScatterReduction.MAX;
            case "min" -> ScatterReduction.MIN;
            default -> throw unsupported(node, "unsupported scatter reduction '" + value + "'");
        };
    }

    private static Tensor reduce(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs,
            ReductionKind kind
    ) {
        requireInputCount(node, 1, 2);
        Tensor out = tensorInput(node, tensors, 0);
        if (kind == ReductionKind.L1) {
            out = out.abs();
        } else if (kind == ReductionKind.L2) {
            out = out.mul(out);
        } else if (kind == ReductionKind.LOG_SUM_EXP) {
            out = out.exp();
        }
        boolean keepDims = attrs.intAttribute("keepdims", 1) != 0;
        int[] axes = node.getInputCount() >= 2 && !node.getInput(1).isBlank()
                ? staticIntVectorInput(node, tensors, int64Constants, constantTensors, 1, "axes")
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
        if (kind == ReductionKind.L2) {
            out = out.sqrt();
        } else if (kind == ReductionKind.LOG_SUM || kind == ReductionKind.LOG_SUM_EXP) {
            out = out.log();
        }
        return out;
    }

    private static Tensor reduceOne(Tensor input, int axis, boolean keepDims, ReductionKind kind) {
        return switch (kind) {
            case SUM, L1, L2, LOG_SUM, LOG_SUM_EXP -> input.sum(axis, keepDims);
            case MEAN -> input.mean(axis, keepDims);
            case MAX -> input.max(axis, keepDims);
            case MIN -> input.min(axis, keepDims);
            case PROD -> input.prod(axis, keepDims);
        };
    }

    private static Tensor argMax(OnnxProto.NodeProto node, Map<String, Tensor> tensors, OnnxAttributeReader attrs) {
        requireInputCount(node, 1, 1);
        ArgMaxTiePolicy tiePolicy = attrs.intAttribute("select_last_index", 0) != 0
                ? ArgMaxTiePolicy.LAST_INDEX
                : ArgMaxTiePolicy.FIRST_INDEX;
        return tensorInput(node, tensors, 0).argMax(attrs.intAttribute("axis", 0), attrs.intAttribute("keepdims", 1) != 0, tiePolicy);
    }

    private static Tensor cumSum(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        requireInputCount(node, 2, 2);
        long[] axisValues = intConstantInput(node, tensors, int64Constants, constantTensors, 1);
        if (axisValues.length != 1) {
            throw unsupported(node, "CumSum axis input must be scalar");
        }
        return tensorInput(node, tensors, 0).cumSum(
                Math.toIntExact(axisValues[0]),
                attrs.intAttribute("exclusive", 0) != 0,
                attrs.intAttribute("reverse", 0) != 0
        );
    }

    private static Tensor globalAveragePool(OnnxProto.NodeProto node, Map<String, Tensor> tensors) {
        requireInputCount(node, 1, 1);
        Tensor out = tensorInput(node, tensors, 0);
        int rank = out.getShapeUnsafe().length;
        if (rank < 3) {
            throw unsupported(node, "GlobalAveragePool requires rank >= 3 with N,C plus spatial axes");
        }
        for (int axis = 2; axis < rank; axis++) {
            out = out.mean(axis, true);
        }
        return out;
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
        try {
            return op.apply(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1));
        } catch (IllegalArgumentException e) {
            throw unsupported(node, e.getMessage());
        }
    }

    private static Tensor variadic(OnnxProto.NodeProto node, Map<String, Tensor> tensors, BinaryOp op) {
        requireInputCount(node, 2, Integer.MAX_VALUE);
        try {
            Tensor out = tensorInput(node, tensors, 0);
            for (int i = 1; i < node.getInputCount(); i++) {
                out = op.apply(out, tensorInput(node, tensors, i));
            }
            return out;
        } catch (IllegalArgumentException e) {
            throw unsupported(node, e.getMessage());
        }
    }

    private static Tensor ternary(OnnxProto.NodeProto node, Map<String, Tensor> tensors, TernaryOp op) {
        requireInputCount(node, 3, 3);
        try {
            return op.apply(tensorInput(node, tensors, 0), tensorInput(node, tensors, 1), tensorInput(node, tensors, 2));
        } catch (IllegalArgumentException e) {
            throw unsupported(node, e.getMessage());
        }
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
            Set<String> constantTensors,
            int index
    ) {
        String name = node.getInput(index);
        long[] int64 = int64Constants.get(name);
        if (int64 != null) {
            return int64;
        }
        if (!constantTensors.contains(name)) {
            throw unsupported(node, "input '" + name + "' must be an INT64/INT32 constant initializer or Constant node");
        }
        Tensor tensor = tensors.get(name);
        if (tensor != null && tensor.getDataType() == DataType.INT32) {
            int[] values = TensorInternalAccess.int32Data(tensor);
            long[] out = new long[values.length];
            for (int i = 0; i < values.length; i++) {
                out[i] = values[i];
            }
            return out;
        }
        throw unsupported(node, "input '" + name + "' must be an INT64/INT32 constant initializer or Constant node");
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
            Set<String> constantTensors,
            OnnxAttributeReader attrs
    ) {
        if (node.getInputCount() >= 2 && !node.getInput(1).isBlank()) {
            return staticIntVectorInput(node, tensors, int64Constants, constantTensors, 1, "axes");
        }
        int[] attrAxes = attrs.intsAttribute("axes");
        return attrAxes == null ? new int[0] : attrAxes;
    }

    private static int[] staticIntVectorInput(
            OnnxProto.NodeProto node,
            Map<String, Tensor> tensors,
            Map<String, long[]> int64Constants,
            Set<String> constantTensors,
            int index,
            String role
    ) {
        return toIntArray(intConstantInput(node, tensors, int64Constants, constantTensors, index), node, role);
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

    private static int[] allAxes(int rank) {
        int[] axes = new int[rank];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = i;
        }
        return axes;
    }

    private static void rejectAutoPad(OnnxProto.NodeProto node, OnnxAttributeReader attrs) {
        String autoPad = attrs.stringAttribute("auto_pad", "NOTSET");
        if (!autoPad.isBlank() && !"NOTSET".equals(autoPad)) {
            throw unsupported(node, node.getOpType() + " auto_pad='" + autoPad + "' is not supported");
        }
    }

    private static int[] pairAttribute(OnnxProto.NodeProto node, OnnxAttributeReader attrs, String name, int[] defaultValue) {
        int[] values = attrs.intsAttribute(name);
        if (values == null) {
            return defaultValue == null ? null : defaultValue.clone();
        }
        if (values.length != 2) {
            throw unsupported(node, name + " must contain exactly two values");
        }
        return values;
    }

    private static int[] symmetricPads(OnnxProto.NodeProto node, OnnxAttributeReader attrs, String opName) {
        int[] pads = pads4(node, attrs, opName);
        if (pads[0] != pads[2] || pads[1] != pads[3]) {
            throw unsupported(node, opName + " supports only symmetric NCHW spatial pads");
        }
        return new int[]{pads[0], pads[1]};
    }

    private static int[] pads4(OnnxProto.NodeProto node, OnnxAttributeReader attrs, String opName) {
        int[] pads = attrs.intsAttribute("pads");
        if (pads == null) {
            return new int[]{0, 0, 0, 0};
        }
        if (pads.length != 4) {
            throw unsupported(node, opName + " pads must contain four values");
        }
        return pads;
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

    private static int[] toSliceIntArray(long[] values) {
        int[] out = new int[values.length];
        for (int i = 0; i < values.length; i++) {
            if (values[i] > Integer.MAX_VALUE) {
                out[i] = Integer.MAX_VALUE;
            } else if (values[i] < Integer.MIN_VALUE) {
                out[i] = Integer.MIN_VALUE;
            } else {
                out[i] = (int) values[i];
            }
        }
        return out;
    }

    private static long[] sliceShapeConstant(
            OnnxProto.NodeProto node,
            long[] values,
            long[] starts,
            long[] ends,
            long[] axes,
            long[] steps
    ) {
        if (starts.length != ends.length) {
            throw unsupported(node, "Slice starts and ends length mismatch");
        }
        int count = starts.length;
        long[] normalizedAxes = axes.length == 0 ? defaultShapeSliceAxes(count) : axes.clone();
        long[] normalizedSteps = steps.length == 0 ? onesLong(count) : steps.clone();
        if (normalizedAxes.length != count || normalizedSteps.length != count) {
            throw unsupported(node, "Slice starts, ends, axes, and steps must have matching lengths");
        }
        if (count == 0) {
            return values.clone();
        }
        if (count != 1) {
            throw unsupported(node, "shape-only Slice supports one-dimensional shape vectors only");
        }
        int axis = normalizeShapeVectorAxis(node, normalizedAxes[0]);
        if (axis != 0) {
            throw unsupported(node, "shape-only Slice supports only axis 0");
        }
        long step = normalizedSteps[0];
        if (step <= 0 || step > Integer.MAX_VALUE) {
            throw unsupported(node, "shape-only Slice supports positive int-sized steps only");
        }
        int[] range = normalizeShapeRange(saturatingInt(starts[0]), saturatingInt(ends[0]), values.length);
        int length = range[0] >= range[1] ? 0 : ((range[1] - range[0] + (int) step - 1) / (int) step);
        if (length <= 0) {
            throw unsupported(node, "shape-only Slice cannot produce an empty shape vector");
        }
        long[] out = new long[length];
        for (int i = 0, p = range[0]; i < length; i++, p += (int) step) {
            out[i] = values[p];
        }
        return out;
    }

    private static int[] normalizeShapeRange(int start, int end, int length) {
        int normalizedStart = start < 0 ? start + length : start;
        int normalizedEnd = end < 0 ? end + length : end;
        normalizedStart = Math.max(0, Math.min(normalizedStart, length));
        normalizedEnd = Math.max(0, Math.min(normalizedEnd, length));
        if (normalizedEnd < normalizedStart) {
            normalizedEnd = normalizedStart;
        }
        return new int[]{normalizedStart, normalizedEnd};
    }

    private static int normalizeShapeVectorAxis(OnnxProto.NodeProto node, long axis) {
        long normalized = axis < 0 ? axis + 1 : axis;
        if (normalized < 0 || normalized >= 1) {
            throw unsupported(node, "shape-only Slice axis out of range: " + axis);
        }
        return (int) normalized;
    }

    private static long[] defaultShapeSliceAxes(int count) {
        long[] out = new long[count];
        for (int i = 0; i < count; i++) {
            out[i] = i;
        }
        return out;
    }

    private static long[] onesLong(int count) {
        long[] out = new long[count];
        Arrays.fill(out, 1L);
        return out;
    }

    private static int[] ones(int count) {
        int[] out = new int[count];
        Arrays.fill(out, 1);
        return out;
    }

    private static int saturatingInt(long value) {
        if (value > Integer.MAX_VALUE) {
            return Integer.MAX_VALUE;
        }
        if (value < Integer.MIN_VALUE) {
            return Integer.MIN_VALUE;
        }
        return (int) value;
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
        MIN,
        PROD,
        L1,
        L2,
        LOG_SUM,
        LOG_SUM_EXP
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
