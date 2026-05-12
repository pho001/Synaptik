package onnx;

import operations.Operation;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import operations.layout.expandDims;
import operations.layout.permute;
import operations.layout.reshape;
import operations.layout.squeeze;
import operations.linalg.linear;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

final class OnnxGraphExporter {
    private OnnxGraphExporter() {
    }

    static OnnxProto.ModelProto export(Tensor output, OnnxExportOptions options) {
        Objects.requireNonNull(output, "output cannot be null");
        options = options == null ? OnnxExportOptions.defaults() : options;

        List<Tensor> graph = output.topologicalSort();
        OnnxNameRegistry names = new OnnxNameRegistry();
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < graph.size(); i++) {
            ids.put(graph.get(i), i);
            names.nameFor(graph.get(i), i);
        }

        OnnxProto.GraphProto.Builder graphBuilder = OnnxProto.GraphProto.newBuilder()
                .setName(options.graphName());
        for (int i = 0; i < graph.size(); i++) {
            Tensor tensor = graph.get(i);
            String name = names.nameFor(tensor, i);
            Operation op = tensor.getOperation();
            if (op == null) {
                if (exportLeafAsInput(tensor, options.leafTensorPolicy())) {
                    graphBuilder.addInput(OnnxTensorProtoUtil.valueInfo(name, tensor.getDataType(), tensor.getShapeUnsafe()));
                } else {
                    graphBuilder.addInitializer(OnnxTensorProtoUtil.tensorInitializer(name, tensor));
                }
                continue;
            }
            graphBuilder.addNode(nodeFor(tensor, name, ids, names, graphBuilder, op));
        }
        graphBuilder.addOutput(OnnxTensorProtoUtil.valueInfo(
                names.nameFor(output, ids.get(output)),
                output.getDataType(),
                output.getShapeUnsafe()
        ));

        return OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .setProducerName(options.producerName())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder()
                        .setDomain("")
                        .setVersion(options.opsetVersion()))
                .setGraph(graphBuilder)
                .build();
    }

    private static boolean exportLeafAsInput(Tensor tensor, OnnxLeafTensorPolicy policy) {
        return switch (policy) {
            case INPUTS -> true;
            case INITIALIZERS -> false;
            case TRAINABLE_INPUTS -> tensor.getRequiresGrad();
        };
    }

    private static OnnxProto.NodeProto nodeFor(
            Tensor tensor,
            String outputName,
            IdentityHashMap<Tensor, Integer> ids,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder,
            Operation op
    ) {
        List<Tensor> inputs = tensor.getPrevTensors();
        OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                .setName("node_" + ids.get(tensor))
                .addOutput(outputName);
        for (Tensor input : inputs) {
            node.addInput(names.nameFor(input, ids.get(input)));
        }

        switch (op.opType()) {
            case ADD -> node.setOpType("Add");
            case SUB -> node.setOpType("Sub");
            case MUL -> node.setOpType("Mul");
            case DIV -> node.setOpType("Div");
            case MIN -> node.setOpType("Min");
            case MAX -> node.setOpType("Max");
            case NEG -> node.setOpType("Neg");
            case ABS -> node.setOpType("Abs");
            case RELU -> node.setOpType("Relu");
            case TANH -> node.setOpType("Tanh");
            case SIGMOID -> node.setOpType("Sigmoid");
            case EXP -> node.setOpType("Exp");
            case LOG -> node.setOpType("Log");
            case SQRT -> node.setOpType("Sqrt");
            case POW -> exportPow(node, op, tensor, names, graphBuilder);
            case CLAMP_MIN -> exportClampMin(node, op, tensor, names, graphBuilder);
            case CLAMP_MAX -> exportClampMax(node, op, tensor, names, graphBuilder);
            case EQ -> node.setOpType("Equal");
            case GT -> node.setOpType("Greater");
            case GE -> node.setOpType("GreaterOrEqual");
            case LT -> node.setOpType("Less");
            case LE -> node.setOpType("LessOrEqual");
            case LOGICAL_AND -> node.setOpType("And");
            case LOGICAL_OR -> node.setOpType("Or");
            case LOGICAL_NOT -> node.setOpType("Not");
            case WHERE -> node.setOpType("Where");
            case MATMUL -> node.setOpType("MatMul");
            case LINEAR -> exportLinear(node, op);
            case PERMUTE -> exportPermute(node, op);
            case RESHAPE -> exportReshape(node, op, names, graphBuilder);
            case SQUEEZE -> exportSqueeze(node, op, names, graphBuilder);
            case EXPAND_DIMS -> exportUnsqueeze(node, op, names, graphBuilder);
            case SUM -> exportReduction(node, "ReduceSum", ((sum) op).getDimension(), ((sum) op).keepDims(), names, graphBuilder);
            case MEAN -> exportReduction(node, "ReduceMean", ((mean) op).getDimension(), ((mean) op).keepDims(), names, graphBuilder);
            case REDUCE_MAX -> exportReduction(node, "ReduceMax", ((reduceMax) op).getDimension(), ((reduceMax) op).keepDims(), names, graphBuilder);
            case REDUCE_MIN -> exportReduction(node, "ReduceMin", ((reduceMin) op).getDimension(), ((reduceMin) op).keepDims(), names, graphBuilder);
            case SOFTMAX -> exportAxisOp(node, "Softmax", ((softmax) op).getDimension());
            case LOG_SOFTMAX -> exportAxisOp(node, "LogSoftmax", ((logSoftmax) op).getDimension());
            case MUL_SCALAR -> exportMulScalar(node, op, tensor, names, graphBuilder);
            default -> throw new OnnxUnsupportedException("Cannot export operation " + op.opType()
                    + " from tensor '" + tensor.getLabel() + "'.");
        }
        return node.build();
    }

    private static void exportLinear(OnnxProto.NodeProto.Builder node, Operation op) {
        linear linear = (linear) op;
        node.setOpType("Gemm");
        if (!linear.hasBias() && node.getInputCount() != 2) {
            throw new OnnxUnsupportedException("Linear without bias must have two inputs.");
        }
        if (linear.hasBias() && node.getInputCount() != 3) {
            throw new OnnxUnsupportedException("Linear with bias must have three inputs.");
        }
    }

    private static void exportPermute(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("Transpose")
                .addAttribute(intsAttr("perm", ((permute) op).getAxes()));
    }

    private static void exportReshape(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Reshape");
        String shapeName = names.auxiliary(node.getOutput(0) + "_shape");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(shapeName, toLong(((reshape) op).getTargetShape())));
        node.addInput(shapeName);
    }

    private static void exportSqueeze(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Squeeze");
        String axesName = names.auxiliary(node.getOutput(0) + "_axes");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(axesName, new long[]{((squeeze) op).getAxis()}));
        node.addInput(axesName);
    }

    private static void exportUnsqueeze(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Unsqueeze");
        String axesName = names.auxiliary(node.getOutput(0) + "_axes");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(axesName, new long[]{((expandDims) op).getAxis()}));
        node.addInput(axesName);
    }

    private static void exportReduction(
            OnnxProto.NodeProto.Builder node,
            String opType,
            int axis,
            boolean keepDims,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType(opType)
                .addAttribute(intAttr("keepdims", keepDims ? 1 : 0));
        if (axis >= 0) {
            String axesName = names.auxiliary(node.getOutput(0) + "_axes");
            graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(axesName, new long[]{axis}));
            node.addInput(axesName);
        }
    }

    private static void exportAxisOp(OnnxProto.NodeProto.Builder node, String opType, int axis) {
        node.setOpType(opType)
                .addAttribute(intAttr("axis", axis));
    }

    private static void exportMulScalar(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Mul");
        mulScalar scalar = (mulScalar) op;
        String scalarName = names.auxiliary(node.getOutput(0) + "_scalar");
        Tensor scalarTensor = Tensor.scalar(scalar.getScalar(), tensor.getDataType());
        graphBuilder.addInitializer(OnnxTensorProtoUtil.tensorInitializer(scalarName, scalarTensor));
        node.addInput(scalarName);
    }

    private static void exportPow(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Pow");
        pow pow = (pow) op;
        String exponentName = names.auxiliary(node.getOutput(0) + "_exponent");
        graphBuilder.addInitializer(scalarInitializer(exponentName, pow.getExponent(), tensor.getDataType()));
        node.addInput(exponentName);
    }

    private static void exportClampMin(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Clip");
        clampMin clamp = (clampMin) op;
        String minName = names.auxiliary(node.getOutput(0) + "_min");
        graphBuilder.addInitializer(scalarInitializer(minName, clamp.getMinValue(), tensor.getDataType()));
        node.addInput(minName);
    }

    private static void exportClampMax(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Clip");
        clampMax clamp = (clampMax) op;
        String maxName = names.auxiliary(node.getOutput(0) + "_max");
        graphBuilder.addInitializer(scalarInitializer(maxName, clamp.getMaxValue(), tensor.getDataType()));
        node.addInput("");
        node.addInput(maxName);
    }

    private static OnnxProto.TensorProto scalarInitializer(String name, double value, DataType dataType) {
        return OnnxTensorProtoUtil.tensorInitializer(name, Tensor.scalar(value, dataType));
    }

    private static OnnxProto.AttributeProto intAttr(String name, long value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setI(value).build();
    }

    private static OnnxProto.AttributeProto intsAttr(String name, int[] values) {
        OnnxProto.AttributeProto.Builder builder = OnnxProto.AttributeProto.newBuilder().setName(name);
        for (int value : values) {
            builder.addInts(value);
        }
        return builder.build();
    }

    private static long[] toLong(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }
}
