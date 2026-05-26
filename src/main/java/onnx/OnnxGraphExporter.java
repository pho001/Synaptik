package onnx;

import operations.Operation;
import operations.dtype.cast;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.elementwise.unary.mulScalar;
import operations.elementwise.unary.pow;
import operations.index.ScatterReduction;
import operations.index.gatherAxis;
import operations.index.gatherNd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.index.takeAlongAxis;
import operations.layout.concat;
import operations.layout.expandDims;
import operations.layout.expand;
import operations.layout.fold2d;
import operations.layout.pad;
import operations.layout.permute;
import operations.layout.reshape;
import operations.layout.squeeze;
import operations.layout.slice;
import operations.layout.tile;
import operations.linalg.linear;
import operations.nn.conv.conv2d;
import operations.nn.pool.avgPool2d;
import operations.nn.pool.maxPool2d;
import operations.reduction.cumSum;
import operations.normalization.layerNorm;
import operations.reduction.logSoftmax;
import operations.reduction.mean;
import operations.reduction.argMax;
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.reduceMax;
import operations.reduction.reduceMin;
import operations.reduction.reduceProd;
import operations.reduction.softmax;
import operations.reduction.sum;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;
import tensor.options.Window2dOptions;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

final class OnnxGraphExporter {
    private OnnxGraphExporter() {
    }

    static OnnxProto.ModelProto export(Tensor output, OnnxExportOptions options) {
        Objects.requireNonNull(output, "output cannot be null");
        return export(List.of(output), options);
    }

    static OnnxProto.ModelProto export(List<Tensor> outputs, OnnxExportOptions options) {
        Objects.requireNonNull(outputs, "outputs cannot be null");
        if (outputs.isEmpty()) {
            throw new IllegalArgumentException("outputs cannot be empty");
        }
        for (Tensor output : outputs) {
            Objects.requireNonNull(output, "outputs cannot contain null tensors");
        }
        options = options == null ? OnnxExportOptions.defaults() : options;

        List<Tensor> graph = topologicalSort(outputs);
        OnnxNameRegistry names = new OnnxNameRegistry();
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < graph.size(); i++) {
            ids.put(graph.get(i), i);
            names.nameFor(graph.get(i), i);
        }
        IdentityHashMap<Tensor, Integer> consumerCounts = consumerCounts(graph);
        Set<Tensor> graphOutputs = Collections.newSetFromMap(new IdentityHashMap<>());
        graphOutputs.addAll(outputs);
        OnnxExportPatternContext patternContext = new OnnxExportPatternContext(graphOutputs, consumerCounts, ids, names);
        IdentityHashMap<Tensor, OnnxExportPatternMatch> patternMatches = new IdentityHashMap<>();
        Set<Tensor> patternConsumed = Collections.newSetFromMap(new IdentityHashMap<>());
        addSplitOutputPatterns(outputs, ids, names, consumerCounts, patternMatches, patternConsumed);
        for (Tensor tensor : graph) {
            if (tensor.getOperation() == null) {
                continue;
            }
            if (patternMatches.containsKey(tensor) || patternConsumed.contains(tensor)) {
                continue;
            }
            Optional<OnnxExportPatternMatch> match = OnnxExportPatternRegistry.match(tensor, patternContext);
            if (match.isPresent()) {
                patternMatches.put(tensor, match.get());
                patternConsumed.addAll(match.get().consumedTensors());
            }
        }

        OnnxProto.GraphProto.Builder graphBuilder = OnnxProto.GraphProto.newBuilder()
                .setName(options.graphName());
        for (int i = 0; i < graph.size(); i++) {
            Tensor tensor = graph.get(i);
            if (patternConsumed.contains(tensor)) {
                continue;
            }
            String name = names.nameFor(tensor, i);
            Operation op = tensor.getOperation();
            if (op == null) {
                exportLeaf(tensor, name, options.leafTensorPolicy(), graphBuilder);
                continue;
            }
            OnnxExportPatternMatch patternMatch = patternMatches.get(tensor);
            if (patternMatch != null) {
                for (OnnxProto.TensorProto initializer : patternMatch.initializers()) {
                    graphBuilder.addInitializer(initializer);
                }
                graphBuilder.addNode(patternMatch.node());
                continue;
            }
            graphBuilder.addNode(nodeFor(tensor, name, ids, names, graphBuilder, op));
        }
        for (Tensor output : outputs) {
            graphBuilder.addOutput(OnnxTensorProtoUtil.valueInfo(
                    names.nameFor(output, ids.get(output)),
                    output.getDataType(),
                    output.getShapeUnsafe()
            ));
        }

        return OnnxProto.ModelProto.newBuilder()
                .setIrVersion(OnnxProto.Version.IR_VERSION_2023_5_5.getNumber())
                .setProducerName(options.producerName())
                .addOpsetImport(OnnxProto.OperatorSetIdProto.newBuilder()
                        .setDomain("")
                        .setVersion(options.opsetVersion()))
                .setGraph(graphBuilder)
                .build();
    }

    private static void exportLeaf(
            Tensor tensor,
            String name,
            OnnxLeafTensorPolicy policy,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        switch (policy) {
            case INPUTS -> graphBuilder.addInput(OnnxTensorProtoUtil.valueInfo(name, tensor.getDataType(), tensor.getShapeUnsafe()));
            case INITIALIZERS -> graphBuilder.addInitializer(OnnxTensorProtoUtil.tensorInitializer(name, tensor));
            case CONSTANT_NODES -> graphBuilder.addNode(constantNode(name, tensor));
            case TRAINABLE_INPUTS -> {
                if (tensor.getRequiresGrad()) {
                    graphBuilder.addInput(OnnxTensorProtoUtil.valueInfo(name, tensor.getDataType(), tensor.getShapeUnsafe()));
                } else {
                    graphBuilder.addInitializer(OnnxTensorProtoUtil.tensorInitializer(name, tensor));
                }
            }
        }
    }

    private static OnnxProto.NodeProto constantNode(String outputName, Tensor tensor) {
        return OnnxProto.NodeProto.newBuilder()
                .setName("const_" + outputName)
                .setOpType("Constant")
                .addOutput(outputName)
                .addAttribute(OnnxProto.AttributeProto.newBuilder()
                        .setName("value")
                        .setT(OnnxTensorProtoUtil.tensorInitializer(outputName + "_value", tensor)))
                .build();
    }

    private static IdentityHashMap<Tensor, Integer> consumerCounts(List<Tensor> graph) {
        IdentityHashMap<Tensor, Integer> counts = new IdentityHashMap<>();
        for (Tensor tensor : graph) {
            for (Tensor input : tensor.getPrevTensors()) {
                counts.merge(input, 1, Integer::sum);
            }
        }
        return counts;
    }

    private static List<Tensor> topologicalSort(List<Tensor> outputs) {
        Deque<Tensor> sorted = new ArrayDeque<>();
        Set<Tensor> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Tensor output : outputs) {
            visit(output, visited, sorted);
        }
        return new ArrayList<>(sorted);
    }

    private static void visit(Tensor tensor, Set<Tensor> visited, Deque<Tensor> sorted) {
        if (visited.contains(tensor)) {
            return;
        }
        visited.add(tensor);
        List<Tensor> previous = tensor.getPrevTensors();
        if (previous != null) {
            for (Tensor prev : previous) {
                visit(prev, visited, sorted);
            }
        }
        if (previous == null) {
            sorted.addFirst(tensor);
        } else {
            sorted.addLast(tensor);
        }
    }

    private static void addSplitOutputPatterns(
            List<Tensor> outputs,
            IdentityHashMap<Tensor, Integer> ids,
            OnnxNameRegistry names,
            IdentityHashMap<Tensor, Integer> consumerCounts,
            IdentityHashMap<Tensor, OnnxExportPatternMatch> patternMatches,
            Set<Tensor> patternConsumed
    ) {
        Map<SplitGroupKey, List<SplitPart>> groups = new HashMap<>();
        for (Tensor output : outputs) {
            if (!(output.getOperation() instanceof slice sliceOp) || output.getPrevTensors().size() != 1) {
                continue;
            }
            SplitPart part = splitPart(output, sliceOp);
            if (part == null) {
                continue;
            }
            if (consumerCounts.getOrDefault(output, 0) != 0) {
                continue;
            }
            groups.computeIfAbsent(new SplitGroupKey(output.getPrevTensors().get(0), part.axis()), ignored -> new ArrayList<>())
                    .add(part);
        }
        for (List<SplitPart> parts : groups.values()) {
            if (parts.size() < 2) {
                continue;
            }
            parts.sort(Comparator.comparingInt(SplitPart::start));
            if (!coversSingleAxis(parts)) {
                continue;
            }
            Tensor key = parts.stream()
                    .min(Comparator.comparingInt(part -> ids.get(part.tensor())))
                    .orElseThrow()
                    .tensor();
            SplitGroupKey groupKey = new SplitGroupKey(key.getPrevTensors().get(0), parts.getFirst().axis());
            OnnxProto.NodeProto.Builder node = OnnxProto.NodeProto.newBuilder()
                    .setName("node_" + ids.get(key))
                    .setOpType("Split")
                    .addInput(names.nameFor(groupKey.input(), ids.get(groupKey.input())));
            String splitName = names.auxiliary(names.nameFor(key, ids.get(key)) + "_split");
            node.addInput(splitName)
                    .addAttribute(intAttr("axis", groupKey.axis()));
            for (SplitPart part : parts) {
                node.addOutput(names.nameFor(part.tensor(), ids.get(part.tensor())));
            }
            List<OnnxProto.TensorProto> initializers = List.of(OnnxTensorProtoUtil.int64Initializer(
                    splitName,
                    parts.stream().mapToLong(SplitPart::size).toArray()
            ));
            Set<Tensor> consumed = Collections.newSetFromMap(new IdentityHashMap<>());
            for (SplitPart part : parts) {
                if (part.tensor() != key) {
                    consumed.add(part.tensor());
                }
            }
            patternMatches.put(key, new OnnxExportPatternMatch(node.build(), consumed, initializers));
            patternConsumed.addAll(consumed);
        }
    }

    private static SplitPart splitPart(Tensor tensor, slice sliceOp) {
        int[] inputShape = tensor.getPrevTensors().get(0).getShapeUnsafe();
        int[] starts = sliceOp.getStarts();
        int[] ends = sliceOp.getEnds();
        int[] axes = sliceOp.getAxes();
        int[] steps = sliceOp.getSteps();
        if (starts.length != inputShape.length || ends.length != inputShape.length
                || axes.length != inputShape.length || steps.length != inputShape.length) {
            return null;
        }
        int splitAxis = -1;
        for (int i = 0; i < inputShape.length; i++) {
            if (axes[i] != i || steps[i] != 1) {
                return null;
            }
            if (starts[i] == 0 && ends[i] == inputShape[i]) {
                continue;
            }
            if (splitAxis >= 0) {
                return null;
            }
            splitAxis = i;
        }
        if (splitAxis < 0 || starts[splitAxis] < 0 || ends[splitAxis] <= starts[splitAxis]
                || ends[splitAxis] > inputShape[splitAxis]) {
            return null;
        }
        return new SplitPart(tensor, splitAxis, starts[splitAxis], ends[splitAxis]);
    }

    private static boolean coversSingleAxis(List<SplitPart> parts) {
        Tensor input = parts.getFirst().tensor().getPrevTensors().get(0);
        int axis = parts.getFirst().axis();
        int offset = 0;
        for (SplitPart part : parts) {
            if (part.tensor().getPrevTensors().get(0) != input || part.axis() != axis || part.start() != offset) {
                return false;
            }
            offset = part.end();
        }
        return offset == input.getShapeUnsafe()[axis];
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
            case INV -> node.setOpType("Reciprocal");
            case ERF -> node.setOpType("Erf");
            case FLOOR -> node.setOpType("Floor");
            case CEIL -> node.setOpType("Ceil");
            case SIGN -> node.setOpType("Sign");
            case POW -> exportPow(node, op, tensor, names, graphBuilder);
            case POW_TENSOR -> node.setOpType("Pow");
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
            case CONV2D -> exportConv(node, op, tensor);
            case MAX_POOL2D -> exportMaxPool(node, op);
            case AVG_POOL2D -> exportAveragePool(node, op);
            case FOLD2D -> exportCol2Im(node, op, names, graphBuilder);
            case LAYER_NORM -> exportLayerNormalization(node, op, tensor);
            case PERMUTE -> exportPermute(node, op);
            case RESHAPE -> exportReshape(node, op, tensor, names, graphBuilder);
            case EXPAND -> exportExpand(node, op, names, graphBuilder);
            case SLICE -> exportSlice(node, op, names, graphBuilder);
            case CONCAT -> exportConcat(node, op);
            case PAD -> exportPad(node, op, tensor, names, graphBuilder);
            case TILE -> exportTile(node, op, names, graphBuilder);
            case CAST -> exportCast(node, op);
            case GATHER_AXIS -> exportGatherAxis(node, op);
            case GATHER_ND -> exportGatherNd(node, op);
            case TAKE_ALONG_AXIS -> exportGatherElements(node, op);
            case SCATTER_ELEMENTS -> exportScatterElements(node, op);
            case SCATTER_ND -> exportScatterNd(node, op);
            case SQUEEZE -> exportSqueeze(node, op, names, graphBuilder);
            case EXPAND_DIMS -> exportUnsqueeze(node, op, names, graphBuilder);
            case SUM -> exportReduction(node, "ReduceSum", ((sum) op).getDimension(), ((sum) op).keepDims(), names, graphBuilder);
            case MEAN -> exportReduction(node, "ReduceMean", ((mean) op).getDimension(), ((mean) op).keepDims(), names, graphBuilder);
            case REDUCE_MAX -> exportReduction(node, "ReduceMax", ((reduceMax) op).getDimension(), ((reduceMax) op).keepDims(), names, graphBuilder);
            case REDUCE_MIN -> exportReduction(node, "ReduceMin", ((reduceMin) op).getDimension(), ((reduceMin) op).keepDims(), names, graphBuilder);
            case REDUCE_PROD -> exportReduction(node, "ReduceProd", ((reduceProd) op).getDimension(), ((reduceProd) op).keepDims(), names, graphBuilder);
            case CUMSUM -> exportCumSum(node, op, names, graphBuilder);
            case ARGMAX -> exportArgMax(node, op);
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

    private static void exportConv(OnnxProto.NodeProto.Builder node, Operation op, Tensor tensor) {
        conv2d conv = (conv2d) op;
        if (!conv.hasBias() && node.getInputCount() != 2) {
            throw new OnnxUnsupportedException("Conv2D without bias must have two inputs.");
        }
        if (conv.hasBias() && node.getInputCount() != 3) {
            throw new OnnxUnsupportedException("Conv2D with bias must have three inputs.");
        }
        Tensor weight = tensor.getPrevTensors().get(1);
        int[] weightShape = weight.getShapeUnsafe();
        if (weightShape.length != 4) {
            throw new OnnxUnsupportedException("Conv2D export requires rank-4 OIHW weights.");
        }
        Conv2dOptions options = conv.getOptions();
        node.setOpType("Conv")
                .addAttribute(intsAttr("kernel_shape", new int[]{weightShape[2], weightShape[3]}))
                .addAttribute(intsAttr("strides", new int[]{options.strideH(), options.strideW()}))
                .addAttribute(intsAttr("pads", new int[]{options.padH(), options.padW(), options.padH(), options.padW()}))
                .addAttribute(intsAttr("dilations", new int[]{options.dilationH(), options.dilationW()}));
        if (options.groups() != 1) {
            node.addAttribute(intAttr("group", options.groups()));
        }
    }

    private static void exportMaxPool(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("MaxPool");
        exportPoolAttributes(node, ((maxPool2d) op).getOptions());
    }

    private static void exportAveragePool(OnnxProto.NodeProto.Builder node, Operation op) {
        avgPool2d pool = (avgPool2d) op;
        node.setOpType("AveragePool");
        exportPoolAttributes(node, pool.getOptions());
        if (pool.getOptions().countIncludePad()) {
            node.addAttribute(intAttr("count_include_pad", 1));
        }
    }

    private static void exportPoolAttributes(OnnxProto.NodeProto.Builder node, Pool2dOptions options) {
        node.addAttribute(intsAttr("kernel_shape", new int[]{options.kernelH(), options.kernelW()}))
                .addAttribute(intsAttr("strides", new int[]{options.strideH(), options.strideW()}))
                .addAttribute(intsAttr("pads", new int[]{options.padH(), options.padW(), options.padH(), options.padW()}));
        if (options.ceilMode()) {
            node.addAttribute(intAttr("ceil_mode", 1));
        }
    }

    private static void exportCol2Im(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        fold2d fold = (fold2d) op;
        int[] outputShape = fold.getOutputShape();
        if (outputShape.length != 4) {
            throw new OnnxUnsupportedException("FOLD2D export requires rank-4 NCHW output shape.");
        }
        Window2dOptions options = fold.getOptions();
        if (options.ceilMode()) {
            throw new OnnxUnsupportedException("FOLD2D export to ONNX Col2Im does not support ceilMode.");
        }
        String imageShapeName = names.auxiliary(node.getOutput(0) + "_image_shape");
        String blockShapeName = names.auxiliary(node.getOutput(0) + "_block_shape");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(
                imageShapeName,
                new long[]{outputShape[2], outputShape[3]}
        ));
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(
                blockShapeName,
                new long[]{options.kernelH(), options.kernelW()}
        ));
        node.setOpType("Col2Im")
                .addInput(imageShapeName)
                .addInput(blockShapeName)
                .addAttribute(intsAttr("strides", new int[]{options.strideH(), options.strideW()}))
                .addAttribute(intsAttr("pads", new int[]{options.padH(), options.padW(), options.padH(), options.padW()}))
                .addAttribute(intsAttr("dilations", new int[]{options.dilationH(), options.dilationW()}));
    }

    private static void exportLayerNormalization(OnnxProto.NodeProto.Builder node, Operation op, Tensor tensor) {
        if (node.getInputCount() != 3) {
            throw new OnnxUnsupportedException("LayerNorm export requires input, scale, and bias tensors.");
        }
        layerNorm layerNorm = (layerNorm) op;
        int axis = tensor.getShapeUnsafe().length - layerNorm.getNormalizedRank();
        node.setOpType("LayerNormalization")
                .addAttribute(intAttr("axis", axis))
                .addAttribute(floatAttr("epsilon", (float) layerNorm.getEpsilon()));
    }

    private static void exportPermute(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("Transpose")
                .addAttribute(intsAttr("perm", ((permute) op).getAxes()));
    }

    private static void exportReshape(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        Integer flattenAxis = flattenAxis(tensor, ((reshape) op).getTargetShape());
        if (flattenAxis != null) {
            node.setOpType("Flatten");
            if (flattenAxis != 1) {
                node.addAttribute(intAttr("axis", flattenAxis));
            }
            return;
        }
        node.setOpType("Reshape");
        String shapeName = names.auxiliary(node.getOutput(0) + "_shape");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(shapeName, toLong(((reshape) op).getTargetShape())));
        node.addInput(shapeName);
    }

    private static Integer flattenAxis(Tensor tensor, int[] targetShape) {
        if (targetShape.length != 2) {
            return null;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.size() != 1) {
            return null;
        }
        int[] inputShape = inputs.get(0).getShapeUnsafe();
        if (inputShape.length <= 2) {
            return null;
        }
        for (int axis = 0; axis <= inputShape.length; axis++) {
            if (product(inputShape, 0, axis) == targetShape[0]
                    && product(inputShape, axis, inputShape.length) == targetShape[1]) {
                return axis;
            }
        }
        return null;
    }

    private static int product(int[] values, int startInclusive, int endExclusive) {
        int product = 1;
        for (int i = startInclusive; i < endExclusive; i++) {
            product *= values[i];
        }
        return product;
    }

    private static void exportExpand(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Expand");
        String shapeName = names.auxiliary(node.getOutput(0) + "_shape");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(shapeName, toLong(((expand) op).getTargetShape())));
        node.addInput(shapeName);
    }

    private static void exportSlice(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Slice");
        slice slice = (slice) op;
        String startsName = names.auxiliary(node.getOutput(0) + "_starts");
        String endsName = names.auxiliary(node.getOutput(0) + "_ends");
        String axesName = names.auxiliary(node.getOutput(0) + "_axes");
        String stepsName = names.auxiliary(node.getOutput(0) + "_steps");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(startsName, toLong(slice.getStarts())));
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(endsName, toLong(slice.getEnds())));
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(axesName, toLong(slice.getAxes())));
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(stepsName, toLong(slice.getSteps())));
        node.addInput(startsName);
        node.addInput(endsName);
        node.addInput(axesName);
        node.addInput(stepsName);
    }

    private static void exportConcat(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("Concat")
                .addAttribute(intAttr("axis", ((concat) op).getAxis()));
    }

    private static void exportPad(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            Tensor tensor,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        pad pad = (pad) op;
        node.setOpType("Pad");
        String padsName = names.auxiliary(node.getOutput(0) + "_pads");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(padsName, concatLong(pad.getBefore(), pad.getAfter())));
        node.addInput(padsName);
        if (pad.getConstantValue() != 0.0d) {
            String valueName = names.auxiliary(node.getOutput(0) + "_constant_value");
            graphBuilder.addInitializer(scalarInitializer(valueName, pad.getConstantValue(), tensor.getDataType()));
            node.addInput(valueName);
        }
    }

    private static void exportTile(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        node.setOpType("Tile");
        String repeatsName = names.auxiliary(node.getOutput(0) + "_repeats");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(repeatsName, toLong(((tile) op).getRepeats())));
        node.addInput(repeatsName);
    }

    private static void exportCast(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("Cast")
                .addAttribute(intAttr("to", OnnxDataTypes.toOnnx(((cast) op).getTargetType())));
    }

    private static void exportGatherAxis(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("Gather")
                .addAttribute(intAttr("axis", ((gatherAxis) op).getAxis()));
    }

    private static void exportGatherElements(OnnxProto.NodeProto.Builder node, Operation op) {
        node.setOpType("GatherElements")
                .addAttribute(intAttr("axis", ((takeAlongAxis) op).getDimension()));
    }

    private static void exportGatherNd(OnnxProto.NodeProto.Builder node, Operation op) {
        if (!(op instanceof gatherNd gatherOp)) {
            throw new OnnxUnsupportedException("GatherND export requires gatherNd operation.");
        }
        node.setOpType("GatherND");
        if (gatherOp.getBatchDims() != 0) {
            node.addAttribute(intAttr("batch_dims", gatherOp.getBatchDims()));
        }
    }

    private static void exportScatterElements(OnnxProto.NodeProto.Builder node, Operation op) {
        scatterElements scatter = (scatterElements) op;
        node.setOpType("ScatterElements")
                .addAttribute(intAttr("axis", scatter.getAxis()));
        if (scatter.getReduction() != ScatterReduction.NONE) {
            node.addAttribute(stringAttr("reduction", scatterReduction(scatter.getReduction())));
        }
    }

    private static void exportScatterNd(OnnxProto.NodeProto.Builder node, Operation op) {
        scatterNd scatter = (scatterNd) op;
        node.setOpType("ScatterND");
        if (scatter.getReduction() != ScatterReduction.NONE) {
            node.addAttribute(stringAttr("reduction", scatterReduction(scatter.getReduction())));
        }
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

    private static void exportArgMax(OnnxProto.NodeProto.Builder node, Operation op) {
        argMax argMax = (argMax) op;
        node.setOpType("ArgMax")
                .addAttribute(intAttr("axis", argMax.getDimension()))
                .addAttribute(intAttr("keepdims", argMax.keepDims() ? 1 : 0))
                .addAttribute(intAttr("select_last_index", argMax.tiePolicy() == ArgMaxTiePolicy.LAST_INDEX ? 1 : 0));
    }

    private static void exportCumSum(
            OnnxProto.NodeProto.Builder node,
            Operation op,
            OnnxNameRegistry names,
            OnnxProto.GraphProto.Builder graphBuilder
    ) {
        cumSum scan = (cumSum) op;
        node.setOpType("CumSum");
        String axisName = names.auxiliary(node.getOutput(0) + "_axis");
        graphBuilder.addInitializer(OnnxTensorProtoUtil.int64Initializer(axisName, new long[]{scan.getAxis()}));
        node.addInput(axisName);
        if (scan.isExclusive()) {
            node.addAttribute(intAttr("exclusive", 1));
        }
        if (scan.isReverse()) {
            node.addAttribute(intAttr("reverse", 1));
        }
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

    private static OnnxProto.AttributeProto floatAttr(String name, float value) {
        return OnnxProto.AttributeProto.newBuilder().setName(name).setF(value).build();
    }

    private static OnnxProto.AttributeProto intsAttr(String name, int[] values) {
        OnnxProto.AttributeProto.Builder builder = OnnxProto.AttributeProto.newBuilder().setName(name);
        for (int value : values) {
            builder.addInts(value);
        }
        return builder.build();
    }

    private static OnnxProto.AttributeProto stringAttr(String name, String value) {
        return OnnxProto.AttributeProto.newBuilder()
                .setName(name)
                .setS(com.google.protobuf.ByteString.copyFromUtf8(value))
                .build();
    }

    private static String scatterReduction(ScatterReduction reduction) {
        return switch (reduction) {
            case NONE -> "none";
            case ADD -> "add";
            case MUL -> "mul";
            case MAX -> "max";
            case MIN -> "min";
        };
    }

    private static long[] toLong(int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out;
    }

    private static long[] concatLong(int[] first, int[] second) {
        long[] out = new long[first.length + second.length];
        for (int i = 0; i < first.length; i++) {
            out[i] = first[i];
        }
        for (int i = 0; i < second.length; i++) {
            out[first.length + i] = second[i];
        }
        return out;
    }

    private record SplitGroupKey(Tensor input, int axis) {
    }

    private record SplitPart(Tensor tensor, int axis, int start, int end) {
        int size() {
            return end - start;
        }
    }
}
