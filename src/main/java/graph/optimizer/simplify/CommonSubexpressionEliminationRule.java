package graph.optimizer.simplify;

import config.optimizer.CseConfig;
import graph.optimizer.OptimizerGraph;
import graph.optimizer.OptimizationRule;
import graph.optimizer.state.OptimizerState;
import operations.Operation;
import operations.nn.pool.avgPool2d;
import operations.elementwise.unary.clampMax;
import operations.elementwise.unary.clampMin;
import operations.loss.crossEntropyLoss;
import operations.loss.crossEntropyLossIndices;
import operations.loss.crossEntropyLossIndicesGrad;
import operations.nn.conv.conv2d;
import operations.layout.expand;
import operations.layout.expandDims;
import operations.layout.concat;
import operations.layout.pad;
import operations.dtype.cast;
import operations.index.gather;
import operations.index.gatherAxis;
import operations.index.gatherAxisGrad;
import operations.index.gatherGrad;
import operations.index.gatherNd;
import operations.index.gatherNdGrad;
import operations.normalization.layerNorm;
import operations.nn.pool.maxPool2d;
import operations.index.scatterAdd;
import operations.index.scatterAxisAdd;
import operations.index.scatterElements;
import operations.index.scatterNd;
import operations.linalg.scaledDotProductAttention;
import operations.linalg.scaledDotProductAttentionBackward;
import operations.index.takeAlongAxis;
import operations.index.takeAlongAxisGrad;
import operations.elementwise.binary.maxGrad;
import operations.reduction.mean;
import operations.elementwise.binary.minGrad;
import operations.elementwise.unary.mulScalar;
import operations.loss.nllLoss;
import operations.layout.noop;
import operations.layout.permute;
import operations.elementwise.unary.pow;
import operations.reduction.cumSum;
import operations.reduction.reduceMax;
import operations.reduction.reduceMaxGrad;
import operations.reduction.reduceMin;
import operations.reduction.reduceMinGrad;
import operations.reduction.reduceAll;
import operations.reduction.reduceAny;
import operations.reduction.reduceProd;
import operations.reduction.argMax;
import operations.layout.reshape;
import operations.reduction.logSoftmax;
import operations.reduction.logSoftmaxGrad;
import operations.linalg.linear;
import operations.normalization.rmsNorm;
import operations.layout.select;
import operations.layout.slice;
import operations.layout.sliceBackward;
import operations.layout.squeeze;
import operations.reduction.sum;
import operations.layout.tile;
import operations.layout.unfoldAxis;
import operations.layout.unfold2d;
import operations.layout.fold2d;
import operations.reduction.softmax;
import operations.reduction.softmaxGrad;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Eliminates duplicate pure subexpressions by structural signature.
 *
 * <p>The rule walks the graph in topological order, rewrites each node's inputs through already discovered
 * replacements, and maps structurally equivalent nodes to the first node with the same signature. It rebuilds the
 * observable closure from the original graph sinks so removed nodes disappear from the optimized graph while the
 * semantic forward output and published gradient bindings remain reachable.
 *
 * <p>Safety boundaries are deliberately conservative: leaf tensors, noop/fused nodes, random/dropout-like operations,
 * and unsupported parameterized operations are not merged. In strict mode, signature keys also include gradient
 * requirement and shape so aliases only occur when graph-semantic execution metadata matches. Backend intent is a
 * compile-layer plan remapped after optimizer rewrites.
 *
 * <p>This rule mutates graph edge metadata while applying replacements and is intended for single-threaded optimizer
 * execution.
 */
public class CommonSubexpressionEliminationRule implements OptimizationRule {
    private final CseConfig config;

    /**
     * Creates a CSE rule.
     *
     * @param config CSE behavior and safety configuration
     */
    public CommonSubexpressionEliminationRule(CseConfig config) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
    }

    /**
     * Returns the CSE configuration used by this rule.
     *
     * @return CSE configuration
     */
    public CseConfig config() {
        return config;
    }

    /**
     * Applies structural common subexpression elimination.
     *
     * @param state optimizer state containing a topologically sorted graph
     * @return state with the replacement graph and preserved graph-level optimizer metadata
     */
    @Override
    public OptimizerState apply(OptimizerState state) {
        List<Tensor> sortedGraph = state.graph();
        List<Tensor> originalRoots = OptimizerGraph.observableRoots(sortedGraph);
        List<Tensor> optimized = new ArrayList<>();
        Map<StructuralSignature, Tensor> seenNodes = new HashMap<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();
        Map<Tensor, SignatureComponent> structuralSignatures = new HashMap<>();

        for (Tensor t : sortedGraph) {
            OptimizerGraph.rewriteInputs(t, replacements);

            StructuralSignature signature = generateSignature(t, structuralSignatures);
            if (signature != null) {
                structuralSignatures.put(t, signature);
            } else {
                structuralSignatures.put(t, leafSignature(t));
            }

            if (signature != null) {
                Tensor existing = seenNodes.get(signature);
                if (existing != null) {
                    if (t.isBackward()) {
                        TensorInternalAccess.setBackward(existing, true);
                    }
                    replacements.put(t, existing);
                    continue;
                }
                seenNodes.put(signature, t);
            }

            optimized.add(t);
        }

        if (!replacements.isEmpty()) {
            for (Tensor tensor : sortedGraph) {
                Tensor resolvedGradient = OptimizerGraph.resolveReplacement(tensor.getGradient(), replacements);
                if (resolvedGradient != null) {
                    TensorInternalAccess.setGradient(tensor, resolvedGradient);
                }
            }
        }

        Tensor resolvedForwardOutput = OptimizerGraph.resolveReplacement(state.forwardOutput(), replacements);
        List<Tensor> rebuilt = OptimizerGraph.rebuildTopologicalClosureFromRoots(
                OptimizerGraph.resolveRoots(originalRoots, replacements)
        );
        return state.withGraph(rebuilt, resolvedForwardOutput == null ? state.forwardOutput() : resolvedForwardOutput)
                .withRewriteMap(state.rewriteMap().withReplacements(replacements));
    }

    private StructuralSignature generateSignature(
            Tensor t,
            Map<Tensor, SignatureComponent> structuralSignatures
    ) {
        Operation op = t.getOperation();
        boolean strictSafety = config.strictSafety();
        if (op == null) {
            return null;
        }

        if (op instanceof noop || op.opType() == Operation.OpType.FUSED) {
            return null;
        }

        String opName = op.getClass().getSimpleName().toLowerCase(Locale.ROOT);
        if (opName.contains("random") || opName.contains("dropout")) {
            return null;
        }

        List<Tensor> inputs = t.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return null;
        }

        List<SignatureComponent> inputKeys = new ArrayList<>(inputs.size());
        for (Tensor input : inputs) {
            inputKeys.add(structuralSignatures.getOrDefault(input, leafSignature(input)));
        }

        if (isCommutative(op.opType())) {
            inputKeys.sort(Comparator.comparing(SignatureComponent::sortKey));
        }

        return new StructuralSignature(
                op.opType(),
                t.isBackward(),
                strictSafety ? t.getRequiresGrad() : null,
                strictSafety ? IntArrayValue.copyOf(t.getShape()) : null,
                parameterKey(op),
                List.copyOf(inputKeys)
        );
    }

    private SignatureComponent leafSignature(Tensor t) {
        if (t.getOperation() == null && t.getRequiresGrad()) {
            return new IdentityLeafSignature(System.identityHashCode(t));
        }

        if (t.getOperation() == null && !t.getRequiresGrad() && t.getFlatDataSize() == 1) {
            return scalarLeafSignature(t);
        }

        return new IdentityLeafSignature(System.identityHashCode(t));
    }

    private SignatureComponent scalarLeafSignature(Tensor t) {
        DataType dtype = t.getDataType();
        long valueBits = switch (dtype) {
            case BOOL -> t.toBooleanArrayCopy()[0] ? 1L : 0L;
            case INT32, INT64 -> t.getIntegralByFlatIndex(0);
            case FLOAT64, FLOAT32, BFLOAT16 -> Double.doubleToLongBits(t.scalarAsDouble());
        };
        return new ScalarLeafSignature(dtype, valueBits, IntArrayValue.copyOf(t.getShape()));
    }

    private boolean isCommutative(Operation.OpType opType) {
        return opType == Operation.OpType.ADD || opType == Operation.OpType.MUL;
    }

    private SignatureComponent parameterKey(Operation op) {
        return switch (op.opType()) {
            case POW -> new DoubleValue(((pow) op).getExponent());
            case MUL_SCALAR -> new DoubleValue(((mulScalar) op).getScalar());
            case CLAMP_MIN -> new DoubleValue(((clampMin) op).getMinValue());
            case CLAMP_MAX -> new DoubleValue(((clampMax) op).getMaxValue());
            case SUM -> new ReductionSignature(((sum) op).getDimension(), ((sum) op).keepDims());
            case MEAN -> new ReductionSignature(((mean) op).getDimension(), ((mean) op).keepDims());
            case SOFTMAX -> new AxisSignature(((softmax) op).getDimension());
            case SOFTMAX_GRAD -> new AxisSignature(((softmaxGrad) op).getDimension());
            case LOG_SOFTMAX -> new AxisSignature(((logSoftmax) op).getDimension());
            case LOG_SOFTMAX_GRAD -> new AxisSignature(((logSoftmaxGrad) op).getDimension());
            case LAYER_NORM -> new NormSignature(((layerNorm) op).getNormalizedRank(), Double.doubleToLongBits(((layerNorm) op).getEpsilon()));
            case RMS_NORM -> new NormSignature(((rmsNorm) op).getNormalizedRank(), Double.doubleToLongBits(((rmsNorm) op).getEpsilon()));
            case NLL_LOSS -> new AxisSignature(((nllLoss) op).getClassDimension());
            case CROSS_ENTROPY_LOSS -> new AxisSignature(((crossEntropyLoss) op).getClassDimension());
            case CROSS_ENTROPY_LOSS_INDICES -> IntArrayValue.copyOf(new int[]{
                    ((crossEntropyLossIndices) op).getClassDimension(),
                    ((crossEntropyLossIndices) op).getReduction().ordinal(),
                    ((crossEntropyLossIndices) op).hasIgnoreIndex() ? 1 : 0,
                    ((crossEntropyLossIndices) op).hasIgnoreIndex() ? ((crossEntropyLossIndices) op).getIgnoreIndex() : 0
            });
            case CROSS_ENTROPY_LOSS_INDICES_GRAD -> new AxisSignature(((crossEntropyLossIndicesGrad) op).getClassDimension());
            case REDUCE_MIN -> new ReductionSignature(((reduceMin) op).getDimension(), ((reduceMin) op).keepDims());
            case REDUCE_MAX -> new ReductionSignature(((reduceMax) op).getDimension(), ((reduceMax) op).keepDims());
            case REDUCE_ALL -> new ReductionSignature(((reduceAll) op).getDimension(), ((reduceAll) op).keepDims());
            case REDUCE_ANY -> new ReductionSignature(((reduceAny) op).getDimension(), ((reduceAny) op).keepDims());
            case REDUCE_PROD -> new ReductionSignature(((reduceProd) op).getDimension(), ((reduceProd) op).keepDims());
            case CUMSUM -> IntArrayValue.copyOf(new int[]{
                    ((cumSum) op).getAxis(),
                    ((cumSum) op).isExclusive() ? 1 : 0,
                    ((cumSum) op).isReverse() ? 1 : 0
            });
            case ARGMAX -> IntArrayValue.copyOf(new int[]{
                    ((argMax) op).getDimension(),
                    ((argMax) op).keepDims() ? 1 : 0,
                    ((argMax) op).tiePolicy().ordinal()
            });
            case MIN_GRAD -> new InputSelectorSignature(((minGrad) op).isForFirstInput());
            case MAX_GRAD -> new InputSelectorSignature(((maxGrad) op).isForFirstInput());
            case REDUCE_MIN_GRAD -> new AxisSignature(((reduceMinGrad) op).getDimension());
            case REDUCE_MAX_GRAD -> new AxisSignature(((reduceMaxGrad) op).getDimension());
            case RESHAPE -> IntArrayValue.copyOf(((reshape) op).getTargetShape());
            case PERMUTE -> IntArrayValue.copyOf(((permute) op).getAxes());
            case EXPAND -> IntArrayValue.copyOf(((expand) op).getTargetShape());
            case SELECT -> IntArrayValue.copyOf(new int[]{((select) op).getDimension(), ((select) op).getIndex()});
            case SLICE -> IntArrayValue.copyOf(sliceSignature((slice) op));
            case SLICE_BACKWARD -> IntArrayValue.copyOf(sliceBackwardSignature((sliceBackward) op));
            case CONCAT -> new AxisSignature(((concat) op).getAxis());
            case PAD -> IntArrayValue.copyOf(padSignature((pad) op));
            case TILE -> IntArrayValue.copyOf(((tile) op).getRepeats());
            case UNFOLD_AXIS -> IntArrayValue.copyOf(new int[]{
                    ((unfoldAxis) op).getAxis(),
                    ((unfoldAxis) op).getSize(),
                    ((unfoldAxis) op).getStep()
            });
            case UNFOLD2D -> window2dSignature(((unfold2d) op).getOptions(), 1);
            case FOLD2D -> fold2dSignature((fold2d) op);
            case CAST -> new AxisSignature(((cast) op).getTargetType().ordinal());
            case EXPAND_DIMS -> new AxisSignature(((expandDims) op).getAxis());
            case GATHER -> new AxisSignature(((gather) op).getDimension());
            case GATHER_GRAD -> new AxisSignature(((gatherGrad) op).getDimension());
            case GATHER_AXIS -> new AxisSignature(((gatherAxis) op).getAxis());
            case GATHER_AXIS_GRAD -> IntArrayValue.copyOf(gatherAxisGradSignature((gatherAxisGrad) op));
            case GATHER_ND -> new AxisSignature(((gatherNd) op).getBatchDims());
            case GATHER_ND_GRAD -> IntArrayValue.copyOf(gatherNdGradSignature((gatherNdGrad) op));
            case TAKE_ALONG_AXIS -> new AxisSignature(((takeAlongAxis) op).getDimension());
            case TAKE_ALONG_AXIS_GRAD -> new AxisSignature(((takeAlongAxisGrad) op).getDimension());
            case SCATTER_ADD -> new AxisSignature(((scatterAdd) op).getDimension());
            case SCATTER_AXIS_ADD -> new AxisSignature(((scatterAxisAdd) op).getAxis());
            case SCATTER_ELEMENTS -> IntArrayValue.copyOf(new int[]{
                    ((scatterElements) op).getAxis(),
                    ((scatterElements) op).getReduction().ordinal()
            });
            case SCATTER_ND -> IntArrayValue.copyOf(new int[]{
                    ((scatterNd) op).getReduction().ordinal(),
                    ((scatterNd) op).getBatchDims()
            });
            case SCALED_DOT_PRODUCT_ATTENTION -> new AttentionSignature(Double.doubleToLongBits(((scaledDotProductAttention) op).getScale()), ((scaledDotProductAttention) op).hasMask());
            case SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> IntArrayValue.copyOf(new int[]{((scaledDotProductAttentionBackward) op).getOutputKind().ordinal()});
            case LINEAR -> new InputSelectorSignature(((linear) op).hasBias());
            case CONV2D -> conv2dSignature(((conv2d) op).getOptions(), ((conv2d) op).hasBias() ? 1 : 0);
            case MAX_POOL2D -> pool2dSignature(((maxPool2d) op).getOptions(), 1);
            case AVG_POOL2D -> pool2dSignature(((avgPool2d) op).getOptions(), 2);
            case SQUEEZE -> new AxisSignature(((squeeze) op).getAxis());
            case ADD, SUB, MUL, DIV, MIN, MAX, GT, GE, LT, LE, EQ, NE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE, MATMUL,
                    NEG, INV, LOG, EXP, FAST_EXP, ERF, TANH, FAST_TANH,
                    POW_TENSOR, SQRT, ABS, FLOOR, CEIL, SIGN, RELU, SIGMOID,
                    CONTIGUOUS, CONST_SCALAR, NOOP, FUSED,
                    SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS -> NoParamsSignature.INSTANCE;
            case UNKNOWN -> new OperationIdentitySignature(op);
        };
    }

    private SignatureComponent conv2dSignature(tensor.options.Conv2dOptions options, int hasBias) {
        return IntArrayValue.copyOf(new int[]{
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.dilationH(), options.dilationW(),
                options.groups(),
                hasBias
        });
    }

    private int[] sliceSignature(slice op) {
        int[] starts = op.getStarts();
        int[] ends = op.getEnds();
        int[] axes = op.getAxes();
        int[] steps = op.getSteps();
        int[] outputShape = op.getOutputShape();
        int[] out = new int[starts.length + ends.length + axes.length + steps.length + outputShape.length + 5];
        int p = 0;
        out[p++] = starts.length;
        for (int value : starts) out[p++] = value;
        out[p++] = ends.length;
        for (int value : ends) out[p++] = value;
        out[p++] = axes.length;
        for (int value : axes) out[p++] = value;
        out[p++] = steps.length;
        for (int value : steps) out[p++] = value;
        out[p++] = outputShape.length;
        for (int value : outputShape) out[p++] = value;
        return out;
    }

    private int[] sliceBackwardSignature(sliceBackward op) {
        int[] starts = op.getStarts();
        int[] axes = op.getAxes();
        int[] steps = op.getSteps();
        int[] inputShape = op.getInputShape();
        int[] out = new int[starts.length + axes.length + steps.length + inputShape.length + 4];
        int p = 0;
        out[p++] = starts.length;
        for (int value : starts) out[p++] = value;
        out[p++] = axes.length;
        for (int value : axes) out[p++] = value;
        out[p++] = steps.length;
        for (int value : steps) out[p++] = value;
        out[p++] = inputShape.length;
        for (int value : inputShape) out[p++] = value;
        return out;
    }

    private int[] padSignature(pad op) {
        int[] before = op.getBefore();
        int[] after = op.getAfter();
        int[] out = new int[before.length + after.length + 3];
        int p = 0;
        out[p++] = before.length;
        for (int value : before) out[p++] = value;
        out[p++] = after.length;
        for (int value : after) out[p++] = value;
        out[p] = Long.hashCode(Double.doubleToLongBits(op.getConstantValue()));
        return out;
    }

    private int[] gatherAxisGradSignature(gatherAxisGrad op) {
        int[] dataShape = op.getDataShape();
        int[] out = new int[dataShape.length + 2];
        out[0] = op.getAxis();
        out[1] = dataShape.length;
        System.arraycopy(dataShape, 0, out, 2, dataShape.length);
        return out;
    }

    private int[] gatherNdGradSignature(gatherNdGrad op) {
        int[] dataShape = op.getDataShape();
        int[] out = new int[dataShape.length + 2];
        out[0] = op.getBatchDims();
        out[1] = dataShape.length;
        System.arraycopy(dataShape, 0, out, 2, dataShape.length);
        return out;
    }

    private SignatureComponent pool2dSignature(tensor.options.Pool2dOptions options, int kind) {
        return IntArrayValue.copyOf(new int[]{
                options.kernelH(), options.kernelW(),
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.countIncludePad() ? 1 : 0,
                options.ceilMode() ? 1 : 0,
                kind
        });
    }

    private SignatureComponent window2dSignature(tensor.options.Window2dOptions options, int kind) {
        return IntArrayValue.copyOf(new int[]{
                options.kernelH(), options.kernelW(),
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.dilationH(), options.dilationW(),
                options.ceilMode() ? 1 : 0,
                kind
        });
    }

    private SignatureComponent fold2dSignature(fold2d op) {
        int[] outputShape = op.getOutputShape();
        tensor.options.Window2dOptions options = op.getOptions();
        int[] out = new int[outputShape.length + 11];
        int p = 0;
        out[p++] = outputShape.length;
        for (int value : outputShape) out[p++] = value;
        out[p++] = options.kernelH();
        out[p++] = options.kernelW();
        out[p++] = options.strideH();
        out[p++] = options.strideW();
        out[p++] = options.padH();
        out[p++] = options.padW();
        out[p++] = options.dilationH();
        out[p++] = options.dilationW();
        out[p++] = options.ceilMode() ? 1 : 0;
        out[p] = 2;
        return IntArrayValue.copyOf(out);
    }

    private sealed interface SignatureComponent
            permits StructuralSignature, IdentityLeafSignature, ScalarLeafSignature,
            NoParamsSignature, AxisSignature, ReductionSignature, InputSelectorSignature,
            DoubleValue, IntArrayValue, NormSignature, AttentionSignature, OperationIdentitySignature {

        String sortKey();
    }

    private record StructuralSignature(
            Operation.OpType opType,
            boolean backward,
            Boolean requiresGrad,
            IntArrayValue outputShape,
            SignatureComponent parameters,
            List<SignatureComponent> inputs
    ) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "node:" + opType + ":" + backward + ":" + requiresGrad + ":" + outputShape + ":" + parameters + ":" + inputs;
        }
    }

    private record IdentityLeafSignature(int identityHash) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "leaf@" + identityHash;
        }
    }

    private record ScalarLeafSignature(DataType dtype, long bits, IntArrayValue shape) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "scalar:" + dtype + ":" + bits + ":" + shape;
        }
    }

    private enum NoParamsSignature implements SignatureComponent {
        INSTANCE;

        @Override
        public String sortKey() {
            return "none";
        }
    }

    private record AxisSignature(int axis) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "axis:" + axis;
        }
    }

    private record ReductionSignature(int dimension, boolean keepDims) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "reduction:" + dimension + ":" + keepDims;
        }
    }

    private record InputSelectorSignature(boolean forFirstInput) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "selector:" + forFirstInput;
        }
    }

    private record DoubleValue(long bits) implements SignatureComponent {
        private DoubleValue(double value) {
            this(Double.doubleToLongBits(value));
        }

        @Override
        public String sortKey() {
            return "double:" + bits;
        }
    }

    private record IntArrayValue(List<Integer> values) implements SignatureComponent {
        static IntArrayValue copyOf(int[] values) {
            if (values == null) {
                return new IntArrayValue(null);
            }
            List<Integer> copy = new ArrayList<>(values.length);
            for (int value : values) {
                copy.add(value);
            }
            return new IntArrayValue(List.copyOf(copy));
        }

        @Override
        public String sortKey() {
            return "ints:" + values;
        }
    }

    private record NormSignature(int normalizedRank, long epsilonBits) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "norm:" + normalizedRank + ":" + epsilonBits;
        }
    }

    private record AttentionSignature(long scaleBits, boolean hasMask) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "attention:" + scaleBits + ":" + hasMask;
        }
    }

    private static final class OperationIdentitySignature implements SignatureComponent {
        private final Operation operation;

        private OperationIdentitySignature(Operation operation) {
            this.operation = operation;
        }

        @Override
        public String sortKey() {
            return "op@" + System.identityHashCode(operation);
        }
    }
}
