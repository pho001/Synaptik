package graph.optimizer.rules;

import config.optimizer.CseConfig;
import graph.optimizer.OptimizerGraphSupport;
import graph.optimizer.OptimizationRule;
import operations.FusedOperation;
import operations.Operation;
import operations.avgPool2d;
import operations.avgPool2dBackwardInput;
import operations.clampMax;
import operations.clampMin;
import operations.crossEntropyLoss;
import operations.conv2d;
import operations.conv2dBackwardInput;
import operations.conv2dBackwardWeight;
import operations.expand;
import operations.expandDims;
import operations.gather;
import operations.gatherGrad;
import operations.maxPool2d;
import operations.maxPool2dBackwardInput;
import operations.scatterAdd;
import operations.takeAlongAxis;
import operations.takeAlongAxisGrad;
import operations.maxGrad;
import operations.mean;
import operations.minGrad;
import operations.mulScalar;
import operations.nllLoss;
import operations.noop;
import operations.permute;
import operations.pow;
import operations.reduceMax;
import operations.reduceMaxGrad;
import operations.reduceMin;
import operations.reduceMinGrad;
import operations.reshape;
import operations.logSoftmax;
import operations.select;
import operations.squeeze;
import operations.sum;
import operations.softmax;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

public class CommonSubexpressionEliminationRule implements OptimizationRule {
    private final CseConfig config;

    public CommonSubexpressionEliminationRule(CseConfig config) {
        this.config = Objects.requireNonNull(config, "config cannot be null");
    }

    public CseConfig config() {
        return config;
    }

    @Override
    public List<Tensor> apply(List<Tensor> sortedGraph) {
        List<Tensor> optimized = new ArrayList<>();
        Map<StructuralSignature, Tensor> seenNodes = new HashMap<>();
        Map<Tensor, Tensor> replacements = new HashMap<>();
        Map<Tensor, SignatureComponent> structuralSignatures = new HashMap<>();

        for (Tensor t : sortedGraph) {
            OptimizerGraphSupport.rewriteInputs(t, replacements);

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
                        existing.setBackward(true);
                    }
                    replacements.put(t, existing);
                    continue;
                }
                seenNodes.put(signature, t);
            }

            optimized.add(t);
        }

        return OptimizerGraphSupport.rebuildTopologicalClosure(optimized);
    }

    private StructuralSignature generateSignature(Tensor t, Map<Tensor, SignatureComponent> structuralSignatures) {
        Operation op = t.getOperation();
        boolean strictSafety = config.strictSafety();
        if (op == null) {
            return null;
        }

        if (op instanceof noop || op instanceof FusedOperation || op.opType() == Operation.OpType.FUSED) {
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
                strictSafety ? t.resolveBackend() : null,
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
            long bits = Double.doubleToLongBits(t.scalarAsDouble());
            return new ScalarLeafSignature(bits, IntArrayValue.copyOf(t.getShape()));
        }

        return new IdentityLeafSignature(System.identityHashCode(t));
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
            case LOG_SOFTMAX -> new AxisSignature(((logSoftmax) op).getDimension());
            case NLL_LOSS -> new AxisSignature(((nllLoss) op).getClassDimension());
            case CROSS_ENTROPY_LOSS -> new AxisSignature(((crossEntropyLoss) op).getClassDimension());
            case REDUCE_MIN -> new ReductionSignature(((reduceMin) op).getDimension(), ((reduceMin) op).keepDims());
            case REDUCE_MAX -> new ReductionSignature(((reduceMax) op).getDimension(), ((reduceMax) op).keepDims());
            case MIN_GRAD -> new InputSelectorSignature(((minGrad) op).isForFirstInput());
            case MAX_GRAD -> new InputSelectorSignature(((maxGrad) op).isForFirstInput());
            case REDUCE_MIN_GRAD -> new AxisSignature(((reduceMinGrad) op).getDimension());
            case REDUCE_MAX_GRAD -> new AxisSignature(((reduceMaxGrad) op).getDimension());
            case RESHAPE -> IntArrayValue.copyOf(((reshape) op).getTargetShape());
            case PERMUTE -> IntArrayValue.copyOf(((permute) op).getAxes());
            case EXPAND -> IntArrayValue.copyOf(((expand) op).getTargetShape());
            case SELECT -> IntArrayValue.copyOf(new int[]{((select) op).getDimension(), ((select) op).getIndex()});
            case EXPAND_DIMS -> new AxisSignature(((expandDims) op).getAxis());
            case GATHER -> new AxisSignature(((gather) op).getDimension());
            case GATHER_GRAD -> new AxisSignature(((gatherGrad) op).getDimension());
            case TAKE_ALONG_AXIS -> new AxisSignature(((takeAlongAxis) op).getDimension());
            case TAKE_ALONG_AXIS_GRAD -> new AxisSignature(((takeAlongAxisGrad) op).getDimension());
            case SCATTER_ADD -> new AxisSignature(((scatterAdd) op).getDimension());
            case CONV2D -> conv2dSignature(((conv2d) op).getOptions(), ((conv2d) op).hasBias() ? 1 : 0);
            case CONV2D_BACKWARD_INPUT -> conv2dBackwardInputSignature((conv2dBackwardInput) op);
            case CONV2D_BACKWARD_WEIGHT -> conv2dBackwardWeightSignature((conv2dBackwardWeight) op);
            case MAX_POOL2D -> pool2dSignature(((maxPool2d) op).getOptions(), 1);
            case MAX_POOL2D_BACKWARD_INPUT -> pool2dBackwardInputSignature(((maxPool2dBackwardInput) op).getOptions(), ((maxPool2dBackwardInput) op).getInputShape(), 1);
            case AVG_POOL2D -> pool2dSignature(((avgPool2d) op).getOptions(), 2);
            case AVG_POOL2D_BACKWARD_INPUT -> pool2dBackwardInputSignature(((avgPool2dBackwardInput) op).getOptions(), ((avgPool2dBackwardInput) op).getInputShape(), 2);
            case SQUEEZE -> new AxisSignature(((squeeze) op).getAxis());
            default -> NoParamsSignature.INSTANCE;
        };
    }

    private SignatureComponent conv2dSignature(tensor.Conv2dOptions options, int hasBias) {
        return IntArrayValue.copyOf(new int[]{
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.dilationH(), options.dilationW(),
                options.groups(),
                hasBias
        });
    }

    private SignatureComponent conv2dBackwardInputSignature(conv2dBackwardInput op) {
        int[] inputShape = op.getInputShape();
        tensor.Conv2dOptions options = op.getOptions();
        return IntArrayValue.copyOf(new int[]{
                inputShape[0], inputShape[1], inputShape[2], inputShape[3],
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.dilationH(), options.dilationW(),
                options.groups()
        });
    }

    private SignatureComponent conv2dBackwardWeightSignature(conv2dBackwardWeight op) {
        int[] weightShape = op.getWeightShape();
        tensor.Conv2dOptions options = op.getOptions();
        return IntArrayValue.copyOf(new int[]{
                weightShape[0], weightShape[1], weightShape[2], weightShape[3],
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.dilationH(), options.dilationW(),
                options.groups()
        });
    }

    private SignatureComponent pool2dSignature(tensor.Pool2dOptions options, int kind) {
        return IntArrayValue.copyOf(new int[]{
                options.kernelH(), options.kernelW(),
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.countIncludePad() ? 1 : 0,
                kind
        });
    }

    private SignatureComponent pool2dBackwardInputSignature(tensor.Pool2dOptions options, int[] inputShape, int kind) {
        return IntArrayValue.copyOf(new int[]{
                inputShape[0], inputShape[1], inputShape[2], inputShape[3],
                options.kernelH(), options.kernelW(),
                options.strideH(), options.strideW(),
                options.padH(), options.padW(),
                options.countIncludePad() ? 1 : 0,
                kind
        });
    }

    private sealed interface SignatureComponent
            permits StructuralSignature, IdentityLeafSignature, ScalarLeafSignature,
            NoParamsSignature, AxisSignature, ReductionSignature, InputSelectorSignature,
            DoubleValue, IntArrayValue {

        String sortKey();
    }

    private record StructuralSignature(
            Operation.OpType opType,
            boolean backward,
            Boolean requiresGrad,
            Object backend,
            IntArrayValue outputShape,
            SignatureComponent parameters,
            List<SignatureComponent> inputs
    ) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "node:" + opType + ":" + backward + ":" + requiresGrad + ":" + backend + ":" + outputShape + ":" + parameters + ":" + inputs;
        }
    }

    private record IdentityLeafSignature(int identityHash) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "leaf@" + identityHash;
        }
    }

    private record ScalarLeafSignature(long bits, IntArrayValue shape) implements SignatureComponent {
        @Override
        public String sortKey() {
            return "scalar:" + bits + ":" + shape;
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
}
