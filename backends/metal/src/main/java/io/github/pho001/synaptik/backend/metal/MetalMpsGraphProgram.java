package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Immutable schema-fourteen Metal program and its canonical bounded image encoder. */
final class MetalMpsGraphProgram {
    static final int SCHEMA_VERSION = 14;
    static final int MAX_RANK = 16;
    static final int MAX_SELECTOR_EXPANSION = 16;
    static final int HEADER_BYTES = 64;
    static final int VALUE_DESCRIPTOR_BYTES = 16;
    static final int NODE_DESCRIPTOR_BYTES = 32;
    static final int MAGIC = 0x33314d53; // little-endian bytes "SM13"
    static final int NO_SECOND_INPUT = -1;
    static final int NO_AXIS = -1;

    enum AttributeKind {
        NONE(0), TARGET_SHAPE(1), PERMUTATION(2), AXIS(3), REDUCTION(4), DEPTH(5),
        WINDOW_AXIS(6), SCALAR_VALUE(7), CLAMP_RANGE(8), SCAN(9), CAST_TARGET(10),
        GATHER_ND(11), SCATTER_ELEMENTS(12), SCATTER_ND(13), SELECT(14),
        CROP_TO_SHAPE(15), PAD(16), SLICE(17), TILE(18), WINDOW_2D(19),
        PADDED_WINDOW_2D(20), FOLD_WINDOW_2D(21), WINDOW_3D(22),
        PADDED_WINDOW_3D(23), FOLD_WINDOW_3D(24), MSE(25),
        DENSE_CROSS_ENTROPY(26), INDEX_CROSS_ENTROPY(27),
        NORMALIZED_SHAPE_EPSILON(28), BATCH_NORM_INFERENCE(29),
        BATCH_NORM_TRAINING(30), SORT(31), TOP_K(32), DROPOUT(33),
        GRAPH_RNG_STATE(34), RECURRENT_DIRECTION(35), ARG_EXTREMA(36),
        MASKED_REDUCTION(37), STATISTICAL_REDUCTION(38), ATTENTION(39),
        CONV_2D(40), CONV_3D(41);

        private final int wireIdentity;
        AttributeKind(int wireIdentity) { this.wireIdentity = wireIdentity; }
        int wireIdentity() { return wireIdentity; }
    }

    enum ReductionForm {
        FULL(1), SINGLE_AXIS(2), MULTI_AXIS(3), SUM_TO_SHAPE(4);
        private final int wireIdentity;
        ReductionForm(int wireIdentity) { this.wireIdentity = wireIdentity; }
        int wireIdentity() { return wireIdentity; }
        static ReductionForm fromWireIdentity(int wireIdentity) {
            for (ReductionForm form : values()) if (form.wireIdentity == wireIdentity) return form;
            throw new IllegalArgumentException("unknown Metal reduction form");
        }
    }

    enum ValueState {
        UNAVAILABLE(0), CANONICAL(1), AFFINE_VIEW(2);
        private final int wireIdentity;
        ValueState(int wireIdentity) { this.wireIdentity = wireIdentity; }
        int wireIdentity() { return wireIdentity; }
    }

    enum NodeKind {
        NEG(1, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        ADD(2, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SUB(3, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        MUL(4, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        DIV(5, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        RESHAPE(6, 1, 1, 1, 1, AttributeKind.TARGET_SHAPE, ValueState.AFFINE_VIEW, true, true),
        EXPAND(7, 1, 1, 1, 1, AttributeKind.TARGET_SHAPE, ValueState.AFFINE_VIEW, true, true),
        PERMUTE(8, 1, 1, 1, 1, AttributeKind.PERMUTATION, ValueState.AFFINE_VIEW, true, true),
        EXPAND_DIMS(9, 1, 1, 1, 1, AttributeKind.AXIS, ValueState.AFFINE_VIEW, true, true),
        SQUEEZE(10, 1, 1, 1, 1, AttributeKind.AXIS, ValueState.AFFINE_VIEW, true, true),
        CONTIGUOUS(11, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, true, true),
        ABS(12, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SUM(13, 1, 1, 1, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false, true),
        MEAN(14, 1, 1, 1, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false, true),
        MATMUL(15, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, true, true),
        GATHER(16, 2, 2, 1, 1, AttributeKind.AXIS, ValueState.CANONICAL, false, true),
        ONE_HOT(17, 1, 1, 1, 1, AttributeKind.DEPTH, ValueState.CANONICAL, false, true),
        SCATTER_ELEMENTS(18, 3, 3, 1, 1, AttributeKind.SCATTER_ELEMENTS, ValueState.CANONICAL, false, true),
        UNFOLD_AXIS(19, 1, 1, 1, 1, AttributeKind.WINDOW_AXIS, ValueState.CANONICAL, false, true),
        GT(20, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        GE(21, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LT(22, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LE(23, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        EQ(24, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        NE(25, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        TENSOR_MIN(26, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        TENSOR_MAX(27, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SCALAR_MIN(28, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        SCALAR_MAX(29, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        CLAMP(30, 1, 1, 1, 1, AttributeKind.CLAMP_RANGE, ValueState.CANONICAL, false, true),
        REDUCTION_MIN(31, 1, 1, 1, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false, true),
        REDUCTION_MAX(32, 1, 1, 1, 1, AttributeKind.REDUCTION, ValueState.CANONICAL, false, true),
        CUM_SUM(33, 1, 1, 1, 1, AttributeKind.SCAN, ValueState.CANONICAL, false, true),
        CUM_PROD(34, 1, 1, 1, 1, AttributeKind.SCAN, ValueState.CANONICAL, false, true),
        SCALED_DOT_PRODUCT_ATTENTION(35, 3, 4, 1, 2, AttributeKind.ATTENTION),
        CONV2D(36, 2, 3, 1, 1, AttributeKind.CONV_2D),
        CONV3D(37, 2, 3, 1, 1, AttributeKind.CONV_3D),
        TENSOR_POW(38, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        CAST(39, 1, 1, 1, 1, AttributeKind.CAST_TARGET),
        IS_FINITE(40, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        IS_NAN(41, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        IS_INF(42, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LOGICAL_AND(43, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LOGICAL_OR(44, 2, 2, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LOGICAL_NOT(45, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SCALAR_ADD(46, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        SCALAR_SUB(47, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        SCALAR_MUL(48, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        SCALAR_DIV(49, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        SCALAR_POW(50, 1, 1, 1, 1, AttributeKind.SCALAR_VALUE, ValueState.CANONICAL, false, true),
        WHERE(51, 3, 3, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        RECIPROCAL(52, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LOG(53, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        LOG1P(54, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        EXP(55, 1, 1, 1, 1, AttributeKind.NONE),
        EXPM1(56, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        ERF(57, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SQRT(58, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        RSQRT(59, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        FLOOR(60, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        CEIL(61, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SIGN(62, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        RELU(63, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SIGMOID(64, 1, 1, 1, 1, AttributeKind.NONE),
        TANH(65, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        GELU(66, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        GELU_TANH_APPROXIMATION(67, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        SILU(68, 1, 1, 1, 1, AttributeKind.NONE, ValueState.CANONICAL, false, true),
        GATHER_ELEMENTS(69, 2, 2, 1, 1, AttributeKind.AXIS),
        SCATTER_ADD(70, 3, 3, 1, 1, AttributeKind.AXIS),
        GATHER_ND(71, 2, 2, 1, 1, AttributeKind.GATHER_ND),
        SCATTER_ND(72, 3, 3, 1, 1, AttributeKind.SCATTER_ND),
        SELECT(73, 1, 1, 1, 1, AttributeKind.SELECT),
        PAD(74, 1, 1, 1, 1, AttributeKind.PAD),
        SLICE(75, 1, 1, 1, 1, AttributeKind.SLICE),
        SLICE_UPDATE(76, 2, 2, 1, 1, AttributeKind.SLICE),
        CONCAT(77, 1, Integer.MAX_VALUE, 1, 1, AttributeKind.AXIS),
        STACK(78, 1, Integer.MAX_VALUE, 1, 1, AttributeKind.AXIS),
        TILE(79, 1, 1, 1, 1, AttributeKind.TILE),
        FOLD_AXIS(80, 1, 1, 1, 1, AttributeKind.WINDOW_AXIS),
        UNFOLD2D(81, 1, 1, 1, 1, AttributeKind.WINDOW_2D),
        FOLD2D(82, 1, 1, 1, 1, AttributeKind.FOLD_WINDOW_2D),
        UNFOLD3D(83, 1, 1, 1, 1, AttributeKind.WINDOW_3D),
        FOLD3D(84, 1, 1, 1, 1, AttributeKind.FOLD_WINDOW_3D),
        MEAN_SQUARED_ERROR(85, 2, 2, 1, 1, AttributeKind.MSE),
        DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS(86, 2, 2, 1, 1, AttributeKind.DENSE_CROSS_ENTROPY),
        INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS(87, 2, 2, 1, 1, AttributeKind.INDEX_CROSS_ENTROPY),
        BATCH_NORM_INFERENCE(88, 5, 5, 1, 1, AttributeKind.BATCH_NORM_INFERENCE),
        BATCH_NORM_TRAINING(89, 5, 5, 5, 5, AttributeKind.BATCH_NORM_TRAINING),
        LAYER_NORM(90, 1, 3, 1, 1, AttributeKind.NORMALIZED_SHAPE_EPSILON),
        RMS_NORM(91, 1, 2, 1, 1, AttributeKind.NORMALIZED_SHAPE_EPSILON),
        SOFTMAX(92, 1, 1, 1, 1, AttributeKind.AXIS),
        LOG_SOFTMAX(93, 1, 1, 1, 1, AttributeKind.AXIS),
        SORT(94, 1, 1, 1, 1, AttributeKind.SORT),
        ARGSORT(95, 1, 1, 1, 1, AttributeKind.SORT),
        TOP_K(96, 1, 1, 2, 2, AttributeKind.TOP_K),
        MAX_POOL2D(97, 1, 1, 1, 1, AttributeKind.WINDOW_2D),
        AVERAGE_POOL2D(98, 1, 1, 1, 1, AttributeKind.WINDOW_2D),
        MAX_POOL3D(99, 1, 1, 1, 1, AttributeKind.WINDOW_3D),
        AVERAGE_POOL3D(100, 1, 1, 1, 1, AttributeKind.WINDOW_3D),
        DROPOUT(101, 2, 2, 3, 3, AttributeKind.DROPOUT),
        INITIAL_STATE(102, 0, 0, 1, 1, AttributeKind.GRAPH_RNG_STATE),
        RNN_TANH(103, 5, 6, 2, 2, AttributeKind.RECURRENT_DIRECTION),
        GRU_RESET_AFTER(104, 5, 6, 2, 2, AttributeKind.RECURRENT_DIRECTION),
        LSTM(105, 6, 7, 3, 3, AttributeKind.RECURRENT_DIRECTION),
        PROD(106, 1, 1, 1, 1, AttributeKind.REDUCTION),
        ALL(107, 1, 1, 1, 1, AttributeKind.REDUCTION),
        ANY(108, 1, 1, 1, 1, AttributeKind.REDUCTION),
        ARG_MAX(109, 1, 1, 1, 1, AttributeKind.ARG_EXTREMA),
        ARG_MIN(110, 1, 1, 1, 1, AttributeKind.ARG_EXTREMA),
        LOG_SUM_EXP(111, 1, 1, 1, 1, AttributeKind.REDUCTION),
        VARIANCE(112, 1, 1, 1, 1, AttributeKind.STATISTICAL_REDUCTION),
        STANDARD_DEVIATION(113, 1, 1, 1, 1, AttributeKind.STATISTICAL_REDUCTION),
        L1_NORM(114, 1, 1, 1, 1, AttributeKind.REDUCTION),
        L2_NORM(115, 1, 1, 1, 1, AttributeKind.REDUCTION);

        private final int wireIdentity;
        private final int minimumInputs;
        private final int maximumInputs;
        private final int minimumOutputs;
        private final int maximumOutputs;
        private final AttributeKind attributeKind;
        private final ValueState outputState;
        private final boolean acceptsAffineView;
        private final boolean executable;

        NodeKind(int wireIdentity, int minimumInputs, int maximumInputs, int minimumOutputs,
                int maximumOutputs, AttributeKind attributeKind) {
            this(wireIdentity, minimumInputs, maximumInputs, minimumOutputs, maximumOutputs,
                    attributeKind, ValueState.CANONICAL, false, false);
        }

        NodeKind(int wireIdentity, int minimumInputs, int maximumInputs, int minimumOutputs,
                int maximumOutputs, AttributeKind attributeKind, ValueState outputState,
                boolean acceptsAffineView, boolean executable) {
            this.wireIdentity = wireIdentity;
            this.minimumInputs = minimumInputs;
            this.maximumInputs = maximumInputs;
            this.minimumOutputs = minimumOutputs;
            this.maximumOutputs = maximumOutputs;
            this.attributeKind = attributeKind;
            this.outputState = outputState;
            this.acceptsAffineView = acceptsAffineView;
            this.executable = executable;
        }

        int wireIdentity() { return wireIdentity; }
        int inputCount() { return minimumInputs == maximumInputs ? minimumInputs : -1; }
        int minimumInputs() { return minimumInputs; }
        int maximumInputs() { return maximumInputs; }
        int minimumOutputs() { return minimumOutputs; }
        int maximumOutputs() { return maximumOutputs; }
        AttributeKind attributeKind() { return attributeKind; }
        ValueState outputState() { return outputState; }
        boolean isAffine() { return outputState == ValueState.AFFINE_VIEW; }
        boolean executable() { return executable; }
        boolean isCustomProgramOperation() {
            return wireIdentity >= 20 && wireIdentity <= 34
                    || wireIdentity >= 40 && wireIdentity <= 45
                    || wireIdentity == 51
                    || wireIdentity >= 60 && wireIdentity <= 63;
        }
        boolean accepts(ValueState inputState) {
            return inputState == ValueState.CANONICAL
                    || (inputState == ValueState.AFFINE_VIEW && acceptsAffineView);
        }
        boolean acceptsCardinality(int inputs, int outputs) {
            if (this == LAYER_NORM && inputs == 2) return false;
            return inputs >= minimumInputs && inputs <= maximumInputs
                    && outputs >= minimumOutputs && outputs <= maximumOutputs;
        }
    }

    static final class ValueDescriptor {
        private final DataType dataType;
        private final long[] dimensions;
        private final boolean requiresGrad;

        ValueDescriptor(DataType dataType, long[] dimensions, boolean requiresGrad) {
            this.dataType = Objects.requireNonNull(dataType, "dataType");
            this.dimensions = Objects.requireNonNull(dimensions, "dimensions").clone();
            if (this.dimensions.length > MAX_RANK) {
                throw new IllegalArgumentException("Metal value rank must be in 0..16");
            }
            long elements = 1L;
            try {
                for (long dimension : this.dimensions) {
                    if (dimension <= 0L) {
                        throw new IllegalArgumentException("Metal value dimensions must be positive");
                    }
                    elements = Math.multiplyExact(elements, dimension);
                }
                Math.multiplyExact(elements, dataType.byteWidth());
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException("Metal value byte geometry overflows", overflow);
            }
            this.requiresGrad = requiresGrad;
        }

        static ValueDescriptor from(TensorDescriptor descriptor) {
            Objects.requireNonNull(descriptor, "descriptor");
            if (!descriptor.shape().isFullyStatic()) {
                throw new IllegalArgumentException("Metal values must have fully static shapes");
            }
            return new ValueDescriptor(descriptor.dataType(), descriptor.shape().toLongArray(),
                    descriptor.requiresGrad());
        }

        DataType dataType() { return dataType; }
        int rank() { return dimensions.length; }
        long[] dimensions() { return dimensions.clone(); }
        boolean requiresGrad() { return requiresGrad; }
        long elementCount() {
            long result = 1L;
            for (long dimension : dimensions) result = Math.multiplyExact(result, dimension);
            return result;
        }
        long byteCount() { return Math.multiplyExact(elementCount(), dataType.byteWidth()); }
    }

    static final class Node {
        private final NodeKind kind;
        private final int[] inputs;
        private final int[] outputs;
        private final AttributeKind attributeKind;
        private final long[] attributeWords;

        private Node(NodeKind kind, int[] inputs, int[] outputs, AttributeKind attributeKind,
                long[] attributeWords) {
            this.kind = Objects.requireNonNull(kind, "kind");
            this.inputs = Objects.requireNonNull(inputs, "inputs").clone();
            this.outputs = Objects.requireNonNull(outputs, "outputs").clone();
            this.attributeKind = Objects.requireNonNull(attributeKind, "attributeKind");
            this.attributeWords = Objects.requireNonNull(attributeWords, "attributeWords").clone();
            if (!kind.acceptsCardinality(this.inputs.length, this.outputs.length)) {
                throw new IllegalArgumentException("node cardinality disagrees with operation kind");
            }
            for (int input : this.inputs) if (input < 0) throw new IllegalArgumentException("input index must be non-negative");
            for (int output : this.outputs) if (output < 0) throw new IllegalArgumentException("output index must be non-negative");
            if (!acceptsAttribute(kind, attributeKind)) {
                throw new IllegalArgumentException("attribute kind disagrees with operation kind");
            }
            validateAttributes();
        }

        static Node generic(NodeKind kind, int[] inputs, int[] outputs,
                AttributeKind attributeKind, long[] attributeWords) {
            return new Node(kind, inputs, outputs, attributeKind, attributeWords);
        }
        static Node neg(int input, int output) { return noAttributes(NodeKind.NEG, input, output); }
        static Node abs(int input, int output) { return noAttributes(NodeKind.ABS, input, output); }
        static Node binary(NodeKind kind, int left, int right, int output) {
            Objects.requireNonNull(kind, "kind");
            if (kind.inputCount() != 2 || kind.attributeKind() != AttributeKind.NONE) {
                throw new IllegalArgumentException("binary node kind must have two inputs");
            }
            return new Node(kind, new int[] {left, right}, new int[] {output},
                    AttributeKind.NONE, new long[0]);
        }
        static Node matmul(int left, int right, int output) { return binary(NodeKind.MATMUL, left, right, output); }
        static Node contiguous(int input, int output) { return noAttributes(NodeKind.CONTIGUOUS, input, output); }
        static Node targetShape(NodeKind kind, int input, int output, long[] dimensions) {
            Objects.requireNonNull(dimensions, "dimensions");
            long[] words = new long[dimensions.length + 1];
            words[0] = dimensions.length;
            System.arraycopy(dimensions, 0, words, 1, dimensions.length);
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.TARGET_SHAPE, words);
        }
        static Node permutation(int input, int output, List<Integer> axes) {
            long[] values = longValues(Objects.requireNonNull(axes, "axes"));
            long[] words = new long[values.length + 1];
            words[0] = values.length;
            System.arraycopy(values, 0, words, 1, values.length);
            return new Node(NodeKind.PERMUTE, new int[] {input}, new int[] {output}, AttributeKind.PERMUTATION, words);
        }
        static Node axis(NodeKind kind, int input, int output, int axis) {
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.AXIS,
                    new long[] {Integer.toUnsignedLong(axis)});
        }
        static Node reduction(NodeKind kind, int input, int output, ReductionForm form,
                List<Integer> axes, boolean keepDimensions) {
            Objects.requireNonNull(form, "form");
            if (form == ReductionForm.SUM_TO_SHAPE) throw new IllegalArgumentException("sum-to-Shape requires dimensions");
            long[] items = longValues(Objects.requireNonNull(axes, "axes"));
            long[] words = new long[3 + items.length];
            words[0] = form.wireIdentity();
            words[1] = keepDimensions ? 1L : 0L;
            words[2] = items.length;
            System.arraycopy(items, 0, words, 3, items.length);
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.REDUCTION, words);
        }
        static Node sumToShape(int input, int output, long[] targetDimensions) {
            Objects.requireNonNull(targetDimensions, "targetDimensions");
            long[] words = new long[3 + targetDimensions.length];
            words[0] = ReductionForm.SUM_TO_SHAPE.wireIdentity();
            words[1] = 0L;
            words[2] = targetDimensions.length;
            System.arraycopy(targetDimensions, 0, words, 3, targetDimensions.length);
            return new Node(NodeKind.SUM, new int[] {input}, new int[] {output}, AttributeKind.REDUCTION, words);
        }
        static Node gather(int data, int indices, int output, int axis) {
            return new Node(NodeKind.GATHER, new int[] {data, indices}, new int[] {output},
                    AttributeKind.AXIS, new long[] {Integer.toUnsignedLong(axis)});
        }
        static Node scatterElements(int data, int indices, int updates, int output, int axis) {
            return new Node(NodeKind.SCATTER_ELEMENTS, new int[] {data, indices, updates},
                    new int[] {output}, AttributeKind.SCATTER_ELEMENTS,
                    new long[] {Integer.toUnsignedLong(axis), 1L});
        }
        static Node oneHot(int indices, int output, long depth) {
            return new Node(NodeKind.ONE_HOT, new int[] {indices}, new int[] {output},
                    AttributeKind.DEPTH, new long[] {depth});
        }
        static Node unfoldAxis(int input, int output, int axis, long size, long step) {
            return new Node(NodeKind.UNFOLD_AXIS, new int[] {input}, new int[] {output},
                    AttributeKind.WINDOW_AXIS,
                    new long[] {Integer.toUnsignedLong(axis), size, step});
        }
        static Node scalarValue(NodeKind kind, int input, int output, int rawBits) {
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.SCALAR_VALUE,
                    new long[] {1L, Integer.toUnsignedLong(rawBits)});
        }
        static Node clamp(int input, int output, int lower, int upper) {
            return new Node(NodeKind.CLAMP, new int[] {input}, new int[] {output}, AttributeKind.CLAMP_RANGE,
                    new long[] {1L, Integer.toUnsignedLong(lower), 1L, Integer.toUnsignedLong(upper)});
        }
        static Node scan(NodeKind kind, int input, int output, int axis,
                boolean exclusive, boolean reverse) {
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.SCAN,
                    new long[] {Integer.toUnsignedLong(axis), exclusive ? 1L : 0L, reverse ? 1L : 0L});
        }
        private static Node noAttributes(NodeKind kind, int input, int output) {
            return new Node(kind, new int[] {input}, new int[] {output}, AttributeKind.NONE, new long[0]);
        }
        private static long[] longValues(List<Integer> values) {
            long[] result = new long[values.size()];
            for (int i = 0; i < values.size(); i++) result[i] = Objects.requireNonNull(values.get(i), "value");
            return result;
        }
        private static boolean acceptsAttribute(NodeKind kind, AttributeKind attribute) {
            if ((kind == NodeKind.SLICE || kind == NodeKind.SLICE_UPDATE)
                    && attribute == AttributeKind.CROP_TO_SHAPE) return true;
            if (kind == NodeKind.UNFOLD2D && attribute == AttributeKind.PADDED_WINDOW_2D) return true;
            if (kind == NodeKind.UNFOLD3D && attribute == AttributeKind.PADDED_WINDOW_3D) return true;
            return kind.attributeKind() == attribute;
        }
        private void validateAttributes() {
            switch (attributeKind) {
                case NONE -> requireWords(0);
                case TARGET_SHAPE -> validateShapeWords(attributeWords);
                case PERMUTATION -> validatePermutation();
                case AXIS, MASKED_REDUCTION -> {
                    requireWords(1);
                    requireAxis(attributeWords[0]);
                }
                case REDUCTION -> validateReduction();
                case DEPTH -> {
                    requireWords(1);
                    if (attributeWords[0] <= 0L) throw malformed();
                }
                case WINDOW_AXIS -> {
                    requireWords(3);
                    requireAxis(attributeWords[0]);
                    if (attributeWords[1] <= 0L || attributeWords[2] <= 0L
                            || (kind == NodeKind.UNFOLD_AXIS
                            && attributeWords[1] > MAX_SELECTOR_EXPANSION)) throw malformed();
                }
                case SCALAR_VALUE -> {
                    requireWords(2);
                    validateScalar(attributeWords, 0);
                    if (kind.executable() && attributeWords[0] != 1L) throw malformed();
                }
                case CLAMP_RANGE -> {
                    requireWords(4);
                    validateScalar(attributeWords, 0);
                    validateScalar(attributeWords, 2);
                    if (kind.executable()
                            && (attributeWords[0] != 1L || attributeWords[2] != 1L)) {
                        throw malformed();
                    }
                }
                case SCAN -> {
                    requireWords(3);
                    requireAxis(attributeWords[0]);
                    requireBoolean(attributeWords[1]);
                    requireBoolean(attributeWords[2]);
                }
                case CAST_TARGET -> {
                    requireWords(1);
                    requireType(attributeWords[0]);
                }
                case GATHER_ND -> {
                    requireWords(1);
                    requireUnsignedInt(attributeWords[0]);
                    if (attributeWords[0] > MAX_RANK) throw malformed();
                }
                case SCATTER_ELEMENTS -> {
                    requireWords(2);
                    requireAxis(attributeWords[0]);
                    requireEnum(attributeWords[1], 1L, 5L);
                    if (kind.executable() && attributeWords[1] != 1L) throw malformed();
                }
                case SCATTER_ND -> {
                    requireWords(2);
                    requireUnsignedInt(attributeWords[0]);
                    requireEnum(attributeWords[1], 1L, 5L);
                }
                case SELECT -> {
                    requireWords(2);
                    requireAxis(attributeWords[0]);
                }
                case CROP_TO_SHAPE -> {
                    int offset = validateShape(attributeWords, 0);
                    if (validateShape(attributeWords, offset) != attributeWords.length) throw malformed();
                }
                case PAD -> validatePad();
                case SLICE -> validateSlice();
                case TILE -> validatePositiveList();
                case WINDOW_2D -> validateWindow(2, false, false);
                case PADDED_WINDOW_2D -> validateWindow(2, true, false);
                case FOLD_WINDOW_2D -> validateWindow(2, false, true);
                case WINDOW_3D -> validateWindow(3, false, false);
                case PADDED_WINDOW_3D -> validateWindow(3, true, false);
                case FOLD_WINDOW_3D -> validateWindow(3, false, true);
                case MSE -> {
                    requireWords(1);
                    requireEnum(attributeWords[0], 1L, 3L);
                }
                case DENSE_CROSS_ENTROPY -> {
                    requireWords(2);
                    requireAxis(attributeWords[0]);
                    requireEnum(attributeWords[1], 1L, 3L);
                }
                case INDEX_CROSS_ENTROPY -> validateIndexCrossEntropy();
                case NORMALIZED_SHAPE_EPSILON -> {
                    int offset = validateShape(attributeWords, 0);
                    if (offset + 2 != attributeWords.length) throw malformed();
                    validateScalar(attributeWords, offset);
                }
                case BATCH_NORM_INFERENCE -> {
                    requireWords(3);
                    requireAxis(attributeWords[0]);
                    validateScalar(attributeWords, 1);
                }
                case BATCH_NORM_TRAINING -> {
                    requireWords(5);
                    requireAxis(attributeWords[0]);
                    validateScalar(attributeWords, 1);
                    validateScalar(attributeWords, 3);
                }
                case SORT -> {
                    requireWords(2);
                    requireAxis(attributeWords[0]);
                    requireBoolean(attributeWords[1]);
                }
                case TOP_K -> {
                    requireWords(4);
                    requireAxis(attributeWords[0]);
                    if (attributeWords[1] <= 0L) throw malformed();
                    requireBoolean(attributeWords[2]);
                    requireBoolean(attributeWords[3]);
                }
                case DROPOUT -> {
                    requireWords(1);
                    double probability = Double.longBitsToDouble(attributeWords[0]);
                    if (!Double.isFinite(probability) || probability < 0.0d
                            || probability >= 1.0d) throw malformed();
                }
                case GRAPH_RNG_STATE -> requireWords(2);
                case RECURRENT_DIRECTION -> {
                    requireWords(1);
                    requireEnum(attributeWords[0], 1L, 2L);
                }
                case ARG_EXTREMA -> {
                    requireWords(3);
                    requireAxis(attributeWords[0]);
                    requireBoolean(attributeWords[1]);
                    requireEnum(attributeWords[2], 1L, 2L);
                }
                case STATISTICAL_REDUCTION -> validateStatisticalReduction();
                case ATTENTION -> validateAttention();
                case CONV_2D -> validateConvolution(2);
                case CONV_3D -> validateConvolution(3);
            }
        }
        private void validatePermutation() {
            int count = checkedCount(attributeWords, 0);
            if (count < 1 || count > MAX_RANK) throw malformed();
            requireWords(count + 1);
            validateAxes(Arrays.copyOfRange(attributeWords, 1, attributeWords.length),
                    count, true);
        }

        private void validatePad() {
            int rank = checkedCount(attributeWords, 0);
            if (rank > MAX_RANK || attributeWords.length != 1 + rank * 2 + 2) {
                throw malformed();
            }
            for (int index = 1; index <= rank * 2; index++) {
                if (attributeWords[index] < 0L) throw malformed();
            }
            validateScalar(attributeWords, 1 + rank * 2);
        }

        private void validateSlice() {
            int count = checkedCount(attributeWords, 0);
            if (count < 1 || count > MAX_RANK || attributeWords.length != 1 + count * 4) {
                throw malformed();
            }
            boolean[] seen = new boolean[MAX_RANK];
            for (int index = 0; index < count; index++) {
                if (attributeWords[1 + count + index] <= 0L) throw malformed();
                long axis = attributeWords[1 + count * 2 + index];
                if (axis < 0L || axis >= MAX_RANK || seen[(int) axis]) throw malformed();
                seen[(int) axis] = true;
                if (attributeWords[1 + count * 3 + index] == 0L) throw malformed();
            }
        }

        private void validatePositiveList() {
            int count = checkedCount(attributeWords, 0);
            if (count > MAX_RANK || attributeWords.length != count + 1) throw malformed();
            for (int index = 1; index < attributeWords.length; index++) {
                if (attributeWords[index] <= 0L) throw malformed();
            }
        }

        private void validateWindow(int dimensions, boolean padded, boolean fold) {
            int offset = fold ? validateShape(attributeWords, 0) : 0;
            int baseWords = dimensions * 4 + 1;
            int words = baseWords + (padded ? 2 : 0);
            if (offset + words != attributeWords.length) throw malformed();
            for (int index = 0; index < dimensions * 2; index++) {
                if (attributeWords[offset + index] <= 0L) throw malformed();
            }
            for (int index = dimensions * 2; index < dimensions * 3; index++) {
                if (attributeWords[offset + index] < 0L) throw malformed();
            }
            for (int index = dimensions * 3; index < dimensions * 4; index++) {
                if (attributeWords[offset + index] <= 0L) throw malformed();
            }
            requireBoolean(attributeWords[offset + dimensions * 4]);
            if (padded) validateScalar(attributeWords, offset + baseWords);
        }

        private void validateIndexCrossEntropy() {
            if (attributeWords.length < 3) throw malformed();
            requireAxis(attributeWords[0]);
            requireEnum(attributeWords[1], 1L, 3L);
            long present = attributeWords[2];
            if (present == 0L) {
                requireWords(3);
            } else if (present == 1L) {
                requireWords(5);
                validateScalar(attributeWords, 3);
            } else {
                throw malformed();
            }
        }

        private void validateStatisticalReduction() {
            int count = checkedCount(attributeWords, 0);
            if (count > MAX_RANK || attributeWords.length != count + 3) throw malformed();
            validateAxes(Arrays.copyOfRange(attributeWords, 1, count + 1),
                    MAX_RANK, false);
            requireBoolean(attributeWords[count + 1]);
            if (attributeWords[count + 2] < 0L) throw malformed();
        }

        private void validateAttention() {
            if (attributeWords.length < 2) throw malformed();
            long present = attributeWords[0];
            int causalOffset;
            if (present == 0L) {
                if (attributeWords.length != 2) throw malformed();
                causalOffset = 1;
            } else if (present == 1L) {
                if (attributeWords.length != 4) throw malformed();
                validateScalar(attributeWords, 1);
                long type = attributeWords[1];
                if (type != 1L && type != 4L && type != 5L) throw malformed();
                causalOffset = 3;
            } else {
                throw malformed();
            }
            requireBoolean(attributeWords[causalOffset]);
        }

        private void validateConvolution(int dimensions) {
            requireWords(dimensions * 3 + 1);
            for (int index = 0; index < dimensions; index++) {
                if (attributeWords[index] <= 0L) throw malformed();
            }
            for (int index = dimensions; index < dimensions * 2; index++) {
                if (attributeWords[index] < 0L) throw malformed();
            }
            for (int index = dimensions * 2; index < dimensions * 3; index++) {
                if (attributeWords[index] <= 0L) throw malformed();
            }
            if (attributeWords[dimensions * 3] <= 0L) throw malformed();
        }
        private void validateReduction() {
            if (attributeWords.length < 3
                    || attributeWords[0] < 1L || attributeWords[0] > 4L) throw malformed();
            ReductionForm form = ReductionForm.fromWireIdentity((int) attributeWords[0]);
            requireBoolean(attributeWords[1]);
            int count = checkedCount(attributeWords, 2);
            if (count > MAX_RANK) throw malformed();
            requireWords(3 + count);
            long[] items = Arrays.copyOfRange(attributeWords, 3, attributeWords.length);
            switch (form) {
                case FULL -> { if (count != 0 || attributeWords[1] != 0L) throw malformed(); }
                case SINGLE_AXIS -> {
                    if (count != 1) throw malformed();
                    validateAxes(items, MAX_RANK, false);
                }
                case MULTI_AXIS -> validateAxes(items, MAX_RANK, false);
                case SUM_TO_SHAPE -> {
                    if (kind != NodeKind.SUM || attributeWords[1] != 0L) throw malformed();
                    for (long dimension : items) if (dimension <= 0L) throw malformed();
                }
            }
        }
        private static void validateShapeWords(long[] words) {
            int end = validateShape(words, 0);
            if (words[0] < 1L || end != words.length) throw malformed();
        }
        private static int validateShape(long[] words, int offset) {
            int rank = checkedCount(words, offset);
            if (rank > MAX_RANK || offset + 1 + rank > words.length) throw malformed();
            for (int index = offset + 1; index < offset + 1 + rank; index++) {
                if (words[index] <= 0L) throw malformed();
            }
            return offset + 1 + rank;
        }
        private static void validateScalar(long[] words, int offset) {
            if (offset + 2 > words.length) throw malformed();
            long type = words[offset];
            long bits = words[offset + 1];
            requireType(type);
            if ((type == 1L || type == 2L) && (bits & ~0xffff_ffffL) != 0L
                    || type == 3L && bits > 1L
                    || type == 5L && (bits & ~0xffffL) != 0L) throw malformed();
        }
        private static void validateAxes(long[] axes, int bound, boolean complete) {
            boolean[] seen = new boolean[bound];
            for (long axis : axes) {
                if (axis < 0L || axis >= bound || seen[(int) axis]) throw malformed();
                seen[(int) axis] = true;
            }
            if (complete && axes.length != bound) throw malformed();
        }
        private static int checkedCount(long[] words, int index) {
            if (index >= words.length || words[index] < 0L || words[index] > Integer.MAX_VALUE) throw malformed();
            return (int) words[index];
        }
        private static void requireAxis(long axis) { if (axis < 0L || axis >= MAX_RANK) throw malformed(); }
        private void requireWords(int count) { if (attributeWords.length != count) throw malformed(); }
        private static IllegalArgumentException malformed() { return new IllegalArgumentException("node attributes are malformed"); }

        NodeKind kind() { return kind; }
        int[] inputs() { return inputs.clone(); }
        int[] outputs() { return outputs.clone(); }
        private static void requireBoolean(long value) {
            if (value != 0L && value != 1L) throw malformed();
        }
        private static void requireEnum(long value, long minimum, long maximum) {
            if (value < minimum || value > maximum) throw malformed();
        }
        private static void requireType(long value) {
            requireEnum(value, 1L, 6L);
        }
        private static void requireUnsignedInt(long value) {
            if ((value & ~0xffff_ffffL) != 0L) throw malformed();
        }
        AttributeKind attributeKind() { return attributeKind; }
        long[] attributeWords() { return attributeWords.clone(); }
        int firstInputIndex() { return inputs[0]; }
        int secondInputIndex() { return inputs.length >= 2 ? inputs[1] : NO_SECOND_INPUT; }
        int outputIndex() { return outputs[0]; }
        int auxiliary() {
            if (kind == NodeKind.SCATTER_ELEMENTS || kind == NodeKind.WHERE) return inputs[2];
            if (attributeKind == AttributeKind.REDUCTION) return Math.toIntExact(attributeWords[1]);
            return 0;
        }
        int axis() {
            return switch (attributeKind) {
                case AXIS, WINDOW_AXIS, SCAN, SCATTER_ELEMENTS -> Math.toIntExact(attributeWords[0]);
                case REDUCTION -> Math.toIntExact(attributeWords[0]);
                default -> NO_AXIS;
            };
        }
        int attributeCount() {
            return switch (attributeKind) {
                case NONE -> 0;
                case TARGET_SHAPE, PERMUTATION -> Math.toIntExact(attributeWords[0]);
                case AXIS, DEPTH, SCALAR_VALUE, SCATTER_ELEMENTS -> 1;
                case REDUCTION -> Math.toIntExact(attributeWords[2]);
                case WINDOW_AXIS -> 3;
                case CLAMP_RANGE, SCAN -> 2;
                default -> attributeWords.length;
            };
        }
        long[] attributeValues() {
            return switch (attributeKind) {
                case TARGET_SHAPE, PERMUTATION -> Arrays.copyOfRange(attributeWords, 1, attributeWords.length);
                case REDUCTION -> Arrays.copyOfRange(attributeWords, 3, attributeWords.length);
                case WINDOW_AXIS -> new long[] {attributeWords[1], attributeWords[2]};
                case SCALAR_VALUE -> new long[] {attributeWords[1]};
                case CLAMP_RANGE -> new long[] {attributeWords[1], attributeWords[3]};
                case SCAN -> new long[] {attributeWords[1], attributeWords[2]};
                case DEPTH -> attributeWords.clone();
                default -> new long[0];
            };
        }
    }

    private final List<Node> nodes;

    MetalMpsGraphProgram(List<Node> nodes) {
        this.nodes = List.copyOf(nodes);
        if (this.nodes.isEmpty()) throw new IllegalArgumentException("Metal program must contain a node");
    }

    List<Node> nodes() { return nodes; }

    MemorySegment encodeNative(
            Arena arena,
            List<ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            MetalPreparedRoute route) {
        Objects.requireNonNull(arena, "arena");
        Layout layout = layout(values, feeds, targets);
        MemorySegment segment = arena.allocate(layout.totalBytes, Long.BYTES);
        write(segment.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN),
                values, feeds, targets, layout, route);
        return segment;
    }

    byte[] encodedProgramImage(List<ValueDescriptor> values, int[] feeds, int[] targets) {
        return encodedProgramImage(values, feeds, targets, MetalPreparedRoute.MPSGRAPH);
    }

    byte[] encodedProgramImage(
            List<ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            MetalPreparedRoute route) {
        Layout layout = layout(values, feeds, targets);
        ByteBuffer buffer = ByteBuffer.allocate(layout.totalBytes).order(ByteOrder.LITTLE_ENDIAN);
        write(buffer, values, feeds, targets, layout, route);
        return buffer.array();
    }

    void updateDigest(
            MessageDigest digest,
            List<ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            MetalPreparedRoute route) {
        Objects.requireNonNull(digest, "digest");
        digest.update(encodedProgramImage(values, feeds, targets, route));
        for (Node node : nodes) {
            switch (node.kind()) {
                case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV -> {
                    updateDigestInt(digest, 0x53434c52); // SCLR
                    updateDigestInt(digest, node.kind().wireIdentity());
                    updateDigestInt(digest, scalarPrimitive(node.kind()).wireIdentity());
                    updateDigestInt(digest, 1); // input is primary, scalar is secondary
                    updateDigestInt(digest, dataTypeWire(DataType.FLOAT32));
                    updateDigestInt(digest, 1); // rank
                    updateDigestInt(digest, 1); // sole extent
                    updateDigestInt(digest, Float.BYTES);
                    updateDigestInt(digest, (int) node.attributeWords[1]);
                }
                case RECIPROCAL -> {
                    updateDigestInt(digest, 0x52435052); // RCPR
                    updateDigestInt(digest, node.kind().wireIdentity());
                    updateDigestInt(digest, NodeKind.DIV.wireIdentity());
                    updateDigestInt(digest, 2); // scalar one is primary, input is secondary
                    updateDigestInt(digest, dataTypeWire(DataType.FLOAT32));
                    updateDigestInt(digest, 1); // rank
                    updateDigestInt(digest, 1); // sole extent
                    updateDigestInt(digest, Float.BYTES);
                    updateDigestInt(digest, 0x3f80_0000);
                }
                default -> {
                    // The schema image completely binds every other lowering.
                }
            }
        }
    }

    private static NodeKind scalarPrimitive(NodeKind kind) {
        return switch (kind) {
            case SCALAR_ADD -> NodeKind.ADD;
            case SCALAR_SUB -> NodeKind.SUB;
            case SCALAR_MUL -> NodeKind.MUL;
            case SCALAR_DIV -> NodeKind.DIV;
            default -> throw new IllegalArgumentException("not scalar arithmetic: " + kind);
        };
    }

    private static void updateDigestInt(MessageDigest digest, int value) {
        digest.update((byte) (value >>> 24));
        digest.update((byte) (value >>> 16));
        digest.update((byte) (value >>> 8));
        digest.update((byte) value);
    }

    private Layout layout(List<ValueDescriptor> values, int[] feeds, int[] targets) {
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(feeds, "feeds");
        Objects.requireNonNull(targets, "targets");
        validateTopology(values.size(), feeds, targets);
        if (values.isEmpty() || targets.length == 0) throw new IllegalArgumentException("program values and targets must be non-empty");
        long dimensions = 0L;
        long references = (long) feeds.length + targets.length;
        long attributes = 0L;
        for (ValueDescriptor value : values) dimensions = Math.addExact(dimensions, Objects.requireNonNull(value, "value").rank());
        for (Node node : nodes) {
            references = Math.addExact(references, Math.addExact(node.inputs.length, node.outputs.length));
            attributes = Math.addExact(attributes, node.attributeWords.length);
        }
        long bytes = HEADER_BYTES;
        bytes = Math.addExact(bytes, Math.multiplyExact((long) values.size(), VALUE_DESCRIPTOR_BYTES));
        bytes = Math.addExact(bytes, Math.multiplyExact((long) nodes.size(), NODE_DESCRIPTOR_BYTES));
        bytes = Math.addExact(bytes, Math.multiplyExact(dimensions, Long.BYTES));
        bytes = Math.addExact(bytes, Math.multiplyExact(references, Integer.BYTES));
        if ((bytes & 7L) != 0L) bytes = Math.addExact(bytes, Integer.BYTES);
        bytes = Math.addExact(bytes, Math.multiplyExact(attributes, Long.BYTES));
        if (bytes > Integer.MAX_VALUE || dimensions > Integer.MAX_VALUE
                || references > Integer.MAX_VALUE || attributes > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Metal program image exceeds signed 32-bit bounds");
        }
        return new Layout((int) bytes, (int) dimensions, (int) references, (int) attributes);
    }

    private void validateTopology(int valueCount, int[] feeds, int[] targets) {
        boolean[] available = new boolean[valueCount];
        boolean[] produced = new boolean[valueCount];
        boolean[] consumed = new boolean[valueCount];
        for (int feed : feeds) {
            requireValueIndex(feed, valueCount);
            if (available[feed]) throw new IllegalArgumentException("duplicate program feed");
            available[feed] = true;
        }
        for (Node node : nodes) {
            for (int input : node.inputs) {
                requireValueIndex(input, valueCount);
                if (!available[input]) {
                    throw new IllegalArgumentException("program input is unavailable");
                }
                consumed[input] = true;
            }
            for (int output : node.outputs) {
                requireValueIndex(output, valueCount);
                if (available[output]) {
                    throw new IllegalArgumentException("program value has multiple producers");
                }
                available[output] = true;
                produced[output] = true;
            }
        }
        boolean[] targeted = new boolean[valueCount];
        for (int target : targets) {
            requireValueIndex(target, valueCount);
            if (!produced[target] || targeted[target]) {
                throw new IllegalArgumentException("program target must be a unique produced value");
            }
            targeted[target] = true;
            consumed[target] = true;
        }
        for (int value = 0; value < valueCount; value++) {
            if (!available[value] || !consumed[value]) {
                throw new IllegalArgumentException("program contains an unused value");
            }
        }
    }

    private static void requireValueIndex(int value, int valueCount) {
        if (value < 0 || value >= valueCount) {
            throw new IllegalArgumentException("program value index is out of range");
        }
    }

    private void write(ByteBuffer out, List<ValueDescriptor> values, int[] feeds, int[] targets,
            Layout layout, MetalPreparedRoute route) {
        Objects.requireNonNull(route, "route");
        if (route == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
            throw new IllegalArgumentException("singleton custom NEG does not use a program image");
        }
        out.putInt(MAGIC).putInt(SCHEMA_VERSION).putInt(layout.totalBytes)
                .putInt(values.size()).putInt(nodes.size()).putInt(feeds.length).putInt(targets.length)
                .putInt(layout.dimensionCount).putInt(layout.referenceCount).putInt(layout.attributeCount)
                .putInt(route.wireIdentity());
        for (int i = 0; i < 5; i++) out.putInt(0);
        int dimensionOffset = 0;
        for (ValueDescriptor value : values) {
            out.putInt(dataTypeWire(value.dataType())).putInt(value.rank()).putInt(dimensionOffset)
                    .putInt(value.requiresGrad() ? 1 : 0);
            dimensionOffset = Math.addExact(dimensionOffset, value.rank());
        }
        int referenceOffset = Math.addExact(feeds.length, targets.length);
        int attributeOffset = 0;
        for (Node node : nodes) {
            out.putInt(node.kind.wireIdentity()).putInt(node.attributeKind.wireIdentity())
                    .putInt(referenceOffset).putInt(node.inputs.length)
                    .putInt(Math.addExact(referenceOffset, node.inputs.length)).putInt(node.outputs.length)
                    .putInt(attributeOffset).putInt(node.attributeWords.length);
            referenceOffset = Math.addExact(referenceOffset, Math.addExact(node.inputs.length, node.outputs.length));
            attributeOffset = Math.addExact(attributeOffset, node.attributeWords.length);
        }
        for (ValueDescriptor value : values) for (long dimension : value.dimensions) out.putLong(dimension);
        for (int feed : feeds) out.putInt(feed);
        for (int target : targets) out.putInt(target);
        for (Node node : nodes) {
            for (int input : node.inputs) out.putInt(input);
            for (int output : node.outputs) out.putInt(output);
        }
        if ((out.position() & 7) != 0) out.putInt(0);
        for (Node node : nodes) for (long word : node.attributeWords) out.putLong(word);
        if (out.position() != layout.totalBytes) throw new AssertionError("program image size mismatch");
    }

    static int dataTypeWire(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> 1;
            case INT32 -> 2;
            case BOOL -> 3;
            case FLOAT64 -> 4;
            case BFLOAT16 -> 5;
            case INT64 -> 6;
        };
    }

    private record Layout(int totalBytes, int dimensionCount, int referenceCount, int attributeCount) {}
}
