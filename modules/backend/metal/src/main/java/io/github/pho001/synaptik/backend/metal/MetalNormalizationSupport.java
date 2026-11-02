package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * FLOAT32 normalization/loss admission, typed wire lowering and independent raw-node preflight.
 * Reads may have checked nonnegative affine strides; outputs must be canonical. No low-valued
 * occurrence is admitted, and canonical saved batch statistics retain their exact Compiler roles.
 */
final class MetalNormalizationSupport {
    private MetalNormalizationSupport() {}

    /** @param operation typed occurrence to classify @return whether this owner recognizes its kind */
    static boolean handles(Operation operation) {
        return operation.kind() instanceof SoftmaxKind || operation.kind() instanceof BatchNormKind
                || operation.kind() == LayerNormKind.LAYER_NORM || operation.kind() == RmsNormKind.RMS_NORM
                || operation.kind() == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                || operation.kind() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
    }

    /**
     * @param operation complete typed occurrence
     * @param inputs ordered logical input descriptors
     * @param outputs all ordered canonical output descriptors, including all five BN siblings
     * @return whether the complete occurrence has the component's FLOAT32 physical contract
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        if (!handles(operation)) return false;
        try {
            int[] in = sequence(0, inputs.size()), out = sequence(inputs.size(), outputs.size());
            lower(operation, in, out, inputs, outputs);
            return true;
        } catch (IllegalArgumentException | ArithmeticException failure) {
            return false;
        }
    }

    /**
     * Encodes a Model occurrence without losing scalar bits or canonical sibling ordering.
     * @param operation non-null recognized typed occurrence
     * @param inputs physical program input value identifiers, in semantic role order
     * @param outputs physical program output identifiers, in canonical sibling order
     * @param inputDescriptors complete ordered Model input descriptors
     * @param outputDescriptors complete ordered Model output descriptors
     * @return authenticated ordinary node with unchanged operation and attribute identities
     * @throws IllegalArgumentException if the occurrence is outside this FLOAT32 domain
     */
    static MetalMpsGraphProgram.Node lower(Operation operation, int[] inputs, int[] outputs,
            List<TensorDescriptor> inputDescriptors, List<TensorDescriptor> outputDescriptors) {
        require(handles(operation) && inputs.length == inputDescriptors.size()
                && outputs.length == outputDescriptors.size());
        MetalMpsGraphProgram.NodeKind kind;
        MetalMpsGraphProgram.AttributeKind attr;
        long[] words;
        if (operation.kind() instanceof SoftmaxKind softmax && operation.attrs() instanceof SoftmaxAttrs a) {
            kind = softmax == SoftmaxKind.SOFTMAX ? MetalMpsGraphProgram.NodeKind.SOFTMAX
                    : MetalMpsGraphProgram.NodeKind.LOG_SOFTMAX;
            attr = MetalMpsGraphProgram.AttributeKind.AXIS;
            words = new long[] {a.axis()};
        } else if (operation.kind() == LayerNormKind.LAYER_NORM && operation.attrs() instanceof LayerNormAttrs a) {
            require(inputs.length == 1);
            kind = MetalMpsGraphProgram.NodeKind.LAYER_NORM;
            attr = MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON;
            words = normalized(a.normalizedShape(), a.epsilon());
        } else if (operation.kind() == LayerNormKind.LAYER_NORM && operation.attrs() instanceof AffineLayerNormAttrs a) {
            require(inputs.length == 3);
            kind = MetalMpsGraphProgram.NodeKind.LAYER_NORM;
            attr = MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON;
            words = normalized(a.normalizedShape(), a.epsilon());
        } else if (operation.kind() == RmsNormKind.RMS_NORM && operation.attrs() instanceof RmsNormAttrs a) {
            kind = MetalMpsGraphProgram.NodeKind.RMS_NORM;
            attr = MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON;
            words = normalized(a.normalizedShape(), a.epsilon());
        } else if (operation.kind() == BatchNormKind.BATCH_NORM_INFERENCE
                && operation.attrs() instanceof BatchNormInferenceAttrs a) {
            kind = MetalMpsGraphProgram.NodeKind.BATCH_NORM_INFERENCE;
            attr = MetalMpsGraphProgram.AttributeKind.BATCH_NORM_INFERENCE;
            words = new long[] {a.channelAxis(), 1, floatingBits(a.epsilon())};
        } else if (operation.kind() == BatchNormKind.BATCH_NORM_TRAINING
                && operation.attrs() instanceof BatchNormTrainingAttrs a) {
            kind = MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING;
            attr = MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING;
            words = new long[] {a.channelAxis(), 1, floatingBits(a.momentum()), 1, floatingBits(a.epsilon())};
        } else if (operation.kind() == LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && operation.attrs() instanceof DenseCategoricalCrossEntropyWithLogitsAttrs a) {
            kind = MetalMpsGraphProgram.NodeKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
            attr = MetalMpsGraphProgram.AttributeKind.DENSE_CROSS_ENTROPY;
            words = new long[] {a.axis(), reduction(a.reduction())};
        } else if (operation.kind() == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && operation.attrs() instanceof IndexCategoricalCrossEntropyWithLogitsAttrs a) {
            kind = MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
            attr = MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY;
            if (a.ignoreIndex().isPresent()) {
                ScalarValue ignore = a.ignoreIndex().orElseThrow();
                require(ignore.dataType() == DataType.INT32 || ignore.dataType() == DataType.INT64);
                words = new long[] {a.axis(), reduction(a.reduction()), 1,
                    MetalMpsGraphProgram.dataTypeWire(ignore.dataType()), ignore.dataType() == DataType.INT32
                            ? Integer.toUnsignedLong(ignore.int32Value()) : ignore.int64Value()};
            } else words = new long[] {a.axis(), reduction(a.reduction()), 0};
        } else throw malformed();
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        for (var descriptor : inputDescriptors) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        for (var descriptor : outputDescriptors) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
        validate(MetalMpsGraphProgram.Node.generic(kind, sequence(0, inputs.length),
                sequence(inputs.length, outputs.length), attr, words), values);
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attr, words);
    }

    /** @param node encoded ordinary node @return whether this component owns its operation wire */
    static boolean handles(MetalMpsGraphProgram.Node node) {
        return switch (node.kind()) {
            case SOFTMAX, LOG_SOFTMAX, LAYER_NORM, RMS_NORM, BATCH_NORM_INFERENCE,
                    BATCH_NORM_TRAINING, DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> true;
            default -> false;
        };
    }

    /**
     * Rechecks raw encoded cardinality, typed scalars, exact shapes, physical layouts and gradients.
     * This entry accepts no capability/provider result and never invokes Model admission.
     * @param node decoded untrusted ordinary node
     * @param values independently decoded program descriptors
     * @throws IllegalArgumentException if any encoded or physical contract is inconsistent
     */
    static void validate(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            validateRaw(node, values);
        } catch (ArithmeticException overflow) {
            throw new IllegalArgumentException("Metal normalization/loss geometry overflows", overflow);
        }
    }

    private static void validateRaw(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        require(handles(node));
        int[] in = node.inputs(), out = node.outputs();
        for (int id : in) require(id >= 0 && id < values.size());
        for (int i = 0; i < out.length; i++) {
            require(out[i] >= 0 && out[i] < values.size());
            for (int id : in) require(id != out[i]);
            for (int j = 0; j < i; j++) require(out[j] != out[i]);
        }
        require(in.length >= 1 && out.length >= 1);
        var x = values.get(in[0]);
        require(x.dataType() == DataType.FLOAT32 && x.rank() >= 1);
        for (int id : in) physical(values.get(id), false);
        for (int id : out) { physical(values.get(id), true); require(values.get(id).dataType() == DataType.FLOAT32); }
        long[] dims = x.dimensions(), words = node.attributeWords();
        boolean gradient = false;
        for (int id : in) gradient |= values.get(id).requiresGrad();
        switch (node.kind()) {
            case SOFTMAX, LOG_SOFTMAX -> {
                require(in.length == 1 && out.length == 1 && words.length == 1
                        && node.attributeKind() == MetalMpsGraphProgram.AttributeKind.AXIS);
                int a = axis(words[0], dims.length);
                if (x.elementCount() != 0) dispatchCount(productExcept(dims, a));
                result(values.get(out[0]), dims, x.requiresGrad());
            }
            case LAYER_NORM, RMS_NORM -> {
                require(out.length == 1 && node.attributeKind() == MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON);
                require(node.kind() == MetalMpsGraphProgram.NodeKind.LAYER_NORM
                        ? in.length == 1 || in.length == 3 : in.length == 1 || in.length == 2);
                require(words.length >= 4 && words[0] >= 1 && words[0] <= dims.length);
                int rank = Math.toIntExact(words[0]);
                require(words.length == rank + 3);
                long[] suffix = Arrays.copyOfRange(dims, dims.length-rank, dims.length);
                long rows = 1;
                for (int i = 0; i < dims.length-rank; i++) rows = Math.multiplyExact(rows, dims[i]);
                if (x.elementCount() != 0) dispatchCount(rows);
                require(Arrays.equals(suffix, Arrays.copyOfRange(words, 1, rank+1)));
                positiveScalar(words, rank+1);
                for (int i = 1; i < in.length; i++) {
                    require(values.get(in[i]).dataType() == DataType.FLOAT32);
                    require(Arrays.equals(values.get(in[i]).dimensions(), suffix));
                }
                result(values.get(out[0]), dims, gradient);
            }
            case BATCH_NORM_INFERENCE, BATCH_NORM_TRAINING -> {
                boolean training = node.kind() == MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING;
                require(in.length == 5 && out.length == (training ? 5 : 1) && dims.length >= 2);
                require(node.attributeKind() == (training ? MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING
                        : MetalMpsGraphProgram.AttributeKind.BATCH_NORM_INFERENCE));
                require(words.length == (training ? 5 : 3));
                int a = axis(words[0], dims.length);
                dispatchCount(dims[a]);
                long[] channelShape = {dims[a]};
                for (int i = 1; i < 5; i++) require(values.get(in[i]).dataType() == DataType.FLOAT32
                        && Arrays.equals(values.get(in[i]).dimensions(), channelShape));
                if (training) {
                    float momentum = scalar(words, 1);
                    require(momentum >= 0 && momentum <= 1);
                    positiveScalar(words, 3);
                    long population = productExcept(dims, a);
                    require(dims[a] == 0 || population >= 2);
                    result(values.get(out[0]), dims, x.requiresGrad() || values.get(in[1]).requiresGrad() || values.get(in[2]).requiresGrad());
                    result(values.get(out[1]), channelShape, x.requiresGrad() || values.get(in[3]).requiresGrad());
                    result(values.get(out[2]), channelShape, x.requiresGrad() || values.get(in[4]).requiresGrad());
                    result(values.get(out[3]), channelShape, x.requiresGrad());
                    result(values.get(out[4]), channelShape, x.requiresGrad());
                } else { positiveScalar(words, 1); result(values.get(out[0]), dims, gradient); }
            }
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> {
                boolean index = node.kind() == MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS;
                require(in.length == 2 && out.length == 1 && words.length >= 2);
                require(node.attributeKind() == (index ? MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY
                        : MetalMpsGraphProgram.AttributeKind.DENSE_CROSS_ENTROPY));
                int a = axis(words[0], dims.length);
                require(words[1] >= 1 && words[1] <= 3);
                if (words[1] == 1) dispatchCount(productExcept(dims, a));
                long[] samples = new long[dims.length-1];
                for (int i = 0, j = 0; i < dims.length; i++) if (i != a) samples[j++] = dims[i];
                var target = values.get(in[1]);
                if (index) {
                    require((target.dataType() == DataType.INT32 || target.dataType() == DataType.INT64)
                            && !target.requiresGrad() && Arrays.equals(target.dimensions(), samples));
                    require(words.length >= 3 && (words[2] == 0 || words[2] == 1));
                    require(words.length == (words[2] == 1 ? 5 : 3));
                    if (words[2] == 1) {
                        require(words[3] == MetalMpsGraphProgram.dataTypeWire(target.dataType()));
                        if (target.dataType() == DataType.INT32) require((words[4] & ~0xffff_ffffL) == 0);
                    }
                } else require(words.length == 2 && target.dataType() == DataType.FLOAT32
                        && Arrays.equals(target.dimensions(), dims));
                require(dims[a] > 0 || productExcept(dims, a) == 0 || index && words[2] == 1);
                result(values.get(out[0]), words[1] == 1 ? samples : new long[0],
                        x.requiresGrad() || !index && target.requiresGrad());
            }
            default -> throw malformed();
        }
    }

    private static void physical(MetalMpsGraphProgram.ValueDescriptor value, boolean output) {
        require(value.rank() <= 16 && value.layout().isPresent());
        var layout = value.layout().orElseThrow();
        long[] dims = value.dimensions();
        require(layout.rank() == dims.length && layout.storageOffset() >= 0);
        for (long d : dims) require(d >= 0);
        for (long stride : layout.strides()) require(stride >= 0);
        require(layout.equals(LayoutDescriptor.of(Shape.of(dims), layout.strides(), layout.storageOffset(), layout.isView())));
        Math.multiplyExact(value.elementCount(), value.dataType().byteWidth());
        Math.multiplyExact(layout.referencedElementSpan(), value.dataType().byteWidth());
        if (output) require(layout.equals(LayoutDescriptor.contiguous(Shape.of(dims))));
    }

    private static long[] normalized(Shape shape, ScalarValue epsilon) {
        require(shape.isFullyStatic());
        long[] dims = shape.toLongArray(), words = new long[dims.length+3];
        words[0] = dims.length;
        System.arraycopy(dims, 0, words, 1, dims.length);
        words[dims.length+1] = 1;
        words[dims.length+2] = floatingBits(epsilon);
        return words;
    }

    private static long floatingBits(ScalarValue value) {
        require(value.dataType() == DataType.FLOAT32);
        return Integer.toUnsignedLong(Float.floatToRawIntBits(value.float32Value()));
    }

    private static float scalar(long[] words, int start) {
        require(start+1 < words.length && words[start] == 1 && (words[start+1] & ~0xffff_ffffL) == 0);
        float value = Float.intBitsToFloat((int) words[start+1]);
        require(Float.isFinite(value));
        return value;
    }

    private static void positiveScalar(long[] words, int start) { require(scalar(words, start) > 0); }
    private static int axis(long word, int rank) { require(word >= 0 && word < rank); return (int) word; }
    private static void dispatchCount(long count) { require(count >= 0 && count <= 0xffff_ffffL); }
    private static long productExcept(long[] dims, int axis) {
        long product = 1;
        for (int i = 0; i < dims.length; i++) if (i != axis) product = Math.multiplyExact(product, dims[i]);
        return product;
    }
    private static void result(MetalMpsGraphProgram.ValueDescriptor value, long[] shape, boolean gradient) {
        require(Arrays.equals(value.dimensions(), shape) && value.requiresGrad() == gradient);
    }
    private static int[] sequence(int first, int count) {
        int[] ids = new int[count];
        for (int i = 0; i < count; i++) ids[i] = first+i;
        return ids;
    }
    private static long reduction(LossReduction reduction) {
        return switch (reduction) { case NONE -> 1; case SUM -> 2; case MEAN -> 3; };
    }
    private static void require(boolean condition) { if (!condition) throw malformed(); }
    private static IllegalArgumentException malformed() {
        return new IllegalArgumentException("unsupported or malformed Metal FLOAT32 normalization/loss occurrence");
    }
}
