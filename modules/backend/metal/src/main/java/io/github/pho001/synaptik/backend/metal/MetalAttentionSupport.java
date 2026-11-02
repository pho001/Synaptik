package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;

/**
 * FLOAT32 attention admission and lowering, with an independent raw-program validator.
 *
 * <p>The original carriers, exact per-port gradient metadata, physical layouts and canonical
 * weights sibling are checked, not inferred from a projected FLOAT32 operation. No low or
 * FLOAT64 arithmetic domain follows from this owner. Mask contents are validated by native
 * invocation preflight before scores, outputs or declared workspace are written.</p>
 */
final class MetalAttentionSupport {
    private MetalAttentionSupport() {}

    /** @param operation actual operation, never a dtype-projected proxy
     * @return whether this owner handles its exact semantic kind */
    static boolean handles(Operation operation) {
        return operation.kind() == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION;
    }

    /**
     * Checks original-carrier capability and resolved physical geometry.
     * @param operation actual semantic operation
     * @param inputs ordered original input descriptors
     * @param outputs ordered original output descriptors
     * @return whether this exact occurrence has a prepared FLOAT32 implementation
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        if (!handles(operation) || !(operation.attrs() instanceof ScaledDotProductAttentionAttrs attrs)
                || inputs.size() < 3 || inputs.size() > 4 || outputs.isEmpty() || outputs.size() > 2) return false;
        try {
            if (attrs.scale().isPresent()) {
                var scale = attrs.scale().orElseThrow();
                if (scale.dataType() != DataType.FLOAT32 || !Float.isFinite(scale.float32Value())
                        || scale.float32Value() <= 0) return false;
            }
            for (int i = 0; i < inputs.size(); i++) {
                TensorDescriptor descriptor = inputs.get(i);
                if (descriptor.dataType() != (i == 3 ? DataType.BOOL : DataType.FLOAT32)
                        || i == 3 && descriptor.requiresGrad()
                        || !physical(descriptor.shape(), descriptor.layout().orElse(null), false,
                                descriptor.dataType().byteWidth())
                        || !MetalMpsGraphProgram.ValueDescriptor.from(descriptor).supportsStorageLayout(0)) return false;
            }
            boolean gradient = inputs.get(0).requiresGrad() || inputs.get(1).requiresGrad();
            for (int i = 0; i < outputs.size(); i++) {
                TensorDescriptor descriptor = outputs.get(i);
                if (descriptor.dataType() != DataType.FLOAT32
                        || descriptor.requiresGrad() != (gradient || i == 0 && inputs.get(2).requiresGrad())
                        || !physical(descriptor.shape(), descriptor.layout().orElse(null), true, Float.BYTES)) return false;
            }
            return shapes(inputs.stream().map(d -> d.shape().toLongArray()).toList(),
                    outputs.stream().map(d -> d.shape().toLongArray()).toList());
        } catch (IllegalArgumentException | ArithmeticException failure) {
            return false;
        }
    }

    /**
     * Encodes the existing exact attention attribute grammar after original occurrence admission.
     * @param operation semantic operation with exact F32 optional scale
     * @param inputs ordered input value indices
     * @param outputs ordered output value indices
     * @param inputDescriptors original input facts
     * @param outputDescriptors original output facts
     * @return the existing ordinary attention node, with no synthetic operation or ports
     * @throws IllegalArgumentException when the occurrence is unsupported
     */
    static MetalMpsGraphProgram.Node lower(Operation operation, int[] inputs, int[] outputs,
            List<TensorDescriptor> inputDescriptors, List<TensorDescriptor> outputDescriptors) {
        if (inputs.length != inputDescriptors.size() || outputs.length != outputDescriptors.size()
                || !supports(operation, inputDescriptors, outputDescriptors))
            throw new IllegalArgumentException("unsupported Metal FLOAT32 attention occurrence");
        var attrs = (ScaledDotProductAttentionAttrs) operation.attrs();
        long[] words = attrs.scale().isPresent()
                ? new long[] {1, 1, Integer.toUnsignedLong(Float.floatToRawIntBits(
                        attrs.scale().orElseThrow().float32Value())), attrs.causal() ? 1 : 0}
                : new long[] {0, attrs.causal() ? 1 : 0};
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION,
                inputs, outputs, MetalMpsGraphProgram.AttributeKind.ATTENTION, words);
    }

    /** @param node independently decoded ordinary node
     * @return whether the exact node kind belongs to this owner */
    static boolean handles(MetalMpsGraphProgram.Node node) {
        return node.kind() == MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION;
    }

    /**
     * Independently rejects malformed raw attention nodes, without capability-provider calls.
     * @param node untrusted decoded operation and attribute words
     * @param values actual raw value descriptors, including physical and gradient facts
     * @throws IllegalArgumentException when any grammar, carrier, geometry or layout fact is invalid
     */
    static void validate(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            if (!handles(node) || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.ATTENTION)
                throw malformed();
            int[] inputs = node.inputs(), outputs = node.outputs();
            if (inputs.length < 3 || inputs.length > 4 || outputs.length < 1 || outputs.length > 2)
                throw malformed();
            long[] words = node.attributeWords();
            if (words.length == 2 && words[0] == 0) {
                if (words[1] != 0 && words[1] != 1) throw malformed();
            } else if (words.length == 4 && words[0] == 1 && words[1] == 1) {
                if ((words[2] & ~0xffff_ffffL) != 0 || words[3] != 0 && words[3] != 1)
                    throw malformed();
                float scale = Float.intBitsToFloat((int) words[2]);
                if (!(scale > 0) || !Float.isFinite(scale)) throw malformed();
            } else throw malformed();
            var destinations = new HashSet<Integer>();
            for (int output : outputs) {
                if (output < 0 || output >= values.size() || !destinations.add(output)) throw malformed();
                for (int input : inputs) if (output == input) throw malformed();
            }
            for (int role = 0; role < inputs.length; role++) {
                int index = inputs[role];
                if (index < 0 || index >= values.size()) throw malformed();
                var d = values.get(index);
                if (d.dataType() != (role == 3 ? DataType.BOOL : DataType.FLOAT32)
                        || role == 3 && d.requiresGrad()
                        || !physical(Shape.of(d.dimensions()), d.layout().orElse(null), false, d.dataType().byteWidth())
                        || !d.supportsStorageLayout(0)
                        || d.densePhysical() && !canonical(d)) throw malformed();
            }
            boolean gradient = values.get(inputs[0]).requiresGrad() || values.get(inputs[1]).requiresGrad();
            for (int port = 0; port < outputs.length; port++) {
                var d = values.get(outputs[port]);
                if (d.dataType() != DataType.FLOAT32 || d.densePhysical()
                        || d.requiresGrad() != (gradient || port == 0 && values.get(inputs[2]).requiresGrad())
                        || !physical(Shape.of(d.dimensions()), d.layout().orElse(null), true, Float.BYTES)) throw malformed();
            }
            if (!shapes(Arrays.stream(inputs).mapToObj(i -> values.get(i).dimensions()).toList(),
                    Arrays.stream(outputs).mapToObj(i -> values.get(i).dimensions()).toList())) throw malformed();
        } catch (ArithmeticException | IndexOutOfBoundsException failure) {
            throw new IllegalArgumentException("malformed Metal FLOAT32 attention geometry", failure);
        }
    }

    private static boolean canonical(MetalMpsGraphProgram.ValueDescriptor d) {
        return d.layout().isPresent()
                && d.layout().orElseThrow().equals(LayoutDescriptor.contiguous(Shape.of(d.dimensions())));
    }

    private static boolean physical(Shape shape, LayoutDescriptor layout, boolean output, int width) {
        if (!shape.isFullyStatic() || shape.rank() > 16 || layout == null || layout.rank() != shape.rank())
            return false;
        long count = 1;
        for (long extent : shape.toLongArray()) {
            if (extent < 0) return false;
            count = Math.multiplyExact(count, extent);
        }
        Math.multiplyExact(count, width);
        if (layout.storageOffset() < 0) return false;
        for (long stride : layout.strides()) if (stride < 0) return false;
        if (!layout.equals(LayoutDescriptor.of(shape, layout.strides(), layout.storageOffset(), layout.isView())))
            return false;
        Math.multiplyExact(layout.referencedElementSpan(), width);
        return !output || layout.equals(LayoutDescriptor.contiguous(shape));
    }

    private static boolean shapes(List<long[]> inputs, List<long[]> outputs) {
        long[] q = inputs.get(0), k = inputs.get(1), v = inputs.get(2);
        if (q.length < 2 || k.length < 2 || v.length < 2) return false;
        long embedding = q[q.length - 1], keyLength = k[k.length - 2];
        if (embedding <= 0 || embedding != k[k.length - 1] || keyLength != v[v.length - 2]) return false;
        int batchRank = Math.max(q.length, Math.max(k.length, v.length)) - 2;
        if (batchRank > 14) return false;
        long[] expected = new long[batchRank + 2];
        Arrays.fill(expected, 1);
        for (long[] role : List.of(q, k, v)) {
            int padding = batchRank - (role.length - 2);
            for (int axis = 0; axis < role.length - 2; axis++) {
                int target = padding + axis;
                if (expected[target] == 1) expected[target] = role[axis];
                else if (role[axis] != 1 && role[axis] != expected[target]) return false;
            }
        }
        expected[batchRank] = q[q.length - 2];
        expected[batchRank + 1] = v[v.length - 1];
        if (!Arrays.equals(expected, outputs.getFirst())) return false;
        expected[batchRank + 1] = keyLength;
        if (outputs.size() == 2 && !Arrays.equals(expected, outputs.get(1))) return false;
        if (inputs.size() == 4) {
            long[] mask = inputs.get(3);
            if (mask.length > expected.length) return false;
            int padding = expected.length - mask.length;
            for (int axis = 0; axis < mask.length; axis++)
                if (mask[axis] != 1 && mask[axis] != expected[padding + axis]) return false;
        }
        long weights = 1;
        for (long extent : expected) weights = Math.multiplyExact(weights, extent);
        Math.multiplyExact(weights, Float.BYTES);
        return true;
    }

    private static IllegalArgumentException malformed() {
        return new IllegalArgumentException("malformed Metal FLOAT32 attention occurrence");
    }
}
