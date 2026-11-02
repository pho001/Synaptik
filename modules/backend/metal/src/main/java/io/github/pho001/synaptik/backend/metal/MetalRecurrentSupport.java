package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Cold qualification and ordinary-wire lowering for canonical FLOAT32 fixed recurrence.
 *
 * <p>This owner admits the forward operation, including lawful gradient-bearing primal metadata;
 * it does not provide or claim the Compiler's independently deferred fixed-family BPTT. Runtime
 * signed INT64 length values remain invocation-preflight inputs, never specialization facts.
 * Zero time/batch extents are admitted only in their semantic recurrent roles. All floating roles
 * are original FLOAT32 descriptors: no low-carrier projection, arithmetic fallback, or software
 * FLOAT64 is involved. Encoded validation is independent of provider answers and selected routes.
 */
final class MetalRecurrentSupport {
    private static final long UINT32_MAX = 0xffff_ffffL;

    private MetalRecurrentSupport() {}

    /**
     * Identifies this owner's closed semantic family, not descriptor admission.
     * @param operation source operation, possibly null
     * @return whether this is one of the three fixed recurrent kinds
     */
    static boolean handles(Operation operation) {
        return operation != null && operation.kind() instanceof RecurrentScanKind;
    }

    /**
     * Qualifies exact original source descriptors through the independently checked wire meaning.
     * @param operation fixed recurrent kind and direction
     * @param inputs ordered original descriptors, including the INT64 length role
     * @param outputs complete canonical sibling descriptors
     * @return whether the original FLOAT32 occurrence satisfies every static native precondition
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        if (!handles(operation) || !(operation.attrs() instanceof RecurrentDirection direction)
                || inputs == null || outputs == null) return false;
        try {
            var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(inputs.size() + outputs.size());
            for (var descriptor : inputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
            for (var descriptor : outputs) values.add(MetalMpsGraphProgram.ValueDescriptor.from(descriptor));
            int[] reads = sequence(0, inputs.size());
            int[] writes = sequence(inputs.size(), outputs.size());
            validate(node((RecurrentScanKind) operation.kind(), direction, reads, writes), values);
            return true;
        } catch (IllegalArgumentException | ArithmeticException | NullPointerException rejected) {
            return false;
        }
    }

    /**
     * Emits one existing ordinary fixed-family node after complete source qualification.
     * @param operation fixed recurrent source kind and direction
     * @param inputs ordered graph-local input value indices
     * @param outputs ordered independent graph-local canonical sibling indices
     * @param inputDescriptors original descriptors in input-role order
     * @param outputDescriptors descriptors in complete sibling order
     * @return the ordinary node with the existing one-word direction encoding
     * @throws IllegalArgumentException if source geometry, carriers, metadata or indices are invalid
     */
    static MetalMpsGraphProgram.Node lower(Operation operation, int[] inputs, int[] outputs,
            List<TensorDescriptor> inputDescriptors, List<TensorDescriptor> outputDescriptors) {
        if (inputs == null || outputs == null || inputDescriptors == null || outputDescriptors == null
                || inputs.length != inputDescriptors.size() || outputs.length != outputDescriptors.size()
                || !supports(operation, inputDescriptors, outputDescriptors)) {
            throw malformed();
        }
        independentIndices(inputs, outputs, Integer.MAX_VALUE);
        return node((RecurrentScanKind) operation.kind(), (RecurrentDirection) operation.attrs(), inputs, outputs);
    }

    /**
     * Selects only the existing fixed-family wire kinds for independent preflight.
     * @param node encoded ordinary node, possibly null
     * @return whether this owner must validate the node
     */
    static boolean handles(MetalMpsGraphProgram.Node node) {
        return node != null && switch (node.kind()) {
            case RNN_TANH, GRU_RESET_AFTER, LSTM -> true;
            default -> false;
        };
    }

    /**
     * Independently authenticates the encoded occurrence against actual value representations.
     *
     * <p>Checks the complete sibling set, ordinary direction grammar, exact role shapes, original
     * carriers, canonical physical spans and gradient OR. No provider, source-support predicate,
     * dtype proxy or prior Java validation is trusted. Shared executable validation additionally
     * checks occurrence ordering, physical buffer identity and invocation lengths before writes.
     * @param node actual encoded fixed recurrent node
     * @param values actual program value descriptors indexed by the encoded node
     * @throws IllegalArgumentException on any malformed or unsupported encoded occurrence
     */
    static void validate(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            validateEncoded(node, values);
        } catch (IllegalArgumentException rejected) {
            throw rejected;
        } catch (RuntimeException rejected) {
            throw new IllegalArgumentException("Malformed Metal fixed recurrence", rejected);
        }
    }

    private static void validateEncoded(MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (!handles(node) || values == null
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION) {
            throw malformed();
        }
        long[] words = node.attributeWords();
        if (words.length != 1 || (words[0] != 1L && words[0] != 2L)) throw malformed();
        int gates = switch (node.kind()) {
            case RNN_TANH -> 1;
            case GRU_RESET_AFTER -> 3;
            case LSTM -> 4;
            default -> throw malformed();
        };
        int[] reads = node.inputs(), writes = node.outputs();
        int baseReads = gates == 4 ? 6 : 5;
        if ((reads.length != baseReads && reads.length != baseReads + 1)
                || writes.length != (gates == 4 ? 3 : 2)) throw malformed();
        independentIndices(reads, writes, values.size());
        var input = values.get(reads[0]);
        var initialHidden = values.get(reads[2]);
        long[] x = input.dimensions(), h = initialHidden.dimensions();
        if (x.length != 3 || h.length != 2) throw malformed();
        long time = x[0], batch = x[1], inputSize = x[2], hiddenSize = h[1];
        if (time < 0L || time > UINT32_MAX || batch < 0L || batch > UINT32_MAX
                || inputSize <= 0L || inputSize > UINT32_MAX
                || hiddenSize <= 0L || hiddenSize > UINT32_MAX) throw malformed();
        long packed = Math.multiplyExact(gates, hiddenSize);
        if (packed > UINT32_MAX) throw malformed();
        check(input, DataType.FLOAT32, time, batch, inputSize);
        check(values.get(reads[1]), DataType.INT64, batch);
        if (values.get(reads[1]).requiresGrad()) throw malformed();
        check(initialHidden, DataType.FLOAT32, batch, hiddenSize);
        int weightRole = 3;
        if (gates == 4) {
            check(values.get(reads[3]), DataType.FLOAT32, batch, hiddenSize);
            weightRole = 4;
        }
        check(values.get(reads[weightRole]), DataType.FLOAT32, packed, inputSize);
        check(values.get(reads[weightRole + 1]), DataType.FLOAT32, packed, hiddenSize);
        if (reads.length == baseReads + 1) {
            check(values.get(reads[weightRole + 2]), DataType.FLOAT32, packed);
        }
        boolean grad = false;
        for (int i = 0; i < reads.length; i++) {
            if (i != 1) grad |= values.get(reads[i]).requiresGrad();
        }
        check(values.get(writes[0]), DataType.FLOAT32, time, batch, hiddenSize);
        check(values.get(writes[1]), DataType.FLOAT32, batch, hiddenSize);
        if (gates == 4) check(values.get(writes[2]), DataType.FLOAT32, batch, hiddenSize);
        for (int output : writes) if (values.get(output).requiresGrad() != grad) throw malformed();
    }

    private static void check(MetalMpsGraphProgram.ValueDescriptor value, DataType type, long... shape) {
        if (value == null || value.dataType() != type || !Arrays.equals(value.dimensions(), shape)
                || value.densePhysical() || value.layout().isEmpty()
                || !value.layout().orElseThrow().equals(LayoutDescriptor.contiguous(Shape.of(shape)))) {
            throw malformed();
        }
        long elements = 1L;
        for (long extent : shape) {
            if (extent < 0L || extent > UINT32_MAX) throw malformed();
            elements = Math.multiplyExact(elements, extent);
        }
        Math.multiplyExact(elements, type.byteWidth());
    }

    private static void independentIndices(int[] reads, int[] writes, int count) {
        for (int input : reads) if (input < 0 || input >= count) throw malformed();
        for (int i = 0; i < writes.length; i++) {
            int output = writes[i];
            if (output < 0 || output >= count) throw malformed();
            for (int input : reads) if (output == input) throw malformed();
            for (int j = 0; j < i; j++) if (output == writes[j]) throw malformed();
        }
    }

    private static MetalMpsGraphProgram.Node node(RecurrentScanKind kind, RecurrentDirection direction,
            int[] inputs, int[] outputs) {
        MetalMpsGraphProgram.NodeKind wire = switch (kind) {
            case RNN_TANH -> MetalMpsGraphProgram.NodeKind.RNN_TANH;
            case GRU_RESET_AFTER -> MetalMpsGraphProgram.NodeKind.GRU_RESET_AFTER;
            case LSTM -> MetalMpsGraphProgram.NodeKind.LSTM;
        };
        return MetalMpsGraphProgram.Node.generic(wire, inputs, outputs,
                MetalMpsGraphProgram.AttributeKind.RECURRENT_DIRECTION,
                new long[] {direction == RecurrentDirection.FORWARD ? 1L : 2L});
    }

    private static int[] sequence(int first, int count) {
        int[] result = new int[count];
        for (int i = 0; i < count; i++) result[i] = first + i;
        return result;
    }

    private static IllegalArgumentException malformed() {
        return new IllegalArgumentException("Malformed Metal fixed recurrence");
    }
}
