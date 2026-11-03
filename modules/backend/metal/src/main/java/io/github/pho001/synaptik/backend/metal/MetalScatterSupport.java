package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Owns the original-carrier output-owned scatter domain and its independent encoded preflight.
 *
 * <p>Arithmetic accepts FLOAT32/BFLOAT16/FLOAT16 and exact modular INT32/INT64;
 * FLOAT64 additionally admits raw MIN/MAX. Low arithmetic works in FLOAT32 and
 * narrows once at the output. Every input uses its real affine read layout.
 * Broadcast axes do not waive the shared storage-layout proof for nonbroadcast axes.
 * Index values and replacement uniqueness are deliberately run-time, pre-write checks.</p>
 */
final class MetalScatterSupport {
    private static final long MAX_BYTES = 0xffff_ffffL;

    private MetalScatterSupport() {}

    /**
     * Selects the scatter operations belonging to this component.
     *
     * @param operation non-null semantic operation
     * @return whether its kind is one of the three scatter kinds
     */
    static boolean handles(Operation operation) {
        return operation.kind() == AxisScatterKind.SCATTER_ADD
                || operation.kind() == AxisScatterKind.SCATTER_ELEMENTS
                || operation.kind() == ScatterNdKind.SCATTER_ND;
    }

    /**
     * Tests the complete static original-carrier occurrence without reading index values.
     *
     * @param operation semantic operation with normalized attributes
     * @param inputs three ordered descriptors, base/index/updates
     * @param outputs one canonical result descriptor
     * @return true only for this component's independently executable domain
     */
    static boolean supports(Operation operation, List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        if (!handles(operation) || inputs.size() != 3 || outputs.size() != 1) return false;
        try {
            var descriptors = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(4);
            for (var input : inputs) descriptors.add(descriptor(input));
            descriptors.add(descriptor(outputs.getFirst()));
            validate(node(operation, new int[] {0, 1, 2}, new int[] {3}), descriptors);
            return true;
        } catch (IllegalArgumentException | ArithmeticException failure) {
            return false;
        }
    }

    /**
     * Lowers the checked occurrence using existing operation and reduction wire identities.
     *
     * @param operation semantic operation with normalized attributes
     * @param inputs three stable value-table indices
     * @param outputs one stable output index, distinct from all input indices
     * @param inputDescriptors three ordered physical input descriptors
     * @param outputDescriptors one canonical physical output descriptor
     * @return the authenticated ordinary scatter node
     * @throws IllegalArgumentException if any descriptor, arity, index or semantic constraint fails
     */
    static MetalMpsGraphProgram.Node lower(Operation operation, int[] inputs, int[] outputs,
            List<TensorDescriptor> inputDescriptors, List<TensorDescriptor> outputDescriptors) {
        if (inputDescriptors.size() != 3 || outputDescriptors.size() != 1) throw invalid();
        var descriptors = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(4);
        for (var input : inputDescriptors) descriptors.add(descriptor(input));
        descriptors.add(descriptor(outputDescriptors.getFirst()));
        var local = node(operation, new int[] {0, 1, 2}, new int[] {3});
        if (handles(local, descriptors)) validate(local, descriptors);
        if (inputs.length != 3 || outputs.length != 1) throw invalid();
        for (int input : inputs) if (input < 0 || input == outputs[0]) throw invalid();
        if (outputs[0] < 0) throw invalid();
        return node(operation, inputs, outputs);
    }

    /**
     * Recognizes encoded scatter kinds; carrier selection is checked independently by validate.
     *
     * @param node non-null encoded ordinary node
     * @return whether the node has a scatter wire kind
     */
    static boolean handles(MetalMpsGraphProgram.Node node) {
        return node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD
                || node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS
                || node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ND;
    }

    /**
     * Selects homogeneous numerical scatter and the retained raw FLOAT64 extrema.
     *
     * @param node encoded scatter node
     * @param values actual stable descriptor table
     * @return true for FLOAT32/BFLOAT16/FLOAT16 scatter, integral arithmetic or FLOAT64 raw extrema
     */
    static boolean handles(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (!handles(node)) return false;
        int[] inputs = node.inputs();
        if (inputs.length != 3 || inputs[0] < 0 || inputs[0] >= values.size()) return false;
        DataType type = values.get(inputs[0]).dataType();
        if (numeric(type)) return true;
        long[] words = node.attributeWords();
        long reduction = node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD ? 2
                : words.length == 2 ? words[1] : 0;
        return (type == DataType.INT32 || type == DataType.INT64) && reduction >= 2
                || type == DataType.FLOAT64 && reduction >= 4;
    }

    /**
     * Independently checks encoded grammar, physical bounds, types, gradients and shape equations.
     * This entry consumes the actual raw value table and never consults provider answers or
     * reconstructed semantic queries. It does not replace complete run-time index validation.
     *
     * @param node encoded scatter node
     * @param values actual stable encoded value descriptor table
     * @throws IllegalArgumentException if grammar, physical storage or occurrence semantics fail
     */
    static void validate(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            if (!handles(node)) throw invalid();
            int[] inputs = node.inputs(), outputs = node.outputs();
            if (inputs.length != 3 || outputs.length != 1) throw invalid();
            for (int input : inputs) if (input < 0 || input >= values.size() || input == outputs[0]) throw invalid();
            if (outputs[0] < 0 || outputs[0] >= values.size()) throw invalid();
            var base = values.get(inputs[0]);
            var index = values.get(inputs[1]);
            var update = values.get(inputs[2]);
            var output = values.get(outputs[0]);
            long[] words = node.attributeWords();
            boolean add = node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD;
            boolean nd = node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ND;
            var expectedAttribute = add ? MetalMpsGraphProgram.AttributeKind.AXIS
                    : nd ? MetalMpsGraphProgram.AttributeKind.SCATTER_ND
                    : MetalMpsGraphProgram.AttributeKind.SCATTER_ELEMENTS;
            if (node.attributeKind() != expectedAttribute || words.length != (add ? 1 : 2)) throw invalid();
            if (words[0] < 0 || words[0] > 16) throw invalid();
            int axis = (int) words[0];
            long reduction = add ? 2 : words[1];
            if (reduction < 1 || reduction > 5) throw invalid();
            DataType type = base.dataType();
            boolean integer = type == DataType.INT32 || type == DataType.INT64;
            boolean floating = numeric(type) || type == DataType.FLOAT64;
            if (!floating && !integer) throw invalid();
            if (!handles(node, values)) throw invalid();
            if (type == DataType.FLOAT64 && (reduction == 2 || reduction == 3)) throw invalid();
            if (update.dataType() != type || output.dataType() != type) throw invalid();
            if (index.dataType() != DataType.INT32 && index.dataType() != DataType.INT64) throw invalid();
            if (index.requiresGrad() || output.requiresGrad() != (base.requiresGrad() || update.requiresGrad())) throw invalid();
            if (!floating && (base.requiresGrad() || update.requiresGrad() || output.requiresGrad())) throw invalid();
            if (type == DataType.FLOAT64 && (base.requiresGrad() || update.requiresGrad() || output.requiresGrad())) throw invalid();
            readLayout(base); readLayout(index); readLayout(update); readLayout(output);
            if (!output.layout().orElseThrow().equals(LayoutDescriptor.contiguous(Shape.of(output.dimensions())))) throw invalid();
            long[] data = base.dimensions(), indices = index.dimensions(), updates = update.dimensions();
            if (data.length < 1 || !Arrays.equals(data, output.dimensions())) throw invalid();
            if (nd) {
                if (indices.length < 1 || axis >= indices.length || axis > data.length) throw invalid();
                long depth = indices[indices.length - 1];
                if (depth < 1 || depth > data.length - axis) throw invalid();
                int k = (int) depth;
                if (updates.length != indices.length - 1 + data.length - axis - k) throw invalid();
                for (int a = 0; a < axis; a++) if (indices[a] != data[a]) throw invalid();
                for (int a = 0; a < indices.length - 1; a++) if (updates[a] != indices[a]) throw invalid();
                for (int a = axis + k; a < data.length; a++) {
                    if (updates[indices.length - 1 + a - axis - k] != data[a]) throw invalid();
                }
            } else if (add) {
                if (axis >= data.length || updates.length != data.length - 1 + indices.length) throw invalid();
                for (int a = 0; a < axis; a++) if (updates[a] != data[a]) throw invalid();
                for (int a = 0; a < indices.length; a++) if (updates[axis + a] != indices[a]) throw invalid();
                for (int a = axis + 1; a < data.length; a++) {
                    if (updates[a - 1 + indices.length] != data[a]) throw invalid();
                }
            } else {
                if (axis >= data.length || !Arrays.equals(indices, updates) || indices.length != data.length) throw invalid();
                for (int a = 0; a < data.length; a++) if (a != axis && updates[a] != data[a]) throw invalid();
            }
        } catch (ArithmeticException | IndexOutOfBoundsException failure) {
            throw invalid();
        }
    }

    private static boolean numeric(DataType type) {
        return type == DataType.FLOAT32 || type == DataType.BFLOAT16 || type == DataType.FLOAT16;
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty()) throw invalid();
        return new MetalMpsGraphProgram.ValueDescriptor(descriptor.dataType(), descriptor.shape().toLongArray(),
                descriptor.layout(), descriptor.requiresGrad(), false);
    }

    private static void readLayout(MetalMpsGraphProgram.ValueDescriptor value) {
        if (value.rank() > 16 || value.layout().isEmpty() || value.byteCount() > MAX_BYTES) throw invalid();
        LayoutDescriptor layout = value.layout().orElseThrow();
        if (!layout.equals(LayoutDescriptor.of(Shape.of(value.dimensions()), layout.strides(), layout.storageOffset(), layout.isView()))) throw invalid();
        if (layout.storageOffset() < 0 || layout.referencedElementSpan() <= 0) throw invalid();
        for (long stride : layout.strides()) if (stride < 0) throw invalid();
        if (layout.hasZeroStride() && !layout.isView()) throw invalid();
        if (!value.supportsStorageLayout(0)) throw invalid();
        if (value.elementCount() > MAX_BYTES / value.dataType().byteWidth()
                || layout.referencedElementSpan() > MAX_BYTES / value.dataType().byteWidth()) throw invalid();
    }

    private static MetalMpsGraphProgram.Node node(Operation operation, int[] inputs, int[] outputs) {
        if (operation.kind() == AxisScatterKind.SCATTER_ADD && operation.attrs() instanceof IndexAxisAttrs attrs) {
            return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ADD, inputs, outputs,
                    MetalMpsGraphProgram.AttributeKind.AXIS, new long[] {attrs.axis()});
        }
        if (operation.kind() == AxisScatterKind.SCATTER_ELEMENTS && operation.attrs() instanceof ScatterElementsAttrs attrs) {
            return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS, inputs, outputs,
                    MetalMpsGraphProgram.AttributeKind.SCATTER_ELEMENTS, new long[] {attrs.axis(), reduction(attrs.reduction())});
        }
        if (operation.kind() == ScatterNdKind.SCATTER_ND && operation.attrs() instanceof ScatterNdAttrs attrs) {
            return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.SCATTER_ND, inputs, outputs,
                    MetalMpsGraphProgram.AttributeKind.SCATTER_ND, new long[] {attrs.batchDimensions(), reduction(attrs.reduction())});
        }
        throw invalid();
    }

    private static long reduction(ScatterReduction reduction) {
        return switch (reduction) { case NONE -> 1; case ADD -> 2; case MUL -> 3; case MAX -> 4; case MIN -> 5; };
    }

    private static IllegalArgumentException invalid() {
        return new IllegalArgumentException("unsupported or malformed original-carrier Metal scatter occurrence");
    }
}
