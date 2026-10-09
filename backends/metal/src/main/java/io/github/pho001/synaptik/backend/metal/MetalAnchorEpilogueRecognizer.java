package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutKind;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Cold, deterministic recognizer for the bounded schema-20 Metal anchor-epilogue domain. */
final class MetalAnchorEpilogueRecognizer {
    private MetalAnchorEpilogueRecognizer() {}

    static List<MetalAnchorEpilogue> recognize(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] targets) {
        List<MetalAnchorEpilogue> result =
                recognizeBounded(program, values, targets);
        return result.size() <= MetalPointwiseFusionPlan.MAX_ANCHOR_UNITS
                ? result : List.of();
    }

    static List<MetalAnchorEpilogue> recognizeForDiagnostics(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] targets) {
        return recognizeBounded(program, values, targets);
    }

    private static List<MetalAnchorEpilogue> recognizeBounded(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] targets) {
        Objects.requireNonNull(program, "program");
        List<MetalMpsGraphProgram.ValueDescriptor> descriptors = List.copyOf(values);

        List<MetalMpsGraphProgram.Node> nodes = program.nodes();
        int[] consumers = new int[descriptors.size()];
        for (MetalMpsGraphProgram.Node node : nodes) {
            for (int input : node.inputs()) consumers[input] = Math.incrementExact(consumers[input]);
        }
        boolean[] target = new boolean[descriptors.size()];
        for (int value : targets) {
            if (value < 0 || value >= target.length) {
                throw new IllegalArgumentException("target index is out of range");
            }
            target[value] = true;
        }

        var result = new ArrayList<MetalAnchorEpilogue>(
                Math.min(nodes.size(), MetalPointwiseFusionPlan.MAX_ANCHOR_UNITS + 1));
        int position = 0;
        while (position < nodes.size()) {
            MetalMpsGraphProgram.Node anchor = nodes.get(position);
            MetalAnchorEpilogue.AnchorKind kind = anchorKind(anchor, descriptors);
            MetalAnchorEpilogue candidate = kind == null ? null : recognizeAt(
                    kind, position, nodes, descriptors, consumers, target);
            if (candidate == null) {
                position++;
            } else {
                result.add(candidate);
                position += candidate.memberCount();
                if (result.size() > MetalPointwiseFusionPlan.MAX_ANCHOR_UNITS) break;
            }
        }
        return List.copyOf(result);
    }

    private static MetalAnchorEpilogue recognizeAt(
            MetalAnchorEpilogue.AnchorKind kind,
            int anchorPosition,
            List<MetalMpsGraphProgram.Node> nodes,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] consumers,
            boolean[] targets) {
        MetalMpsGraphProgram.Node anchor = nodes.get(anchorPosition);
        int currentValue = anchor.outputIndex();
        if (!privateIntermediate(currentValue, consumers, targets)) return null;

        var operations = new ArrayList<MetalAnchorEpilogue.Operation>(3);
        boolean scalar = false;
        boolean add = false;
        boolean terminal = false;
        int position = anchorPosition + 1;
        while (position < nodes.size()) {
            MetalMpsGraphProgram.Node node = nodes.get(position);
            if (!isEpilogueKind(node.kind()) || !consumes(node, currentValue)) break;
            if (terminal || operations.size() == 3) return null;

            MetalAnchorEpilogue.Operation operation;
            switch (node.kind()) {
                case SCALAR_MUL -> {
                    if (kind == MetalAnchorEpilogue.AnchorKind.CONV2D) return null;
                    if (scalar || add || !operations.isEmpty()) return null;
                    operation = scalarMultiply(position, node, currentValue, values);
                    scalar = true;
                }
                case ADD -> {
                    if (add) return null;
                    operation = add(position, node, currentValue, values);
                    add = true;
                }
                case RELU -> {
                    operation = relu(position, node, currentValue, values);
                    terminal = true;
                }
                case CLAMP -> {
                    operation = clamp(position, node, currentValue, values);
                    terminal = true;
                }
                default -> throw new AssertionError("unreachable epilogue operation");
            }
            if (operation == null) return null;
            operations.add(operation);
            currentValue = operation.outputValue();
            position++;
            if (position < nodes.size()
                    && isEpilogueKind(nodes.get(position).kind())
                    && consumes(nodes.get(position), currentValue)
                    && (terminal || operations.size() == 3)) {
                return null;
            }
            if (!terminal && position < nodes.size()
                    && isEpilogueKind(nodes.get(position).kind())
                    && consumes(nodes.get(position), currentValue)
                    && !privateIntermediate(currentValue, consumers, targets)) {
                return null;
            }
            if (terminal) break;
        }
        if (operations.isEmpty()) return null;
        if (!privateIntermediate(anchor.outputIndex(), consumers, targets)) return null;
        for (int index = 0; index + 1 < operations.size(); index++) {
            if (!privateIntermediate(operations.get(index).outputValue(), consumers, targets)) {
                return null;
            }
        }
        for (MetalAnchorEpilogue.Operation operation : operations) {
            if (operation instanceof MetalAnchorEpilogue.Add external) {
                for (int anchorInput : anchor.inputs()) {
                    if (external.externalValue() == anchorInput) return null;
                }
            }
        }
        return new MetalAnchorEpilogue(kind, anchorPosition, anchor.outputIndex(), operations);
    }

    private static MetalAnchorEpilogue.AnchorKind anchorKind(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.kind() == MetalMpsGraphProgram.NodeKind.MATMUL
                && node.inputs().length == 2
                && node.outputs().length == 1
                && node.attributeKind() == MetalMpsGraphProgram.AttributeKind.NONE
                && node.attributeWords().length == 0) {
            MetalMpsGraphProgram.ValueDescriptor left = values.get(node.firstInputIndex());
            MetalMpsGraphProgram.ValueDescriptor right = values.get(node.secondInputIndex());
            MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
            boolean carriers = left.dataType() == DataType.FLOAT32
                    && right.dataType() == DataType.FLOAT32
                    || left.dataType() == DataType.FLOAT32
                            && right.dataType() == DataType.BFLOAT16
                    || left.dataType() == DataType.BFLOAT16
                            && right.dataType() == DataType.FLOAT32;
            boolean mixed = left.dataType() != right.dataType();
            if (carriers
                    && MetalCapabilityProvider.matmulShapeMatches(
                            left.dimensions(), right.dimensions(), output.dimensions())
                    && canonical(left)
                    && canonical(right)
                    && canonicalFloat32(output)
                    && (mixed
                            ? !left.requiresGrad() && !right.requiresGrad()
                                    && !output.requiresGrad()
                            : output.requiresGrad()
                                    == (left.requiresGrad() || right.requiresGrad()))) {
                return MetalAnchorEpilogue.AnchorKind.MATMUL;
            }
            return null;
        }
        if (node.kind() == MetalMpsGraphProgram.NodeKind.CONV2D
                && node.inputs().length >= 2
                && node.inputs().length <= 3
                && node.outputs().length == 1
                && node.attributeKind() == MetalMpsGraphProgram.AttributeKind.CONV_2D
                && convolutionCarriers(node, values)) {
            return MetalAnchorEpilogue.AnchorKind.CONV2D;
        }
        return null;
    }


    private static boolean convolutionCarriers(
            MetalMpsGraphProgram.Node node,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        if (!canonicalFloat32(output)) return false;
        boolean anyFloat32 = false;
        boolean anyBfloat16 = false;
        boolean inputGradient = false;
        for (int inputIndex : node.inputs()) {
            MetalMpsGraphProgram.ValueDescriptor input = values.get(inputIndex);
            if (!canonical(input)
                    || input.dataType() != DataType.FLOAT32
                            && input.dataType() != DataType.BFLOAT16) {
                return false;
            }
            anyFloat32 |= input.dataType() == DataType.FLOAT32;
            anyBfloat16 |= input.dataType() == DataType.BFLOAT16;
            inputGradient |= input.requiresGrad();
        }
        return anyFloat32 && (anyBfloat16
                ? !inputGradient && !output.requiresGrad()
                : output.requiresGrad() == inputGradient);
    }

    private static MetalAnchorEpilogue.Operation scalarMultiply(
            int position,
            MetalMpsGraphProgram.Node node,
            int currentValue,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        long[] words = node.attributeWords();
        if (node.inputs().length != 1 || node.outputs().length != 1
                || node.firstInputIndex() != currentValue
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE
                || words.length != 2 || words[0] != 1L
                || (words[1] & ~0xffff_ffffL) != 0L
                || !sameCanonicalFloat32NoGrad(currentValue, node.outputIndex(), values)) {
            return null;
        }
        return new MetalAnchorEpilogue.ScalarMultiply(
                position, currentValue, node.outputIndex(), (int) words[1]);
    }

    private static MetalAnchorEpilogue.Operation add(
            int position,
            MetalMpsGraphProgram.Node node,
            int currentValue,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.inputs().length != 2 || node.outputs().length != 1
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.NONE
                || node.attributeWords().length != 0) {
            return null;
        }
        int left = node.firstInputIndex();
        int right = node.secondInputIndex();
        boolean currentOnLeft = left == currentValue;
        boolean currentOnRight = right == currentValue;
        if (currentOnLeft == currentOnRight) return null;
        int external = currentOnLeft ? right : left;
        MetalMpsGraphProgram.ValueDescriptor current = values.get(currentValue);
        MetalMpsGraphProgram.ValueDescriptor addend = values.get(external);
        MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
        if (!canonicalFloat32(current) || !canonicalFloat32(addend) || !canonicalFloat32(output)
                || current.requiresGrad() != addend.requiresGrad()
                || current.requiresGrad() != output.requiresGrad()
                || !Arrays.equals(current.dimensions(), output.dimensions())
                || !broadcastsTo(addend.dimensions(), current.dimensions())) {
            return null;
        }
        return new MetalAnchorEpilogue.Add(
                position,
                currentValue,
                node.outputIndex(),
                left,
                right,
                external,
                !currentOnLeft);
    }

    private static MetalAnchorEpilogue.Operation relu(
            int position,
            MetalMpsGraphProgram.Node node,
            int currentValue,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        if (node.inputs().length != 1 || node.outputs().length != 1
                || node.firstInputIndex() != currentValue
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.NONE
                || node.attributeWords().length != 0
                || !sameCanonicalFloat32(currentValue, node.outputIndex(), values, false)) {
            return null;
        }
        return new MetalAnchorEpilogue.Relu(position, currentValue, node.outputIndex());
    }

    private static MetalAnchorEpilogue.Operation clamp(
            int position,
            MetalMpsGraphProgram.Node node,
            int currentValue,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        long[] words = node.attributeWords();
        if (node.inputs().length != 1 || node.outputs().length != 1
                || node.firstInputIndex() != currentValue
                || node.attributeKind() != MetalMpsGraphProgram.AttributeKind.CLAMP_RANGE
                || words.length != 4 || words[0] != 1L || words[2] != 1L
                || (words[1] & ~0xffff_ffffL) != 0L
                || (words[3] & ~0xffff_ffffL) != 0L
                || !sameCanonicalFloat32NoGrad(currentValue, node.outputIndex(), values)) {
            return null;
        }
        return new MetalAnchorEpilogue.Clamp(
                position, currentValue, node.outputIndex(), (int) words[1], (int) words[3]);
    }

    private static boolean sameCanonicalFloat32NoGrad(
            int input,
            int output,
            List<MetalMpsGraphProgram.ValueDescriptor> values) {
        return sameCanonicalFloat32(input, output, values, true);
    }

    private static boolean sameCanonicalFloat32(
            int input,
            int output,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            boolean requireNoGrad) {
        MetalMpsGraphProgram.ValueDescriptor source = values.get(input);
        MetalMpsGraphProgram.ValueDescriptor target = values.get(output);
        return canonicalFloat32(source)
                && canonicalFloat32(target)
                && Arrays.equals(source.dimensions(), target.dimensions())
                && source.requiresGrad() == target.requiresGrad()
                && (!requireNoGrad || !source.requiresGrad());
    }

    private static boolean broadcastsTo(long[] source, long[] output) {
        if (source.length > output.length) return false;
        int padding = output.length - source.length;
        for (int axis = 0; axis < source.length; axis++) {
            long extent = source[axis];
            long target = output[padding + axis];
            if (extent != 1L && extent != target) return false;
        }
        return true;
    }

    private static boolean consumes(MetalMpsGraphProgram.Node node, int value) {
        for (int input : node.inputs()) if (input == value) return true;
        return false;
    }

    private static boolean isEpilogueKind(MetalMpsGraphProgram.NodeKind kind) {
        return kind == MetalMpsGraphProgram.NodeKind.SCALAR_MUL
                || kind == MetalMpsGraphProgram.NodeKind.ADD
                || kind == MetalMpsGraphProgram.NodeKind.RELU
                || kind == MetalMpsGraphProgram.NodeKind.CLAMP;
    }

    private static boolean privateIntermediate(int value, int[] consumers, boolean[] targets) {
        return consumers[value] == 1 && !targets[value];
    }

    private static boolean canonicalFloat32(MetalMpsGraphProgram.ValueDescriptor value) {
        return value.dataType() == DataType.FLOAT32 && canonical(value);
    }

    private static boolean canonical(MetalMpsGraphProgram.ValueDescriptor value) {
        return value.layout().filter(layout -> layout.kind() == LayoutKind.DENSE_CONTIGUOUS
                && layout.storageOffset() == 0L
                && !layout.isView()
                && layout.referencedElementSpan() == value.elementCount()).isPresent();
    }
}
