package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;

/**
 * Closes the final logical layouts of eligible fully static convolution results.
 *
 * <p>The pass assigns canonical contiguous layout if and only if a final
 * {@link Conv2dKind#CONV2D} or {@link Conv3dKind#CONV3D} occurrence has its validated single
 * output, that output has a fully static shape, and its layout is unresolved. It additionally
 * derives the exact singleton-height view layout for an unresolved direct
 * {@link AxisTransformKind#SQUEEZE} at axis two whose input is a Conv2d output newly closed by
 * this invocation, provided that the input is static rank four with extent one at axis two and
 * the validated output is the matching rank-three shape. Dynamic or partially dynamic results,
 * already resolved results, and every other operation remain unchanged. This is logical
 * descriptor closure only: it does not infer bindings, propagate layout generally, classify a
 * composition as Conv1d, or introduce physical or backend state.</p>
 */
final class ConvolutionLogicalLayoutClosure {
    /** Prevents construction of this stateless transformation owner. */
    private ConvolutionLogicalLayoutClosure() {}

    /**
     * Closes eligible final descriptors while preserving all graph structure and sidecars.
     *
     * <p>Nodes are inspected in stored order, so only a direct later squeeze can consume a
     * Conv2d result closed during this invocation. If nothing qualifies, the exact input object
     * is returned. Otherwise only changed {@link GraphValue} objects are replaced; every
     * unchanged value reference, node and operation reference, identity, boundary, phase,
     * constant fact, bindable Tensor identity, constraint, and derivative-order value is
     * preserved. Derivative metadata is rebuilt only to own the rebuilt exact graph reference.</p>
     *
     * @param validatedGraph non-null final optimized, validated, and published-constant-closed
     *     graph state; it is not mutated
     * @return the exact input if no descriptor changes, otherwise a non-null validated result
     *     containing only the eligible logical-layout replacements
     * @throws NullPointerException if {@code validatedGraph} is {@code null}
     * @throws IllegalArgumentException if checked canonical or squeeze-view layout arithmetic
     *     overflows; the failure identifies the relevant operation occurrence and values and
     *     retains the arithmetic failure as its cause
     */
    static ValidatedGraph close(ValidatedGraph validatedGraph) {
        java.util.Objects.requireNonNull(validatedGraph, "validatedGraph");

        CompiledGraphModel graph = validatedGraph.graph();
        Map<ValueId, GraphValue> selectedValues = new HashMap<>();
        for (GraphValue value : graph.values()) {
            selectedValues.put(value.id(), value);
        }
        Set<ValueId> newlyClosedConv2dOutputs = new HashSet<>();
        boolean changed = false;

        for (CompiledNode node : graph.nodes()) {
            Object kind = node.operation().kind();
            if (kind == Conv2dKind.CONV2D || kind == Conv3dKind.CONV3D) {
                ValueId outputId = node.outputs().getFirst();
                GraphValue output = selectedValues.get(outputId);
                TensorDescriptor descriptor = output.descriptor();
                if (descriptor.shape().isFullyStatic() && descriptor.layout().isEmpty()) {
                    LayoutDescriptor layout;
                    try {
                        layout = LayoutDescriptor.contiguous(descriptor.shape());
                    } catch (ArithmeticException failure) {
                        throw new IllegalArgumentException(
                                "cannot close " + node.operation().kind().name()
                                        + " logical layout for " + node.id()
                                        + " output " + outputId,
                                failure);
                    }
                    TensorDescriptor closed = withLayout(descriptor, layout);
                    selectedValues.put(outputId, new GraphValue(outputId, closed));
                    if (kind == Conv2dKind.CONV2D) {
                        newlyClosedConv2dOutputs.add(outputId);
                    }
                    changed = true;
                }
                continue;
            }

            if (kind != AxisTransformKind.SQUEEZE
                    || !(node.operation().attrs() instanceof AxisTransformAttrs attrs)
                    || attrs.axis() != 2
                    || node.inputs().size() != 1
                    || node.outputs().size() != 1
                    || !newlyClosedConv2dOutputs.contains(node.inputs().getFirst())) {
                continue;
            }

            ValueId inputId = node.inputs().getFirst();
            ValueId outputId = node.outputs().getFirst();
            TensorDescriptor input = selectedValues.get(inputId).descriptor();
            GraphValue outputValue = selectedValues.get(outputId);
            TensorDescriptor output = outputValue.descriptor();
            if (!eligibleSqueeze(input.shape(), output)) {
                continue;
            }

            LayoutDescriptor inputLayout = input.layout().orElseThrow();
            long[] inputStrides = inputLayout.strides();
            long[] outputStrides = new long[] {
                    inputStrides[0], inputStrides[1], inputStrides[3]
            };
            LayoutDescriptor outputLayout;
            try {
                outputLayout = LayoutDescriptor.of(
                        output.shape(), outputStrides, inputLayout.storageOffset(), true);
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(
                        "cannot close SQUEEZE logical layout for " + node.id()
                                + " input " + inputId + " output " + outputId,
                        failure);
            }
            selectedValues.put(outputId, new GraphValue(outputId, withLayout(output, outputLayout)));
            changed = true;
        }

        if (!changed) {
            return validatedGraph;
        }

        List<GraphValue> values = new ArrayList<>(graph.values().size());
        for (GraphValue value : graph.values()) {
            values.add(selectedValues.get(value.id()));
        }
        CompiledGraphModel rebuilt = new CompiledGraphModel(
                values,
                graph.nodes(),
                graph.inputs(),
                graph.outputs(),
                graph.nodePhases());
        CompileTimeConstantGraph constantGraph = new CompileTimeConstantGraph(
                rebuilt,
                validatedGraph.constantGraph().constants(),
                validatedGraph.constantGraph().bindableTensorIds());
        DerivativeGraphMetadata derivatives = new DerivativeGraphMetadata(
                rebuilt, validatedGraph.derivatives().derivativeOrderByNode());
        return new ValidatedGraph(constantGraph, validatedGraph.constraints(), derivatives);
    }

    private static boolean eligibleSqueeze(Shape inputShape, TensorDescriptor output) {
        if (!inputShape.isFullyStatic()
                || inputShape.rank() != 4
                || !(inputShape.dimension(2) instanceof StaticDimension singleton)
                || singleton.size() != 1
                || output.layout().isPresent()
                || output.shape().rank() != 3) {
            return false;
        }
        Shape expected = Shape.ofDimensions(
                inputShape.dimension(0), inputShape.dimension(1), inputShape.dimension(3));
        return output.shape().equals(expected);
    }

    private static TensorDescriptor withLayout(
            TensorDescriptor descriptor, LayoutDescriptor layout) {
        return new TensorDescriptor(
                descriptor.dataType(),
                descriptor.shape(),
                Optional.of(layout),
                descriptor.requiresGrad());
    }
}
