package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.tensor.RecurrentScan;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Owns bounded cold source admission and reconstruction, separate from ordinary compilation.
 *
 * <p>Model construction already proves structural closure, signature counts, attributes classes
 * and descriptor layout geometry. This boundary adds static checked counts, FORWARD-only closed
 * family admission and existing exact descriptor inference. No Tensor is constructed until all
 * stored facts and positional substitutions pass. The only ValueId-to-Tensor table is local to
 * reconstruction after preflight; nothing caches it or attaches it to returned expressions.</p>
 * <p>The closed slice includes the narrow default CNN/static recurrent compositions and fixed
 * flat scans. Reconstruction invokes each public family constructor once and retains all its
 * canonical ports. Index and length values are never read; raw unresolved layouts stay unresolved.</p>
 */
final class SourceGraphReconstruction {
    private SourceGraphReconstruction() {}

    /**
     * Verifies all stored source facts without rewriting descriptors or constructing Tensors.
     *
     * @param graph non-null immutable, structurally validated graph
     * @return non-null call-local descriptor index for subsequent positional association checks
     * @throws NullPointerException if graph is null
     * @throws IllegalArgumentException if static geometry, phase, family admission, inference or
     *     exact descriptor validation fails; checked count overflow is wrapped with value context
     */
    static Map<ValueId, TensorDescriptor> preflight(CompiledGraphModel graph) {
        Objects.requireNonNull(graph, "graph");
        var descriptors = new HashMap<ValueId, TensorDescriptor>();
        for (GraphValue value : graph.values()) {
            TensorDescriptor descriptor = value.descriptor();
            if (!descriptor.shape().isFullyStatic()) {
                throw new IllegalArgumentException(value.id() + ": source Shape must be fully static");
            }
            try {
                descriptor.shape().knownElementCount();
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(value.id() + ": source element count overflow", failure);
            }
            descriptors.put(value.id(), descriptor);
        }
        for (CompiledNode node : graph.nodes()) {
            if (graph.nodePhases().get(node.id()) != GraphPhase.FORWARD) {
                throw new IllegalArgumentException(node.id() + ": source phase must be FORWARD");
            }
            var kind = node.operation().kind();
            if (kind == BinaryArithmeticKind.SUB && node.inputs().stream()
                    .anyMatch(id -> !descriptors.get(id).dataType().isFloating()))
                throw new IllegalArgumentException(node.id() + ": source SUB requires floating operands");
            boolean parameterless = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL
                    || kind == UnaryElementwiseKind.NEG
                    || kind == UnaryElementwiseKind.RELU
                    || kind == UnaryElementwiseKind.TANH
                    || kind == UnaryElementwiseKind.SIGMOID
                    || kind == MatmulKind.MATMUL;
            boolean transpose = kind == AxisTransformKind.PERMUTE
                    && node.operation().attrs() instanceof PermutationAttrs attrs
                    && attrs.axes().equals(List.of(1, 0));
            if (!(parameterless && node.operation().attrs() == NoOperationAttrs.INSTANCE)
                    && !transpose && !additional(node, descriptors)) {
                throw new IllegalArgumentException(node.id() + ": unsupported source operation "
                        + node.operation());
            }
        }
        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);
        if (!validated.constraints().isEmpty()) {
            throw new IllegalArgumentException("source has deferred obligations: " + validated.constraints());
        }
        return descriptors;
    }

    /**
     * Admits only the selected builder geometry and fixed scan signatures, without storage access.
     * @param node structurally validated occurrence
     * @param descriptors complete static descriptor index
     * @return whether the occurrence belongs to the additional closed source slice
     * @throws IllegalArgumentException if a target attribute's checked element count overflows
     */
    private static boolean additional(CompiledNode node, Map<ValueId, TensorDescriptor> descriptors) {
        if (node.inputs().isEmpty()) return false;
        var kind = node.operation().kind();
        var attrs = node.operation().attrs();
        var first = descriptors.get(node.inputs().getFirst());
        if (kind == ShapeTransformKind.RESHAPE && attrs instanceof TargetShapeAttrs target) {
            try {
                return target.targetShape().isFullyStatic()
                    && java.util.Arrays.stream(target.targetShape().toLongArray()).allMatch(n -> n > 0)
                    && target.targetShape().knownElementCount().orElseThrow()
                        == first.shape().knownElementCount().orElseThrow();
            } catch (ArithmeticException failure) {
                throw new IllegalArgumentException(node.id() + ": source target element count overflow", failure);
            }
        }
        if ((kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE)
                && attrs instanceof AxisTransformAttrs axis) return axis.axis() == 2;
        if (kind == SelectKind.SELECT && attrs instanceof SelectAttrs select)
            return select.axis() == 0 && first.shape().rank() == 3;
        if (kind == AxisGatherKind.GATHER && attrs instanceof IndexAxisAttrs axis) {
            var indices = descriptors.get(node.inputs().get(1));
            return axis.axis() == 0 && first.shape().rank() == 2 && indices.shape().rank() == 1
                    && indices.dataType() == DataType.INT64 && !indices.requiresGrad();
        }
        if (kind == SliceKind.SLICE && attrs instanceof SliceAttrs slice)
            return first.shape().rank() == 2 && slice.axes().equals(List.of(1))
                    && slice.lengths().getFirst() > 0 && slice.steps().equals(List.of(1L));
        if (kind == TensorCompositionKind.CONCAT && attrs instanceof CompositionAxisAttrs axis)
            return axis.axis() == 0 && node.inputs().stream()
                    .allMatch(id -> descriptors.get(id).shape().rank() == 3);
        if (kind == Conv2dKind.CONV2D) return node.inputs().size() == 3 && attrs.equals(Conv2dAttrs.defaults());
        if (kind == Conv3dKind.CONV3D) return node.inputs().size() == 3 && attrs.equals(Conv3dAttrs.defaults());
        if (kind == Pool2dKind.MAX_POOL2D && attrs instanceof MaxPool2dAttrs pool)
            return pool.strideHeight() == pool.kernelHeight() && pool.strideWidth() == pool.kernelWidth()
                    && pool.paddingHeight() == 0 && pool.paddingWidth() == 0
                    && pool.dilationHeight() == 1 && pool.dilationWidth() == 1 && !pool.ceilMode();
        return kind instanceof RecurrentScanKind && attrs instanceof RecurrentDirection;
    }

    /**
     * Checks a snapshotted positional association without inspecting storage or upstream DAGs.
     *
     * @param graph non-null preflighted source graph
     * @param descriptors non-null complete descriptor index for that graph
     * @param leaves non-null snapshot without null elements
     * @param requireSourceLeaves whether to require provenance-free unique exact references;
     *     reconstruction instead permits expressions and repeated substitutions
     * @throws IllegalArgumentException if count, full descriptors or required leaf identity fails
     */
    static void validateLeaves(
            CompiledGraphModel graph,
            Map<ValueId, TensorDescriptor> descriptors,
            List<Tensor> leaves,
            boolean requireSourceLeaves) {
        if (leaves.size() != graph.inputs().size()) {
            throw new IllegalArgumentException("leaves count " + leaves.size()
                    + " must match graph inputs count " + graph.inputs().size());
        }
        var seen = requireSourceLeaves ? new IdentityHashMap<Tensor, Boolean>() : null;
        for (int index = 0; index < leaves.size(); index++) {
            Tensor tensor = leaves.get(index);
            ValueId input = graph.inputs().get(index);
            String context = "leaves[" + index + "] for " + input + ": ";
            if (!descriptors.get(input).equals(tensor.descriptor())) {
                throw new IllegalArgumentException(context + "descriptor mismatch; expected="
                        + descriptors.get(input) + ", supplied=" + tensor.descriptor());
            }
            if (requireSourceLeaves && (tensor.provenance().isPresent()
                    || seen.put(tensor, Boolean.TRUE) != null)) {
                throw new IllegalArgumentException(context + "requires unique provenance-free leaves");
            }
        }
    }

    /**
     * Preflights a complete source and substitutions before rebuilding each occurrence once.
     *
     * @param graph non-null raw source graph; never mutated
     * @param leaves non-null input-aligned substitutions without nulls; snapshotted and borrowed
     * @return non-null immutable ordered ordinary outputs, retaining substitutions by provenance
     * @throws NullPointerException if an argument or substitution is null
     * @throws IllegalArgumentException if source preflight or positional association fails
     * @throws IllegalStateException if a family constructor exhausts Tensor identity allocation
     */
    static List<Tensor> reconstruct(CompiledGraphModel graph, List<Tensor> leaves) {
        Objects.requireNonNull(graph, "graph");
        List<Tensor> snapshot = List.copyOf(Objects.requireNonNull(leaves, "leaves"));
        var descriptors = preflight(graph);
        validateLeaves(graph, descriptors, snapshot, false);

        var tensors = new HashMap<ValueId, Tensor>();
        for (int index = 0; index < snapshot.size(); index++) {
            tensors.put(graph.inputs().get(index), snapshot.get(index));
        }
        for (CompiledNode node : graph.nodes()) {
            Tensor first = tensors.get(node.inputs().getFirst());
            var kind = node.operation().kind();
            Tensor output;
            if (kind == BinaryArithmeticKind.ADD) {
                output = first.add(tensors.get(node.inputs().get(1)));
            } else if (kind == BinaryArithmeticKind.SUB) {
                output = first.sub(tensors.get(node.inputs().get(1)));
            } else if (kind == BinaryArithmeticKind.MUL) {
                output = first.mul(tensors.get(node.inputs().get(1)));
            } else if (kind == UnaryElementwiseKind.NEG) {
                output = first.neg();
            } else if (kind == MatmulKind.MATMUL) {
                output = first.matmul(tensors.get(node.inputs().get(1)));
            } else if (kind == UnaryElementwiseKind.RELU) {
                output = first.relu();
            } else if (kind == UnaryElementwiseKind.TANH) {
                output = first.tanh();
            } else if (kind == UnaryElementwiseKind.SIGMOID) {
                output = first.sigmoid();
            } else if (kind == AxisTransformKind.PERMUTE) {
                output = first.permute(1, 0);
            } else if (kind == ShapeTransformKind.RESHAPE) {
                output = first.reshape(((TargetShapeAttrs) node.operation().attrs()).targetShape());
            } else if (kind == AxisTransformKind.EXPAND_DIMS) {
                output = first.expandDims(2);
            } else if (kind == AxisTransformKind.SQUEEZE) {
                output = first.squeeze(2);
            } else if (kind == SelectKind.SELECT) {
                var attrs = (SelectAttrs) node.operation().attrs();
                output = first.select(attrs.axis(), attrs.index());
            } else if (kind == AxisGatherKind.GATHER) {
                output = first.gather(tensors.get(node.inputs().get(1)), 0);
            } else if (kind == SliceKind.SLICE) {
                var attrs = (SliceAttrs) node.operation().attrs();
                output = first.sliceByLength(new long[] {attrs.starts().getFirst()},
                        new long[] {attrs.lengths().getFirst()}, new int[] {1}, new long[] {1});
            } else if (kind == TensorCompositionKind.CONCAT) {
                output = Tensor.concat(0, node.inputs().stream().map(tensors::get).toArray(Tensor[]::new));
            } else if (kind == Conv2dKind.CONV2D) {
                output = first.conv2d(tensors.get(node.inputs().get(1)), tensors.get(node.inputs().get(2)),
                        (Conv2dAttrs) node.operation().attrs());
            } else if (kind == Conv3dKind.CONV3D) {
                output = first.conv3d(tensors.get(node.inputs().get(1)), tensors.get(node.inputs().get(2)),
                        (Conv3dAttrs) node.operation().attrs());
            } else if (kind == Pool2dKind.MAX_POOL2D) {
                output = first.maxPool2d((MaxPool2dAttrs) node.operation().attrs());
            } else if (kind instanceof RecurrentScanKind scan) {
                List<Tensor> args = node.inputs().stream().map(tensors::get).toList();
                var direction = (RecurrentDirection) node.operation().attrs();
                List<Tensor> ports;
                if (scan == RecurrentScanKind.LSTM) {
                    var result = args.size() == 6
                            ? RecurrentScan.lstm(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction)
                            : RecurrentScan.lstm(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), args.get(6), direction);
                    ports = List.of(result.outputs(), result.finalHidden(), result.finalCell());
                } else {
                    var result = scan == RecurrentScanKind.RNN_TANH
                            ? (args.size() == 5
                                ? RecurrentScan.rnn(first, args.get(1), args.get(2), args.get(3), args.get(4), direction)
                                : RecurrentScan.rnn(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction))
                            : (args.size() == 5
                                ? RecurrentScan.gru(first, args.get(1), args.get(2), args.get(3), args.get(4), direction)
                                : RecurrentScan.gru(first, args.get(1), args.get(2), args.get(3), args.get(4), args.get(5), direction));
                    ports = List.of(result.outputs(), result.finalHidden());
                }
                for (int j = 0; j < ports.size(); j++) tensors.put(node.outputs().get(j), ports.get(j));
                continue;
            } else {
                throw new IllegalArgumentException("Unsupported preflighted source operation: " + kind);
            }
            tensors.put(node.outputs().getFirst(), output);
        }
        var outputs = new ArrayList<Tensor>(graph.outputs().size());
        for (ValueId output : graph.outputs()) outputs.add(tensors.get(output));
        return List.copyOf(outputs);
    }
}
