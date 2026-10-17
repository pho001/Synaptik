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
            boolean parameterless = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.MUL
                    || kind == UnaryElementwiseKind.NEG
                    || kind == MatmulKind.MATMUL;
            boolean transpose = kind == AxisTransformKind.PERMUTE
                    && node.operation().attrs() instanceof PermutationAttrs attrs
                    && attrs.axes().equals(List.of(1, 0));
            if (!(parameterless && node.operation().attrs() == NoOperationAttrs.INSTANCE)
                    && !transpose) {
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
            } else if (kind == BinaryArithmeticKind.MUL) {
                output = first.mul(tensors.get(node.inputs().get(1)));
            } else if (kind == UnaryElementwiseKind.NEG) {
                output = first.neg();
            } else if (kind == MatmulKind.MATMUL) {
                output = first.matmul(tensors.get(node.inputs().get(1)));
            } else {
                output = first.permute(1, 0);
            }
            tensors.put(node.outputs().getFirst(), output);
        }
        var outputs = new ArrayList<Tensor>(graph.outputs().size());
        for (ValueId output : graph.outputs()) outputs.add(tensors.get(output));
        return List.copyOf(outputs);
    }
}
