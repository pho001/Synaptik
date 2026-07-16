package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Converts public Tensor expression provenance into immutable forward graph structure.
 *
 * <p>This package-private compiler boundary translates exact leaf and producer occurrences into
 * graph-local identities for one capture call. It preserves deterministic operand and
 * producer-output ordering, including output positions with no public Tensor wrapper and opaque
 * state positions, but performs no semantic inference, transformation, differentiation,
 * planning, preparation, or execution.</p>
 */
final class GraphCapture {
    private GraphCapture() {
    }

    /**
     * Captures one ordered selection of Tensor results as a structurally closed forward graph.
     *
     * <p>The request is traversed in caller order using exact Tensor and producer identity.
     * Provenance-free leaves become inputs on first encounter, producers are emitted once after
     * their inputs, and all declared producer output positions become graph values. Requested
     * results retain caller order. Every emitted node has phase {@link GraphPhase#FORWARD}. The
     * returned model owns immutable collection snapshots and retains exact descriptor and
     * operation references, never a requested Tensor, producer, or provenance reference.</p>
     *
     * @param outputs non-null, non-empty ordered requested results; elements must be non-null and
     *     no exact Tensor reference or resolved logical output may occur more than once; the list
     *     is observed synchronously and is not mutated
     * @return a non-null immutable, structurally closed forward graph whose node and value
     *     identifiers restart at zero and follow deterministic depth-first postorder allocation
     * @throws NullPointerException if {@code outputs} is null, or if its first null element is
     *     encountered; the message identifies the reference or zero-based element position
     * @throws IllegalArgumentException if {@code outputs} is empty, if a later element repeats an
     *     earlier exact Tensor reference, or if two requested wrappers resolve to the same graph
     *     value; duplicate messages identify both request positions
     */
    static CompiledGraphModel capture(List<Tensor> outputs) {
        return capture(outputs, CompileTimeConstantGraph.Ingress.empty()).graph();
    }

    /**
     * Captures one ordered Tensor selection and maps only explicitly supplied leaf splats.
     *
     * <p>Output validation and traversal are identical to {@link #capture(List)}. Ingress matches
     * exact Tensor object identity only. Every binding must be encountered as a reachable
     * provenance-free leaf; capture never reads host storage or infers constants from factory
     * history, descriptors, layouts, labels, or other Tensor metadata.</p>
     *
     * @param outputs non-null, non-empty ordered requested results with no null element, repeated
     *     exact Tensor reference, or duplicate resolved graph output
     * @param ingress non-null ordered immutable explicit leaf bindings
     * @return a non-null immutable captured graph plus exact source facts; unbound leaves remain
     *     bindable and binding order does not change graph input order
     * @throws NullPointerException if output validation encounters a null first, or if
     *     {@code ingress} is null after output validation succeeds
     * @throws IllegalArgumentException if output validation fails or an ingress binding is not a
     *     reachable leaf
     */
    static CompileTimeConstantGraph capture(
            List<Tensor> outputs, CompileTimeConstantGraph.Ingress ingress) {
        Objects.requireNonNull(outputs, "outputs");
        if (outputs.isEmpty()) {
            throw new IllegalArgumentException("outputs must not be empty");
        }
        for (int index = 0; index < outputs.size(); index++) {
            Objects.requireNonNull(outputs.get(index), "outputs[" + index + "]");
        }
        var requestedPositions = new IdentityHashMap<Tensor, Integer>();
        for (int index = 0; index < outputs.size(); index++) {
            Integer firstIndex = requestedPositions.putIfAbsent(outputs.get(index), index);
            if (firstIndex != null) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] duplicates outputs[" + firstIndex + "]");
            }
        }
        Objects.requireNonNull(ingress, "ingress");

        var requestedConstants = new IdentityHashMap<
                Tensor, CompileTimeConstantGraph.Splat>();
        for (CompileTimeConstantGraph.Binding binding : ingress.bindings()) {
            requestedConstants.put(binding.tensor(), binding.splat());
        }
        var encounteredIngress = new IdentityHashMap<Tensor, Boolean>();
        var constants = new HashMap<ValueId, CompileTimeConstantGraph.Splat>();

        var values = new ArrayList<GraphValue>();
        var nodes = new ArrayList<CompiledNode>();
        var graphInputs = new ArrayList<ValueId>();
        var nodePhases = new LinkedHashMap<NodeId, GraphPhase>();
        var leafValues = new IdentityHashMap<Tensor, ValueId>();
        var producerOutputs = new IdentityHashMap<TensorProducer, List<ValueId>>();
        var visiting = new IdentityHashMap<TensorProducer, Boolean>();
        long nextValueId = 0;
        long nextNodeId = 0;

        for (Tensor requested : outputs) {
            TensorProvenance requestedProvenance = requested.provenance().orElse(null);
            if (requestedProvenance == null) {
                if (!leafValues.containsKey(requested)) {
                    ValueId valueId = new ValueId(nextValueId++);
                    leafValues.put(requested, valueId);
                    graphInputs.add(valueId);
                    values.add(new GraphValue(valueId, requested.descriptor()));
                    addConstant(
                            requested,
                            valueId,
                            requestedConstants,
                            encounteredIngress,
                            constants);
                }
                continue;
            }
            TensorProducer root = requestedProvenance.producer();
            if (producerOutputs.containsKey(root)) {
                continue;
            }

            var stack = new ArrayDeque<TraversalFrame>();
            visiting.put(root, Boolean.TRUE);
            stack.push(new TraversalFrame(root, 0));
            while (!stack.isEmpty()) {
                TraversalFrame frame = stack.peek();
                TensorProducer producer = frame.producer();
                List<Tensor> producerInputs = producer.inputs();
                if (frame.nextInputIndex() < producerInputs.size()) {
                    Tensor input = producerInputs.get(frame.nextInputIndex());
                    stack.pop();
                    stack.push(new TraversalFrame(producer, frame.nextInputIndex() + 1));
                    TensorProvenance inputProvenance = input.provenance().orElse(null);
                    if (inputProvenance == null) {
                        if (!leafValues.containsKey(input)) {
                            ValueId valueId = new ValueId(nextValueId++);
                            leafValues.put(input, valueId);
                            graphInputs.add(valueId);
                            values.add(new GraphValue(valueId, input.descriptor()));
                            addConstant(
                                    input,
                                    valueId,
                                    requestedConstants,
                                    encounteredIngress,
                                    constants);
                        }
                    } else {
                        TensorProducer inputProducer = inputProvenance.producer();
                        if (!producerOutputs.containsKey(inputProducer)
                                && !visiting.containsKey(inputProducer)) {
                            visiting.put(inputProducer, Boolean.TRUE);
                            stack.push(new TraversalFrame(inputProducer, 0));
                        }
                    }
                    continue;
                }

                var inputIds = new ArrayList<ValueId>(producerInputs.size());
                for (Tensor input : producerInputs) {
                    inputIds.add(resolve(input, leafValues, producerOutputs));
                }
                var outputIds = new ArrayList<ValueId>(producer.outputCount());
                for (var descriptor : producer.outputDescriptors()) {
                    ValueId valueId = new ValueId(nextValueId++);
                    outputIds.add(valueId);
                    values.add(new GraphValue(valueId, descriptor));
                }
                List<ValueId> outputSnapshot = List.copyOf(outputIds);
                producerOutputs.put(producer, outputSnapshot);
                NodeId nodeId = new NodeId(nextNodeId++);
                nodes.add(new CompiledNode(nodeId, producer.operation(), inputIds, outputSnapshot));
                nodePhases.put(nodeId, GraphPhase.FORWARD);
                visiting.remove(producer);
                stack.pop();
            }
        }

        var graphOutputs = new ArrayList<ValueId>(outputs.size());
        Map<ValueId, Integer> outputPositions = new HashMap<>();
        for (int index = 0; index < outputs.size(); index++) {
            ValueId valueId = resolve(outputs.get(index), leafValues, producerOutputs);
            Integer firstIndex = outputPositions.putIfAbsent(valueId, index);
            if (firstIndex != null) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] duplicates outputs[" + firstIndex
                                + "] at " + valueId);
            }
            graphOutputs.add(valueId);
        }
        for (int index = 0; index < ingress.bindings().size(); index++) {
            Tensor tensor = ingress.bindings().get(index).tensor();
            if (!encounteredIngress.containsKey(tensor)) {
                throw new IllegalArgumentException(
                        "ingress.bindings()[" + index + "] is not a reachable leaf");
            }
        }
        CompiledGraphModel graph =
                new CompiledGraphModel(values, nodes, graphInputs, graphOutputs, nodePhases);
        return new CompileTimeConstantGraph(graph, constants);
    }

    private static void addConstant(
            Tensor tensor,
            ValueId valueId,
            IdentityHashMap<Tensor, CompileTimeConstantGraph.Splat> requestedConstants,
            IdentityHashMap<Tensor, Boolean> encounteredIngress,
            Map<ValueId, CompileTimeConstantGraph.Splat> constants) {
        CompileTimeConstantGraph.Splat splat = requestedConstants.get(tensor);
        if (splat != null) {
            encounteredIngress.put(tensor, Boolean.TRUE);
            constants.put(valueId, splat);
        }
    }

    private static ValueId resolve(
            Tensor tensor,
            IdentityHashMap<Tensor, ValueId> leafValues,
            IdentityHashMap<TensorProducer, List<ValueId>> producerOutputs) {
        TensorProvenance provenance = tensor.provenance().orElse(null);
        if (provenance == null) {
            return leafValues.get(tensor);
        }
        return producerOutputs.get(provenance.producer()).get(provenance.outputIndex());
    }

    private record TraversalFrame(TensorProducer producer, int nextInputIndex) {
    }
}
