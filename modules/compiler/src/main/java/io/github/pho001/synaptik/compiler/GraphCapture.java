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
import java.util.Set;

/**
 * Converts public Tensor-expression provenance into immutable forward-only or combined graph
 * structure.
 *
 * <p>This package-private compiler boundary translates exact leaf and producer occurrences into
 * graph-local identities for one capture call. It preserves deterministic operand and
 * producer-output ordering, including hidden canonical wrappers and opaque state positions.
 * Forward-only entries classify every producer as {@link GraphPhase#FORWARD}; combined capture
 * uses original producer object identity to classify each occurrence as {@code FORWARD} or
 * {@link GraphPhase#BACKWARD} while assigning every {@link NodeId} and {@link ValueId} once.</p>
 *
 * <p>Capture performs no semantic inference, optimization, derivative-rule selection, numerical
 * evaluation, planning, storage access, preparation, backend work, or execution. Returned graph
 * state retains operations and descriptors but no Tensor, producer, or provenance reference.</p>
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
        validateForwardOutputs(outputs);
        Objects.requireNonNull(ingress, "ingress");
        return captureInternal(outputs, List.of(), null, null, null, ingress).constantGraph();
    }

    /**
     * Captures one combined forward/backward expression and records stable gradient-boundary
     * ordinals.
     *
     * <p>Traversal roots are the ordered forward outputs followed by gradient roots in target-role
     * order. Each exact producer occurrence is emitted once in deterministic input-first
     * depth-first postorder, and every declared output slot receives a graph value. The public
     * graph boundary is the distinct forward prefix followed by each gradient value absent from
     * that prefix at its first role occurrence. Every target role retains the ordinal of its
     * resolved boundary value, including roles that share a gradient or reuse a forward value.</p>
     *
     * @param forwardOutputs non-null, non-empty ordered forward boundary with no repeated exact
     *     Tensor reference or resolved logical value
     * @param targetGradients non-null ordered exact target/gradient roles; elements are observed
     *     but not mutated
     * @param originalProducers non-null identity set of every producer in the complete original
     *     forward request; membership selects {@link GraphPhase#FORWARD}
     * @param stageOneProducers non-null identity set of producers first owned by stage one
     * @param stageTwoProducers non-null identity set of producers first owned by stage two
     * @param ingress non-null ordered merged caller and generated logical-splat bindings; every
     *     bound Tensor must be encountered as a reachable provenance-free leaf
     * @return a non-null immutable combined graph with derivative metadata, a forward-boundary
     *     count, and one graph-output ordinal per target role
     * @throws NullPointerException if a required argument or target-role element is {@code null}
     * @throws IllegalArgumentException if the forward boundary is empty or duplicated, an ingress
     *     binding is not a reachable leaf, or structural graph construction rejects the result
     */
    static CombinedCapture captureCombined(
            List<Tensor> forwardOutputs,
            List<FirstOrderAutograd.TargetGradient> targetGradients,
            Set<TensorProducer> originalProducers,
            Set<TensorProducer> stageOneProducers,
            Set<TensorProducer> stageTwoProducers,
            CompileTimeConstantGraph.Ingress ingress) {
        validateForwardOutputs(forwardOutputs);
        Objects.requireNonNull(targetGradients, "targetGradients");
        for (int index = 0; index < targetGradients.size(); index++) {
            Objects.requireNonNull(targetGradients.get(index), "targetGradients[" + index + "]");
        }
        Objects.requireNonNull(originalProducers, "originalProducers");
        Objects.requireNonNull(stageOneProducers, "stageOneProducers");
        Objects.requireNonNull(stageTwoProducers, "stageTwoProducers");
        Objects.requireNonNull(ingress, "ingress");
        return captureInternal(
                forwardOutputs,
                targetGradients,
                originalProducers,
                stageOneProducers,
                stageTwoProducers,
                ingress);
    }

    private static void validateForwardOutputs(List<Tensor> outputs) {
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
    }

    private static CombinedCapture captureInternal(
            List<Tensor> forwardOutputs,
            List<FirstOrderAutograd.TargetGradient> targetGradients,
            Set<TensorProducer> originalProducers,
            Set<TensorProducer> stageOneProducers,
            Set<TensorProducer> stageTwoProducers,
            CompileTimeConstantGraph.Ingress ingress) {
        List<Tensor> traversalRoots =
                new ArrayList<>(forwardOutputs.size() + targetGradients.size());
        traversalRoots.addAll(forwardOutputs);
        for (FirstOrderAutograd.TargetGradient role : targetGradients) {
            traversalRoots.add(role.gradient());
        }

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
        var derivativeOrders = new LinkedHashMap<NodeId, Integer>();
        var leafValues = new IdentityHashMap<Tensor, ValueId>();
        var producerOutputs = new IdentityHashMap<TensorProducer, List<ValueId>>();
        var visiting = new IdentityHashMap<TensorProducer, Boolean>();
        long nextValueId = 0;
        long nextNodeId = 0;

        for (Tensor requested : traversalRoots) {
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
                int derivativeOrder;
                if (originalProducers == null || originalProducers.contains(producer)) {
                    derivativeOrder = 0;
                } else if (stageOneProducers.contains(producer)) {
                    derivativeOrder = 1;
                } else if (stageTwoProducers.contains(producer)) {
                    derivativeOrder = 2;
                } else {
                    throw new IllegalArgumentException(
                            "captured producer has no derivative-order owner");
                }
                derivativeOrders.put(nodeId, derivativeOrder);
                nodePhases.put(
                        nodeId,
                        derivativeOrder == 0 ? GraphPhase.FORWARD : GraphPhase.BACKWARD);
                visiting.remove(producer);
                stack.pop();
            }
        }

        var graphOutputs =
                new ArrayList<ValueId>(forwardOutputs.size() + targetGradients.size());
        Map<ValueId, Integer> outputPositions = new HashMap<>();
        for (int index = 0; index < forwardOutputs.size(); index++) {
            ValueId valueId = resolve(forwardOutputs.get(index), leafValues, producerOutputs);
            Integer firstIndex = outputPositions.putIfAbsent(valueId, index);
            if (firstIndex != null) {
                throw new IllegalArgumentException(
                        "outputs[" + index + "] duplicates outputs[" + firstIndex
                                + "] at " + valueId);
            }
            graphOutputs.add(valueId);
        }
        List<Integer> gradientOutputOrdinals = new ArrayList<>(targetGradients.size());
        for (FirstOrderAutograd.TargetGradient role : targetGradients) {
            ValueId valueId = resolve(role.gradient(), leafValues, producerOutputs);
            Integer ordinal = outputPositions.get(valueId);
            if (ordinal == null) {
                ordinal = graphOutputs.size();
                outputPositions.put(valueId, ordinal);
                graphOutputs.add(valueId);
            }
            gradientOutputOrdinals.add(ordinal);
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
        return new CombinedCapture(
                new CompileTimeConstantGraph(graph, constants),
                forwardOutputs.size(),
                gradientOutputOrdinals,
                new DerivativeGraphMetadata(graph, derivativeOrders));
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

    /**
     * Combined-capture state whose positional roles survive graph-local identifier rebuilds.
     *
     * @param constantGraph non-null captured immutable graph and exact source facts
     * @param forwardOutputCount number of leading forward boundary positions
     * @param gradientOutputOrdinals non-null ordered graph-output ordinal per target role;
     *     snapshotted and permitted to contain repeated ordinals
     * @param derivatives non-null exact derivative-order sidecar for the captured graph
     */
    record CombinedCapture(
            CompileTimeConstantGraph constantGraph,
            int forwardOutputCount,
            List<Integer> gradientOutputOrdinals,
            DerivativeGraphMetadata derivatives) {
        /**
         * Validates and snapshots one combined-capture result.
         *
         * @param constantGraph non-null captured immutable graph and exact source facts
         * @param forwardOutputCount number of leading forward boundary positions
     * @param gradientOutputOrdinals non-null ordered graph-output ordinal per target role
     * @param derivatives non-null exact derivative-order sidecar for the captured graph
     * @throws NullPointerException if {@code constantGraph},
     *     {@code gradientOutputOrdinals}, or an ordinal element is {@code null}
     * @throws IllegalArgumentException if the derivative metadata owns a different graph
         */CombinedCapture {
            Objects.requireNonNull(constantGraph, "constantGraph");
            gradientOutputOrdinals = List.copyOf(gradientOutputOrdinals);
            Objects.requireNonNull(derivatives, "derivatives");
            if (derivatives.graph() != constantGraph.graph()) {
                throw new IllegalArgumentException(
                        "derivatives graph must be the exact captured graph");
            }
        }
    }
}
