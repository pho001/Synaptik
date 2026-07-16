package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Associates an immutable compiler graph with exact logical-splat facts for fixed graph sources.
 *
 * <p>Every fact applies one exact typed {@link ScalarValue} to every logical coordinate of one
 * structural graph input. Facts do not contain Tensor state, storage, shapes, buffers, backend
 * values, or physical materialization. Inputs without facts remain caller-bindable; inputs with
 * facts are fixed compile-time sources and are excluded from {@link #bindableInputs()}.</p>
 *
 * @param graph non-null immutable structural graph retained by exact reference
 * @param constants non-null mapping from graph-input IDs to exact splats, snapshotted on creation
 */
record CompileTimeConstantGraph(
        CompiledGraphModel graph,
        Map<ValueId, CompileTimeConstantGraph.Splat> constants) {
    /**
     * Creates immutable logical constant state and validates every source role.
     *
     * @param graph non-null immutable structural graph retained by exact reference
     * @param constants non-null source-fact map; keys and values must be non-null, each key must
     *     name a graph input, and every splat type must equal a non-gradient input descriptor type
     * @throws NullPointerException if {@code graph}, {@code constants}, a key, or a value is null
     * @throws IllegalArgumentException if a fact names a non-input value, disagrees with its input
     *     descriptor type, or fixes a gradient-eligible input
     */
    CompileTimeConstantGraph {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(constants, "constants");

        List<Map.Entry<ValueId, Splat>> orderedEntries = new ArrayList<>(constants.entrySet());
        orderedEntries.sort(Comparator.comparing(
                Map.Entry<ValueId, Splat>::getKey,
                Comparator.nullsFirst(Comparator.comparingLong(ValueId::value))));
        for (Map.Entry<ValueId, Splat> entry : orderedEntries) {
            Objects.requireNonNull(entry.getKey(), "constants contains null key");
            Objects.requireNonNull(entry.getValue(), "constants[" + entry.getKey() + "]");
        }

        Map<ValueId, TensorDescriptor> inputDescriptors = inputDescriptors(graph);
        for (ValueId input : graph.inputs()) {
            Splat splat = constants.get(input);
            if (splat == null) {
                continue;
            }
            TensorDescriptor descriptor = inputDescriptors.get(input);
            if (splat.value().dataType() != descriptor.dataType()) {
                throw new IllegalArgumentException("constant " + input + " data type "
                        + splat.value().dataType() + " does not match input descriptor "
                        + descriptor.dataType());
            }
            if (descriptor.requiresGrad()) {
                throw new IllegalArgumentException(
                        "constant " + input + " input descriptor requires gradients");
            }
        }
        for (Map.Entry<ValueId, Splat> entry : orderedEntries) {
            if (!inputDescriptors.containsKey(entry.getKey())) {
                throw new IllegalArgumentException(
                        "constant " + entry.getKey() + " is not a graph input");
            }
        }
        constants = Map.copyOf(constants);
    }

    /**
     * Wraps a graph whose complete source boundary remains bindable.
     *
     * @param graph non-null immutable graph retained by exact reference
     * @return non-null immutable sidecar with no constant facts
     * @throws NullPointerException if {@code graph} is null
     */
    static CompileTimeConstantGraph withoutConstants(CompiledGraphModel graph) {
        return new CompileTimeConstantGraph(graph, Map.of());
    }

    /**
     * Derives the ordered caller-bindable source boundary.
     *
     * @return a new non-null immutable list containing exactly graph inputs without splat facts,
     *     in graph-input order
     */
    List<ValueId> bindableInputs() {
        List<ValueId> result = new ArrayList<>(graph.inputs().size() - constants.size());
        for (ValueId input : graph.inputs()) {
            if (!constants.containsKey(input)) {
                result.add(input);
            }
        }
        return List.copyOf(result);
    }

    /**
     * Remaps source roles across a graph-only transformation that preserves the ordered boundary.
     *
     * @param replacement non-null immutable graph with the same input count and equal descriptor
     *     at every corresponding input position
     * @return this exact sidecar when {@code replacement == graph}; otherwise a new immutable
     *     sidecar with facts remapped by input position
     * @throws NullPointerException if {@code replacement} is null
     * @throws IllegalArgumentException if the replacement source boundary differs in count or any
     *     ordered descriptor
     */
    CompileTimeConstantGraph replaceGraphPreservingInputRoles(CompiledGraphModel replacement) {
        Objects.requireNonNull(replacement, "replacement");
        if (replacement == graph) {
            return this;
        }
        if (replacement.inputs().size() != graph.inputs().size()) {
            throw new IllegalArgumentException("replacement input count "
                    + replacement.inputs().size() + " does not match " + graph.inputs().size());
        }

        Map<ValueId, TensorDescriptor> currentDescriptors = inputDescriptors(graph);
        Map<ValueId, TensorDescriptor> replacementDescriptors = inputDescriptors(replacement);
        Map<ValueId, Splat> remapped = new HashMap<>();
        for (int index = 0; index < graph.inputs().size(); index++) {
            ValueId currentInput = graph.inputs().get(index);
            ValueId replacementInput = replacement.inputs().get(index);
            TensorDescriptor current = currentDescriptors.get(currentInput);
            TensorDescriptor next = replacementDescriptors.get(replacementInput);
            if (!current.equals(next)) {
                throw new IllegalArgumentException("replacement input[" + index
                        + "] descriptor does not match current input descriptor");
            }
            Splat splat = constants.get(currentInput);
            if (splat != null) {
                remapped.put(replacementInput, splat);
            }
        }
        return new CompileTimeConstantGraph(replacement, remapped);
    }

    private static Map<ValueId, TensorDescriptor> inputDescriptors(CompiledGraphModel graph) {
        Map<ValueId, TensorDescriptor> values = new HashMap<>();
        for (GraphValue value : graph.values()) {
            values.put(value.id(), value.descriptor());
        }
        Map<ValueId, TensorDescriptor> result = new HashMap<>();
        for (ValueId input : graph.inputs()) {
            result.put(input, values.get(input));
        }
        return result;
    }

    /**
     * Holds one exact typed value that applies at every logical coordinate of a graph source.
     *
     * <p>Equality and hashing delegate to {@link ScalarValue}'s exact data-type-and-bit contract.
     * The immutable value reference is retained without conversion or materialization.</p>
     *
     * @param value non-null exact scalar retained by reference
     */
    record Splat(ScalarValue value) {
        /**
         * Creates one immutable logical splat.
         *
         * @param value non-null exact typed scalar retained by reference
         * @throws NullPointerException if {@code value} is null
         */
        Splat {
            Objects.requireNonNull(value, "value");
        }
    }

    /**
     * Binds one exact provenance-free non-gradient Tensor leaf identity to one logical splat.
     *
     * @param tensor non-null provenance-free non-gradient leaf retained only by the ingress request
     * @param splat non-null immutable logical splat retained by reference
     */
    record Binding(Tensor tensor, Splat splat) {
        /**
         * Creates one explicit leaf binding.
         *
         * @param tensor non-null provenance-free non-gradient Tensor leaf
         * @param splat non-null immutable logical splat
         * @throws NullPointerException if {@code tensor} or {@code splat} is null
         * @throws IllegalArgumentException if the Tensor has provenance or requires gradients
         */
        Binding {
            Objects.requireNonNull(tensor, "tensor");
            Objects.requireNonNull(splat, "splat");
            if (tensor.provenance().isPresent()) {
                throw new IllegalArgumentException("tensor must be a provenance-free leaf");
            }
            if (tensor.descriptor().requiresGrad()) {
                throw new IllegalArgumentException("tensor descriptor must not require gradients");
            }
        }
    }

    /**
     * Owns an ordered immutable request for explicit compile-time constant leaf ingress.
     *
     * <p>Tensor uniqueness uses exact object identity. Request order controls deterministic
     * diagnostics only; capture encounter order controls graph inputs and fact IDs.</p>
     *
     * @param bindings non-null ordered bindings, snapshotted on construction without null elements
     */
    record Ingress(List<Binding> bindings) {
        /**
         * Creates an immutable identity-unique ingress request.
         *
         * @param bindings non-null ordered bindings; elements must be non-null and Tensor
         *     references must be identity-unique
         * @throws NullPointerException if {@code bindings} or its first null element is null
         * @throws IllegalArgumentException if a later binding repeats an earlier exact Tensor
         */
        Ingress {
            Objects.requireNonNull(bindings, "bindings");
            IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
            for (int index = 0; index < bindings.size(); index++) {
                Binding binding = Objects.requireNonNull(
                        bindings.get(index), "bindings[" + index + "]");
                Integer first = positions.putIfAbsent(binding.tensor(), index);
                if (first != null) {
                    throw new IllegalArgumentException(
                            "bindings[" + index + "] duplicates bindings[" + first + "] tensor");
                }
            }
            bindings = List.copyOf(bindings);
        }

        /**
         * Creates an ingress request with no fixed leaves.
         *
         * @return non-null immutable empty ingress
         */
        static Ingress empty() {
            return new Ingress(List.of());
        }
    }
}
