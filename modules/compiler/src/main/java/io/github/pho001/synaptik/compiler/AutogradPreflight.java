package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Preflights the closed {@code SUPPORTED_0004} first-order rule matrix before constructing any
 * derivative Tensor expression.
 *
 * <p>The iterative inventory covers every producer and canonical output wrapper reachable from
 * the requested forward boundary. Objective ancestry, target reachability, occurrence selection,
 * and ingress membership use exact Tensor and producer object identity rather than identifiers,
 * record equality, labels, or storage. A successful plan is request-local compiler bookkeeping;
 * it retains deterministic producer postorder and the original-producer identity set needed by
 * reverse accumulation and phase-aware capture.</p>
 *
 * <p>This owner selects rules and rejects unsupported operation, attribute, role, data-type,
 * Shape, and policy combinations. It neither performs captured-graph inference nor reads Tensor
 * payloads, captures a graph, allocates storage, lowers work, or executes computation.</p>
 */
final class AutogradPreflight {
    private AutogradPreflight() {}

    /**
     * Immutable scalar-objective and ordered-target request for one first-order expansion.
     *
     * <p>Construction snapshots the target list and enforces non-null, non-empty, exact-object-
     * identity-unique targets. {@link #preflight(CompileMode, List, FirstOrderRequest,
     * CompileTimeConstantGraph.Ingress)} later verifies that the objective is an exact requested
     * forward output with scalar Shape, floating type, and gradient eligibility, and that every
     * target belongs to a selected differentiable route in that objective's ancestry.</p>
     *
     * @param objective non-null exact Tensor requested as the scalar objective
     * @param targets non-null, non-empty, exact-object-identity-unique ordered targets; the list
     *     is snapshotted and target order becomes result-role order
     */
    record FirstOrderRequest(Tensor objective, List<Tensor> targets) {
        /**
         * Validates and snapshots one first-order request.
         *
         * @param objective non-null exact Tensor requested as the scalar objective
         * @param targets non-null, non-empty, exact-object-identity-unique ordered targets; the
         *     list is snapshotted
         * @throws NullPointerException if {@code objective}, {@code targets}, or the first target
         *     element encountered is {@code null}
         * @throws IllegalArgumentException if {@code targets} is empty or repeats an exact Tensor
         *     reference
         */FirstOrderRequest {
            Objects.requireNonNull(objective, "objective");
            Objects.requireNonNull(targets, "targets");
            if (targets.isEmpty()) {
                throw new IllegalArgumentException("targets must not be empty");
            }
            IdentityHashMap<Tensor, Integer> positions = new IdentityHashMap<>();
            for (int index = 0; index < targets.size(); index++) {
                Tensor target = Objects.requireNonNull(targets.get(index), "targets[" + index + "]");
                Integer first = positions.putIfAbsent(target, index);
                if (first != null) {
                    throw new IllegalArgumentException(
                            "targets[" + index + "] duplicates targets[" + first + "]");
                }
            }
            targets = List.copyOf(targets);
        }
    }

    /**
     * Validates a backward-capable request and its complete selected slice without constructing a
     * seed, derivative constant, or formula Tensor.
     *
     * <p>The method inventories all requested forward producers, then the objective ancestry. It
     * validates exact objective and target membership, retains only routes that can reach at
     * least one requested target through a differentiable input role, and validates every
     * selected occurrence against the closed matrix. The first known failure is reported before
     * derivative Tensor identity allocation. Full captured-graph inference and validation remain
     * a later compiler stage.</p>
     *
     * @param mode non-null backward-capable compile mode; {@link CompileMode#FORWARD_ONLY} is
     *     rejected
     * @param forwardOutputs non-null, non-empty ordered forward boundary already validated by
     *     {@link GraphCompiler}; the list and Tensors are observed but not mutated
     * @param request non-null scalar-objective and ordered-target request
     * @param forwardIngress non-null ordered explicit constant bindings; every bound Tensor must
     *     be a reachable leaf in the complete forward inventory
     * @return a non-null request-local plan containing exact Tensor/producer references for
     *     immediate reverse accumulation and combined capture
     * @throws NullPointerException if a top-level argument is {@code null}
     * @throws IllegalArgumentException if the mode, objective, target membership, differentiable
     *     route, canonical wrapper, ingress membership, or any selected operation, attribute,
     *     role, data-type, Shape, or derivative-policy fact is unsupported
     */
    static Plan preflight(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            FirstOrderRequest request,
            CompileTimeConstantGraph.Ingress forwardIngress) {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(forwardOutputs, "forwardOutputs");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(forwardIngress, "forwardIngress");
        if (mode == CompileMode.FORWARD_ONLY) {
            throw new IllegalArgumentException("FORWARD_ONLY must not include a first-order request");
        }

        Tensor objective = request.objective();
        int objectivePosition = identityIndexOf(forwardOutputs, objective);
        if (objectivePosition < 0) {
            throw new IllegalArgumentException(
                    "firstOrderRequest.objective must be an exact forward output");
        }
        if (objective.descriptor().shape().rank() != 0) {
            throw new IllegalArgumentException(
                    "firstOrderRequest.objective must be scalar");
        }
        DataType objectiveType = objective.descriptor().dataType();
        if (!objectiveType.isFloating() || !objective.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "firstOrderRequest.objective must be floating and require gradients");
        }

        Inventory complete = inventory(forwardOutputs);
        for (int index = 0; index < forwardIngress.bindings().size(); index++) {
            if (!complete.tensors().containsKey(
                    forwardIngress.bindings().get(index).tensor())) {
                throw new IllegalArgumentException(
                        "forwardIngress.bindings()[" + index + "] is not a reachable leaf");
            }
        }
        Inventory ancestry = inventory(List.of(objective));
        for (int index = 0; index < request.targets().size(); index++) {
            Tensor target = request.targets().get(index);
            if (!complete.tensors().containsKey(target)) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index + "] is not in the forward graph");
            }
            if (!ancestry.routeTensors().containsKey(target)) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index + "] is not in objective ancestry");
            }
            if (target.descriptor().dataType() != objectiveType
                    || !target.descriptor().requiresGrad()) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index
                                + "] must have the objective floating type and require gradients");
            }
        }

        IdentityHashMap<Tensor, Boolean> containsTarget = new IdentityHashMap<>();
        for (Tensor target : request.targets()) {
            containsTarget.put(target, Boolean.TRUE);
        }
        List<SelectedOccurrence> selected = new ArrayList<>();
        for (int producerIndex = 0;
                producerIndex < ancestry.producerPostorder().size();
                producerIndex++) {
            TensorProducer producer = ancestry.producerPostorder().get(producerIndex);
            boolean[] selectedInputs = new boolean[producer.inputs().size()];
            boolean anySelectedInput = false;
            for (int input = 0; input < producer.inputs().size(); input++) {
                selectedInputs[input] = containsTarget.containsKey(producer.inputs().get(input));
                anySelectedInput |= selectedInputs[input];
            }
            int selectedOutput = ancestry.selectedOutput(producer);
            boolean selectedOutputContains =
                    selectedOutput >= 0
                            && containsTarget.containsKey(producer.output(selectedOutput));
            if (anySelectedInput && selectedOutput >= 0) {
                containsTarget.put(producer.output(selectedOutput), Boolean.TRUE);
                selectedOutputContains = true;
            }
            if (!anySelectedInput) {
                continue;
            }
            if (selectedOutput < 0) {
                throw new IllegalArgumentException(
                        occurrence(producerIndex, producer, -1, -1)
                                + "missing selected canonical output");
            }
            validateOccurrence(
                    producerIndex, producer, selectedOutput, selectedInputs, objectiveType);
            selected.add(new SelectedOccurrence(
                    producerIndex, producer, selectedOutput, selectedInputs));
            if (!selectedOutputContains) {
                throw new IllegalArgumentException(
                        occurrence(producerIndex, producer, selectedOutput, -1)
                                + "selected route is not differentiable");
            }
        }
        for (int index = 0; index < request.targets().size(); index++) {
            if (!containsTarget.containsKey(objective)) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index
                                + "] has no differentiable path from objective");
            }
        }

        return new Plan(
                objective,
                request.targets(),
                ancestry.producerPostorder(),
                selected,
                complete.producers());
    }

    private static void validateOccurrence(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs,
            DataType objectiveType) {
        Operation operation = producer.operation();
        if (producer.outputCount() != 1 || outputIndex != 0) {
            throw unsupported(producerIndex, producer, outputIndex, -1, "requires one output");
        }
        Tensor output = producer.output(outputIndex);
        if (output.descriptor().dataType() != objectiveType
                || !output.descriptor().dataType().isFloating()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1, "output type is not the objective type");
        }
        if (!output.descriptor().requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1, "selected output must require gradients");
        }
        for (int input = 0; input < selectedInputs.length; input++) {
            if (selectedInputs[input]
                    && !producer.inputs().get(input).descriptor().requiresGrad()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, input,
                        "selected input must require gradients");
            }
        }

        if (operation.kind() instanceof BinaryArithmeticKind kind) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE
                    || (kind != BinaryArithmeticKind.ADD
                            && kind != BinaryArithmeticKind.SUB
                            && kind != BinaryArithmeticKind.MUL)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported binary variant");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, false);
            return;
        }
        if (operation.kind() instanceof ScalarElementwiseKind kind) {
            if (!(operation.attrs() instanceof ScalarValueAttrs attrs)
                    || (kind != ScalarElementwiseKind.ADD
                            && kind != ScalarElementwiseKind.SUB
                            && kind != ScalarElementwiseKind.MUL)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported scalar variant");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            if (attrs.value().dataType() != output.descriptor().dataType()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0, "scalar type must match input/output");
            }
            return;
        }
        if (operation.kind() == WhereSelectionKind.WHERE) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported WHERE attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 3);
            if (selectedInputs[0]) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0, "condition role is non-differentiable");
            }
            if (producer.inputs().get(0).descriptor().dataType() != DataType.BOOL) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0, "condition must be BOOL");
            }
            for (int input = 1; input < 3; input++) {
                if (producer.inputs().get(input).descriptor().dataType()
                        != output.descriptor().dataType()) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, input,
                            "branch type must exactly match output");
                }
            }
            return;
        }
        if (operation.kind() == CastKind.CAST) {
            if (!(operation.attrs() instanceof CastAttrs attrs)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported CAST attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            DataType inputType = producer.inputs().getFirst().descriptor().dataType();
            if (!inputType.isFloating()
                    || inputType != output.descriptor().dataType()
                    || attrs.targetDataType() != inputType) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0, "CAST must be same floating type");
            }
            if (!producer.inputs().getFirst().descriptor().shape()
                    .equals(output.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0, "CAST Shape must be unchanged");
            }
            return;
        }
        if (operation.kind() instanceof UnaryElementwiseKind kind) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE
                    || (kind != UnaryElementwiseKind.NEG
                            && kind != UnaryElementwiseKind.EXP
                            && kind != UnaryElementwiseKind.EXPM1
                            && kind != UnaryElementwiseKind.SIGMOID
                            && kind != UnaryElementwiseKind.TANH)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported unary variant");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            return;
        }
        if (operation.kind() == AggregateReductionKind.SUM) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE
                    && !(operation.attrs() instanceof AxisReductionAttrs)
                    && !(operation.attrs() instanceof MultiAxisReductionAttrs)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported SUM attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == CumulativeScanKind.CUM_SUM) {
            if (!(operation.attrs() instanceof CumulativeScanAttrs)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported CUM_SUM attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            return;
        }
        if (operation.kind() == ContiguousKind.CONTIGUOUS) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported CONTIGUOUS attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            return;
        }
        if (operation.kind() instanceof ShapeTransformKind kind) {
            if (!(operation.attrs() instanceof TargetShapeAttrs)
                    || (kind != ShapeTransformKind.RESHAPE && kind != ShapeTransformKind.EXPAND)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported shape transform");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof AxisTransformKind kind) {
            boolean attrsSupported = switch (kind) {
                case EXPAND_DIMS, SQUEEZE -> operation.attrs() instanceof AxisTransformAttrs;
                case PERMUTE -> operation.attrs() instanceof PermutationAttrs;
            };
            if (!attrsSupported) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported axis transform attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            return;
        }
        throw unsupported(
                producerIndex, producer, outputIndex, firstSelected(selectedInputs),
                "operation is outside SUPPORTED_0004");
    }

    private static void requireInputs(
            int producerIndex, TensorProducer producer, int outputIndex, int count) {
        if (producer.inputs().size() != count) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "expected " + count + " inputs, got " + producer.inputs().size());
        }
    }

    private static void requireSameFloatingDescriptors(
            int producerIndex, TensorProducer producer, int outputIndex, boolean sameShape) {
        requireSameFloatingTypes(producerIndex, producer, outputIndex);
        if (sameShape) {
            var outputShape = producer.output(outputIndex).descriptor().shape();
            for (int input = 0; input < producer.inputs().size(); input++) {
                if (!producer.inputs().get(input).descriptor().shape().equals(outputShape)) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, input,
                            "input/output Shape must match exactly");
                }
            }
        }
    }

    private static void requireSameFloatingTypes(
            int producerIndex, TensorProducer producer, int outputIndex) {
        DataType outputType = producer.output(outputIndex).descriptor().dataType();
        if (!outputType.isFloating()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1, "output must be floating");
        }
        for (int input = 0; input < producer.inputs().size(); input++) {
            if (producer.inputs().get(input).descriptor().dataType() != outputType) {
                throw unsupported(
                        producerIndex, producer, outputIndex, input,
                        "input/output floating types must match exactly");
            }
        }
    }

    private static IllegalArgumentException unsupported(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            int inputIndex,
            String reason) {
        return new IllegalArgumentException(
                occurrence(producerIndex, producer, outputIndex, inputIndex) + reason);
    }

    private static String occurrence(
            int producerIndex, TensorProducer producer, int outputIndex, int inputIndex) {
        return "producerPostorder[" + producerIndex + "] output[" + outputIndex + "] input["
                + inputIndex + "] " + producer.operation().kind().getClass().getName() + "."
                + producer.operation().kind().name() + " attrs="
                + producer.operation().attrs().getClass().getName() + ": ";
    }

    private static int firstSelected(boolean[] selected) {
        for (int index = 0; index < selected.length; index++) {
            if (selected[index]) {
                return index;
            }
        }
        return -1;
    }

    private static int identityIndexOf(List<Tensor> tensors, Tensor candidate) {
        for (int index = 0; index < tensors.size(); index++) {
            if (tensors.get(index) == candidate) {
                return index;
            }
        }
        return -1;
    }

    private static Inventory inventory(List<Tensor> roots) {
        IdentityHashMap<Tensor, Boolean> tensors = new IdentityHashMap<>();
        IdentityHashMap<Tensor, Boolean> routeTensors = new IdentityHashMap<>();
        List<Tensor> encountered = new ArrayList<>();
        Set<TensorProducer> producers =
                Collections.newSetFromMap(new IdentityHashMap<>());
        IdentityHashMap<TensorProducer, Integer> selectedOutputs = new IdentityHashMap<>();
        List<TensorProducer> postorder = new ArrayList<>();
        IdentityHashMap<TensorProducer, Boolean> complete = new IdentityHashMap<>();
        IdentityHashMap<TensorProducer, Boolean> visiting = new IdentityHashMap<>();

        for (Tensor root : roots) {
            validateCanonicalWrapper(root);
            encounter(root, tensors, encountered);
            routeTensors.put(root, Boolean.TRUE);
            TensorProvenance provenance = root.provenance().orElse(null);
            if (provenance == null) {
                continue;
            }
            selectedOutputs.putIfAbsent(provenance.producer(), provenance.outputIndex());
            if (complete.containsKey(provenance.producer())) {
                continue;
            }
            ArrayDeque<TraversalFrame> stack = new ArrayDeque<>();
            visiting.put(provenance.producer(), Boolean.TRUE);
            stack.push(new TraversalFrame(provenance.producer(), 0));
            while (!stack.isEmpty()) {
                TraversalFrame frame = stack.peek();
                TensorProducer producer = frame.producer();
                producers.add(producer);
                if (frame.nextInput() < producer.inputs().size()) {
                    Tensor input = producer.inputs().get(frame.nextInput());
                    validateCanonicalWrapper(input);
                    stack.pop();
                    stack.push(new TraversalFrame(producer, frame.nextInput() + 1));
                    encounter(input, tensors, encountered);
                    routeTensors.put(input, Boolean.TRUE);
                    TensorProvenance inputProvenance = input.provenance().orElse(null);
                    if (inputProvenance != null) {
                        selectedOutputs.putIfAbsent(
                                inputProvenance.producer(), inputProvenance.outputIndex());
                        if (!complete.containsKey(inputProvenance.producer())
                                && !visiting.containsKey(inputProvenance.producer())) {
                            visiting.put(inputProvenance.producer(), Boolean.TRUE);
                            stack.push(new TraversalFrame(inputProvenance.producer(), 0));
                        }
                    }
                    continue;
                }
                for (int output = 0; output < producer.outputCount(); output++) {
                    encounter(producer.output(output), tensors, encountered);
                }
                postorder.add(producer);
                complete.put(producer, Boolean.TRUE);
                visiting.remove(producer);
                stack.pop();
            }
        }
        return new Inventory(
                tensors,
                routeTensors,
                List.copyOf(encountered),
                Collections.unmodifiableSet(producers),
                List.copyOf(postorder),
                selectedOutputs);
    }

    private static void encounter(
            Tensor tensor,
            IdentityHashMap<Tensor, Boolean> tensors,
            List<Tensor> encountered) {
        if (tensors.putIfAbsent(tensor, Boolean.TRUE) == null) {
            encountered.add(tensor);
        }
    }

    private static void validateCanonicalWrapper(Tensor tensor) {
        TensorProvenance provenance = tensor.provenance().orElse(null);
        if (provenance != null
                && provenance.producer().output(provenance.outputIndex()) != tensor) {
            throw new IllegalArgumentException(
                    "non-canonical Tensor wrapper for producer output["
                            + provenance.outputIndex() + "]");
        }
    }

    /**
     * Immutable selected producer occurrence for reverse accumulation.
     *
     * @param postorderIndex deterministic objective-ancestry producer postorder position
     * @param producer exact original producer occurrence; never a reconstructed or captured node
     * @param outputIndex zero-based selected canonical output position
     * @param selectedInputs input-position-aligned differentiable-route flags; cloned on input
     *     and access
     */
    record SelectedOccurrence(
            int postorderIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs) {
        /**
         * Snapshots one selected occurrence and its input-role flags.
         *
         * @param postorderIndex deterministic objective-ancestry producer postorder position
         * @param producer exact original producer occurrence
         * @param outputIndex zero-based selected canonical output position
         * @param selectedInputs non-null input-position-aligned differentiable-route flags
         * @throws NullPointerException if {@code selectedInputs} is {@code null}
         */SelectedOccurrence {
            selectedInputs = selectedInputs.clone();
        }

        /**
         * Returns an independent snapshot of the selected input-role flags.
         *
         * @return a new non-null boolean array aligned with producer input positions
         */
        @Override
        public boolean[] selectedInputs() {
            return selectedInputs.clone();
        }

        /**
         * Reports whether one producer input is on the selected differentiable route.
         *
         * @param index zero-based producer input position
         * @return {@code true} exactly when that input is selected
         * @throws ArrayIndexOutOfBoundsException if {@code index} is outside the input range
         */boolean selectedInput(int index) {
            return selectedInputs[index];
        }
    }

    /**
     * Complete successful preflight plan consumed within the same compile request.
     *
     * <p>List components are immutable snapshots. Tensor and producer references deliberately
     * preserve exact pre-capture identity and must be discarded after combined capture; they are
     * not graph IR or persistent compiler output.</p>
     *
     * @param objective exact scalar objective Tensor
     * @param targets ordered immutable target snapshot
     * @param producerPostorder complete objective-ancestry producer postorder
     * @param selectedOccurrences closed-matrix occurrences in deterministic postorder
     * @param originalProducers unmodifiable identity set for the complete forward boundary,
     *     including producer occurrences outside the objective ancestry
     */
    record Plan(
            Tensor objective,
            List<Tensor> targets,
            List<TensorProducer> producerPostorder,
            List<SelectedOccurrence> selectedOccurrences,
            Set<TensorProducer> originalProducers) {
        /**
         * Snapshots the ordered list components of a successful preflight plan.
         *
         * @param objective non-null exact scalar objective Tensor
         * @param targets non-null ordered target list
         * @param producerPostorder non-null complete objective-ancestry producer postorder
         * @param selectedOccurrences non-null selected occurrences in deterministic postorder
         * @param originalProducers non-null identity set of original forward producers
         * @throws NullPointerException if a list or one of its elements is {@code null}
         */Plan {
            targets = List.copyOf(targets);
            producerPostorder = List.copyOf(producerPostorder);
            selectedOccurrences = List.copyOf(selectedOccurrences);
        }
    }

    private record Inventory(
            IdentityHashMap<Tensor, Boolean> tensors,
            IdentityHashMap<Tensor, Boolean> routeTensors,
            List<Tensor> encounteredOrder,
            Set<TensorProducer> producers,
            List<TensorProducer> producerPostorder,
            IdentityHashMap<TensorProducer, Integer> selectedOutputs) {
        int selectedOutput(TensorProducer producer) {
            return selectedOutputs.getOrDefault(producer, -1);
        }
    }

    private record TraversalFrame(TensorProducer producer, int nextInput) {}
}
