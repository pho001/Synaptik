package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.shape.StaticDimension;
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
 * Preflights the closed {@code SUPPORTED_0004} and {@code SUPPORTED_0004A} first-order rule
 * matrices before constructing any derivative Tensor expression.
 *
 * <p>The iterative inventory covers every producer and canonical output wrapper reachable from
 * the requested forward boundary. Objective ancestry, target reachability, occurrence selection,
 * and ingress membership use exact Tensor and producer object identity rather than identifiers,
 * record equality, labels, or storage. A successful plan is request-local compiler bookkeeping;
 * it retains deterministic producer postorder and the original-producer identity set needed by
 * reverse accumulation and phase-aware capture.</p>
 *
 * <p>The additive {@code SUPPORTED_0004A} rows are exact-type {@code ERF}; masked
 * {@code SUM}; locally invertible {@code SUM_TO_SHAPE}; role-aware floating {@code MATMUL};
 * normalized {@code SLICE} and both {@code SLICE_UPDATE} data roles; {@code SELECT},
 * {@code PAD}, {@code TILE}, {@code CONCAT}, and {@code STACK}. Matrix-multiplication and slice
 * replacement are validated per selected input role: an unselected promoted operand does not
 * require a cotangent conversion, and only a selected slice-update input requires static selected
 * base extents. Binding-dependent shape inversion and every operation or role outside the two
 * closed matrices still fail.</p>
 *
 * <p>This owner selects rules and rejects unsupported operation, attribute, role, data-type,
 * Shape, and policy combinations. A known rejection occurs before the seed, a derivative
 * constant, or a formula Tensor is constructed, so it consumes no derivative {@code TensorId}.
 * The guarantee ends after a successful plan is returned: later public Tensor construction,
 * capture, inference, validation, or optimization may consume IDs before failing. This owner
 * neither performs captured-graph inference nor reads Tensor payloads, captures a graph,
 * allocates storage, lowers work, or executes computation.</p>
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
     * a later compiler stage. The returned plan retains exact original object references and is
     * owned by the current compile request; callers must hand it directly to reverse accumulation
     * and discard it after combined capture.</p>
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
                            && kind != UnaryElementwiseKind.TANH
                            && kind != UnaryElementwiseKind.ERF)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported unary variant");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            return;
        }
        if (operation.kind() == AggregateReductionKind.SUM) {
            if (operation.attrs() instanceof MaskedReductionAttrs attrs) {
                requireInputs(producerIndex, producer, outputIndex, 2);
                if (selectedInputs[1]) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked SUM mask role is non-differentiable");
                }
                Tensor data = producer.inputs().get(0);
                Tensor mask = producer.inputs().get(1);
                requireSameFloatingType(producerIndex, producer, outputIndex, 0);
                if (mask.descriptor().dataType() != DataType.BOOL) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked SUM mask must be BOOL");
                }
                Shape broadcast;
                try {
                    broadcast = ShapeBroadcast.broadcast(
                            mask.descriptor().shape(), data.descriptor().shape());
                } catch (IllegalArgumentException exception) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked SUM mask cannot broadcast to data Shape");
                }
                if (!broadcast.equals(data.descriptor().shape())
                        || attrs.axis() >= data.descriptor().shape().rank()
                        || !output.descriptor().shape().equals(
                                removeAxis(data.descriptor().shape(), attrs.axis()))) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 0,
                            "masked SUM descriptor or normalized axis is inconsistent");
                }
                return;
            }
            if (operation.attrs() instanceof SumToShapeAttrs attrs) {
                requireInputs(producerIndex, producer, outputIndex, 1);
                requireSameFloatingType(producerIndex, producer, outputIndex, 0);
                Shape inputShape = producer.inputs().getFirst().descriptor().shape();
                Shape targetShape = attrs.targetShape();
                if (!output.descriptor().shape().equals(targetShape)
                        || targetShape.rank() > inputShape.rank()) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 0,
                            "SUM_TO_SHAPE target rank or output Shape is inconsistent");
                }
                int offset = inputShape.rank() - targetShape.rank();
                for (int axis = 0; axis < targetShape.rank(); axis++) {
                    Dimension target = targetShape.dimension(axis);
                    Dimension input = inputShape.dimension(offset + axis);
                    if (!target.equals(input) && !isStaticOne(target)) {
                        throw unsupported(
                                producerIndex, producer, outputIndex, 0,
                                "SUM_TO_SHAPE aligned target must be exact input Dimension or static one");
                    }
                }
                return;
            }
            if (operation.attrs() != NoOperationAttrs.INSTANCE
                    && !(operation.attrs() instanceof AxisReductionAttrs)
                    && !(operation.attrs() instanceof MultiAxisReductionAttrs)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported SUM attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == MatmulKind.MATMUL) {
            if (operation.attrs() != NoOperationAttrs.INSTANCE) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported MATMUL attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            Tensor left = producer.inputs().get(0);
            Tensor right = producer.inputs().get(1);
            if (!left.descriptor().dataType().isFloating()
                    || !right.descriptor().dataType().isFloating()
                    || DataTypePromotion.promoteFloating(
                                    left.descriptor().dataType(), right.descriptor().dataType())
                            != output.descriptor().dataType()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "MATMUL operands and output must have the current floating promotion");
            }
            for (int input = 0; input < selectedInputs.length; input++) {
                if (selectedInputs[input]
                        && producer.inputs().get(input).descriptor().dataType()
                                != output.descriptor().dataType()) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, input,
                            "selected MATMUL input must have the output floating type");
                }
            }
            Shape expected;
            try {
                expected = matmulShape(
                        left.descriptor().shape(), right.descriptor().shape());
            } catch (IllegalArgumentException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "MATMUL rank, contraction, or batch Shape is inconsistent");
            }
            if (!expected.equals(output.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "MATMUL output Shape is inconsistent");
            }
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
        if (operation.kind() instanceof SliceKind kind) {
            if (!(operation.attrs() instanceof SliceAttrs attrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "only normalized SliceAttrs are supported");
            }
            int expectedInputs = kind == SliceKind.SLICE ? 1 : 2;
            requireInputs(producerIndex, producer, outputIndex, expectedInputs);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            Tensor base = producer.inputs().getFirst();
            Shape selectedShape;
            try {
                selectedShape = sliceShape(base.descriptor().shape(), attrs);
            } catch (IllegalArgumentException | ArithmeticException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "normalized slice geometry is inconsistent");
            }
            if (kind == SliceKind.SLICE) {
                if (!selectedShape.equals(output.descriptor().shape())) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 0,
                            "SLICE output Shape is inconsistent");
                }
                return;
            }
            Tensor update = producer.inputs().get(1);
            if (!update.descriptor().shape().equals(selectedShape)
                    || !output.descriptor().shape().equals(base.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "SLICE_UPDATE base, update, or output Shape is inconsistent");
            }
            if (selectedInputs[1]) {
                for (int axis : attrs.axes()) {
                    if (!(base.descriptor().shape().dimension(axis)
                            instanceof StaticDimension)) {
                        throw unsupported(
                                producerIndex, producer, outputIndex, 1,
                                "SLICE_UPDATE update role requires static selected base Dimensions");
                    }
                }
            }
            return;
        }
        if (operation.kind() == SelectKind.SELECT) {
            if (!(operation.attrs() instanceof SelectAttrs attrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported SELECT attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            Shape inputShape = producer.inputs().getFirst().descriptor().shape();
            if (attrs.axis() >= inputShape.rank()
                    || !output.descriptor().shape().equals(removeAxis(inputShape, attrs.axis()))
                    || (inputShape.dimension(attrs.axis()) instanceof StaticDimension selected
                            && attrs.index() >= selected.size())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "SELECT normalized index or output Shape is inconsistent");
            }
            return;
        }
        if (operation.kind() == PadKind.PAD) {
            if (!(operation.attrs() instanceof PadAttrs attrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported PAD attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            Tensor input = producer.inputs().getFirst();
            if (attrs.before().size() != input.descriptor().shape().rank()
                    || attrs.constantValue().dataType() != input.descriptor().dataType()
                    || !output.descriptor().shape().equals(
                            padShape(input.descriptor().shape(), attrs))) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "PAD attributes or output Shape are inconsistent");
            }
            return;
        }
        if (operation.kind() == TileKind.TILE) {
            if (!(operation.attrs() instanceof TileAttrs attrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported TILE attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            Tensor input = producer.inputs().getFirst();
            if (attrs.repeats().size() != input.descriptor().shape().rank()
                    || !output.descriptor().shape().equals(
                            tileShape(input.descriptor().shape(), attrs))) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "TILE repeats or output Shape are inconsistent");
            }
            return;
        }
        if (operation.kind() instanceof TensorCompositionKind kind) {
            if (!(operation.attrs() instanceof CompositionAxisAttrs attrs)
                    || producer.inputs().isEmpty()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported composition attributes or empty inputs");
            }
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            Shape expected;
            try {
                expected = compositionShape(kind, attrs.axis(), producer.inputs());
            } catch (IllegalArgumentException | ArithmeticException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "composition input Shapes or normalized axis are inconsistent");
            }
            if (!expected.equals(output.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "composition output Shape is inconsistent");
            }
            return;
        }
        throw unsupported(
                producerIndex, producer, outputIndex, firstSelected(selectedInputs),
                "operation is outside SUPPORTED_0004 and SUPPORTED_0004A");
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

    /**
     * Requires one selected input and its producer output to share one exact floating type.
     *
     * @param producerIndex deterministic producer postorder position used in diagnostics
     * @param producer non-null original producer occurrence
     * @param outputIndex valid selected producer-output position
     * @param inputIndex valid selected producer-input position
     * @throws IllegalArgumentException if the output is not floating or the two types differ
     */
    private static void requireSameFloatingType(
            int producerIndex, TensorProducer producer, int outputIndex, int inputIndex) {
        DataType outputType = producer.output(outputIndex).descriptor().dataType();
        if (!outputType.isFloating()
                || producer.inputs().get(inputIndex).descriptor().dataType() != outputType) {
            throw unsupported(
                    producerIndex, producer, outputIndex, inputIndex,
                    "selected input/output floating types must match exactly");
        }
    }

    /**
     * Tests whether one non-null Dimension is the resolved singleton extent.
     *
     * @param dimension non-null Dimension to inspect without binding or mutation
     * @return {@code true} only for a static extent of exactly one
     */
    private static boolean isStaticOne(Dimension dimension) {
        return dimension instanceof StaticDimension staticDimension
                && staticDimension.size() == 1;
    }

    /**
     * Returns a Shape with one existing axis removed and all other Dimension references retained.
     *
     * @param shape non-null source Shape
     * @param removedAxis zero-based axis in {@code shape}
     * @return a new Shape without {@code removedAxis}, including the scalar Shape when rank was one
     * @throws IndexOutOfBoundsException if {@code removedAxis} is outside the source rank
     */
    private static Shape removeAxis(Shape shape, int removedAxis) {
        List<Dimension> dimensions = new ArrayList<>(shape.dimensions());
        dimensions.remove(removedAxis);
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Re-derives the current vector/matrix MATMUL result Shape for preflight comparison.
     *
     * @param left non-null left operand Shape with positive rank
     * @param right non-null right operand Shape with positive rank
     * @return a new Shape retaining the selected exact operand Dimension references
     * @throws IllegalArgumentException if a rank, static contraction, or batch pair is invalid
     */
    private static Shape matmulShape(Shape left, Shape right) {
        int leftRank = left.rank();
        int rightRank = right.rank();
        if (leftRank < 1 || rightRank < 1) {
            throw new IllegalArgumentException("MATMUL rank must be positive");
        }
        Dimension leftInner = left.dimension(leftRank - 1);
        Dimension rightInner = right.dimension(rightRank == 1 ? 0 : rightRank - 2);
        if (leftInner instanceof StaticDimension leftStatic
                && rightInner instanceof StaticDimension rightStatic
                && leftStatic.size() != rightStatic.size()) {
            throw new IllegalArgumentException("MATMUL contraction mismatch");
        }
        int leftBatchRank = Math.max(0, leftRank - 2);
        int rightBatchRank = Math.max(0, rightRank - 2);
        int resultBatchRank = Math.max(leftBatchRank, rightBatchRank);
        int leftOffset = resultBatchRank - leftBatchRank;
        int rightOffset = resultBatchRank - rightBatchRank;
        List<Dimension> dimensions = new ArrayList<>();
        for (int axis = 0; axis < resultBatchRank; axis++) {
            Dimension leftDimension =
                    axis < leftOffset ? null : left.dimension(axis - leftOffset);
            Dimension rightDimension =
                    axis < rightOffset ? null : right.dimension(axis - rightOffset);
            dimensions.add(matmulBatchDimension(leftDimension, rightDimension));
        }
        if (leftRank != 1) {
            dimensions.add(left.dimension(leftRank - 2));
        }
        if (rightRank != 1) {
            dimensions.add(right.dimension(rightRank - 1));
        }
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Selects one current right-aligned MATMUL batch Dimension.
     *
     * @param left nullable left batch Dimension; {@code null} means an absent leading axis
     * @param right nullable right batch Dimension; {@code null} means an absent leading axis
     * @return the non-null exact Dimension retained by current MATMUL batch broadcasting
     * @throws IllegalArgumentException if two present Dimensions cannot broadcast under the
     *     current local rules
     */
    private static Dimension matmulBatchDimension(Dimension left, Dimension right) {
        if (left == null) {
            return right;
        }
        if (right == null || left.equals(right)) {
            return left;
        }
        if (isStaticOne(left)) {
            return right;
        }
        if (isStaticOne(right)) {
            return left;
        }
        if (left instanceof StaticDimension && !(right instanceof StaticDimension)) {
            return left;
        }
        if (right instanceof StaticDimension && !(left instanceof StaticDimension)) {
            return right;
        }
        throw new IllegalArgumentException("MATMUL batch mismatch");
    }

    /**
     * Re-derives the selected Shape of normalized finite slice metadata.
     *
     * @param base non-null base Shape whose unselected Dimension references are retained
     * @param attrs non-null normalized slice attributes
     * @return a new same-rank Shape with selected extents replaced by recorded static lengths
     * @throws IllegalArgumentException if an axis or statically provable coordinate is invalid
     * @throws ArithmeticException if a recorded final coordinate overflows
     */
    private static Shape sliceShape(Shape base, SliceAttrs attrs) {
        List<Dimension> dimensions = new ArrayList<>(base.dimensions());
        for (int index = 0; index < attrs.axes().size(); index++) {
            int axis = attrs.axes().get(index);
            if (axis < 0 || axis >= base.rank()) {
                throw new IllegalArgumentException("slice axis out of range");
            }
            long length = attrs.lengths().get(index);
            if (length > 0) {
                long last = Math.addExact(
                        attrs.starts().get(index),
                        Math.multiplyExact(length - 1, attrs.steps().get(index)));
                if (last < 0
                        || (base.dimension(axis) instanceof StaticDimension extent
                                && (attrs.starts().get(index) >= extent.size()
                                        || last >= extent.size()))) {
                    throw new IllegalArgumentException("slice coordinate out of range");
                }
            }
            dimensions.set(axis, new StaticDimension(length));
        }
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Re-derives a constant-pad output Shape through checked symbolic extent addition.
     *
     * @param input non-null input Shape
     * @param attrs non-null rank-aligned normalized pad attributes
     * @return a new Shape containing the exact checked padded extent expressions
     * @throws ArithmeticException if a combined static width overflows
     */
    private static Shape padShape(Shape input, PadAttrs attrs) {
        Dimension[] dimensions = new Dimension[input.rank()];
        for (int axis = 0; axis < input.rank(); axis++) {
            dimensions[axis] = DimensionExpressions.addConstant(
                    input.dimension(axis),
                    Math.addExact(attrs.before().get(axis), attrs.after().get(axis)));
        }
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Re-derives a tile output Shape through canonical symbolic extent multiplication.
     *
     * @param input non-null input Shape
     * @param attrs non-null rank-aligned positive repeat attributes
     * @return a new Shape containing the exact repeated extent expressions
     * @throws ArithmeticException if a static repeated extent overflows
     */
    private static Shape tileShape(Shape input, TileAttrs attrs) {
        Dimension[] dimensions = new Dimension[input.rank()];
        for (int axis = 0; axis < input.rank(); axis++) {
            dimensions[axis] =
                    DimensionExpressions.multiply(input.dimension(axis), attrs.repeats().get(axis));
        }
        return Shape.ofDimensions(dimensions);
    }

    /**
     * Re-derives the output Shape of one normalized CONCAT or STACK occurrence.
     *
     * @param kind non-null composition kind
     * @param axis normalized existing CONCAT axis or inserted STACK axis
     * @param inputs non-null, non-empty ordered input list; observed but not mutated
     * @return a new exact composition Shape
     * @throws IllegalArgumentException if the axis, ranks, or required input Dimensions disagree
     * @throws ArithmeticException if a static concatenated extent overflows
     */
    private static Shape compositionShape(
            TensorCompositionKind kind, int axis, List<Tensor> inputs) {
        Shape first = inputs.getFirst().descriptor().shape();
        if (kind == TensorCompositionKind.STACK) {
            if (axis < 0 || axis > first.rank()) {
                throw new IllegalArgumentException("STACK axis out of range");
            }
            List<Dimension> dimensions = new ArrayList<>(first.dimensions());
            for (Tensor input : inputs) {
                if (!input.descriptor().shape().equals(first)) {
                    throw new IllegalArgumentException("STACK Shape mismatch");
                }
            }
            dimensions.add(axis, new StaticDimension(inputs.size()));
            return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        }
        if (axis < 0 || axis >= first.rank()) {
            throw new IllegalArgumentException("CONCAT axis out of range");
        }
        Dimension extent = new StaticDimension(0);
        for (Tensor input : inputs) {
            Shape shape = input.descriptor().shape();
            if (shape.rank() != first.rank()) {
                throw new IllegalArgumentException("CONCAT rank mismatch");
            }
            for (int candidate = 0; candidate < first.rank(); candidate++) {
                if (candidate != axis
                        && !shape.dimension(candidate).equals(first.dimension(candidate))) {
                    throw new IllegalArgumentException("CONCAT Shape mismatch");
                }
            }
            extent = DimensionExpressions.add(extent, shape.dimension(axis));
        }
        List<Dimension> dimensions = new ArrayList<>(first.dimensions());
        dimensions.set(axis, extent);
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
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
