package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
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
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
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
import java.util.BitSet;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Preflights the source-closed first-order rule matrix before constructing any derivative Tensor
 * expression.
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
 * require a cotangent conversion. Compiler 0005C removes the former static-selected-base
 * restriction by validating length-defined extraction and target-relative placement through the
 * exact occurrence constraints retained by layout inference.</p>
 *
 * <p>The additive {@code SUPPORTED_0004B} rows admit mixed-floating {@code ADD}/{@code SUB}/
 * {@code MUL}/{@code DIV}, branch-only {@code WHERE}, floating {@code CAST}, and role-aware
 * {@code MATMUL} when each selected contribution can reverse broadcasting and convert to its
 * selected input type through ordinary Tensor operations. They also admit exact-type scalar
 * {@code DIV}, direct-zero {@code FLOOR}/{@code CEIL}/{@code SIGN}, and ordinary or masked
 * floating {@code MEAN}. Mask roles remain non-differentiable. The checks select local
 * differentiation rules and normalization paths; they do not create a gradient-specific
 * arithmetic, cast, comparison, exceptional-value, validation, or optimization contract.</p>
 *
 * <p>Compiler 0005A completes the exact current 48-kind elementwise and activation inventory:
 * seven binary arithmetic kinds, eight scalar elementwise kinds, nineteen unary kinds, one
 * {@code WHERE}, one {@code CAST}, six comparisons, three Boolean logical kinds, and three
 * floating-classification kinds. Floating binary {@code MIN}/{@code MAX}/{@code POW}, scalar
 * {@code MIN}/{@code MAX}/{@code POW}/{@code CLAMP}, and the remaining unary and activation
 * formulas are accepted only with their exact current signatures and same-type or promoted
 * floating descriptors. Comparisons, Boolean logic, classifications, the {@code WHERE}
 * condition, scalar attributes and bounds, and non-floating cast roles are non-differentiable.
 * The selected extrema-tie, clamp-endpoint, discontinuity, NaN, infinity, and raw-domain
 * conventions belong to the compiler rules; preflight selects those fixed rows without
 * inspecting represented Tensor values.</p>
 *
 * <p>Compiler 0005B adds floating products, reduction extrema, cumulative product,
 * softmax/log-softmax, statistics, norms, and Layer/RMS/batch-normalization routes. It also admits
 * binding-dependent {@code EXPAND} and {@code SUM_TO_SHAPE} only when the inverse uses the same
 * occurrence-local source-one-or-source-equal predicate retained by forward inference. Batch
 * normalization is output-slot-aware: public result slots zero through two select their exact
 * input roles in ascending slot order, while saved mean and saved inverse-standard-deviation
 * slots three and four remain same-occurrence auxiliaries and cannot seed an independent
 * cotangent route.</p>
 *
 * <p>Compiler 0005C completes the assigned floating layout/window, Gather/scatter, ordering, and
 * explicit-state dropout rows. It distinguishes both {@code SliceAttrs} and
 * {@code CropToShapeAttrs} slice variants, proves or retains their occurrence-local bounds, and
 * applies the same rule to unresolved two-dimensional window domains. Gather and functional
 * scatter validate exact index geometry and the fixed replacement, addition, multiplication, and
 * extrema policies while leaving index roles non-differentiable. {@code SORT} proves that one
 * matching stable {@code ARGSORT} occurrence can be constructed; {@code TOP_K} and
 * {@code DROPOUT} require the exact canonical indices or mask wrapper from the original producer.
 * Ordering indices, one-hot results, dropout masks, and graph random-number-generator state remain
 * non-differentiable.</p>
 *
 * <p>Compiler 0005D adds the remaining representable structured-neural rows: both public outputs
 * of exact two-output scaled-dot-product attention, grouped NCHW convolution, fixed-count average
 * pooling, exact first-winner maximum pooling, mean-squared error, and dense-target categorical
 * cross-entropy. Index-target categorical cross-entropy selects only logits and requires a
 * positive static class depth. Attention requires the canonical same-occurrence weights output;
 * the one-output overload therefore fails closed. Attention masks and index targets remain
 * non-differentiable configuration roles.</p>
 *
 * <p>{@link FirstOrderGradientCoverage} supplies the current source-backed disposition and one
 * formula-family owner for each selected output/input role. This class retains the larger typed
 * occurrence-validation matrix: a conditional differentiable disposition becomes usable only
 * after the exact Shape, data-type, cardinality, canonical-auxiliary, normalization, and
 * construction prerequisites pass. Unknown signatures and unsupported roles fail closed. The
 * recorded family owner is carried in each {@link SelectedOccurrence}, so preflight selection
 * and formula dispatch cannot choose different families.</p>
 *
 * <p>This owner selects rules and rejects unsupported operation, input/output signature,
 * attribute, role, data-type, Shape, and policy combinations. A known rejection occurs before the
 * seed, a derivative constant, a matching {@code ARGSORT}, or another formula Tensor is
 * constructed, so it consumes no derivative {@code TensorId}. The guarantee ends after a
 * successful plan is returned: later public Tensor construction, capture, inference, validation,
 * or optimization may consume IDs before failing. This owner neither reads Tensor payloads,
 * captures a graph, allocates storage, binds a dynamic Dimension, lowers work, nor executes
 * computation.</p>
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
     * selected output-slot occurrence, including its exact input/output signature and inferred
     * descriptors, against the closed matrix. Selected slots of one multi-output producer are
     * retained in ascending numeric order. Saved batch-normalization slots are rejected both as
     * independent targets and as selected cotangent roots. The first known failure is reported
     * before derivative Tensor identity allocation. Full captured-graph inference and validation
     * remain a later compiler stage. The returned plan retains exact original object references
     * and is owned by the current compile request; callers must hand it directly to reverse
     * accumulation and discard it after combined capture.</p>
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
     *     route, canonical wrapper, ingress membership, saved batch-normalization target, or any
     *     selected operation, input/output signature, attribute, role, data-type, Shape, or
     *     derivative-policy fact is unsupported
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
            if (!target.descriptor().dataType().isFloating()
                    || !target.descriptor().requiresGrad()) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index
                                + "] must be floating and require gradients");
            }
            TensorProvenance targetProvenance = target.provenance().orElse(null);
            if (targetProvenance != null
                    && targetProvenance.producer().operation().kind()
                            == BatchNormKind.BATCH_NORM_TRAINING
                    && targetProvenance.outputIndex() >= 3) {
                throw new IllegalArgumentException(
                        "firstOrderRequest.targets[" + index
                                + "] is a batch-normalization saved auxiliary output");
            }
        }

        IdentityHashMap<Tensor, BitSet> containsTargets = new IdentityHashMap<>();
        for (int targetIndex = 0; targetIndex < request.targets().size(); targetIndex++) {
            containsTargets
                    .computeIfAbsent(request.targets().get(targetIndex), ignored -> new BitSet())
                    .set(targetIndex);
        }
        String[] blockedRoutes = new String[request.targets().size()];
        List<SelectedOccurrence> selected = new ArrayList<>();
        for (int producerIndex = 0;
                producerIndex < ancestry.producerPostorder().size();
                producerIndex++) {
            TensorProducer producer = ancestry.producerPostorder().get(producerIndex);
            List<Integer> selectedOutputs = ancestry.selectedOutputs(producer);
            for (int selectedOutput : selectedOutputs) {
                boolean[] selectedInputs = new boolean[producer.inputs().size()];
                FirstOrderGradientCoverage.FamilyOwner familyOwner = null;
                BitSet outputTargets = copyOf(containsTargets.get(producer.output(selectedOutput)));
                boolean anySelectedInput = false;
                for (int input = 0; input < producer.inputs().size(); input++) {
                    BitSet inputTargets = containsTargets.get(producer.inputs().get(input));
                    if (inputTargets == null || inputTargets.isEmpty()) {
                        continue;
                    }
                    FirstOrderGradientCoverage.Decision decision =
                            FirstOrderGradientCoverage.classify(
                                    producer, selectedOutput, input);
                    if (decision.disposition()
                            == FirstOrderGradientCoverage.Disposition.D) {
                        if (familyOwner != null && familyOwner != decision.owner()) {
                            throw new IllegalStateException(
                                    "one occurrence selected more than one gradient family owner");
                        }
                        familyOwner = decision.owner();
                        selectedInputs[input] = true;
                        anySelectedInput = true;
                        outputTargets.or(inputTargets);
                    } else {
                        for (int targetIndex = inputTargets.nextSetBit(0);
                                targetIndex >= 0;
                                targetIndex = inputTargets.nextSetBit(targetIndex + 1)) {
                            if (blockedRoutes[targetIndex] == null) {
                                blockedRoutes[targetIndex] = occurrence(
                                                producerIndex,
                                                producer,
                                                selectedOutput,
                                                input)
                                        + decision.reason();
                            }
                        }
                    }
                }
                if (anySelectedInput) {
                    validateOccurrence(
                            producerIndex, producer, selectedOutput, selectedInputs);
                    selected.add(new SelectedOccurrence(
                            producerIndex,
                            producer,
                            selectedOutput,
                            selectedInputs,
                            familyOwner));
                }
                if (!outputTargets.isEmpty()) {
                    containsTargets.put(producer.output(selectedOutput), outputTargets);
                }
            }
        }
        BitSet objectiveTargets = containsTargets.get(objective);
        for (int index = 0; index < request.targets().size(); index++) {
            if (objectiveTargets == null || !objectiveTargets.get(index)) {
                String detail = blockedRoutes[index];
                throw new IllegalArgumentException(detail != null
                        ? detail
                        : "firstOrderRequest.targets[" + index
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

    private static BitSet copyOf(BitSet source) {
        return source == null ? new BitSet() : (BitSet) source.clone();
    }

    private static void validateOccurrence(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs) {
        Operation operation = producer.operation();
        if (operation.kind() == BatchNormKind.BATCH_NORM_TRAINING) {
            if (producer.outputCount() != 5) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "batch-normalization training requires five outputs");
            }
            if (outputIndex < 0 || outputIndex >= 5) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "selected output slot is outside the five-output occurrence");
            }
            if (outputIndex >= 3) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "saved auxiliary output cannot be an independent cotangent root");
            }
        } else if (operation.kind() instanceof TopKKind) {
            if (producer.outputCount() != 2 || outputIndex != 0) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "TOP_K requires values slot zero and canonical indices slot one");
            }
        } else if (operation.kind() instanceof DropoutKind) {
            if (producer.outputCount() != 3 || outputIndex != 0) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "DROPOUT requires values slot zero, mask slot one, and state slot two");
            }
        } else if (operation.kind()
                == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION) {
            if (producer.outputCount() != 2 || outputIndex < 0 || outputIndex > 1) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "attention gradients require output and canonical weights slots");
            }
        } else if (producer.outputCount() != 1 || outputIndex != 0) {
            throw unsupported(producerIndex, producer, outputIndex, -1, "requires one output");
        }
        Tensor output = producer.output(outputIndex);
        if (!output.descriptor().dataType().isFloating()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1, "output type must be floating");
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
                            && kind != BinaryArithmeticKind.MUL
                            && kind != BinaryArithmeticKind.DIV
                            && kind != BinaryArithmeticKind.MIN
                            && kind != BinaryArithmeticKind.MAX
                            && kind != BinaryArithmeticKind.POW)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported binary variant");
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
                        "binary operands and output must have the current floating promotion");
            }
            Shape expected;
            try {
                expected = ShapeBroadcast.broadcast(
                        left.descriptor().shape(), right.descriptor().shape());
            } catch (IllegalArgumentException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "binary input Shapes are not broadcast-compatible");
            }
            if (!expected.equals(output.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "binary output Shape is inconsistent");
            }
            return;
        }
        if (operation.kind() instanceof ScalarElementwiseKind kind) {
            ScalarValueAttrs scalarAttrs = operation.attrs() instanceof ScalarValueAttrs attrs
                    ? attrs
                    : null;
            ClampRangeAttrs clampAttrs = operation.attrs() instanceof ClampRangeAttrs attrs
                    ? attrs
                    : null;
            if ((kind == ScalarElementwiseKind.CLAMP && clampAttrs == null)
                    || (kind != ScalarElementwiseKind.CLAMP && scalarAttrs == null)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported scalar kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            if (kind == ScalarElementwiseKind.CLAMP) {
                if (clampAttrs.minValue().dataType() != output.descriptor().dataType()
                        || clampAttrs.maxValue().dataType() != output.descriptor().dataType()) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, -1,
                            "CLAMP bounds must match input/output floating type");
                }
                return;
            }
            if (scalarAttrs.value().dataType() != output.descriptor().dataType()) {
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
            Tensor ifTrue = producer.inputs().get(1);
            Tensor ifFalse = producer.inputs().get(2);
            if (!ifTrue.descriptor().dataType().isFloating()
                    || !ifFalse.descriptor().dataType().isFloating()
                    || DataTypePromotion.promoteFloating(
                                    ifTrue.descriptor().dataType(),
                                    ifFalse.descriptor().dataType())
                            != output.descriptor().dataType()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "WHERE branches and output must have the current floating promotion");
            }
            Shape expected;
            try {
                expected = ShapeBroadcast.broadcast(
                        ShapeBroadcast.broadcast(
                                producer.inputs().get(0).descriptor().shape(),
                                ifTrue.descriptor().shape()),
                        ifFalse.descriptor().shape());
            } catch (IllegalArgumentException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "WHERE input Shapes are not broadcast-compatible");
            }
            if (!expected.equals(output.descriptor().shape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "WHERE output Shape is inconsistent");
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
                    || !output.descriptor().dataType().isFloating()
                    || attrs.targetDataType() != output.descriptor().dataType()) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "CAST source and target must be floating and match the descriptor");
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
                    || (kind != UnaryElementwiseKind.ABS
                            && kind != UnaryElementwiseKind.NEG
                            && kind != UnaryElementwiseKind.RECIPROCAL
                            && kind != UnaryElementwiseKind.LOG
                            && kind != UnaryElementwiseKind.LOG1P
                            && kind != UnaryElementwiseKind.EXP
                            && kind != UnaryElementwiseKind.EXPM1
                            && kind != UnaryElementwiseKind.ERF
                            && kind != UnaryElementwiseKind.SQRT
                            && kind != UnaryElementwiseKind.RSQRT
                            && kind != UnaryElementwiseKind.FLOOR
                            && kind != UnaryElementwiseKind.CEIL
                            && kind != UnaryElementwiseKind.SIGN
                            && kind != UnaryElementwiseKind.RELU
                            && kind != UnaryElementwiseKind.SIGMOID
                            && kind != UnaryElementwiseKind.TANH
                            && kind != UnaryElementwiseKind.GELU
                            && kind != UnaryElementwiseKind.GELU_TANH_APPROXIMATION
                            && kind != UnaryElementwiseKind.SILU)) {
                throw unsupported(producerIndex, producer, outputIndex, -1, "unsupported unary variant");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            return;
        }
        if (operation.kind() instanceof AggregateReductionKind kind) {
            if (kind == AggregateReductionKind.ALL
                    || kind == AggregateReductionKind.ANY
                    || kind == AggregateReductionKind.ARG_MIN
                    || kind == AggregateReductionKind.ARG_MAX) {
                throw unsupported(
                        producerIndex,
                        producer,
                        outputIndex,
                        firstSelected(selectedInputs),
                        "aggregate roles are non-differentiable");
            }
            boolean mean = kind == AggregateReductionKind.MEAN;
            if (operation.attrs() instanceof MaskedReductionAttrs attrs) {
                if (kind != AggregateReductionKind.SUM
                        && kind != AggregateReductionKind.MEAN) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, -1,
                            "masked attributes require SUM or MEAN");
                }
                requireInputs(producerIndex, producer, outputIndex, 2);
                if (selectedInputs[1]) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked reduction mask role is non-differentiable");
                }
                Tensor data = producer.inputs().get(0);
                Tensor mask = producer.inputs().get(1);
                requireSameFloatingType(producerIndex, producer, outputIndex, 0);
                if (mask.descriptor().dataType() != DataType.BOOL) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked reduction mask must be BOOL");
                }
                Shape broadcast;
                try {
                    broadcast = ShapeBroadcast.broadcast(
                            mask.descriptor().shape(), data.descriptor().shape());
                } catch (IllegalArgumentException exception) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 1,
                            "masked reduction mask cannot broadcast to data Shape");
                }
                if (!broadcast.equals(data.descriptor().shape())
                        || attrs.axis() >= data.descriptor().shape().rank()
                        || !output.descriptor().shape().equals(
                                removeAxis(data.descriptor().shape(), attrs.axis()))) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 0,
                            "masked reduction descriptor or normalized axis is inconsistent");
                }
                return;
            }
            if (operation.attrs() instanceof SumToShapeAttrs attrs) {
                if (kind != AggregateReductionKind.SUM) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, -1,
                            "SUM_TO_SHAPE attributes require SUM");
                }
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
                    if (!target.equals(input)
                            && !isStaticOne(target)
                            && target instanceof StaticDimension
                            && input instanceof StaticDimension) {
                        throw unsupported(
                                producerIndex, producer, outputIndex, 0,
                                "SUM_TO_SHAPE aligned target contradicts source-one-or-equal proof");
                    }
                }
                validateReductionNormalizationOccurrence(
                        producerIndex, producer, outputIndex);
                return;
            }
            if (operation.attrs() != NoOperationAttrs.INSTANCE
                    && !(operation.attrs() instanceof AxisReductionAttrs)
                    && !(operation.attrs() instanceof MultiAxisReductionAttrs)
                    && !(operation.attrs()
                            instanceof io.github.pho001.synaptik.model.operation.reduction
                                    .StatisticalReductionAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported aggregate attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
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
        if (operation.kind()
                == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION) {
            if (!(operation.attrs() instanceof ScaledDotProductAttentionAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported attention attributes");
            }
            if (producer.inputs().size() != 3 && producer.inputs().size() != 4) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "attention requires query, key, value, and optional mask inputs");
            }
            if (producer.inputs().size() == 4) {
                requireNonDifferentiableRole(
                        producerIndex, producer, outputIndex, selectedInputs, 3, "mask");
            }
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == Conv2dKind.CONV2D) {
            if (!(operation.attrs() instanceof Conv2dAttrs attrs)
                    || (producer.inputs().size() != 2 && producer.inputs().size() != 3)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported CONV2D signature or attributes");
            }
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            try {
                Shape weight = producer.inputs().get(1).descriptor().shape();
                long kernelHeight = ((StaticDimension) weight.dimension(2)).size();
                long kernelWidth = ((StaticDimension) weight.dimension(3)).size();
                Math.multiplyExact(kernelHeight, kernelWidth);
                new Window2dAttrs(
                        kernelHeight, kernelWidth,
                        attrs.strideHeight(), attrs.strideWidth(),
                        attrs.paddingHeight(), attrs.paddingWidth(),
                        attrs.dilationHeight(), attrs.dilationWidth(), false);
            } catch (RuntimeException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "CONV2D grouped window geometry is not constructible");
            }
            return;
        }
        if (operation.kind() instanceof Pool2dKind kind) {
            boolean attrsMatch = kind == Pool2dKind.MAX_POOL2D
                    ? operation.attrs() instanceof MaxPool2dAttrs
                    : operation.attrs() instanceof AveragePool2dAttrs;
            if (!attrsMatch) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported pooling kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            try {
                long height = kind == Pool2dKind.MAX_POOL2D
                        ? ((MaxPool2dAttrs) operation.attrs()).kernelHeight()
                        : ((AveragePool2dAttrs) operation.attrs()).kernelHeight();
                long width = kind == Pool2dKind.MAX_POOL2D
                        ? ((MaxPool2dAttrs) operation.attrs()).kernelWidth()
                        : ((AveragePool2dAttrs) operation.attrs()).kernelWidth();
                Math.multiplyExact(height, width);
            } catch (ArithmeticException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "pool kernel element count overflows");
            }
            return;
        }
        if (operation.kind() instanceof LossKind kind) {
            boolean attrsMatch = switch (kind) {
                case MEAN_SQUARED_ERROR ->
                        operation.attrs() instanceof MeanSquaredErrorAttrs;
                case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                        operation.attrs()
                                instanceof DenseCategoricalCrossEntropyWithLogitsAttrs;
                case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ->
                        operation.attrs()
                                instanceof IndexCategoricalCrossEntropyWithLogitsAttrs;
            };
            if (!attrsMatch) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported loss kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS) {
                requireNonDifferentiableRole(
                        producerIndex, producer, outputIndex, selectedInputs, 1, "index target");
                int axis = ((IndexCategoricalCrossEntropyWithLogitsAttrs) operation.attrs()).axis();
                Dimension classes =
                        producer.inputs().get(0).descriptor().shape().dimension(axis);
                if (!(classes instanceof StaticDimension staticClasses)
                        || staticClasses.size() == 0) {
                    throw unsupported(
                            producerIndex, producer, outputIndex, 0,
                            "index loss requires a statically positive class extent");
                }
            }
            return;
        }
        if (operation.kind() instanceof CumulativeScanKind) {
            if (!(operation.attrs() instanceof CumulativeScanAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported cumulative-scan attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof SoftmaxKind) {
            if (!(operation.attrs() instanceof SoftmaxAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported softmax attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof LayerNormKind) {
            int expectedInputs;
            if (operation.attrs() instanceof LayerNormAttrs) {
                expectedInputs = 1;
            } else if (operation.attrs() instanceof AffineLayerNormAttrs) {
                expectedInputs = 3;
            } else {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported layer-normalization attrs");
            }
            requireInputs(producerIndex, producer, outputIndex, expectedInputs);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof RmsNormKind) {
            if (!(operation.attrs() instanceof RmsNormAttrs)
                    || (producer.inputs().size() != 1
                            && producer.inputs().size() != 2)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported RMS-normalization signature");
            }
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof BatchNormKind) {
            requireInputs(producerIndex, producer, outputIndex, 5);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
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
            if (!(operation.attrs() instanceof TargetShapeAttrs attrs)
                    || (kind != ShapeTransformKind.RESHAPE && kind != ShapeTransformKind.EXPAND)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1, "unsupported shape transform");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            Tensor input = producer.inputs().getFirst();
            if (!producer.output(0).descriptor().shape().equals(attrs.targetShape())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "shape-transform target and output Shape differ");
            }
            if (kind == ShapeTransformKind.EXPAND) {
                validateExpandInverse(
                        producerIndex,
                        producer,
                        outputIndex,
                        input.descriptor().shape(),
                        attrs.targetShape());
            }
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
            if (!(operation.attrs() instanceof SliceAttrs)
                    && !(operation.attrs() instanceof CropToShapeAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported slice attributes");
            }
            int expectedInputs = kind == SliceKind.SLICE ? 1 : 2;
            requireInputs(producerIndex, producer, outputIndex, expectedInputs);
            requireSameFloatingTypes(producerIndex, producer, outputIndex);
            validateLayoutOccurrence(producerIndex, producer, outputIndex);
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
        if (operation.kind() instanceof WindowTransformKind kind) {
            boolean supportedAttrs = switch (kind) {
                case UNFOLD_AXIS -> operation.attrs() instanceof UnfoldAxisAttrs;
                case FOLD_AXIS -> operation.attrs() instanceof FoldAxisAttrs;
                case UNFOLD2D -> operation.attrs() instanceof Window2dAttrs
                        || operation.attrs() instanceof Unfold2dAttrs;
                case FOLD2D -> operation.attrs() instanceof Fold2dAttrs;
            };
            if (!supportedAttrs) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported window kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateLayoutOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof AxisGatherKind kind) {
            if (!(operation.attrs() instanceof IndexAxisAttrs)
                    || (kind != AxisGatherKind.GATHER
                            && kind != AxisGatherKind.GATHER_ELEMENTS)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported Gather kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            requireNonDifferentiableRole(
                    producerIndex, producer, outputIndex, selectedInputs, 1, "indices");
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateIndexingOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == GatherNdKind.GATHER_ND) {
            if (!(operation.attrs() instanceof GatherNdAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported Gather-ND attributes");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            requireNonDifferentiableRole(
                    producerIndex, producer, outputIndex, selectedInputs, 1, "indices");
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateIndexingOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() instanceof AxisScatterKind kind) {
            boolean supportedAttrs = kind == AxisScatterKind.SCATTER_ADD
                    ? operation.attrs() instanceof IndexAxisAttrs
                    : kind == AxisScatterKind.SCATTER_ELEMENTS
                            && operation.attrs() instanceof ScatterElementsAttrs;
            if (!supportedAttrs) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported axis-scatter kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 3);
            requireNonDifferentiableRole(
                    producerIndex, producer, outputIndex, selectedInputs, 1, "indices");
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            requireSameFloatingType(producerIndex, producer, outputIndex, 2);
            validateIndexingOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == ScatterNdKind.SCATTER_ND) {
            if (!(operation.attrs() instanceof ScatterNdAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported Scatter-ND attributes");
            }
            requireInputs(producerIndex, producer, outputIndex, 3);
            requireNonDifferentiableRole(
                    producerIndex, producer, outputIndex, selectedInputs, 1, "indices");
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            requireSameFloatingType(producerIndex, producer, outputIndex, 2);
            validateIndexingOccurrence(producerIndex, producer, outputIndex);
            return;
        }
        if (operation.kind() == OrderingKind.SORT) {
            if (!(operation.attrs() instanceof SortAttrs attrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported SORT attributes");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingDescriptors(producerIndex, producer, outputIndex, true);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            validateMatchingArgsortConstructibility(
                    producerIndex, producer, outputIndex, attrs);
            return;
        }
        if (operation.kind() == TopKKind.TOP_K) {
            if (!(operation.attrs() instanceof TopKAttrs)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported TOP_K attributes");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateReductionNormalizationOccurrence(
                    producerIndex, producer, outputIndex);
            validateCanonicalAuxiliary(
                    producerIndex,
                    producer,
                    outputIndex,
                    1,
                    DataType.INT64,
                    producer.output(0).descriptor().shape(),
                    "TOP_K indices");
            return;
        }
        if (operation.kind() == DropoutKind.DROPOUT) {
            if (!(operation.attrs() instanceof DropoutAttrs attrs)
                    || !Double.isFinite(attrs.probability())
                    || attrs.probability() < 0.0d
                    || attrs.probability() >= 1.0d) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported DROPOUT probability attributes");
            }
            requireInputs(producerIndex, producer, outputIndex, 2);
            requireNonDifferentiableRole(
                    producerIndex, producer, outputIndex, selectedInputs, 1, "RNG state");
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            validateCanonicalAuxiliary(
                    producerIndex,
                    producer,
                    outputIndex,
                    1,
                    DataType.BOOL,
                    producer.output(0).descriptor().shape(),
                    "DROPOUT mask");
            validateCanonicalAuxiliary(
                    producerIndex,
                    producer,
                    outputIndex,
                    2,
                    DataType.INT64,
                    Shape.of(2),
                    "DROPOUT next state");
            return;
        }
        if (operation.kind() instanceof BinaryComparisonKind
                || operation.kind() instanceof BooleanLogicalKind
                || operation.kind() instanceof FloatingClassificationKind
                || operation.kind() instanceof OneHotKind
                || operation.kind() == OrderingKind.ARGSORT
                || operation.kind() instanceof GraphRngKind) {
            throw unsupported(
                    producerIndex, producer, outputIndex, firstSelected(selectedInputs),
                    "operation roles are non-differentiable");
        }
        throw unsupported(
                producerIndex, producer, outputIndex, firstSelected(selectedInputs),
                "operation is outside the supported first-order matrices");
    }

    private static void requireInputs(
            int producerIndex, TensorProducer producer, int outputIndex, int count) {
        if (producer.inputs().size() != count) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "expected " + count + " inputs, got " + producer.inputs().size());
        }
    }

    private static void requireNonDifferentiableRole(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs,
            int inputIndex,
            String role) {
        if (selectedInputs[inputIndex]
                || producer.inputs().get(inputIndex).descriptor().requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, inputIndex,
                    role + " role must be non-differentiable");
        }
    }

    private static void validateLayoutOccurrence(
            int producerIndex, TensorProducer producer, int outputIndex) {
        CapturedGraphInference.InferenceResult inferred;
        try {
            inferred = LayoutInference.infer(
                    producer.operation(),
                    producer.inputs().stream().map(Tensor::descriptor).toList());
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "layout inference rejected the occurrence");
        }
        validateInferredOccurrence(producerIndex, producer, outputIndex, inferred);
    }

    private static void validateIndexingOccurrence(
            int producerIndex, TensorProducer producer, int outputIndex) {
        CapturedGraphInference.InferenceResult inferred;
        try {
            inferred = IndexingInference.infer(
                    producer.operation(),
                    producer.inputs().stream().map(Tensor::descriptor).toList());
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "indexing inference rejected the occurrence");
        }
        validateInferredOccurrence(producerIndex, producer, outputIndex, inferred);
    }

    private static void validateStructuredOccurrence(
            int producerIndex, TensorProducer producer, int outputIndex) {
        CapturedGraphInference.InferenceResult inferred;
        try {
            inferred = StructuredOperationInference.infer(
                    producer.operation(),
                    producer.inputs().stream().map(Tensor::descriptor).toList(),
                    producer.outputCount());
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "structured inference rejected the occurrence");
        }
        validateInferredOccurrence(producerIndex, producer, outputIndex, inferred);
    }

    private static void validateInferredOccurrence(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            CapturedGraphInference.InferenceResult inferred) {
        if (inferred.outputs().size() != producer.outputCount()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "derived output count differs from the original occurrence");
        }
        for (int output = 0; output < producer.outputCount(); output++) {
            if (!inferred.outputs().get(output).equals(producer.output(output).descriptor())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "derived output descriptor differs at output[" + output + "]");
            }
        }
        for (CapturedGraphInference.ConstraintRequest constraint : inferred.constraints()) {
            if (GraphPredicateProof.evaluate(constraint.predicate()) == ProofStatus.DISPROVEN) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "occurrence-local constraint is contradicted: " + constraint.subject());
            }
        }
    }

    private static void validateMatchingArgsortConstructibility(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            SortAttrs attrs) {
        CapturedGraphInference.InferenceResult inferred;
        try {
            inferred = ReductionNormalizationInference.infer(
                    new Operation(OrderingKind.ARGSORT, attrs),
                    List.of(producer.inputs().getFirst().descriptor()));
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "matching stable ARGSORT is not constructible");
        }
        if (inferred.outputs().size() != 1) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "matching stable ARGSORT must have one output");
        }
        var descriptor = inferred.outputs().getFirst();
        if (descriptor.dataType() != DataType.INT64
                || !descriptor.shape().equals(producer.inputs().getFirst().descriptor().shape())
                || descriptor.requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "matching stable ARGSORT descriptor is inconsistent");
        }
    }

    private static void validateCanonicalAuxiliary(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            int auxiliaryIndex,
            DataType dataType,
            Shape shape,
            String role) {
        Tensor auxiliary = producer.output(auxiliaryIndex);
        TensorProvenance provenance = auxiliary.provenance().orElse(null);
        if (provenance == null
                || provenance.producer() != producer
                || provenance.outputIndex() != auxiliaryIndex
                || auxiliary.descriptor().dataType() != dataType
                || !auxiliary.descriptor().shape().equals(shape)
                || auxiliary.descriptor().requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    role + " canonical auxiliary is inconsistent");
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
     * Re-derives one reduction, scan, or normalization occurrence without constructing Tensors.
     *
     * @param producerIndex deterministic producer postorder position used in diagnostics
     * @param producer non-null original producer occurrence
     * @param outputIndex valid selected producer-output position
     * @throws IllegalArgumentException if inference rejects the occurrence, a derived output
     *     differs, or an occurrence-local predicate is statically contradicted
     */
    private static void validateReductionNormalizationOccurrence(
            int producerIndex, TensorProducer producer, int outputIndex) {
        CapturedGraphInference.InferenceResult inferred;
        try {
            inferred = ReductionNormalizationInference.infer(
                    producer.operation(),
                    producer.inputs().stream().map(Tensor::descriptor).toList());
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "reduction, scan, or normalization inference rejected the occurrence");
        }
        if (inferred.outputs().size() != producer.outputCount()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1,
                    "derived output count differs from the original occurrence");
        }
        for (int output = 0; output < producer.outputCount(); output++) {
            if (!inferred.outputs().get(output).equals(producer.output(output).descriptor())) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "derived output descriptor differs at output[" + output + "]");
            }
        }
        for (CapturedGraphInference.ConstraintRequest constraint : inferred.constraints()) {
            if (GraphPredicateProof.evaluate(constraint.predicate()) == ProofStatus.DISPROVEN) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "occurrence-local constraint is contradicted: " + constraint.subject());
            }
        }
    }

    /**
     * Proves that reversing one EXPAND with SUM_TO_SHAPE uses the exact forward predicate.
     *
     * @param producerIndex deterministic producer postorder position used in diagnostics
     * @param producer non-null original producer occurrence
     * @param outputIndex valid selected producer-output position
     * @param source non-null exact pre-expand Shape
     * @param target non-null exact expanded Shape
     * @throws IllegalArgumentException if the target rank is smaller or a static aligned pair
     *     contradicts source-one-or-equal
     */
    private static void validateExpandInverse(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            Shape source,
            Shape target) {
        if (target.rank() < source.rank()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, 0,
                    "EXPAND target rank is smaller than its source rank");
        }
        int offset = target.rank() - source.rank();
        for (int sourceAxis = 0; sourceAxis < source.rank(); sourceAxis++) {
            Dimension sourceDimension = source.dimension(sourceAxis);
            Dimension targetDimension = target.dimension(sourceAxis + offset);
            if (sourceDimension.equals(targetDimension) || isStaticOne(sourceDimension)) {
                continue;
            }
            if (sourceDimension instanceof StaticDimension
                    && targetDimension instanceof StaticDimension) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "EXPAND aligned dimensions contradict source-one-or-equal proof");
            }
            /*
             * Layout inference and SUM_TO_SHAPE use this same ordered predicate. A deferred
             * binding therefore proves the inverse by construction without resolving a symbol.
             */
            var forward = new AnyOf(List.of(
                    new DimensionEqual(sourceDimension, new StaticDimension(1)),
                    new DimensionEqual(sourceDimension, targetDimension)));
            var inverse = new AnyOf(List.of(
                    new DimensionEqual(sourceDimension, new StaticDimension(1)),
                    new DimensionEqual(sourceDimension, targetDimension)));
            if (!forward.equals(inverse)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, 0,
                        "EXPAND and SUM_TO_SHAPE inverse predicates differ");
            }
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
     * Derives the expected Shape for one already normalized ordinary MEAN occurrence.
     *
     * @param input non-null exact input Shape
     * @param axes non-null ordered axes; every axis must be normalized and distinct
     * @param keepDimensions whether each reduced axis remains as a static singleton
     * @return a non-null Shape retaining every unaffected exact Dimension reference
     * @throws IllegalArgumentException if an axis is out of range or repeated
     */
    private static Shape reductionShape(
            Shape input, List<Integer> axes, boolean keepDimensions) {
        boolean[] reduced = new boolean[input.rank()];
        for (int axis : axes) {
            if (axis < 0 || axis >= input.rank() || reduced[axis]) {
                throw new IllegalArgumentException("axes must be normalized and distinct");
            }
            reduced[axis] = true;
        }
        List<Dimension> dimensions = new ArrayList<>();
        for (int axis = 0; axis < input.rank(); axis++) {
            if (!reduced[axis]) {
                dimensions.add(input.dimension(axis));
            } else if (keepDimensions) {
                dimensions.add(new StaticDimension(1));
            }
        }
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
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
        IdentityHashMap<TensorProducer, BitSet> selectedOutputs = new IdentityHashMap<>();
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
            selectOutput(selectedOutputs, provenance);
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
                        selectOutput(selectedOutputs, inputProvenance);
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

    private static void selectOutput(
            IdentityHashMap<TensorProducer, BitSet> selectedOutputs,
            TensorProvenance provenance) {
        selectedOutputs
                .computeIfAbsent(
                        provenance.producer(),
                        ignored -> new BitSet(provenance.producer().outputCount()))
                .set(provenance.outputIndex());
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
     * @param familyOwner non-null closed formula-family owner shared by every selected input role
     */
    record SelectedOccurrence(
            int postorderIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs,
            FirstOrderGradientCoverage.FamilyOwner familyOwner) {
        /**
         * Snapshots one selected occurrence and its input-role flags.
         *
         * @param postorderIndex deterministic objective-ancestry producer postorder position
         * @param producer exact original producer occurrence
         * @param outputIndex zero-based selected canonical output position
         * @param selectedInputs non-null input-position-aligned differentiable-route flags
         * @param familyOwner non-null closed formula-family owner shared by every selected role
         * @throws NullPointerException if {@code selectedInputs} or {@code familyOwner} is
         *     {@code null}
         */SelectedOccurrence {
            Objects.requireNonNull(familyOwner, "familyOwner");
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
            IdentityHashMap<TensorProducer, BitSet> selectedOutputs) {
        List<Integer> selectedOutputs(TensorProducer producer) {
            BitSet selected = selectedOutputs.get(producer);
            if (selected == null) {
                return List.of();
            }
            List<Integer> ordered = new ArrayList<>(selected.cardinality());
            for (int output = selected.nextSetBit(0);
                    output >= 0;
                    output = selected.nextSetBit(output + 1)) {
                ordered.add(output);
            }
            return List.copyOf(ordered);
        }
    }

    private record TraversalFrame(TensorProducer producer, int nextInput) {}
}
