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
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
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
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
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
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
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
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import java.math.BigInteger;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Preflights one bounded functional reverse-mode stage before constructing derivative Tensor
 * expressions.
 *
 * <p>The iterative inventory covers every producer and canonical output wrapper reachable from the
 * requested forward boundary. Output ancestry, target reachability, occurrence selection, and
 * ingress membership use exact Tensor and producer object identity rather than identifiers, record
 * equality, labels, or storage. A successful plan is request-local compiler bookkeeping; it retains
 * deterministic producer postorder and the original-producer identity set needed by reverse
 * accumulation and phase-aware capture.
 *
 * <p>The additive {@code SUPPORTED_0004A} rows are exact-type {@code ERF}; masked {@code SUM};
 * locally invertible {@code SUM_TO_SHAPE}; role-aware floating {@code MATMUL}; normalized {@code
 * SLICE} and both {@code SLICE_UPDATE} data roles; {@code SELECT}, {@code PAD}, {@code TILE},
 * {@code CONCAT}, and {@code STACK}. Matrix-multiplication and slice replacement are validated per
 * selected input role: an unselected promoted operand does not require a cotangent conversion.
 * Compiler 0005C removes the former static-selected-base restriction by validating length-defined
 * extraction and target-relative placement through the exact occurrence constraints retained by
 * layout inference.
 *
 * <p>The additive {@code SUPPORTED_0004B} rows admit mixed-floating {@code ADD}/{@code SUB}/ {@code
 * MUL}/{@code DIV}, branch-only {@code WHERE}, floating {@code CAST}, and role-aware {@code MATMUL}
 * when each selected contribution can reverse broadcasting and convert to its selected input type
 * through ordinary Tensor operations. They also admit exact-type scalar {@code DIV}, direct-zero
 * {@code FLOOR}/{@code CEIL}/{@code SIGN}, and ordinary or masked floating {@code MEAN}. Mask roles
 * remain non-differentiable. The checks select local differentiation rules and normalization paths;
 * they do not create a gradient-specific arithmetic, cast, comparison, exceptional-value,
 * validation, or optimization contract.
 *
 * <p>The current inventory contains 48 elementwise and activation kinds: seven binary arithmetic
 * kinds, eight scalar elementwise kinds, nineteen unary kinds, one {@code WHERE}, one {@code
 * CAST}, six comparisons, three Boolean logical kinds, and three floating-classification kinds.
 * Floating binary {@code MIN}/{@code MAX}/{@code POW}, scalar {@code MIN}/{@code MAX}/{@code
 * POW}/{@code CLAMP}, and the remaining unary and activation formulas are accepted only with their
 * exact current signatures and same-type or promoted floating descriptors. Comparisons, Boolean
 * logic, classifications, the {@code WHERE} condition, scalar attributes and bounds, and
 * non-floating cast roles are non-differentiable. The selected extrema-tie, clamp-endpoint,
 * discontinuity, NaN, infinity, and raw-domain conventions belong to the compiler rules; preflight
 * selects those fixed rows without inspecting represented Tensor values.
 *
 * <p>Compiler 0005B adds floating products, reduction extrema, cumulative product,
 * softmax/log-softmax, statistics, norms, and Layer/RMS/batch-normalization routes. It also admits
 * binding-dependent {@code EXPAND} and {@code SUM_TO_SHAPE} only when the inverse uses the same
 * occurrence-local source-one-or-source-equal predicate retained by forward inference. Batch
 * normalization is output-slot-aware: public result slots zero through two select their exact input
 * roles in ascending slot order, while saved mean and saved inverse-standard-deviation slots three
 * and four remain same-occurrence auxiliaries and cannot seed an independent cotangent route.
 *
 * <p>Compiler 0005C completes the assigned floating layout/window, Gather/scatter, ordering, and
 * explicit-state dropout rows. It distinguishes both {@code SliceAttrs} and {@code
 * CropToShapeAttrs} slice variants, proves or retains their occurrence-local bounds, and applies
 * the same rule to unresolved two-dimensional window domains. Gather and functional scatter
 * validate exact index geometry and the fixed replacement, addition, multiplication, and extrema
 * policies while leaving index roles non-differentiable. {@code SORT} proves that one matching
 * stable {@code ARGSORT} occurrence can be constructed; {@code TOP_K} and {@code DROPOUT} require
 * the exact canonical indices or mask wrapper from the original producer. Ordering indices, one-hot
 * results, dropout masks, and graph random-number-generator state remain non-differentiable.
 *
 * <p>Compiler 0005D adds the remaining representable structured-neural rows: both public outputs of
 * exact two-output scaled-dot-product attention, grouped NCHW/NCDHW convolution, fixed-count average
 * pooling, exact first-winner maximum pooling, mean-squared error, and dense-target categorical
 * cross-entropy. Index-target categorical cross-entropy selects only logits and requires a positive
 * static class depth. Attention requires the canonical same-occurrence weights output; the
 * one-output overload therefore fails closed. Attention masks and index targets remain
 * non-differentiable configuration roles.
 *
 * <p>{@link FirstOrderGradientCoverage} supplies the current source-backed disposition and one
 * formula-family owner for each selected output/input role. This class retains the larger typed
 * occurrence-validation matrix: a conditional differentiable disposition becomes usable only after
 * the exact Shape, data-type, cardinality, canonical-auxiliary, normalization, and construction
 * prerequisites pass. Unknown signatures and unsupported roles fail closed. The recorded family
 * owner is carried in each {@link SelectedOccurrence}, so preflight selection and formula dispatch
 * cannot choose different families.
 * Required cotangent paths are traced from selected stage outputs through differentiable and
 * fail-closed roles, stopping at intentionally non-differentiable roles. A fail-closed role on a
 * required target path rejects the stage even when another sibling path connects that target,
 * regardless of disconnected policy. A saved statistic used only to compute a mask or index does
 * not require its own derivative; ordinary first-order use of saved values is unchanged.
 *
 * <p>This owner selects rules and rejects unsupported operation, input/output signature, attribute,
 * role, data-type, Shape, and policy combinations. A known rejection occurs before the seed, a
 * derivative constant, a matching {@code ARGSORT}, or another formula Tensor is constructed, so it
 * consumes no derivative {@code TensorId}. Fixed recurrent first-order routes validate the
 * ordinary fixed descriptors and dispatch joint canonical seeds through the shared length-aware
 * Scan recipe. Higher-order recurrent requests reject before derivative construction.
 * Convolution, pooling, and image-window transforms prove their matching public unfold/fold
 * geometry and structural kernel volume before allocation. The guarantee ends
 * after a successful plan is returned: later public Tensor construction, capture, inference,
 * validation, or optimization may consume IDs before failing. This owner neither reads Tensor
 * payloads, captures a graph, allocates storage, binds a dynamic Dimension, lowers work, nor
 * executes computation.
 *
 * <p>General scans retain whole-source facts and selected finite routes in {@link ScanGradientRules}.
 * Every original scan, including unrelated branches, is checked before derivative allocation.
 * Selected D/ND policy is aggregated per requested target at each legal prefix, across all
 * selected roots, tied actual positions and ordinary routes outside Scan. Syntactic ancestry
 * remains distinct from differentiable ancestry, so an unused computed actual cannot create
 * an ND-only diagnostic merely by belonging to the outer inventory.
 * Original length scans must have only ROW(0) carries/sequences, including unused and nongradient
 * declarations; this original-source gate is not applied to generated reverse mixed-role inputs.
 * Residual roots/paths/targets and scan higher-order requests reject. Body-local plans alone may
 * traverse false-flag carried values: their flags stay unchanged, and public output, target and
 * explicit-seed guards are not relaxed.</p>
 */
final class AutogradPreflight {
    private AutogradPreflight() {}

    /**
     * Performs allocation-free validation for the complete forward inventory and first stage.
     *
     * @param mode non-null backward-capable compile mode
     * @param forwardOutputs non-null validated ordered forward boundary
     * @param request non-null immutable one- or two-stage request
     * @param ingress non-null explicit logical-splat ingress
     * @return non-null initial request plan containing exact ephemeral Tensor identities
     * @throws NullPointerException if a required argument is {@code null}
     * @throws IllegalArgumentException if request membership, descriptors, routes, seeds,
     *     disconnected policy, ingress, or a selected differentiation fact is invalid
     */
    static InitialPlan preflight(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            FunctionalGradientRequest request,
            CompileTimeConstantGraph.Ingress ingress) {
        Objects.requireNonNull(mode, "mode");
        Objects.requireNonNull(forwardOutputs, "forwardOutputs");
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(ingress, "ingress");
        if (mode == CompileMode.FORWARD_ONLY) {
            throw new IllegalArgumentException(
                    "FORWARD_ONLY must not include a functional gradient request");
        }

        Inventory original = inventory(forwardOutputs);
        ScanBodyInference.validateScopes(forwardOutputs);
        var scanSources = preflightScanSources(original);
        for (TensorProducer producer : original.producerPostorder()) {
            if (producer.scanBody().isPresent() && request.stages().size() != 1) {
                throw new IllegalArgumentException("general scan supports first-order only; createGraph is unsupported");
            }
            if (producer.operation().kind() instanceof RecurrentScanKind && request.stages().size() != 1) {
                throw new IllegalArgumentException("fixed recurrent scan supports first-order only; createGraph is unsupported");
            }
        }
        FunctionalGradientRequest.Stage firstRequest = request.stages().getFirst();
        List<Tensor> firstOutputs = new ArrayList<>(firstRequest.outputs().size());
        IdentityHashMap<Tensor, Integer> outputPositions = new IdentityHashMap<>();
        for (int index = 0; index < firstRequest.outputs().size(); index++) {
            Tensor output = ((FunctionalGradientRequest.ForwardTensorReference)
                    firstRequest.outputs().get(index)).tensor();
            int forwardIndex = identityIndexOf(forwardOutputs, output);
            if (forwardIndex < 0) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[0].outputs[" + index
                                + "] must reference an exact forward output");
            }
            Integer first = outputPositions.putIfAbsent(output, index);
            if (first != null) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[0].outputs[" + index
                                + "] duplicates outputs[" + first + "]");
            }
            firstOutputs.add(output);
        }

        List<Tensor> ingressRoots = new ArrayList<>(forwardOutputs);
        for (FunctionalGradientRequest.Stage stage : request.stages()) {
            for (Optional<Tensor> seed : stage.cotangentSeeds()) {
                seed.ifPresent(ingressRoots::add);
            }
        }
        Inventory withSeeds = inventory(ingressRoots);
        for (int index = 0; index < ingress.bindings().size(); index++) {
            if (!withSeeds.tensors().containsKey(ingress.bindings().get(index).tensor())) {
                throw new IllegalArgumentException(
                        "forwardConstants.bindings[" + index
                                + "] is not reachable from forward outputs or explicit seeds");
            }
        }

        StagePlan first = preflightStage(
                0,
                firstOutputs,
                firstRequest.cotangentSeeds(),
                firstRequest.targets(),
                firstRequest.disconnectedPolicy(),
                original, false, scanSources);
        if (request.stages().size() == 2) {
            validateSecondStageClosure(request.stages().get(1), first);
        }
        return new InitialPlan(first, original.producers(), original.tensors());
    }

    /**
     * Preflights one compatibility stage through the bounded functional-request path.
     *
     * @param mode non-null backward-capable compile mode
     * @param forwardOutputs non-null validated ordered forward boundary
     * @param stage non-null one-stage request description
     * @param ingress non-null explicit logical-splat ingress
     * @return non-null allocation-free stage plan
     */
    static StagePlan preflight(
            CompileMode mode,
            List<Tensor> forwardOutputs,
            FunctionalGradientRequest.Stage stage,
            CompileTimeConstantGraph.Ingress ingress) {
        return preflight(
                        mode,
                        forwardOutputs,
                        new FunctionalGradientRequest(List.of(stage)),
                        ingress)
                .stageOne();
    }

    /**
     * Resolves and validates the second stage after successful first-stage construction.
     *
     * @param request non-null two-stage functional request
     * @param initial non-null successful first-stage preflight state
     * @param firstGradients non-null exact generated first-stage target-gradient roles
     * @return non-null allocation-free second-stage plan
     * @throws NullPointerException if a required argument is {@code null}
     * @throws IllegalArgumentException if references, seeds, targets, routes, or policy fail
     */
    static StagePlan preflightSecondStage(
            FunctionalGradientRequest request,
            InitialPlan initial,
            List<FirstOrderAutograd.TargetGradient> firstGradients) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(initial, "initial");
        Objects.requireNonNull(firstGradients, "firstGradients");
        if (request.stages().size() != 2) {
            throw new IllegalArgumentException("request must contain exactly two stages");
        }
        FunctionalGradientRequest.Stage stage = request.stages().get(1);
        List<Tensor> outputs = new ArrayList<>(stage.outputs().size());
        Set<Integer> referenced = new java.util.HashSet<>();
        for (int index = 0; index < stage.outputs().size(); index++) {
            int targetIndex = ((FunctionalGradientRequest.FirstStageGradientReference)
                    stage.outputs().get(index)).targetIndex();
            if (!referenced.add(targetIndex)) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[1].outputs[" + index
                                + "] repeats first-stage target index " + targetIndex);
            }
            Tensor output = firstGradients.get(targetIndex).gradient();
            Tensor target = initial.stageOne().targets().get(targetIndex);
            if (!output.descriptor().shape().equals(target.descriptor().shape())
                    || output.descriptor().dataType() != target.descriptor().dataType()) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[1].outputs[" + index
                                + "] gradient descriptor does not match the first-stage target");
            }
            outputs.add(output);
        }
        return preflightStage(
                1,
                outputs,
                stage.cotangentSeeds(),
                stage.targets(),
                stage.disconnectedPolicy(),
                new Inventory(
                        initial.originalTensors(),
                        initial.originalTensors(),
                        List.of(),
                        initial.originalProducers(),
                        List.of(),
                        new IdentityHashMap<>()));
    }

    private static StagePlan preflightStage(
            int stageIndex,
            List<Tensor> outputs,
            List<Optional<Tensor>> seeds,
            List<Tensor> targets,
            FunctionalGradientRequest.DisconnectedPolicy policy,
            Inventory original) {
        return preflightStage(stageIndex, outputs, seeds, targets, policy, original, false);
    }

    /**
     * Selects only exact proved local root/formal pairs, retaining ordinary policy per pair.
     * Unioning endpoint lists first would accidentally qualify their Cartesian product.
     * @param body independently validated lexical template
     * @param pairs root/formal declaration pairs used by selected finite outer routes
     * @param rootPorts compact root declaration positions
     * @param targetPositions compact formal declaration positions
     * @return one ordinary stage with the union of the exact pair-selected positional edges
     */
    static StagePlan preflightBody(io.github.pho001.synaptik.model.tensor.ScanBody body,
            List<ScanGradientRules.Pair> pairs, List<Integer> rootPorts, List<Integer> targetPositions) {
        var allRoots = new ArrayList<>(body.nextCarry());
        allRoots.addAll(body.emissions());
        List<Tensor> formals = ScanBodyInference.formals(body);
        List<Tensor> roots = rootPorts.stream().map(allRoots::get).toList();
        List<Tensor> targets = targetPositions.stream().map(formals::get).toList();
        Inventory ancestry = inventory(roots);
        var selected = new ArrayList<SelectedOccurrence>();
        for (ScanGradientRules.Pair pair : pairs) {
            List<Tensor> oneRoot = List.of(allRoots.get(pair.seedOutputPort()));
            StagePlan one = preflightStage(0, oneRoot, List.of(Optional.empty()),
                    List.of(formals.get(pair.actualInputPosition())),
                    FunctionalGradientRequest.DisconnectedPolicy.ERROR, inventory(oneRoot), true);
            for (SelectedOccurrence occurrence : one.selectedOccurrences()) {
                int existing = -1;
                for (int i = 0; i < selected.size(); i++) {
                    SelectedOccurrence candidate = selected.get(i);
                    if (candidate.producer() == occurrence.producer()
                            && candidate.outputIndex() == occurrence.outputIndex()) { existing = i; break; }
                }
                boolean[] flags = occurrence.selectedInputs();
                if (existing >= 0) {
                    boolean[] previous = selected.get(existing).selectedInputs();
                    for (int i = 0; i < flags.length; i++) flags[i] |= previous[i];
                }
                int postorder = ancestry.producerPostorder().indexOf(occurrence.producer());
                var merged = new SelectedOccurrence(postorder, occurrence.producer(),
                        occurrence.outputIndex(), flags, occurrence.familyOwner());
                if (existing < 0) selected.add(merged); else selected.set(existing, merged);
            }
        }
        selected.sort(java.util.Comparator.comparingInt(SelectedOccurrence::postorderIndex)
                .thenComparingInt(SelectedOccurrence::outputIndex));
        BitSet connected = new BitSet();
        connected.set(0, targets.size());
        return new StagePlan(0, roots, Collections.nCopies(roots.size(), Optional.empty()), targets,
                ancestry.producerPostorder(), selected, connected,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO, ancestry.producers(), Map.of());
    }

    /**
     * Checks a compiler-owned recipe before binding its generated cotangents. Like a Scan body,
     * these roots are internal flow, not new public explicit-seed declarations. The enclosing
     * public request has already checked eligibility; actual seeds bind only during expansion.
     */
    static StagePlan preflightRecipe(List<Tensor> roots, List<Tensor> targets) {
        return preflightStage(0, roots, Collections.nCopies(roots.size(), Optional.empty()),
                targets, FunctionalGradientRequest.DisconnectedPolicy.ZERO, inventory(roots), true);
    }

    /**
     * Selects existing rules for a public stage or a closed internal body flow.
     * @param stageIndex zero-based derivative stage
     * @param outputs ordered cotangent roots, repeated roles permitted only internally
     * @param seeds aligned explicit public seeds or empty internal allocation-free placeholders
     * @param targets ordered original targets or all declared body formals
     * @param policy public disconnected policy; internal unused formals receive typed zeros
     * @param original allocation-free original inventory
     * @param internalBody whether source eligibility guards are bypassed only for temporal flow
     * @return successful allocation-free role/rule plan
     * @throws IllegalArgumentException if public guards or existing occurrence rules fail
     */
    private static StagePlan preflightStage(
            int stageIndex, List<Tensor> outputs, List<Optional<Tensor>> seeds,
            List<Tensor> targets, FunctionalGradientRequest.DisconnectedPolicy policy,
            Inventory original, boolean internalBody) {
        return preflightStage(stageIndex, outputs, seeds, targets, policy, original, internalBody, null);
    }

    /** Reuses the whole-original source facts in both selected-route passes. */
    private static StagePlan preflightStage(
            int stageIndex, List<Tensor> outputs, List<Optional<Tensor>> seeds,
            List<Tensor> targets, FunctionalGradientRequest.DisconnectedPolicy policy,
            Inventory original, boolean internalBody,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> originalScanSources) {
        String prefix = "functionalGradientRequest.stages[" + stageIndex + "]";
        for (int index = 0; index < outputs.size(); index++) {
            Tensor output = outputs.get(index);
            DataType outputType = output.descriptor().dataType();
            if (!outputType.isFloating()
                    || (!internalBody && stageIndex == 0 && !output.descriptor().requiresGrad())) {
                throw new IllegalArgumentException(
                        prefix + ".outputs[" + index
                                + (stageIndex == 0
                                        ? "] must be floating and require gradients"
                                        : "] must be floating"));
            }
            Optional<Tensor> seed = seeds.get(index);
            if (internalBody) continue; // Real internal seeds bind only after whole-body preflight.
            if (seed.isEmpty()) {
                if (!output.descriptor().requiresGrad()) {
                    throw new IllegalArgumentException(
                            prefix + ".cotangentSeeds[" + index
                                    + "] default seed requires a gradient-eligible output");
                }
                if (!output.descriptor().shape().equals(Shape.scalar())) {
                    throw new IllegalArgumentException(
                            prefix + ".cotangentSeeds[" + index
                                    + "] requires an explicit seed for a non-scalar output");
                }
                continue;
            }
            Tensor explicit = seed.orElseThrow();
            if (!explicit.descriptor().dataType().isFloating()) {
                throw new IllegalArgumentException(
                        prefix + ".cotangentSeeds[" + index + "] must be floating");
            }
            if (explicit.descriptor().dataType() != outputType) {
                throw new IllegalArgumentException(
                        prefix + ".cotangentSeeds[" + index
                                + "] data type must equal the selected output data type");
            }
            if (!explicit.descriptor().shape().equals(output.descriptor().shape())) {
                throw new IllegalArgumentException(
                        prefix + ".cotangentSeeds[" + index
                                + "] Shape must equal the selected output Shape");
            }
            if (explicit.descriptor().requiresGrad()) {
                throw new IllegalArgumentException(
                        prefix + ".cotangentSeeds[" + index
                                + "] must not require gradients");
            }
            inventory(List.of(explicit));
        }

        for (int index = 0; index < targets.size(); index++) {
            Tensor target = targets.get(index);
            if (!internalBody && !original.tensors().containsKey(target)) {
                throw new IllegalArgumentException(
                        prefix + ".targets[" + index
                                + "] is not in the complete original forward inventory");
            }
            if (!target.descriptor().dataType().isFloating()
                    || (!internalBody && !target.descriptor().requiresGrad())) {
                throw new IllegalArgumentException(
                        prefix + ".targets[" + index
                                + "] must be floating and require gradients");
            }
            TensorProvenance targetProvenance = target.provenance().orElse(null);
            if (targetProvenance != null && targetProvenance.producer().scanBody().isPresent()) {
                var attrs = (io.github.pho001.synaptik.model.operation.scan.ScanAttrs)
                        targetProvenance.producer().operation().attrs();
                if (targetProvenance.outputIndex() >= attrs.carryCount() + attrs.emissionCount()) {
                    throw new IllegalArgumentException(prefix + ".targets[" + index + "] is a scan residual port");
                }
            }
            if (targetProvenance != null
                    && targetProvenance.producer().operation().kind()
                            == BatchNormKind.BATCH_NORM_TRAINING
                    && targetProvenance.outputIndex() >= 3) {
                throw new IllegalArgumentException(
                        prefix + ".targets[" + index
                                + "] is a batch-normalization saved auxiliary output");
            }
        }

        Inventory ancestry = inventory(outputs);
        var scanSources = originalScanSources;
        if (scanSources == null) {
            scanSources = new IdentityHashMap<>();
            for (TensorProducer producer : ancestry.producerPostorder()) if (producer.scanBody().isPresent()) {
                scanSources.put(producer, ScanGradientRules.preflightSource(producer));
            }
        }
        Set<Tensor> requiredCotangents = requiredCotangents(ancestry, outputs, scanSources);
        for (TensorProducer producer : ancestry.producerPostorder()) {
            if (producer.scanBody().isEmpty()) continue;
            var attrs = (io.github.pho001.synaptik.model.operation.scan.ScanAttrs) producer.operation().attrs();
            for (int output : ancestry.selectedOutputs(producer)) {
                if (output >= attrs.carryCount() + attrs.emissionCount()
                        && requiredCotangents.contains(producer.output(output))) {
                    throw new IllegalArgumentException("scan residual ports cannot be differentiation roots or paths");
                }
            }
        }
        FinitePolicy finitePolicy = ancestry.producerPostorder().stream().anyMatch(p -> p.scanBody().isPresent())
                ? finitePolicy(ancestry, outputs, targets, scanSources) : null;
        IdentityHashMap<Tensor, BitSet> containsTargets = new IdentityHashMap<>();
        for (int targetIndex = 0; targetIndex < targets.size(); targetIndex++) {
            containsTargets
                    .computeIfAbsent(targets.get(targetIndex), ignored -> new BitSet())
                    .set(targetIndex);
        }
        String[] nonDifferentiableRoutes = new String[targets.size()];
        List<SelectedOccurrence> selected = new ArrayList<>();
        var scanPairs = new IdentityHashMap<TensorProducer, List<ScanGradientRules.Pair>>();
        for (int producerIndex = 0;
                producerIndex < ancestry.producerPostorder().size();
                producerIndex++) {
            TensorProducer producer = ancestry.producerPostorder().get(producerIndex);
            for (int selectedOutput : ancestry.selectedOutputs(producer)) {
                boolean[] selectedInputs = new boolean[producer.inputs().size()];
                FirstOrderGradientCoverage.FamilyOwner familyOwner = null;
                BitSet outputTargets = copyOf(containsTargets.get(producer.output(selectedOutput)));
                boolean anySelectedInput = false;
                for (int input = 0; input < producer.inputs().size(); input++) {
                    BitSet inputTargets = finitePolicy == null
                            ? containsTargets.get(producer.inputs().get(input))
                            : finitePolicy.support(producer, selectedOutput, input);
                    if (inputTargets == null || inputTargets.isEmpty()) {
                        continue;
                    }
                    ScanGradientRules.SourcePlan scanSource = scanSources.get(producer);
                    ScanGradientRules.PairQualification scanRoute = scanSource != null
                            && selectedOutput < scanSource.roots().size()
                            ? ScanGradientRules.qualify(scanSource, selectedOutput, input) : null;
                    if (scanRoute != null && !scanRoute.syntacticRoute()) continue;
                    FirstOrderGradientCoverage.Decision decision =
                            classifyRoute(producer, selectedOutput, input, scanSources);
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
                        if (scanRoute != null && requiredCotangents.contains(producer.output(selectedOutput))) {
                            scanPairs.computeIfAbsent(producer, ignored -> new ArrayList<>())
                                    .add(new ScanGradientRules.Pair(selectedOutput, input));
                        }
                    } else if (decision.disposition()
                            == FirstOrderGradientCoverage.Disposition.FC) {
                        if (requiredCotangents.contains(producer.output(selectedOutput))) {
                            throw new IllegalArgumentException(prefix + "." + occurrence(
                                    producerIndex, producer, selectedOutput, input) + decision.reason());
                        }
                    } else if (finitePolicy == null) {
                        for (int targetIndex = inputTargets.nextSetBit(0);
                                targetIndex >= 0;
                                targetIndex = inputTargets.nextSetBit(targetIndex + 1)) {
                            if (nonDifferentiableRoutes[targetIndex] == null) {
                                nonDifferentiableRoutes[targetIndex] = occurrence(
                                                producerIndex,
                                                producer,
                                                selectedOutput,
                                                input)
                                        + decision.reason();
                            }
                        }
                    }
                }
                if (anySelectedInput && requiredCotangents.contains(producer.output(selectedOutput))) {
                    try {
                        validateOccurrence(
                                producerIndex, producer, selectedOutput, selectedInputs, internalBody);
                    } catch (IllegalArgumentException failure) {
                        throw new IllegalArgumentException(
                                prefix + "." + failure.getMessage(), failure);
                    }
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

        BitSet connectedTargets = new BitSet(targets.size());
        for (Tensor output : outputs) {
            BitSet outputTargets = containsTargets.get(output);
            if (outputTargets != null) {
                connectedTargets.or(outputTargets);
            }
        }
        if (finitePolicy != null) connectedTargets = finitePolicy.connectedTargets;
        for (int index = 0; index < targets.size(); index++) {
            if (finitePolicy != null && finitePolicy.failClosedTargets.get(index)) {
                throw new IllegalArgumentException(prefix + ".targets[" + index
                        + "] has a required unsupported finite derivative route");
            }
            if (finitePolicy != null && finitePolicy.nonDifferentiableTargets.get(index)) {
                throw new IllegalArgumentException(prefix + ".targets[" + index
                        + "] has only non-differentiable routes at a permitted finite Scan prefix");
            }
            if (connectedTargets.get(index)) {
                continue;
            }
            if (finitePolicy == null && ancestry.routeTensors().containsKey(targets.get(index))
                    && nonDifferentiableRoutes[index] != null) {
                throw new IllegalArgumentException(prefix + "." + nonDifferentiableRoutes[index]);
            }
            if (policy == FunctionalGradientRequest.DisconnectedPolicy.ERROR) {
                throw new IllegalArgumentException(
                        prefix + ".targets[" + index
                                + "] is disconnected from the selected stage outputs");
            }
        }
        var scanPlans = new IdentityHashMap<TensorProducer, ScanGradientRules.BackwardPlan>();
        for (TensorProducer producer : ancestry.producerPostorder()) {
            List<ScanGradientRules.Pair> pairs = scanPairs.get(producer);
            if (pairs != null) scanPlans.put(producer,
                    ScanGradientRules.plan(scanSources.get(producer), new ScanGradientRules.Demand(pairs)));
        }
        return new StagePlan(
                stageIndex,
                outputs,
                seeds,
                targets,
                ancestry.producerPostorder(),
                selected,
                connectedTargets,
                policy,
                original.producers(), scanPlans,
                finitePolicy == null ? Map.of() : finitePolicy.retainedPairs);
    }

    /** Exact canonical-output/input declaration edge, also used for ordinary occurrences. */
    private record PortPair(int output, int input) {}

    /**
     * One immutable uniform cut plus the independently qualified materialized-boundary policy.
     * Target support is positional: retaining an edge for one target never restores another
     * target's impossible route.
     */
    private record FinitePolicy(BitSet connectedTargets, BitSet nonDifferentiableTargets,
            BitSet failClosedTargets, Map<TensorProducer, Map<PortPair, BitSet>> retainedPairs) {
        BitSet support(TensorProducer producer, int output, int input) {
            Map<PortPair, BitSet> pairs = retainedPairs.get(producer);
            return pairs == null ? null : pairs.get(new PortPair(output, input));
        }
    }

    /** A chosen canonical row, or an unproved/mixing coordinate reached from that row. */
    private sealed interface RowCoordinate permits RootRowCoordinate, MixedRowCoordinate {}
    private enum RootRowCoordinate implements RowCoordinate { ROW }

    /**
     * Opaque coordinate contexts are shared only for the exact same contributing boundary.
     * They do not assert that a gather/reduction/matrix contributor uses its consumer's row.
     */
    private record MixedRowCoordinate(RowCoordinate parent, Tensor actual, int actualAxis,
            Tensor origin, int originAxis) implements RowCoordinate {
        @Override public boolean equals(Object other) {
            return other instanceof MixedRowCoordinate row && parent.equals(row.parent)
                    && actual == row.actual && actualAxis == row.actualAxis
                    && origin == row.origin && originAxis == row.originAxis;
        }
        @Override public int hashCode() {
            int hash = 31 * parent.hashCode() + System.identityHashCode(actual);
            hash = 31 * hash + actualAxis;
            hash = 31 * hash + System.identityHashCode(origin);
            return 31 * hash + originAxis;
        }
    }

    /** Length identity alone is deliberately insufficient to identify a control variable. */
    private record ControlKey(Tensor lengths, RowCoordinate row) {
        @Override public boolean equals(Object other) {
            return other instanceof ControlKey key && lengths == key.lengths && row.equals(key.row);
        }
        @Override public int hashCode() {
            return 31 * System.identityHashCode(lengths) + row.hashCode();
        }
    }

    /**
     * A singleton (period zero), or a bounded arithmetic progression of legal exact prefixes.
     * Periods arise only from the existing finite relation's transient/period quotient.
     */
    private record PrefixDomain(long first, long last, long period) {
        PrefixDomain {
            if (period > 0) last = first + ((last - first) / period) * period;
            if (first == last) period = 0;
        }
        boolean contains(long value) {
            return value >= first && value <= last
                    && (period == 0 ? value == first : (value - first) % period == 0);
        }

        PrefixDomain intersect(PrefixDomain other) {
            if (period == 0) return other.contains(first) ? this : null;
            if (other.period == 0) return contains(other.first) ? other : null;
            long lower = Math.max(first, other.first), upper = Math.min(last, other.last);
            if (lower > upper) return null;
            if (period == other.period) {
                return (other.first - first) % period == 0 ? new PrefixDomain(lower, upper, period) : null;
            }
            // CRT is metadata arithmetic, not Tensor math. BigInteger is needed only when two
            // periodic domains meet: their exact least common period need not fit a long.
            BigInteger a = BigInteger.valueOf(period), b = BigInteger.valueOf(other.period);
            BigInteger gcd = a.gcd(b);
            BigInteger difference = BigInteger.valueOf(other.first).subtract(BigInteger.valueOf(first));
            if (difference.remainder(gcd).signum() != 0) return null;
            BigInteger reduced = b.divide(gcd);
            BigInteger multiplier = reduced.equals(BigInteger.ONE) ? BigInteger.ZERO
                    : difference.divide(gcd).multiply(a.divide(gcd).modInverse(reduced)).mod(reduced);
            BigInteger step = a.multiply(reduced);
            BigInteger start = BigInteger.valueOf(first).add(a.multiply(multiplier)).mod(step);
            BigInteger minimum = BigInteger.valueOf(lower);
            if (start.compareTo(minimum) < 0) {
                start = start.add(minimum.subtract(start).add(step).subtract(BigInteger.ONE)
                        .divide(step).multiply(step));
            }
            if (start.compareTo(BigInteger.valueOf(upper)) > 0) return null;
            long exact = start.longValueExact();
            return step.compareTo(BigInteger.valueOf(upper - exact)) > 0
                    ? new PrefixDomain(exact, exact, 0)
                    : new PrefixDomain(exact, upper, step.longValueExact());
        }
    }

    private record PrefixBinding(ControlKey key, ScanGradientRules.SourcePlan source,
            PrefixDomain domain, PrefixBinding previous) {}
    private record DemandEdge(TensorProducer producer, PortPair pair) {
        @Override public boolean equals(Object other) {
            return other instanceof DemandEdge edge && producer == edge.producer && pair.equals(edge.pair);
        }
        @Override public int hashCode() {
            return 31 * System.identityHashCode(producer) + pair.hashCode();
        }
    }
    private record ScanBoundary(ScanGradientRules.SourcePlan source, int input, RowCoordinate row) {
        @Override public boolean equals(Object other) {
            return other instanceof ScanBoundary boundary && source.producer() == boundary.source.producer()
                    && input == boundary.input && row.equals(boundary.row);
        }
        @Override public int hashCode() {
            return 31 * (31 * System.identityHashCode(source.producer()) + input) + row.hashCode();
        }
    }

    /** A quotient state; ordinary DAG diamonds share their suffix analysis. */
    private record RouteStateKey(ScanGradientRules.Route route, ScanBoundary boundary,
            Map<ControlKey, PrefixDomain> domains, BitSet visitedScans, boolean numerical) {}
    private record RouteLink(RouteState parent, DemandEdge edge) {}
    private static final class RouteState {
        final RouteStateKey key;
        final PrefixBinding bindings;
        final List<RouteLink> parents = new ArrayList<>();
        RouteState(RouteStateKey key, PrefixBinding bindings) {
            this.key = key;
            this.bindings = bindings;
        }
    }

    /**
     * A complete feasible path, not an independent union of its endpoint roots or inputs.
     * Its prefix predicates are compile-local proof facts; no predicate crosses a Tensor boundary.
     */
    private record RouteWitness(ScanGradientRules.Route route, PrefixBinding bindings,
            List<DemandEdge> edges, boolean numerical) {
        boolean matches(Map<ControlKey, Long> assignment) {
            for (PrefixBinding binding = bindings; binding != null; binding = binding.previous) {
                Long prefix = assignment.get(binding.key);
                if (prefix == null || !binding.domain.contains(prefix)) return false;
            }
            return true;
        }

        boolean traverses(TensorProducer producer) {
            for (DemandEdge edge : edges) {
                if (edge.producer == producer) return true;
            }
            return false;
        }
    }

    /** Geometry is independent of selected targets and finite control states. */
    private record RowProofQuery(Tensor origin, int originAxis, Tensor actual, int actualAxis, Dimension batch) {
        @Override public boolean equals(Object other) {
            return other instanceof RowProofQuery query && origin == query.origin && originAxis == query.originAxis
                    && actual == query.actual && actualAxis == query.actualAxis && batch.equals(query.batch);
        }
        @Override public int hashCode() {
            int hash = 31 * System.identityHashCode(origin) + originAxis;
            hash = 31 * hash + System.identityHashCode(actual);
            hash = 31 * hash + actualAxis;
            return 31 * hash + batch.hashCode();
        }
    }

    /**
     * Computes one source-feasible positional cut, then qualifies retained seeds as PRESENT over
     * the consuming local domain. Both passes use the ordinary S/D/FC owner; neither constructs
     * derivative Tensors. The ordinary no-Scan path does not create any of these facts.
     */
    private static FinitePolicy finitePolicy(Inventory ancestry, List<Tensor> outputs, List<Tensor> targets,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> sources) {
        IdentityHashMap<Tensor, BitSet> syntacticTargets = syntacticTargets(ancestry, targets, sources);
        var retained = new IdentityHashMap<TensorProducer, Map<PortPair, BitSet>>();
        BitSet connected = new BitSet(), nd = new BitSet(), fc = new BitSet();
        var witnesses = new ArrayList<List<RouteWitness>>();
        var rowProofs = new java.util.HashMap<RowProofQuery, Boolean>();
        for (int target = 0; target < targets.size(); target++) {
            List<RouteWitness> paths = routeWitnesses(ancestry, outputs, targets.get(target), target,
                    syntacticTargets, sources, rowProofs);
            witnesses.add(paths);
            for (RouteWitness path : paths) {
                if (path.route.failClosed()) fc.set(target);
                if (!path.route.differentiable()) continue;
                connected.set(target);
                for (DemandEdge edge : path.edges) {
                    retained.computeIfAbsent(edge.producer, ignored -> new LinkedHashMap<>())
                            .computeIfAbsent(edge.pair, ignored -> new BitSet()).set(target);
                }
            }
            if (ndOnly(paths, null)) nd.set(target);
        }

        // Once retained, an external cotangent Tensor is present at every legal local prefix.
        // A complete original-path witness is still required for a rescuing D bypass at the
        // SAME assignment; another branch's existential D or published zero is not a rescue.
        for (TensorProducer producer : ancestry.producerPostorder()) {
            ScanGradientRules.SourcePlan source = sources.get(producer);
            Map<PortPair, BitSet> pairs = retained.get(producer);
            if (source == null || pairs == null) continue;
            for (int target = 0; target < targets.size(); target++) {
                var selectedPairs = new ArrayList<PortPair>();
                for (var entry : pairs.entrySet()) {
                    if (!entry.getValue().get(target)) continue;
                    selectedPairs.add(entry.getKey());
                    if (ScanGradientRules.qualify(source, entry.getKey().output, entry.getKey().input)
                            .failClosed()) fc.set(target);
                }
                if (selectedPairs.isEmpty() || fc.get(target)) continue;
                var bypass = new ArrayList<RouteWitness>();
                for (RouteWitness path : witnesses.get(target)) {
                    if (path.route.differentiable() && !path.traverses(producer)) bypass.add(path);
                }
                // Restart retained roots at the materialized boundary. Numerical input requests
                // and all their suffixes stay inside the cut; genuine S/ND ancestry remains
                // available separately for target policy.
                var rootPorts = new BitSet();
                for (PortPair pair : selectedPairs) rootPorts.set(pair.output);
                var roots = new ArrayList<Tensor>();
                for (int port = rootPorts.nextSetBit(0); port >= 0; port = rootPorts.nextSetBit(port + 1)) {
                    roots.add(producer.output(port));
                }
                var local = new ArrayList<>(routeWitnesses(ancestry, roots, targets.get(target), target,
                        syntacticTargets, sources, rowProofs, retained));
                local.addAll(bypass);
                if (ndOnly(local, source)) nd.set(target);
            }
        }
        var frozen = new IdentityHashMap<TensorProducer, Map<PortPair, BitSet>>();
        retained.forEach((producer, pairs) -> {
            var copy = new LinkedHashMap<PortPair, BitSet>();
            pairs.forEach((pair, support) -> copy.put(pair, (BitSet) support.clone()));
            frozen.put(producer, Collections.unmodifiableMap(copy));
        });
        return new FinitePolicy(connected, nd, fc, Collections.unmodifiableMap(frozen));
    }

    /** Ordinary syntax is retained through ND roles, but unused Scan actuals are not ancestry. */
    private static IdentityHashMap<Tensor, BitSet> syntacticTargets(Inventory ancestry, List<Tensor> targets,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> sources) {
        var result = new IdentityHashMap<Tensor, BitSet>();
        for (int target = 0; target < targets.size(); target++) {
            result.computeIfAbsent(targets.get(target), ignored -> new BitSet()).set(target);
        }
        for (TensorProducer producer : ancestry.producerPostorder()) {
            ScanGradientRules.SourcePlan source = sources.get(producer);
            for (int output : ancestry.selectedOutputs(producer)) {
                BitSet support = copyOf(result.get(producer.output(output)));
                for (int input = 0; input < producer.inputs().size(); input++) {
                    if (source != null && !ScanGradientRules.hasPrimalRoute(source, output, input)) continue;
                    BitSet inputSupport = result.get(producer.inputs().get(input));
                    if (inputSupport != null) support.or(inputSupport);
                }
                if (!support.isEmpty()) result.put(producer.output(output), support);
            }
        }
        return result;
    }

    /**
     * Computes a finite path quotient in reverse producer postorder. Equal ordinary diamonds
     * share a state and retain a provenance DAG, rather than enumerating exponentially many
     * equivalent paths. Different Scan ancestry is not merged: an independent D bypass must
     * remain distinguishable from a path through the occurrence whose ND obligation it rescues.
     */
    private static List<RouteWitness> routeWitnesses(Inventory ancestry, List<Tensor> outputs,
            Tensor target, int targetIndex, IdentityHashMap<Tensor, BitSet> syntacticTargets,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> sources,
            Map<RowProofQuery, Boolean> rowProofs) {
        return routeWitnesses(ancestry, outputs, target, targetIndex, syntacticTargets, sources,
                rowProofs, null);
    }

    /**
     * A local restart may inspect original S/ND ancestry, but its numerical witnesses must use
     * the frozen target-specific cut throughout the suffix. Removing numerical eligibility
     * does not convert an excluded original D path into an intentional ND path.
     */
    private static List<RouteWitness> routeWitnesses(Inventory ancestry, List<Tensor> outputs,
            Tensor target, int targetIndex, IdentityHashMap<Tensor, BitSet> syntacticTargets,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> sources,
            Map<RowProofQuery, Boolean> rowProofs,
            Map<TensorProducer, Map<PortPair, BitSet>> cut) {
        var states = new IdentityHashMap<Tensor, Map<RouteStateKey, RouteState>>();
        for (Tensor output : outputs) {
            addRouteState(states, output, new RouteStateKey(ScanGradientRules.Route.identity(),
                    null, Map.of(), new BitSet(), true), null, null, null);
        }
        for (int producerIndex = ancestry.producerPostorder().size() - 1; producerIndex >= 0; producerIndex--) {
            TensorProducer producer = ancestry.producerPostorder().get(producerIndex);
            ScanGradientRules.SourcePlan source = sources.get(producer);
            for (int output : ancestry.selectedOutputs(producer)) {
                Tensor value = producer.output(output);
                if (value == target) continue;
                BitSet support = syntacticTargets.get(value);
                Map<RouteStateKey, RouteState> incoming = states.get(value);
                if (incoming == null || support == null || !support.get(targetIndex)) continue;
                for (RouteState state : incoming.values()) {
                    if (source == null) {
                        for (int input = 0; input < producer.inputs().size(); input++) {
                            BitSet inputSupport = syntacticTargets.get(producer.inputs().get(input));
                            if (inputSupport == null || !inputSupport.get(targetIndex)) continue;
                            ScanGradientRules.Route edge = ScanGradientRules.Route.ordinary(
                                    FirstOrderGradientCoverage.classify(producer, output, input));
                            PortPair pair = new PortPair(output, input);
                            addRouteState(states, producer.inputs().get(input),
                                    new RouteStateKey(state.key.route.then(edge), state.key.boundary,
                                            state.key.domains, state.key.visitedScans,
                                            state.key.numerical && retainsNumericalEdge(cut, producer, pair, targetIndex)),
                                    state.bindings, state, new DemandEdge(producer, pair));
                        }
                        continue;
                    }
                    boolean residual = output >= source.roots().size();
                    if (residual && (state.key.route.differentiable() || state.key.route.failClosed())) {
                        throw new IllegalArgumentException("scan residual ports cannot be differentiation roots or paths");
                    }
                    RowCoordinate row = rowCoordinate(source, output, state.key.boundary, rowProofs);
                    BitSet visited = (BitSet) state.key.visitedScans.clone();
                    visited.set(producerIndex);
                    List<ScanGradientRules.FinitePrefix> prefixes = source.finitePrefixes().states();
                    int first = source.attrs().lengthAware() ? 0
                            : source.finitePrefixes().indexAt(source.attrs().timeCount());
                    int end = source.attrs().lengthAware() ? prefixes.size() : first + 1;
                    for (int prefix = first; prefix < end; prefix++) {
                        PrefixBinding bindings = state.bindings;
                        Map<ControlKey, PrefixDomain> domains = state.key.domains;
                        if (source.attrs().lengthAware()) {
                            ControlKey key = new ControlKey(producer.inputs().getLast(), row);
                            PrefixDomain domain = prefixDomain(source, prefix);
                            PrefixDomain previous = domains.get(key);
                            if (previous != null) domain = domain.intersect(previous);
                            if (domain == null) continue;
                            var extended = new LinkedHashMap<>(domains);
                            extended.put(key, domain);
                            domains = Collections.unmodifiableMap(extended);
                            bindings = new PrefixBinding(key, source, domain, bindings);
                        }
                        for (int input = 0; input < producer.inputs().size(); input++) {
                            PortPair pair = new PortPair(output, input);
                            BitSet inputSupport = syntacticTargets.get(producer.inputs().get(input));
                            if (inputSupport == null || !inputSupport.get(targetIndex)) continue;
                            ScanGradientRules.Route edge = residual
                                    ? ScanGradientRules.primalRoute(source, prefixes.get(prefix), output, input)
                                    : ScanGradientRules.route(source, prefixes.get(prefix), output, input).withoutUses();
                            if (!edge.syntactic()) continue;
                            addRouteState(states, producer.inputs().get(input),
                                    new RouteStateKey(state.key.route.then(edge),
                                            new ScanBoundary(source, input, row), domains, visited,
                                            state.key.numerical && retainsNumericalEdge(cut, producer, pair, targetIndex)),
                                    bindings, state, new DemandEdge(producer, pair));
                        }
                    }
                }
            }
        }
        Map<RouteStateKey, RouteState> complete = states.get(target);
        if (complete == null) return List.of();
        var result = new ArrayList<RouteWitness>();
        for (RouteState state : complete.values()) {
            var edges = new java.util.LinkedHashSet<DemandEdge>();
            Set<RouteState> seen = Collections.newSetFromMap(new IdentityHashMap<>());
            var pending = new ArrayDeque<RouteState>();
            pending.add(state);
            while (!pending.isEmpty()) {
                RouteState next = pending.removeLast();
                if (!seen.add(next)) continue;
                for (RouteLink link : next.parents) {
                    edges.add(link.edge);
                    pending.add(link.parent);
                }
            }
            result.add(new RouteWitness(state.key.route, state.bindings, List.copyOf(edges), state.key.numerical));
        }
        return List.copyOf(result);
    }

    private static boolean retainsNumericalEdge(Map<TensorProducer, Map<PortPair, BitSet>> cut,
            TensorProducer producer, PortPair pair, int target) {
        if (cut == null) return true; // Whole-source feasibility has not selected a cut yet.
        Map<PortPair, BitSet> pairs = cut.get(producer);
        BitSet support = pairs == null ? null : pairs.get(pair);
        return support != null && support.get(target);
    }

    private static void addRouteState(IdentityHashMap<Tensor, Map<RouteStateKey, RouteState>> states,
            Tensor value, RouteStateKey key, PrefixBinding bindings, RouteState parent, DemandEdge edge) {
        RouteState state = states.computeIfAbsent(value, ignored -> new LinkedHashMap<>())
                .computeIfAbsent(key, ignored -> new RouteState(key, bindings));
        if (parent != null) state.parents.add(new RouteLink(parent, edge));
    }

    /** A correlation is a qualified contributor mapping, never equality of batch extents. */
    private static RowCoordinate rowCoordinate(ScanGradientRules.SourcePlan source, int output,
            ScanBoundary parent, Map<RowProofQuery, Boolean> rowProofs) {
        if (parent == null) return RootRowCoordinate.ROW;
        var fromAxis = ScanGradientRules.outerRowAxis(source, output);
        var toAxis = ScanGradientRules.actualRowAxis(parent.source, parent.input);
        Tensor origin = source.producer().output(output);
        Tensor actual = parent.source.producer().inputs().get(parent.input);
        if (fromAxis.isPresent() && toAxis.isPresent()) {
            Dimension batch = source.producer().inputs().getLast().descriptor().shape().dimension(0);
            RowProofQuery query = new RowProofQuery(origin, fromAxis.getAsInt(), actual, toAxis.getAsInt(), batch);
            if (rowProofs.computeIfAbsent(query, key -> ScanGradientRules.sameRow(
                    key.origin, key.originAxis, key.actual, key.actualAxis, key.batch))) {
                return parent.row;
            }
        }
        return new MixedRowCoordinate(parent.row, actual, toAxis.orElse(-1), origin, fromAxis.orElse(-1));
    }

    private static PrefixDomain prefixDomain(ScanGradientRules.SourcePlan source, int index) {
        int start = source.finitePrefixes().cycleStart();
        return start < 0 || index < start ? new PrefixDomain(index, index, 0)
                : new PrefixDomain(index, source.attrs().timeCount(),
                        source.finitePrefixes().states().size() - start);
    }

    private record ControlChoices(ControlKey key, List<Long> prefixes) {}

    private static void retainSource(List<ScanGradientRules.SourcePlan> sources,
            ScanGradientRules.SourcePlan candidate) {
        for (var source : sources) if (source.producer() == candidate.producer()) return;
        sources.add(candidate);
    }

    /**
     * Builds a finite joint partition for each coordinate-qualified control. An optional local
     * source supplies its OWN complete domain, including prefixes whose producer publishes zero.
     */
    private static List<ControlChoices> controlChoices(List<RouteWitness> paths,
            ScanGradientRules.SourcePlan local) {
        var sources = new LinkedHashMap<ControlKey, List<ScanGradientRules.SourcePlan>>();
        for (RouteWitness path : paths) {
            for (PrefixBinding binding = path.bindings; binding != null; binding = binding.previous) {
                List<ScanGradientRules.SourcePlan> group =
                        sources.computeIfAbsent(binding.key, ignored -> new ArrayList<>());
                retainSource(group, binding.source);
            }
        }
        ControlKey localKey = local != null && local.attrs().lengthAware()
                ? new ControlKey(local.producer().inputs().getLast(), RootRowCoordinate.ROW) : null;
        if (localKey != null) {
            List<ScanGradientRules.SourcePlan> group =
                    sources.computeIfAbsent(localKey, ignored -> new ArrayList<>());
            retainSource(group, local);
        }
        var choices = new ArrayList<ControlChoices>();
        for (var entry : sources.entrySet()) {
            long end = entry.getKey().equals(localKey) ? local.attrs().timeCount() : Long.MAX_VALUE;
            if (!entry.getKey().equals(localKey)) {
                for (var source : entry.getValue()) end = Math.min(end, source.attrs().timeCount());
            }
            choices.add(new ControlChoices(entry.getKey(), prefixRepresentatives(entry.getValue(), end)));
        }
        return List.copyOf(choices);
    }

    /**
     * A joint transient/period partition. Time-domain ends are explicit partition boundaries;
     * no early periodic repeat may hide a later source's checked T boundary.
     */
    private static List<Long> prefixRepresentatives(List<ScanGradientRules.SourcePlan> sources, long end) {
        var boundaries = new java.util.TreeSet<Long>();
        boundaries.add(0L);
        for (var source : sources) {
            long time = source.attrs().timeCount();
            if (time < end) boundaries.add(time + 1);
        }
        var starts = new ArrayList<>(boundaries);
        var result = new ArrayList<Long>();
        for (int segment = 0; segment < starts.size(); segment++) {
            long last = segment + 1 < starts.size() ? starts.get(segment + 1) - 1 : end;
            Set<List<Integer>> seen = new HashSet<>();
            for (long prefix = starts.get(segment); ; prefix++) {
                var indices = new ArrayList<Integer>();
                for (var source : sources) {
                    indices.add(prefix <= source.attrs().timeCount()
                            ? source.finitePrefixes().indexAt(prefix) : -1);
                }
                if (!seen.add(List.copyOf(indices))) break;
                result.add(prefix);
                if (prefix == last) break;
            }
        }
        return List.copyOf(result);
    }

    /** Tests a complete assignment; short-circuiting is only over immutable proof facts. */
    private static boolean ndOnly(List<RouteWitness> paths, ScanGradientRules.SourcePlan local) {
        if (paths.stream().noneMatch(path -> !path.route.differentiable() && !path.route.failClosed())
                || paths.stream().anyMatch(path ->
                        path.numerical && path.route.differentiable() && path.bindings == null)) return false;
        return anyAssignment(controlChoices(paths, local), assignment -> {
            boolean nonDifferentiable = false, differentiable = false, failClosed = false;
            for (RouteWitness path : paths) {
                if (!path.matches(assignment)) continue;
                nonDifferentiable |= path.route.syntactic()
                        && !path.route.differentiable() && !path.route.failClosed();
                differentiable |= path.numerical && path.route.differentiable();
                failClosed |= path.numerical && path.route.failClosed();
            }
            return nonDifferentiable && !differentiable && !failClosed;
        });
    }


    /** Iterative Cartesian traversal of finite metadata partitions; no recursion or time DAG. */
    private static boolean anyAssignment(List<ControlChoices> choices,
            java.util.function.Predicate<Map<ControlKey, Long>> predicate) {
        var assignment = new LinkedHashMap<ControlKey, Long>();
        if (choices.isEmpty()) return predicate.test(assignment);
        int[] positions = new int[choices.size()];
        while (true) {
            for (int index = 0; index < choices.size(); index++) {
                ControlChoices choice = choices.get(index);
                assignment.put(choice.key, choice.prefixes.get(positions[index]));
            }
            if (predicate.test(assignment)) return true;
            int index = positions.length - 1;
            while (index >= 0 && ++positions[index] == choices.get(index).prefixes.size()) {
                positions[index--] = 0;
            }
            if (index < 0) return false;
        }
    }

    /**
     * Marks the stage-output ancestry in which a target path would require a cotangent.
     *
     * <p>Fail-closed roles remain traversable here: lack of a formula is a required rejection,
     * not permission to discard that path. Intentionally non-differentiable roles cut traversal,
     * including comparison, mask, and index roles. The forward target pass then intersects this
     * demand with exact target reachability before validating or rejecting an occurrence.</p>
     *
     * @param ancestry non-null selected-output inventory in producer postorder
     * @param outputs non-null exact selected stage roots
     * @return a new identity set of Tensors whose cotangents may be needed; never {@code null}
     */
    private static Set<Tensor> requiredCotangents(Inventory ancestry, List<Tensor> outputs,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> scanSources) {
        Set<Tensor> required = Collections.newSetFromMap(new IdentityHashMap<>());
        required.addAll(outputs);
        for (TensorProducer producer : ancestry.producerPostorder().reversed()) {
            for (int output : ancestry.selectedOutputs(producer)) {
                if (!required.contains(producer.output(output))) {
                    continue;
                }
                for (int input = 0; input < producer.inputs().size(); input++) {
                    if (classifyRoute(producer, output, input, scanSources).disposition()
                            != FirstOrderGradientCoverage.Disposition.ND) {
                        required.add(producer.inputs().get(input));
                    }
                }
            }
        }
        return required;
    }

    /** Shares one finite source classifier between the required-route and target-route passes. */
    private static FirstOrderGradientCoverage.Decision classifyRoute(TensorProducer producer,
            int output, int input,
            IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan> sources) {
        if (producer.scanBody().isEmpty()) return FirstOrderGradientCoverage.classify(producer, output, input);
        var attrs = (io.github.pho001.synaptik.model.operation.scan.ScanAttrs) producer.operation().attrs();
        if (output >= attrs.carryCount() + attrs.emissionCount()) {
            return new FirstOrderGradientCoverage.Decision(FirstOrderGradientCoverage.Disposition.FC,
                    null, "scan residual ports cannot be differentiated");
        }
        var route = ScanGradientRules.qualify(sources.get(producer), output, input);
        if (route.failClosed()) return new FirstOrderGradientCoverage.Decision(
                FirstOrderGradientCoverage.Disposition.FC, null,
                "required finite Scan route crosses an unsupported ordinary derivative role");
        if (route.differentiableRoute()) return new FirstOrderGradientCoverage.Decision(
                FirstOrderGradientCoverage.Disposition.D, FirstOrderGradientCoverage.FamilyOwner.SCAN, "");
        return new FirstOrderGradientCoverage.Decision(FirstOrderGradientCoverage.Disposition.ND, null,
                route.syntacticRoute() ? "intentional non-differentiable finite Scan route"
                        : "disconnected finite Scan route");
    }

    private static void validateSecondStageClosure(
            FunctionalGradientRequest.Stage second, StagePlan first) {
        Set<Integer> referenced = new java.util.HashSet<>();
        for (int index = 0; index < second.outputs().size(); index++) {
            int targetIndex = ((FunctionalGradientRequest.FirstStageGradientReference)
                    second.outputs().get(index)).targetIndex();
            if (!referenced.add(targetIndex)) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[1].outputs[" + index
                                + "] repeats first-stage target index " + targetIndex);
            }
            if (targetIndex >= first.targets().size()) {
                throw new IllegalArgumentException(
                        "functionalGradientRequest.stages[1].outputs[" + index
                                + "] references an unknown first-stage target");
            }
        }
        // Compiler 0005E proves the transitive operation closure for every retained family owner.
        // Keeping the selected occurrences in the plan makes formula dispatch use those same
        // owners; no second operation algebra or unchecked family path is introduced here.
        for (SelectedOccurrence occurrence : first.selectedOccurrences()) {
            Objects.requireNonNull(occurrence.familyOwner(), "familyOwner");
        }
    }

    private static BitSet copyOf(BitSet source) {
        return source == null ? new BitSet() : (BitSet) source.clone();
    }

    /**
     * Proves construction prerequisites without allocating derivative expressions.
     * @param producerIndex deterministic diagnostic position
     * @param producer original occurrence with canonical outputs
     * @param outputIndex selected output position
     * @param selectedInputs ordered input-route flags
     * @param internalBody whether false-flag local carry flow is permitted
     * @throws IllegalArgumentException if a rule or its descriptor prerequisites are unsupported
     */
    private static void validateOccurrence(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            boolean[] selectedInputs,
            boolean internalBody) {
        Operation operation = producer.operation();
        // Whole-source validity is already checked. Selected finite construction is proved only
        // after exact root/input pairs have been retained, never through a Cartesian body plan.
        if (producer.scanBody().isPresent()) return;
        if (operation.kind() instanceof RecurrentScanKind) {
            var inferred = RecurrentScanInference.infer(operation,
                    producer.inputs().stream().map(Tensor::descriptor).toList(), producer.outputCount());
            if (!inferred.outputs().equals(producer.outputDescriptors())) {
                throw unsupported(producerIndex, producer, outputIndex, -1,
                        "recurrent output descriptors differ from the fixed semantic contract");
            }
            return;
        }
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
        if (!internalBody && !output.descriptor().requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, -1, "selected output must require gradients");
        }
        for (int input = 0; input < selectedInputs.length; input++) {
            if (!internalBody && selectedInputs[input]
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
        if (operation.kind() == Conv2dKind.CONV2D
                || operation.kind() == Conv3dKind.CONV3D) {
            boolean attrsMatch = operation.kind() == Conv2dKind.CONV2D
                    ? operation.attrs() instanceof Conv2dAttrs
                    : operation.attrs() instanceof Conv3dAttrs;
            if (!attrsMatch
                    || (producer.inputs().size() != 2 && producer.inputs().size() != 3)) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported convolution signature or attributes");
            }
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            try {
                Shape weight = producer.inputs().get(1).descriptor().shape();
                long kernelElements = 1;
                for (int axis = 2; axis < weight.rank(); axis++) {
                    kernelElements = Math.multiplyExact(
                            kernelElements, ((StaticDimension) weight.dimension(axis)).size());
                }
                ConvolutionGradientRules.window(operation.attrs(), weight);
            } catch (RuntimeException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        operation.kind().name() + " grouped window geometry is not constructible");
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
        if (operation.kind() instanceof Pool3dKind kind) {
            boolean attrsMatch = kind == Pool3dKind.MAX_POOL3D
                    ? operation.attrs() instanceof MaxPool3dAttrs
                    : operation.attrs() instanceof AveragePool3dAttrs;
            if (!attrsMatch) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported Pool3d kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            validateStructuredOccurrence(producerIndex, producer, outputIndex);
            long depth = kind == Pool3dKind.MAX_POOL3D
                    ? ((MaxPool3dAttrs) operation.attrs()).kernelDepth()
                    : ((AveragePool3dAttrs) operation.attrs()).kernelDepth();
            long height = kind == Pool3dKind.MAX_POOL3D
                    ? ((MaxPool3dAttrs) operation.attrs()).kernelHeight()
                    : ((AveragePool3dAttrs) operation.attrs()).kernelHeight();
            long width = kind == Pool3dKind.MAX_POOL3D
                    ? ((MaxPool3dAttrs) operation.attrs()).kernelWidth()
                    : ((AveragePool3dAttrs) operation.attrs()).kernelWidth();
            try {
                Math.multiplyExact(Math.multiplyExact(depth, height), width);
                validatePool3dFormulaConstructibility(producer, kind);
            } catch (RuntimeException exception) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "Pool3d public gradient formula is not constructible");
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
                case UNFOLD3D -> operation.attrs() instanceof Window3dAttrs
                        || operation.attrs() instanceof Unfold3dAttrs;
                case FOLD3D -> operation.attrs() instanceof Fold3dAttrs;
            };
            if (!supportedAttrs) {
                throw unsupported(
                        producerIndex, producer, outputIndex, -1,
                        "unsupported window kind/attributes pairing");
            }
            requireInputs(producerIndex, producer, outputIndex, 1);
            requireSameFloatingType(producerIndex, producer, outputIndex, 0);
            validateLayoutOccurrence(producerIndex, producer, outputIndex);
            if (kind == WindowTransformKind.UNFOLD3D
                    || kind == WindowTransformKind.FOLD3D) {
                validateWindow3dAdjointConstructibility(
                        producerIndex, producer, outputIndex, kind);
            }
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

    /**
     * Proves that a Pool3d occurrence's public unfold/fold path restores the exact captured input
     * descriptor without constructing a Tensor.
     *
     * @param producer non-null validated one-input Pool3d occurrence
     * @param kind exact Pool3d kind used to select its immutable geometry type
     * @throws RuntimeException if layout inference rejects an intermediate or descriptor identity is
     * not preserved semantically
     */
    private static void validatePool3dFormulaConstructibility(
            TensorProducer producer, Pool3dKind kind) {
        Window3dAttrs window = kind == Pool3dKind.MAX_POOL3D
                ? window((MaxPool3dAttrs) producer.operation().attrs())
                : window((AveragePool3dAttrs) producer.operation().attrs());
        var input = producer.inputs().getFirst().descriptor();
        var unfolded = LayoutInference.infer(
                new Operation(WindowTransformKind.UNFOLD3D, window), List.of(input));
        if (unfolded.outputs().size() != 1) {
            throw new IllegalArgumentException("Pool3d unfold formula must have one output");
        }
        var folded = LayoutInference.infer(
                new Operation(
                        WindowTransformKind.FOLD3D,
                        new Fold3dAttrs(input.shape(), window)),
                unfolded.outputs());
        if (folded.outputs().size() != 1) {
            throw new IllegalArgumentException("Pool3d fold formula must have one output");
        }
        var canonicalized = LayoutInference.infer(
                new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                folded.outputs());
        if (canonicalized.outputs().size() != 1) {
            throw new IllegalArgumentException("Pool3d canonicalization must have one output");
        }
        TensorDescriptor restored = canonicalized.outputs().getFirst();
        if (restored.dataType() != input.dataType()
                || !restored.shape().equals(input.shape())
                || restored.requiresGrad() != input.requiresGrad()) {
            throw new IllegalArgumentException("Pool3d fold formula must restore input semantics");
        }
    }

    /**
     * Proves the exact public adjoint counterpart for a three-dimensional window occurrence.
     *
     * @param producerIndex deterministic producer postorder position used in diagnostics
     * @param producer non-null validated one-input window occurrence
     * @param outputIndex valid selected output slot
     * @param kind exact UNFOLD3D or FOLD3D kind
     * @throws IllegalArgumentException if the generated counterpart is not constructible or does not
     * reproduce the original input data type, shape, and gradient metadata
     */
    private static void validateWindow3dAdjointConstructibility(
            int producerIndex,
            TensorProducer producer,
            int outputIndex,
            WindowTransformKind kind) {
        var input = producer.inputs().getFirst().descriptor();
        var output = producer.output(outputIndex).descriptor();
        CapturedGraphInference.InferenceResult inferred;
        try {
            if (kind == WindowTransformKind.UNFOLD3D) {
                Window3dAttrs window = producer.operation().attrs() instanceof Unfold3dAttrs attrs
                        ? attrs.window()
                        : (Window3dAttrs) producer.operation().attrs();
                inferred = LayoutInference.infer(
                        new Operation(
                                WindowTransformKind.FOLD3D,
                                new Fold3dAttrs(input.shape(), window)),
                        List.of(output));
            } else {
                Fold3dAttrs attrs = (Fold3dAttrs) producer.operation().attrs();
                inferred = LayoutInference.infer(
                        new Operation(WindowTransformKind.UNFOLD3D, attrs.window()),
                        List.of(output));
            }
        } catch (RuntimeException exception) {
            throw unsupported(
                    producerIndex, producer, outputIndex, 0,
                    "three-dimensional window adjoint is not constructible");
        }
        if (inferred.outputs().size() != 1 || inferred.outputs().getFirst().dataType() != input.dataType()
        || !inferred.outputs().getFirst().shape().equals(input.shape())
        || inferred.outputs().getFirst().requiresGrad() != input.requiresGrad()) {
            throw unsupported(
                    producerIndex, producer, outputIndex, 0,
          "three-dimensional window adjoint semantic descriptor differs from input");
        }
    }

    /**
     * Projects maximum Pool3d geometry into the exact public volumetric window value used by its
     * derivative formula.
     *
     * @param attrs non-null validated maximum Pool3d attributes
     * @return a new immutable window retaining every geometry component
     */
    private static Window3dAttrs window(MaxPool3dAttrs attrs) {
        return new Window3dAttrs(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
    }

    /**
     * Projects average Pool3d geometry into the exact public volumetric window value used by its
     * derivative formula.
     *
     * @param attrs non-null validated average Pool3d attributes
     * @return a new immutable window retaining every geometry component
     */
    private static Window3dAttrs window(AveragePool3dAttrs attrs) {
        return new Window3dAttrs(
                attrs.kernelDepth(), attrs.kernelHeight(), attrs.kernelWidth(),
                attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth(),
                attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth(),
                attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth(),
                attrs.ceilMode());
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
     * @throws IllegalArgumentException if inference rejects the occurrence, a derived output     differs,
   * or an occurrence-local predicate is statically contradicted
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
     * @throws IllegalArgumentException if two present Dimensions cannot broadcast under the current
     * local rules
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

    /**
     * Checks complete lexical Scan sources independently of selected derivative demand.
     * @param original complete deterministic source inventory
     * @return request-local exact-occurrence source facts
     */
    private static IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan>
            preflightScanSources(Inventory original) {
        var sources = new IdentityHashMap<TensorProducer, ScanGradientRules.SourcePlan>();
        for (int producerIndex = 0;
                producerIndex < original.producerPostorder().size();
                producerIndex++) {
            TensorProducer producer = original.producerPostorder().get(producerIndex);
            if (producer.scanBody().isPresent()) sources.put(producer, ScanGradientRules.preflightSource(producer));
        }
        return sources;
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
     * @param selectedInputs input-position-aligned differentiable-route flags; cloned on input and
     * access
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
         * @throws NullPointerException if {@code selectedInputs} or {@code familyOwner} is     {@code null}
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
     * One successful allocation-free reverse-mode stage plan.
     *
     * @param stageIndex zero-based public stage index
     * @param outputs exact ordered reverse roots
     * @param cotangentSeeds output-aligned optional exact explicit seeds
     * @param targets exact ordered original-forward targets
     * @param producerPostorder complete selected-output ancestry postorder
     * @param selectedOccurrences checked differentiable occurrences
     * @param connectedTargets stage-local connected target positions
     * @param disconnectedPolicy selected disconnected target policy
     * @param originalProducers complete original forward producer identity set
     * @param scanPlans exact-occurrence finite selected Scan plans proved before materialization
     * @param selectedPairSupport immutable exact positional cut, with original target-role support
     */
    record StagePlan(
            int stageIndex,
            List<Tensor> outputs,
            List<Optional<Tensor>> cotangentSeeds,
            List<Tensor> targets,
            List<TensorProducer> producerPostorder,
            List<SelectedOccurrence> selectedOccurrences,
            BitSet connectedTargets,
            FunctionalGradientRequest.DisconnectedPolicy disconnectedPolicy,
            Set<TensorProducer> originalProducers,
            Map<TensorProducer, ScanGradientRules.BackwardPlan> scanPlans,
            Map<TensorProducer, Map<PortPair, BitSet>> selectedPairSupport) {
        /**
         * Creates a local body plan without an outer finite-path cut.
         * @param stageIndex zero-based public stage index
         * @param outputs exact ordered reverse roots
         * @param cotangentSeeds output-aligned optional explicit cotangents
         * @param targets exact ordered target declarations
         * @param producerPostorder complete selected-output ancestry in producer postorder
         * @param selectedOccurrences checked differentiable occurrence/input selections
         * @param connectedTargets target positions having a differentiable route
         * @param disconnectedPolicy policy for wholly disconnected targets
         * @param originalProducers complete original-forward producer identity set
         * @param scanPlans exact-occurrence checked Scan plans, empty for an ordinary body
         */
        StagePlan(int stageIndex, List<Tensor> outputs, List<Optional<Tensor>> cotangentSeeds,
                List<Tensor> targets, List<TensorProducer> producerPostorder,
                List<SelectedOccurrence> selectedOccurrences, BitSet connectedTargets,
                FunctionalGradientRequest.DisconnectedPolicy disconnectedPolicy,
                Set<TensorProducer> originalProducers,
                Map<TensorProducer, ScanGradientRules.BackwardPlan> scanPlans) {
            this(stageIndex, outputs, cotangentSeeds, targets, producerPostorder, selectedOccurrences,
                    connectedTargets, disconnectedPolicy, originalProducers, scanPlans, Map.of());
        }
        /**
         * Creates an ordinary stage without controlled Scan plans or a finite-path cut.
         * @param stageIndex zero-based public stage index
         * @param outputs exact ordered reverse roots
         * @param cotangentSeeds output-aligned optional explicit cotangents
         * @param targets exact ordered target declarations
         * @param producerPostorder complete selected-output ancestry in producer postorder
         * @param selectedOccurrences checked differentiable occurrence/input selections
         * @param connectedTargets target positions having a differentiable route
         * @param disconnectedPolicy policy for wholly disconnected targets
         * @param originalProducers complete original-forward producer identity set
         */
        StagePlan(int stageIndex, List<Tensor> outputs, List<Optional<Tensor>> cotangentSeeds,
                List<Tensor> targets, List<TensorProducer> producerPostorder,
                List<SelectedOccurrence> selectedOccurrences, BitSet connectedTargets,
                FunctionalGradientRequest.DisconnectedPolicy disconnectedPolicy,
                Set<TensorProducer> originalProducers) {
            this(stageIndex, outputs, cotangentSeeds, targets, producerPostorder, selectedOccurrences,
                    connectedTargets, disconnectedPolicy, originalProducers, Map.of());
        }
        /**
         * Snapshots the ordered components of one successful stage plan.
         *
         * @param stageIndex zero-based public stage index
         * @param outputs non-null ordered stage outputs
         * @param cotangentSeeds non-null output-aligned optional seeds
         * @param targets non-null ordered target list
         * @param producerPostorder non-null complete selected-output producer postorder
         * @param selectedOccurrences non-null selected occurrences in deterministic postorder
         * @param connectedTargets non-null connected-target flags
         * @param disconnectedPolicy non-null disconnected target policy
         * @param originalProducers non-null complete original forward producer identity set
         * @param scanPlans identity-keyed checked Scan plans, empty only for ordinary stages
         * @param selectedPairSupport compiler-owned frozen positional cut, or empty for ordinary AD
         * @throws NullPointerException if a required component is {@code null}
         */StagePlan {
            outputs = List.copyOf(outputs);
            cotangentSeeds = List.copyOf(cotangentSeeds);
            targets = List.copyOf(targets);
            producerPostorder = List.copyOf(producerPostorder);
            selectedOccurrences = List.copyOf(selectedOccurrences);
            connectedTargets = (BitSet) Objects.requireNonNull(
                    connectedTargets, "connectedTargets").clone();
            Objects.requireNonNull(disconnectedPolicy, "disconnectedPolicy");
            Objects.requireNonNull(originalProducers, "originalProducers");
            scanPlans = Collections.unmodifiableMap(new IdentityHashMap<>(scanPlans));
            Objects.requireNonNull(selectedPairSupport, "selectedPairSupport");
        }

        /**
         * Checks the same positional cut used by occurrence selection and Scan planning before
         * a real contribution is appended. Ordinary plans without a cut allow their checked roles.
         * @param producer exact original producing occurrence
         * @param output original canonical output position
         * @param input original input declaration position
         * @return whether at least one requested target retains this edge, or no cut applies
         */
        boolean retainsRoute(TensorProducer producer, int output, int input) {
            if (selectedPairSupport.isEmpty()) return true;
            Map<PortPair, BitSet> pairs = selectedPairSupport.get(producer);
            BitSet support = pairs == null ? null : pairs.get(new PortPair(output, input));
            return support != null && !support.isEmpty();
        }

        /**
         * Returns an independent snapshot of connected target positions.
         *
         * @return non-null cloned bit set
         */
        @Override
        public BitSet connectedTargets() {
            return (BitSet) connectedTargets.clone();
        }

        /**
         * Reports whether one target has a differentiable route.
         *
         * @param index zero-based target index
         * @return whether the target is connected
         */
        boolean connectedTarget(int index) {
            return connectedTargets.get(index);
        }
    }

    /**
     * Successful complete-original-inventory and first-stage preflight handoff.
     *
     * @param stageOne non-null successful first-stage plan
     * @param originalProducers non-null identity set of complete original forward producers
     * @param originalTensors non-null identity inventory of complete original forward Tensors
     */
    record InitialPlan(
            StagePlan stageOne,
            Set<TensorProducer> originalProducers,
            IdentityHashMap<Tensor, Boolean> originalTensors) {
        /**
         * Validates the successful initial handoff.
         *
         * @param stageOne non-null successful first-stage plan
         * @param originalProducers non-null complete original producer identity set
         * @param originalTensors non-null complete original Tensor identity inventory
         */
        InitialPlan {
            Objects.requireNonNull(stageOne, "stageOne");
            Objects.requireNonNull(originalProducers, "originalProducers");
            Objects.requireNonNull(originalTensors, "originalTensors");
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
