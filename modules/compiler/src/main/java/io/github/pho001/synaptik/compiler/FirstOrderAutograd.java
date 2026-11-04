package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.ArrayDeque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.function.BiFunction;

/**
 * Constructs one deterministic reverse-mode Tensor-expression stage after successful preflight.
 *
 * <p>Incoming contributions and accumulated cotangents are keyed by exact Tensor identity.
 * Contributions are appended in reverse producer-postorder, ascending selected-output-slot, and
 * ascending input-position order, then merged through left-associated public Tensor addition.
 * This keeps existing one-output order stable while allowing several selected public outputs of
 * one batch-normalization occurrence to contribute to the same input. Each selected output uses
 * its validated explicit cotangent seed or an implicit exact typed positive one for an eligible
 * scalar output. Request-local floating scalar leaves, including both OFP8 types, are cached by
 * exact {@link ScalarValue} type and represented bits, remain storage-free, and are registered
 * explicitly as logical splats in deterministic first-use order. Shape-specific values are
 * ordinary public {@code expand} expressions.</p>
 *
 * <p>Formula ownership remains split among {@link ElementwiseGradientRules},
 * {@link ReductionGradientRules}, {@link NormalizationGradientRules},
 * {@link LinearAlgebraGradientRules}, {@link LayoutGradientRules},
 * {@link IndexingGradientRules}, {@link OrderingGradientRules}, and
 * {@link StochasticGradientRules}, with structured-neural formulas owned by
 * {@link AttentionGradientRules}, {@link ConvolutionGradientRules},
 * {@link PoolingGradientRules}, and {@link LossGradientRules}. This class switches only on the
 * {@link FirstOrderGradientCoverage.FamilyOwner} retained by each preflight-approved occurrence;
 * it does not reclassify an operation kind. It preserves every selected positional contribution,
 * including repeated MATMUL, slice-update, composition, scatter, and batch-normalization output
 * routes. Canonical TOP_K indices and the canonical dropout mask, attention weights, and
 * maximum-pool output remain attached to their exact original producer; the sole matching
 * ARGSORT and every other generated formula occurrence are handed to the combined capture as
 * generated expressions. The exact typed positive and negative coefficients used by
 * mean-squared error remain scalar-operation metadata. This class does not absorb family
 * formulas or infer support.</p>
 *
 * <p>The ordinary owners and central contribution orchestration use one body for direct Tensor
 * construction and checked descriptor-only construction through {@link OrdinaryConstruction}.
 * Descriptor reads preserve original source roles independently of lazy primal bindings.
 * Synchronous optional observation includes eager generated operations, normalization, cast-back,
 * ordered merges and disconnected publication. Unobserved construction retains direct Tensor
 * values and allocates no activity wrappers, scope tables or observers. Stochastic and joint Scan
 * formulas retain their separate protocols within the same contribution ordering.</p>
 *
 * <p>The returned Tensor roles and original-producer identities are an ephemeral handoff to one
 * combined capture. Generated logical-splat bindings, including extrema or clamp bounds needed
 * as Tensor comparison operands, are retained only when reachable from a returned gradient role.
 * A direct local-zero result therefore does not expose an otherwise unreachable seed binding as
 * graph ingress. This owner does not retain global state, mutate an original
 * Tensor, infer a constant from storage or provenance, capture or validate a graph, materialize
 * values, lower work, or execute computation.</p>
 *
 * <p>General scans dispatch once per occurrence with joint carry/emission seeds through
 * {@link ScanGradientRules}. A body expansion reuses these same family rules and original
 * contribution identities with an explicit primal binding. Internal carry flow ignores source
 * eligibility only after local preflight; outer eligibility is unchanged. Scalar body constants
 * bind to non-gradient invariants backed by registered outer splats. Length-mode shape literals
 * and disconnected cotangents with an explicit row coordinate instead bind non-gradient ROW
 * sequences. Neither path invents a row by expanding a SHARED leaf or reads a primal for zero.</p>
 */
final class FirstOrderAutograd {
    /** Shared identity primal resolver for ordinary expansion without body-local rebinding. */
    static final Function<Tensor, Tensor> IDENTITY = Function.identity();
    private FirstOrderAutograd() {}

    /**
     * Expands one stage with original producers as the already-owned producer set.
     *
     * @param plan non-null successful stage preflight
     * @param forwardIngress non-null caller-ordered explicit ingress
     * @return non-null expansion handoff
     */
    static Expansion expand(
            AutogradPreflight.StagePlan plan,
            CompileTimeConstantGraph.Ingress forwardIngress) {
        return expand(plan, forwardIngress, plan.originalProducers());
    }

    /**
     * Expands one successful plan and merges generated typed constants after caller ingress.
     *
     * @param plan non-null successful Tensor-allocation-free preflight plan
     * @param forwardIngress non-null caller-ordered explicit forward constant bindings; observed
     *     but not mutated
     * @param previouslyOwnedProducers non-null identity set of producers owned by earlier stages
     * @return a non-null expansion handoff with target roles in request order, the original
     *     producer identity set, and ingress ordered as caller bindings followed by reachable
     *     generated derivative bindings in deterministic first-use order
     * @throws NullPointerException if {@code plan} or {@code forwardIngress} is {@code null}
     * @throws IllegalArgumentException if a generated derivative leaf collides by exact Tensor
     *     identity with caller ingress
     * @throws IllegalStateException if the successful preflight plan and closed rule dispatch are
     *     internally inconsistent
     */
    static Expansion expand(
            AutogradPreflight.StagePlan plan,
            CompileTimeConstantGraph.Ingress forwardIngress,
            Set<TensorProducer> previouslyOwnedProducers) {
        return expand(plan, forwardIngress, previouslyOwnedProducers,
                new DerivativeConstants(), IDENTITY, (target, incoming) -> incoming);
    }

    /**
     * Expands a preflight-approved lexical body with exact seeds and primal/constant bindings.
     * @param plan allocation-free local rule plan
     * @param seeds output-position-aligned cotangents, including repeated root roles
     * @param primal total original-value to reverse-formal binding
     * @param constants closed local constant owner
     * @return target-position-aligned cotangents, including false-flag and disconnected formals
     */
    static List<Tensor> expandBody(AutogradPreflight.StagePlan plan, List<Tensor> seeds,
            Function<Tensor, Tensor> primal, DerivativeConstants constants) {
        return expandBody(plan, seeds, primal, constants, (target, incoming) -> incoming);
    }

    /**
     * Expands a length body with role-aware normalization of every incoming term before append.
     * The callback also handles shared intermediates, before they acquire mixed-role sums.
     * @param plan allocation-free ordinary family plan
     * @param seeds ordered explicit local cotangents
     * @param primal total exact primal binding
     * @param constants closed constant/zero binding owner
     * @param normalize original-target/incoming-term normalization, preserving ordinary formulas
     * @return formal-position-aligned cotangents
     */
    static List<Tensor> expandBody(AutogradPreflight.StagePlan plan, List<Tensor> seeds,
            Function<Tensor, Tensor> primal, DerivativeConstants constants,
            BiFunction<Tensor, Tensor, Tensor> normalize) {
        return expandOrdinaryBody(plan, seeds, IDENTITY, primal,
                OrdinaryConstruction.real(constants, null), normalize, null);
    }

    /**
     * Expands a fixed-cell Scan recipe using the enclosing request's constant/activity owner.
     * @param plan validated internal recipe plan
     * @param seeds actual cotangents aligned with the recipe's ordered output roles
     * @param constants enclosing request's scalar-splat and activity owner
     * @return cotangents aligned with the plan's original target roles
     */
    static List<Tensor> expandRecipe(AutogradPreflight.StagePlan plan, List<Tensor> seeds,
            DerivativeConstants constants) {
        return expandContributions(plan, seeds, IDENTITY, IDENTITY,
                OrdinaryConstruction.real(constants, null), (target, incoming) -> incoming,
                null, REAL_RULES, constants);
    }

    /**
     * Accumulates one approved reverse stage through ordinary or lexically bound existing rules.
     * @param plan successful stage plan
     * @param forwardIngress explicit caller constant bindings
     * @param previouslyOwnedProducers identities owned by previous stages
     * @param constants request-local or closed lexical scalar-splat owner
     * @param primal identity for ordinary AD, otherwise total original-primal to local binding
     * @param normalize incoming normalization applied before every contribution append
     * @return ordered target roles and reachable outer ingress/producer identities
     * @throws IllegalStateException if a successful plan has no selected contribution
     */
    private static Expansion expand(AutogradPreflight.StagePlan plan,
            CompileTimeConstantGraph.Ingress forwardIngress, Set<TensorProducer> previouslyOwnedProducers,
            DerivativeConstants constants, Function<Tensor, Tensor> primal,
            BiFunction<Tensor, Tensor, Tensor> normalize) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(forwardIngress, "forwardIngress");
        Objects.requireNonNull(previouslyOwnedProducers, "previouslyOwnedProducers");
        var seeds = new ArrayList<Tensor>(plan.outputs().size());
        for (int index = 0; index < plan.outputs().size(); index++) {
            Tensor output = plan.outputs().get(index);
            seeds.add(plan.cotangentSeeds().get(index).orElseGet(
                    () -> constants.oneBase(output.descriptor().dataType())));
        }
        List<Tensor> gradients = expandContributions(plan, seeds, IDENTITY, primal,
                OrdinaryConstruction.real(constants, null), normalize, null, REAL_RULES, constants);
        List<TargetGradient> roles = new ArrayList<>(plan.targets().size());
        for (int index = 0; index < plan.targets().size(); index++) {
            Tensor gradient = gradients.get(index);
            if (gradient == null) {
                throw new IllegalStateException(
                        "successful preflight target has no gradient contribution");
            }
            roles.add(new TargetGradient(plan.targets().get(index), gradient));
        }

        List<CompileTimeConstantGraph.Binding> merged =
                new ArrayList<>(forwardIngress.bindings().size() + constants.bindings().size());
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        for (CompileTimeConstantGraph.Binding binding : forwardIngress.bindings()) {
            seen.put(binding.tensor(), Boolean.TRUE);
            merged.add(binding);
        }
        IdentityHashMap<Tensor, Boolean> reachableDerivativeTensors =
                reachableDerivativeTensors(roles);
        for (CompileTimeConstantGraph.Binding binding : constants.bindings()) {
            if (!reachableDerivativeTensors.containsKey(binding.tensor())) {
                continue;
            }
            if (seen.putIfAbsent(binding.tensor(), Boolean.TRUE) != null) {
                throw new IllegalArgumentException(
                        "generated derivative constant collides with forward ingress");
            }
            merged.add(binding);
        }
        return new Expansion(
                roles,
                ownedProducers(roles, previouslyOwnedProducers),
                new CompileTimeConstantGraph.Ingress(merged),
                constants.scanActivityDrafts());
    }

    private static Set<TensorProducer> ownedProducers(
            List<TargetGradient> roles,
            Set<TensorProducer> previouslyOwnedProducers) {
        Set<TensorProducer> result = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        for (TargetGradient role : roles) {
            pending.addLast(role.gradient());
        }
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (seen.put(tensor, Boolean.TRUE) != null) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> {
                TensorProducer producer = provenance.producer();
                if (!previouslyOwnedProducers.contains(producer)) {
                    result.add(producer);
                }
                producer.inputs().forEach(pending::addLast);
            });
        }
        return java.util.Collections.unmodifiableSet(result);
    }

    /**
     * Inventories the exact Tensor-expression ancestry of the returned gradient roles.
     *
     * @param roles non-null ordered target-gradient roles whose gradient expressions are roots
     * @return a non-null identity-keyed set containing each exact reachable gradient Tensor,
     *     including provenance-free leaves; owned by the current expansion and never retained
     */
    private static IdentityHashMap<Tensor, Boolean> reachableDerivativeTensors(
            List<TargetGradient> roles) {
        IdentityHashMap<Tensor, Boolean> reachable = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        for (TargetGradient role : roles) {
            pending.addLast(role.gradient());
        }
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (reachable.put(tensor, Boolean.TRUE) != null) {
                continue;
            }
            tensor.provenance().ifPresent(
                    provenance -> provenance.inputs().forEach(pending::addLast));
        }
        return reachable;
    }

    /**
     * Optional synchronous owner scopes for controlled contribution construction.
     *
     * <p>Scopes surround construction, not just returned expressions, so eager primal-only and
     * constant operations inherit the selected VJP's incoming scope. Merge preparation precedes
     * the actual ADD and completion retains its two distinct operand scopes. No observer is
     * created by the ordinary unobserved route.</p>
     *
     * @param <V> direct real or metadata operand type
     */
    interface ContributionObserver<V> {
        /**
         * Enters one incoming owner scope.
         * @param incoming exact incoming cotangent, or null for disconnected-zero publication
         */
        void entering(V incoming);
        /**
         * Enters the union owner scope before constructing an optional contribution-list ADD.
         * @param left exact left accumulated cotangent
         * @param right exact next positional contribution
         */
        void merging(V left, V right);
        /**
         * Records the represented optional ADD with separate left and right scopes.
         * @param left exact left operand
         * @param right exact right operand
         * @param result exact newly constructed ADD result
         */
        void merge(V left, V right, V result);
        /** Leaves the current incoming or merge scope, including on construction failure. */
        void leaving();
    }

    /**
     * Expands an approved ordinary body using the same orchestration as ordinary real autograd.
     *
     * <p>Original Tensor identities key contributions; their values are direct {@code V}s.
     * Null seeds are structurally absent, never implicit ones or numerical zeros. All enclosing
     * operations use the supplied construction route, including normalization performed by the
     * callback, cast-back, left-associated contribution ADDs and disconnected positive zeros.
     * The caller supplies the exact selected roots/targets and owns any activity bookkeeping.
     * Disconnected targets must be floating roles; no primal is resolved to publish their zero.</p>
     *
     * @param <V> direct real or metadata operand type
     * @param plan successful ordinary selected body plan
     * @param seeds output-position-aligned cotangents; null elements denote structural absence
     * @param role total identity-preserving original-source to operand mapping
     * @param primal lazy source-operand to actual primal binding
     * @param construction shared ordinary operation and literal owner
     * @param normalize per-term normalization before every append and disconnected publication
     * @param observer nullable synchronous owner/merge observer
     * @return target-position-aligned cotangents; a connected target with no present seed route
     *     is null, while a disconnected target has its explicitly constructed typed positive zero
     * @throws IllegalArgumentException if seed positions do not match the selected roots
     * @throws IllegalStateException if the plan requires a stochastic or joint Scan rule
     */
    static <V> List<V> expandOrdinaryBody(AutogradPreflight.StagePlan plan, List<V> seeds,
            Function<Tensor, V> role, Function<V, V> primal,
            OrdinaryConstruction<V> construction, BiFunction<Tensor, V, V> normalize,
            ContributionObserver<V> observer) {
        return expandContributions(plan, seeds, role, primal, construction, normalize,
                observer, null, null);
    }

    /**
     * Executes the one source-identity contribution algorithm for real and metadata operands.
     * @param <V> direct operand type
     * @param plan selected successful plan
     * @param seeds nullable positional explicit seeds, never implicit ones
     * @param role total original-role mapping
     * @param primal lazy actual-primal resolver
     * @param construction operation owner shared with the eleven formula bodies
     * @param normalize role-aware per-term normalization through that construction
     * @param observer optional synchronous activity owner
     * @param special optional real-only stochastic/joint Scan dispatch
     * @param constants real-only closed or request-local constants, null for ordinary bodies
     * @return ordered target values, retaining structural nulls without copying operand objects
     */
    private static <V> List<V> expandContributions(AutogradPreflight.StagePlan plan,
            List<V> seeds, Function<Tensor, V> role, Function<V, V> primal,
            OrdinaryConstruction<V> construction, BiFunction<Tensor, V, V> normalize,
            ContributionObserver<V> observer, SpecialRules<V> special,
            DerivativeConstants constants) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(seeds, "seeds");
        Objects.requireNonNull(role, "role");
        Objects.requireNonNull(primal, "primal");
        Objects.requireNonNull(construction, "construction");
        Objects.requireNonNull(normalize, "normalize");
        if (seeds.size() != plan.outputs().size()) {
            throw new IllegalArgumentException("body seeds must match selected output positions");
        }
        IdentityHashMap<Tensor, List<V>> contributions = new IdentityHashMap<>();
        IdentityHashMap<Tensor, V> accumulated = new IdentityHashMap<>();
        for (int index = 0; index < seeds.size(); index++) {
            V seed = seeds.get(index);
            if (seed == null) continue;
            Tensor output = plan.outputs().get(index);
            if (observer != null) observer.entering(seed);
            try {
                append(contributions, output, normalize.apply(output, seed));
            } finally {
                if (observer != null) observer.leaving();
            }
        }

        List<AutogradPreflight.SelectedOccurrence> selected = plan.selectedOccurrences();
        for (int end = selected.size(); end > 0; ) {
            int start = end - 1;
            int postorderIndex = selected.get(start).postorderIndex();
            while (start > 0 && selected.get(start - 1).postorderIndex() == postorderIndex) {
                start--;
            }
            TensorProducer grouped = selected.get(start).producer();
            if (selected.get(start).familyOwner() == FirstOrderGradientCoverage.FamilyOwner.RECURRENT) {
                if (special == null) throw new IllegalStateException("ordinary body cannot dispatch fixed recurrence");
                var recurrentSeeds = new ArrayList<V>(grouped.outputCount());
                boolean present = false;
                for (int port = 0; port < grouped.outputCount(); port++) {
                    Tensor output = grouped.output(port);
                    V seed = accumulated.get(output);
                    if (seed == null) {
                        seed = accumulate(contributions.get(output), construction, observer);
                        if (seed != null) accumulated.put(output, seed);
                    }
                    recurrentSeeds.add(seed);
                    present |= seed != null;
                }
                if (present) {
                    boolean[] requested = new boolean[grouped.inputs().size()];
                    for (int index = start; index < end; index++) {
                        var occurrence = selected.get(index);
                        for (int input = 0; input < requested.length; input++) {
                            requested[input] |= occurrence.selectedInput(input)
                                    && plan.retainsRoute(grouped, occurrence.outputIndex(), input);
                        }
                    }
                    V[] inputs = special.recurrent(grouped, recurrentSeeds, requested, constants);
                    for (int input = 0; input < inputs.length; input++) if (requested[input]) {
                        appendTerm(contributions, grouped.inputs().get(input), inputs[input],
                                construction, normalize);
                    }
                }
                end = start;
                continue;
            }
            if (grouped.scanBody().isPresent()) {
                if (special == null) {
                    throw new IllegalStateException("ordinary body cannot dispatch a joint Scan");
                }
                ScanGradientRules.BackwardPlan backward = plan.scanPlans().get(grouped);
                if (backward == null) {
                    throw new IllegalStateException("successful stage has no checked Scan plan");
                }
                List<V> selectedSeeds = new ArrayList<>(backward.selectedSeedPorts().size());
                boolean present = false;
                for (int port : backward.selectedSeedPorts()) {
                    Tensor output = grouped.output(port);
                    V seed = accumulated.get(output);
                    if (seed == null) {
                        seed = accumulate(contributions.get(output), construction, observer);
                        if (seed != null) accumulated.put(output, seed);
                    }
                    selectedSeeds.add(seed);
                    present |= seed != null;
                }
                if (present) {
                    V[] inputs = special.scan(backward, selectedSeeds, constants);
                    for (int input = 0; input < grouped.inputs().size(); input++) {
                        boolean requested = false;
                        for (int index = start; index < end; index++) {
                            AutogradPreflight.SelectedOccurrence occurrence = selected.get(index);
                            requested |= occurrence.selectedInput(input)
                                    && plan.retainsRoute(grouped, occurrence.outputIndex(), input);
                        }
                        if (requested && inputs[input] != null) {
                            appendTerm(contributions, grouped.inputs().get(input), inputs[input],
                                    construction, normalize);
                        }
                    }
                }
                end = start;
                continue;
            }
            for (int index = start; index < end; index++) {
                AutogradPreflight.SelectedOccurrence occurrence = selected.get(index);
                TensorProducer producer = occurrence.producer();
                Tensor output = producer.output(occurrence.outputIndex());
                V gradient = accumulated.get(output);
                if (gradient == null) {
                    gradient = accumulate(contributions.get(output), construction, observer);
                    if (gradient == null) continue;
                    accumulated.put(output, gradient);
                }
                if (observer != null) observer.entering(gradient);
                try {
                    V[] inputGradients;
                    if (occurrence.familyOwner() == FirstOrderGradientCoverage.FamilyOwner.STOCHASTIC) {
                        if (special == null) {
                            throw new IllegalStateException("ordinary body cannot dispatch stochastic rules");
                        }
                        inputGradients = special.stochastic(occurrence, gradient, constants);
                    } else {
                        inputGradients = applyOrdinary(occurrence.familyOwner(),
                                construction.sourceRoles(producer, role), occurrence.outputIndex(),
                                gradient, occurrence.selectedInputs(), construction, primal);
                    }
                    for (int input = 0; input < producer.inputs().size(); input++) {
                        if (!occurrence.selectedInput(input)
                                || !plan.retainsRoute(producer, occurrence.outputIndex(), input)) continue;
                        if (inputGradients[input] == null) {
                            throw new IllegalStateException(
                                    "preflight selected a non-differentiable input role " + input);
                        }
                        appendTerm(contributions, producer.inputs().get(input), inputGradients[input],
                                construction, normalize);
                    }
                } finally {
                    if (observer != null) observer.leaving();
                }
            }
            end = start;
        }

        List<V> results = new ArrayList<>(plan.targets().size());
        for (int index = 0; index < plan.targets().size(); index++) {
            Tensor target = plan.targets().get(index);
            V gradient;
            if (!plan.connectedTarget(index)) {
                if (observer != null) observer.entering(null);
                try {
                    gradient = normalize.apply(
                            target, construction.disconnectedZeroLike(role.apply(target)));
                } finally {
                    if (observer != null) observer.leaving();
                }
            } else {
                gradient = accumulated.get(target);
                if (gradient == null) {
                    gradient = accumulate(contributions.get(target), construction, observer);
                    if (gradient != null) accumulated.put(target, gradient);
                }
            }
            results.add(gradient);
        }
        return java.util.Collections.unmodifiableList(results);
    }

    /** Real-only formula adapters; this singleton allocates no request-local observation state. */
    private static final SpecialRules<Tensor> REAL_RULES = new SpecialRules<>() {
        @Override public Tensor[] stochastic(AutogradPreflight.SelectedOccurrence occurrence,
                Tensor gradient, DerivativeConstants constants) {
            return StochasticGradientRules.apply(occurrence.producer(), gradient, constants);
        }
        @Override public Tensor[] scan(ScanGradientRules.BackwardPlan plan,
                List<Tensor> seeds, DerivativeConstants constants) {
            return ScanGradientRules.apply(plan, seeds, constants);
        }
        @Override public Tensor[] recurrent(TensorProducer producer, List<Tensor> seeds,
                boolean[] selected, DerivativeConstants constants) {
            return RecurrentGradientRules.apply(producer, seeds, selected, constants);
        }
    };

    /**
     * Keeps nonordinary formulas out of the ordinary body entry while sharing contribution math.
     * @param <V> direct operand type
     */
    private interface SpecialRules<V> {
        /**
         * Constructs a selected stochastic positional VJP.
         * @param occurrence checked selected output and inputs
         * @param gradient exact accumulated incoming
         * @param constants real request-local literal owner
         * @return input-position-aligned cotangents
         */
        V[] stochastic(AutogradPreflight.SelectedOccurrence occurrence,
                V gradient, DerivativeConstants constants);
        /**
         * Constructs one checked joint Scan VJP.
         * @param plan retained selected finite-time proof
         * @param seeds selected-port-aligned cotangents, with absent positions null
         * @param constants real request-local literal owner
         * @return original-input-position-aligned nullable cotangents
         */
        V[] scan(ScanGradientRules.BackwardPlan plan, List<V> seeds, DerivativeConstants constants);
        /** Joint canonical fixed-cell output seeds, preserving selected positional input roles. */
        V[] recurrent(TensorProducer producer, List<V> seeds, boolean[] selected,
                DerivativeConstants constants);
    }

    /**
     * Constructs one approved ordinary positional VJP through its existing formula owner.
     *
     * <p>The value type is Tensor itself for real construction and descriptor-only metadata for
     * checked construction. Both routes execute the same eager formula body. Source roles are
     * read directly for descriptor decisions and resolved only where the formula requests an
     * actual primal operand; unselected nullable positions and canonical saved slots are retained.
     * The caller owns preflight, original contribution identities and any synchronous observer.
     * Stochastic and joint Scan construction deliberately remain outside this entry.</p>
     *
     * @param <V> direct real or metadata operand type
     * @param owner existing preflight-selected ordinary family owner
     * @param producer canonical source occurrence with original positional role identities
     * @param outputIndex selected canonical output slot
     * @param gradient accumulated cotangent for that slot
     * @param selectedInputs input-position-aligned flags, not mutated
     * @param construction compile-local construction and exact literal owner
     * @param primal explicit lazy resolver of value operands and saved outputs
     * @return an actually typed input-position-aligned array, null for unselected roles
     * @throws IllegalStateException if the selected owner requires its separate joint/effect path
     */
    static <V> V[] applyOrdinary(
            FirstOrderGradientCoverage.FamilyOwner owner,
            OrdinaryConstruction.Source<V> producer, int outputIndex, V gradient,
            boolean[] selectedInputs, OrdinaryConstruction<V> construction,
            Function<V, V> primal) {
        return switch (owner) {
            case ELEMENTWISE -> construction.elementwise().apply(
                    producer, outputIndex, gradient, selectedInputs, primal);
            case REDUCTION -> construction.reduction().apply(
                    producer, outputIndex, gradient, selectedInputs, primal);
            case NORMALIZATION -> construction.normalization().apply(
                    producer, outputIndex, gradient, selectedInputs, primal);
            case LINEAR_ALGEBRA -> construction.linearAlgebra().apply(
                    producer, gradient, selectedInputs, primal);
            case ATTENTION -> construction.attention().apply(
                    producer, outputIndex, gradient, selectedInputs, primal);
            case CONVOLUTION -> construction.convolution().apply(
                    producer, gradient, selectedInputs, primal);
            case POOLING -> construction.pooling().apply(producer, gradient, primal);
            case LOSS -> construction.loss().apply(
                    producer, gradient, selectedInputs, primal);
            case LAYOUT -> construction.layout().apply(
                    producer, gradient, selectedInputs, primal);
            case INDEXING -> construction.indexing().apply(
                    producer, gradient, selectedInputs, primal);
            case ORDERING -> construction.ordering().apply(producer, gradient, primal);
            case STOCHASTIC, SCAN, RECURRENT -> throw new IllegalStateException(
                    "ordinary construction excludes stochastic, joint Scan and fixed recurrence rules");
        };
    }

    /**
     * Casts a selected positional term back before role normalization and deterministic append.
     * @param <V> direct operand type
     * @param contributions exact-source-identity contribution lists
     * @param target exact original target role
     * @param term present positional formula result
     * @param construction same operation owner used by the formula
     * @param normalize role-aware normalization before append
     */
    private static <V> void appendTerm(IdentityHashMap<Tensor, List<V>> contributions,
            Tensor target, V term, OrdinaryConstruction<V> construction,
            BiFunction<Tensor, V, V> normalize) {
        DataType type = target.descriptor().dataType();
        if (construction.descriptor(term).dataType() != type) {
            term = construction.cast(term, type);
        }
        append(contributions, target, normalize.apply(target, term));
    }

    /**
     * Appends one present term without deduplicating repeated identities or declaration positions.
     * @param <V> direct operand type
     * @param contributions source-identity-keyed ordered lists
     * @param tensor exact source role
     * @param contribution present normalized term
     */
    private static <V> void append(IdentityHashMap<Tensor, List<V>> contributions,
            Tensor tensor, V contribution) {
        if (contribution != null) {
            contributions.computeIfAbsent(tensor, ignored -> new ArrayList<>()).add(contribution);
        }
    }

    /**
     * Merges present positional terms in original order; no terms is absence, one is raw identity.
     * @param <V> direct operand type
     * @param contributions nullable source contribution list
     * @param construction operation owner shared with the formula
     * @param observer optional two-sided merge observation
     * @return absent null, sole exact value, or the ordered left-associated ADD result
     */
    private static <V> V accumulate(List<V> contributions, OrdinaryConstruction<V> construction,
            ContributionObserver<V> observer) {
        if (contributions == null || contributions.isEmpty()) return null;
        V result = contributions.getFirst();
        for (int index = 1; index < contributions.size(); index++) {
            V right = contributions.get(index);
            if (observer != null) observer.merging(result, right);
            try {
                V merged = construction.add(result, right);
                if (observer != null) observer.merge(result, right, merged);
                result = merged;
            } finally {
                if (observer != null) observer.leaving();
            }
        }
        return result;
    }

    /**
     * Request-local exact typed scalar-splat owner used by derivative rules.
     *
     * <p>At most one scalar base is created for each exact floating {@link ScalarValue} data-type
     * and represented-bit identity, in deterministic first-use order. This distinction preserves
     * signed zeros and NaN payloads. Each base is provenance-free, storage-free, unlabeled,
     * non-gradient Tensor metadata paired with one explicit logical-splat binding. Shape-specific
     * constants are ordinary public
     * {@link Tensor#expand(io.github.pho001.synaptik.model.shape.Shape) expand} expressions.</p>
     * <p>A closed body-local instance instead obtains each base through an explicit invariant
     * binding factory. Length-mode row-shaped literals and disconnected zeros use an explicit
     * typed sequence binding with its checked row axis; ordinary/all-valid expansion is unchanged.
     * It owns no outer ingress and never creates a free body constant leaf.</p>
     */
    static final class DerivativeConstants {
        private final Function<ScalarValue, Tensor> closedBase;
        private final Function<Tensor, Tensor> closedZero;
        private final Function<Tensor, Integer> originalRowRole;
        private final BiFunction<ScalarValue, Tensor, Tensor> closedLike;
        private final Map<ScalarValue, Tensor> bases = new HashMap<>();
        private final Map<DisconnectedZeroKey, Tensor> disconnectedZeros = new HashMap<>();
        private IdentityHashMap<TensorProducer, ScanGradientRules.ActivityDraft> scanActivityDrafts;
        private final List<CompileTimeConstantGraph.Binding> bindings = new ArrayList<>();

        /**
         * Creates one empty request-local cache with deterministic first-use binding order.
         */
        DerivativeConstants() {
            this.closedBase = null; this.closedZero = null; this.originalRowRole = null; this.closedLike = null;
        }

        /**
         * Routes scalar leaves through explicit non-gradient lexical invariant declarations.
         * @param closedBase non-null factory binding typed outer splats to local scalar formals
         */
        DerivativeConstants(Function<ScalarValue, Tensor> closedBase) {
            this(closedBase, null, null, null);
        }

        /**
         * Binds length-mode zeros as declared row sequences rather than inventing rows by expand.
         * @param closedBase non-null explicit scalar invariant binder
         * @param closedZero optional exact row-zero binder; null results use ordinary shared zero
         * @param originalRowRole optional checked original lexical roles for layout restoration
         * @param closedLike optional role-aware typed shape-literal binder, null results use expand
         */
        DerivativeConstants(Function<ScalarValue, Tensor> closedBase,
                Function<Tensor, Tensor> closedZero, Function<Tensor, Integer> originalRowRole,
                BiFunction<ScalarValue, Tensor, Tensor> closedLike) {
            this.closedBase = Objects.requireNonNull(closedBase);
            this.closedZero = closedZero;
            this.originalRowRole = originalRowRole;
            this.closedLike = closedLike;
        }

        /**
         * Resolves an explicitly declared shape literal without constructing an EXPAND.
         * Construction adapters use this same binding decision before naming an ordinary
         * EXPAND site, so descriptor-only proof and optional operation observation preserve
         * lexical ownership without creating or observing a fictitious constant leaf.
         *
         * @param value exact typed scalar literal
         * @param original exact original source role, not a resolved primal
         * @return explicit bound operand, or null when ordinary scalar expansion is required
         */
        Tensor boundLike(ScalarValue value, Tensor original) {
            return closedLike == null ? null : closedLike.apply(value, original);
        }

        /**
         * Supplies a checked original role to the bounded SUM restoration entry only.
         * @param original exact original body operand
         * @return explicit local row axis, or -1 for shared/ordinary all-valid dispatch
         */
        int originalRowRole(Tensor original) {
            return originalRowRole == null ? -1 : originalRowRole.apply(original);
        }

        /**
         * Returns an exact typed positive-zero expression with the supplied Tensor's Shape.
         *
         * @param tensor non-null floating Tensor whose exact data type and Shape are reused
         * @return a non-null public expand expression, or an explicitly bound row-zero formal
         */
        Tensor zeroLike(Tensor tensor) {
            return valueLike(scalar(tensor.descriptor().dataType(), false), tensor);
        }

        /**
         * Returns an exact typed positive-zero cotangent without consuming a disconnected primal.
         * Ordinary expressions are cached by descriptor; lexical ROW sequences retain their
         * explicit role and are not inferred or shared by equal shape alone.
         *
         * @param tensor non-null target whose exact floating type and Shape are reused
         * @return non-null request-local zero expression or explicitly bound row-zero formal;
         *     a lexical row binding is not cached by shape alone
         */
        Tensor disconnectedZeroLike(Tensor tensor) {
            return disconnectedZeroLike(tensor, null);
        }

        /**
         * Retains the existing disconnected-zero recipe while observing only newly created sites.
         * @param tensor exact original floating target
         * @param construction optional operation route; null keeps the direct ordinary route
         * @return the cached exact zero or explicitly declared lexical zero binding
         */
        Tensor disconnectedZeroLike(Tensor tensor, OrdinaryConstruction<Tensor> construction) {
            if (closedZero != null) {
                Tensor bound = closedZero.apply(tensor);
                if (bound != null) return bound;
            }
            DisconnectedZeroKey key = new DisconnectedZeroKey(
                    tensor.descriptor().dataType(), tensor.descriptor().shape());
            return disconnectedZeros.computeIfAbsent(
                    key,
                    ignored -> key.shape().equals(Shape.scalar())
                            ? zeroBase(key.dataType())
                            : construction == null
                                    ? zeroBase(key.dataType()).expand(key.shape())
                                    : construction.expand(zeroBase(key.dataType()), key.shape()));
        }

        /**
         * Returns an exact typed positive-one expression with the supplied Tensor's Shape.
         *
         * @param tensor non-null floating Tensor whose exact data type and Shape are reused
         * @return a non-null public expand expression, or an explicitly bound row-one formal
         */
        Tensor oneLike(Tensor tensor) {
            return valueLike(scalar(tensor.descriptor().dataType(), true), tensor);
        }

        /**
         * Returns an exact typed scalar-value expression with the supplied Tensor's Shape.
         *
         * @param value non-null floating scalar value whose exact typed bits identify the base
         * @param tensor non-null Tensor whose exact Shape is reused
         * @return a non-null public expand expression or explicitly bound typed row-literal formal
         * @throws IllegalArgumentException if {@code value} is not floating
         */
        Tensor valueLike(ScalarValue value, Tensor tensor) {
            Tensor bound = boundLike(value, tensor);
            return bound != null ? bound : base(value).expand(tensor.descriptor().shape());
        }

        /**
         * Returns the request-local scalar positive-zero leaf for one floating data type.
         *
         * @param dataType non-null floating type, including either OFP8 storage type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor zeroBase(DataType dataType) {
            return base(scalar(dataType, false));
        }

        /**
         * Returns the request-local scalar positive-one leaf for one floating data type.
         *
         * @param dataType non-null floating type, including either OFP8 storage type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor oneBase(DataType dataType) {
            return base(scalar(dataType, true));
        }

        /**
         * Returns the exact fixed typed scalar-operation value {@code 2}.
         *
         * @param dataType non-null floating type, including either OFP8 storage type
         * @return the exact represented coefficient selected by Compiler 0005A
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        static ScalarValue two(DataType dataType) {
            return switch (dataType) {
                case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) 0x40);
                case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) 0x40);
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x4000);
                case FLOAT16 -> ScalarValue.float16Bits((short) 0x4000);
                case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0x40000000));
                case FLOAT64 ->
                        ScalarValue.float64(Double.longBitsToDouble(0x4000000000000000L));
                case INT32, INT64, BOOL -> throw new IllegalArgumentException(
                        "derivative constants require floating data type: " + dataType);
            };
        }

        /**
         * Returns the exact fixed typed scalar-operation value {@code -2}.
         *
         * @param dataType non-null floating type, including either OFP8 storage type
         * @return the exact represented negative coefficient selected for loss formulas
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        static ScalarValue negativeTwo(DataType dataType) {
            return switch (dataType) {
                case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) 0xC0);
                case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) 0xC0);
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xC000);
                case FLOAT16 -> ScalarValue.float16Bits((short) 0xC000);
                case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0xC0000000));
                case FLOAT64 ->
                        ScalarValue.float64(Double.longBitsToDouble(0xC000000000000000L));
                case INT32, INT64, BOOL -> throw new IllegalArgumentException(
                        "derivative constants require floating data type: " + dataType);
            };
        }

        /**
         * Returns the exact fixed typed scalar-operation value {@code -0.5}.
         *
         * @param dataType non-null floating type, including either OFP8 storage type
         * @return the exact represented coefficient selected by Compiler 0005A
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        static ScalarValue negativeHalf(DataType dataType) {
            return switch (dataType) {
                case FLOAT8_E4M3 -> ScalarValue.float8E4m3Bits((byte) 0xB0);
                case FLOAT8_E5M2 -> ScalarValue.float8E5m2Bits((byte) 0xB8);
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xBF00);
                case FLOAT16 -> ScalarValue.float16Bits((short) 0xB800);
                case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0xBF000000));
                case FLOAT64 ->
                        ScalarValue.float64(Double.longBitsToDouble(0xBFE0000000000000L));
                case INT32, INT64, BOOL -> throw new IllegalArgumentException(
                        "derivative constants require floating data type: " + dataType);
            };
        }

        /**
         * Returns the request-local scalar leaf for one exact floating typed value.
         *
         * @param value non-null exact floating scalar, including either OFP8 storage type
         * @return the exact cached leaf, created and explicitly bound on first request
         * @throws IllegalArgumentException if {@code value} is not floating
         */
        Tensor base(ScalarValue value) {
            Objects.requireNonNull(value, "value");
            if (!value.dataType().isFloating()) {
                throw new IllegalArgumentException(
                        "derivative constants require floating data type: " + value.dataType());
            }
            return bases.computeIfAbsent(value, closedBase == null ? this::create : closedBase);
        }

        /**
         * Snapshots generated leaf bindings in deterministic creation order.
         *
         * @return a non-null immutable snapshot; later constant creation does not mutate it
         */
        List<CompileTimeConstantGraph.Binding> bindings() {
            return List.copyOf(bindings);
        }

        /**
         * Retains compile-local activity correspondence for one generated containing Scan.
         * @param producer exact newly generated containing occurrence, never a body-template key
         * @param draft ephemeral activity owner finished against that occurrence's captured body
         * @throws NullPointerException if either argument is null
         * @throws IllegalStateException if an occurrence already has a different draft
         */
        void recordScanActivity(TensorProducer producer, ScanGradientRules.ActivityDraft draft) {
            Objects.requireNonNull(producer, "producer");
            Objects.requireNonNull(draft, "draft");
            if (scanActivityDrafts == null) scanActivityDrafts = new IdentityHashMap<>();
            ScanGradientRules.ActivityDraft previous = scanActivityDrafts.putIfAbsent(producer, draft);
            if (previous != null && previous != draft) {
                throw new IllegalStateException("generated Scan already has an activity draft");
            }
        }

        /**
         * Snapshots activity drafts for immediate exact-occurrence combined capture.
         * @return immutable identity-keyed map; neither Tensor correspondence nor drafts are
         *     retained in final compiled metadata
         */
        Map<TensorProducer, ScanGradientRules.ActivityDraft> scanActivityDrafts() {
            return scanActivityDrafts == null ? Map.of()
                    : java.util.Collections.unmodifiableMap(new IdentityHashMap<>(scanActivityDrafts));
        }

        private Tensor create(ScalarValue value) {
            Tensor tensor = TensorFactory.create(new TensorDescriptor(
                    value.dataType(), Shape.scalar(), Optional.empty(), false));
            bindings.add(new CompileTimeConstantGraph.Binding(
                    tensor, new CompileTimeConstantGraph.Splat(value)));
            return tensor;
        }

        /**
         * Selects the existing exact represented positive-zero or positive-one coefficient.
         * No Tensor, producer, storage or identity is constructed by this bit-table lookup.
         * @param dataType non-null existing floating scalar type
         * @param one true for exact positive one, false for exact positive zero
         * @return exact typed scalar metadata shared by real and metadata literal ownership
         * @throws IllegalArgumentException if the data type is not floating
         */
        static ScalarValue scalar(DataType dataType, boolean one) {
            return switch (dataType) {
                case FLOAT8_E4M3 ->
                        ScalarValue.float8E4m3Bits((byte) (one ? 0x38 : 0x00));
                case FLOAT8_E5M2 ->
                        ScalarValue.float8E5m2Bits((byte) (one ? 0x3C : 0x00));
                case BFLOAT16 ->
                        ScalarValue.bfloat16Bits((short) (one ? 0x3F80 : 0x0000));
                case FLOAT16 ->
                        ScalarValue.float16Bits((short) (one ? 0x3C00 : 0x0000));
                case FLOAT32 -> ScalarValue.float32(one ? 1.0f : 0.0f);
                case FLOAT64 -> ScalarValue.float64(one ? 1.0d : 0.0d);
                case INT32, INT64, BOOL ->
                        throw new IllegalArgumentException(
                                "derivative constants require floating data type: " + dataType);
            };
        }

        private record DisconnectedZeroKey(DataType dataType, Shape shape) {}
    }

    /**
     * One ordered exact target-to-gradient Tensor role before combined capture.
     *
     * @param target non-null exact requested target Tensor reference
     * @param gradient non-null exact generated gradient-root Tensor reference
     */
    record TargetGradient(Tensor target, Tensor gradient) {
        /**
         * Validates one target-to-gradient role.
         *
         * @param target non-null exact requested target Tensor reference
         * @param gradient non-null exact generated gradient-root Tensor reference
         * @throws NullPointerException if either component is {@code null}
         */TargetGradient {
            Objects.requireNonNull(target, "target");
            Objects.requireNonNull(gradient, "gradient");
        }
    }

    /**
     * Complete ephemeral expansion result consumed immediately by combined capture.
     *
     * @param targetGradients non-null ordered target-to-gradient roles; snapshotted
     * @param stageProducers non-null unmodifiable identity set of producers first owned by this
     *     derivative stage
     * @param ingress non-null caller bindings followed by generated derivative bindings
     * @param scanActivityDrafts ephemeral generated-containing-Scan identity-keyed capture drafts
     */
    record Expansion(
            List<TargetGradient> targetGradients,
            Set<TensorProducer> stageProducers,
            CompileTimeConstantGraph.Ingress ingress,
            Map<TensorProducer, ScanGradientRules.ActivityDraft> scanActivityDrafts) {
        /**
         * Creates a genuinely unannotated ordinary expansion.
         * @param targetGradients exact ordered target roles
         * @param stageProducers exact generated producer identities
         * @param ingress ordered caller and derivative leaf bindings
         */
        Expansion(List<TargetGradient> targetGradients, Set<TensorProducer> stageProducers,
                CompileTimeConstantGraph.Ingress ingress) {
            this(targetGradients, stageProducers, ingress, Map.of());
        }
        /**
         * Validates and snapshots one complete expansion result.
         *
         * @param targetGradients non-null ordered target-to-gradient roles
         * @param stageProducers non-null identity set first owned by this derivative stage
         * @param ingress non-null caller and derivative constant ingress
         * @param scanActivityDrafts non-null ephemeral identity-keyed containing-Scan drafts
         * @throws NullPointerException if a required component or role element is {@code null}
        */Expansion {
            targetGradients = List.copyOf(targetGradients);
            Objects.requireNonNull(stageProducers, "stageProducers");
            Objects.requireNonNull(ingress, "ingress");
            Objects.requireNonNull(scanActivityDrafts, "scanActivityDrafts");
            scanActivityDrafts = scanActivityDrafts.isEmpty() ? Map.of()
                    : java.util.Collections.unmodifiableMap(new IdentityHashMap<>(scanActivityDrafts));
        }
    }
}
