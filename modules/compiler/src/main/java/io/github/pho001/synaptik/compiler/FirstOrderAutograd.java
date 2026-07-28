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

/**
 * Constructs one deterministic first-order Tensor expression after successful preflight.
 *
 * <p>Incoming contributions and accumulated cotangents are keyed by exact Tensor identity.
 * Contributions are appended in reverse producer-postorder, ascending selected-output-slot, and
 * ascending input-position order, then merged through left-associated public Tensor addition.
 * This keeps existing one-output order stable while allowing several selected public outputs of
 * one batch-normalization occurrence to contribute to the same input. The only seed is an
 * implicit exact typed positive one for the scalar objective. Request-local BFLOAT16, FLOAT32, or
 * FLOAT64 scalar leaves are cached by exact {@link ScalarValue} type and represented bits, remain
 * storage-free, and are registered explicitly as logical splats in deterministic first-use
 * order. Shape-specific values are ordinary public {@code expand} expressions.</p>
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
 * <p>The returned Tensor roles and original-producer identities are an ephemeral handoff to one
 * combined capture. Generated logical-splat bindings, including extrema or clamp bounds needed
 * as Tensor comparison operands, are retained only when reachable from a returned gradient role.
 * A direct local-zero result therefore does not expose an otherwise unreachable seed binding as
 * graph ingress. This owner does not retain global state, mutate an original
 * Tensor, infer a constant from storage or provenance, capture or validate a graph, materialize
 * values, lower work, or execute computation.</p>
 */
final class FirstOrderAutograd {
    private FirstOrderAutograd() {}

    /**
     * Expands one successful plan and merges generated typed constants after caller ingress.
     *
     * @param plan non-null successful Tensor-allocation-free preflight plan
     * @param forwardIngress non-null caller-ordered explicit forward constant bindings; observed
     *     but not mutated
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
            AutogradPreflight.Plan plan,
            CompileTimeConstantGraph.Ingress forwardIngress) {
        Objects.requireNonNull(plan, "plan");
        Objects.requireNonNull(forwardIngress, "forwardIngress");

        DerivativeConstants constants = new DerivativeConstants();
        IdentityHashMap<Tensor, List<Tensor>> contributions = new IdentityHashMap<>();
        IdentityHashMap<Tensor, Tensor> accumulated = new IdentityHashMap<>();
        append(contributions, plan.objective(), constants.oneBase(
                plan.objective().descriptor().dataType()));

        List<AutogradPreflight.SelectedOccurrence> selected = plan.selectedOccurrences();
        for (int end = selected.size(); end > 0; ) {
            int start = end - 1;
            int postorderIndex = selected.get(start).postorderIndex();
            while (start > 0
                    && selected.get(start - 1).postorderIndex() == postorderIndex) {
                start--;
            }
            for (int index = start; index < end; index++) {
                AutogradPreflight.SelectedOccurrence occurrence = selected.get(index);
                TensorProducer producer = occurrence.producer();
                Tensor output = producer.output(occurrence.outputIndex());
                List<Tensor> incoming = contributions.get(output);
                if (incoming == null || incoming.isEmpty()) {
                    continue;
                }
                Tensor gradient = accumulate(incoming);
                accumulated.put(output, gradient);
                Tensor[] inputGradients = apply(occurrence, gradient, constants);
                for (int input = 0; input < producer.inputs().size(); input++) {
                    if (!occurrence.selectedInput(input)) {
                        continue;
                    }
                    Tensor inputGradient = inputGradients[input];
                    if (inputGradient == null) {
                        throw new IllegalStateException(
                                "preflight selected a non-differentiable input role " + input);
                    }
                    append(contributions, producer.inputs().get(input), inputGradient);
                }
            }
            end = start;
        }

        List<TargetGradient> roles = new ArrayList<>(plan.targets().size());
        for (Tensor target : plan.targets()) {
            Tensor gradient = accumulated.get(target);
            if (gradient == null) {
                List<Tensor> incoming = contributions.get(target);
                if (incoming == null || incoming.isEmpty()) {
                    throw new IllegalStateException(
                            "successful preflight target has no gradient contribution");
                }
                gradient = accumulate(incoming);
                accumulated.put(target, gradient);
            }
            roles.add(new TargetGradient(target, gradient));
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
                plan.originalProducers(),
                new CompileTimeConstantGraph.Ingress(merged));
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

    private static Tensor[] apply(
            AutogradPreflight.SelectedOccurrence occurrence,
            Tensor gradient,
            DerivativeConstants constants) {
        TensorProducer producer = occurrence.producer();
        return switch (occurrence.familyOwner()) {
            case ELEMENTWISE -> ElementwiseGradientRules.apply(
                    producer,
                    occurrence.outputIndex(),
                    gradient,
                    occurrence.selectedInputs(),
                    constants);
            case REDUCTION -> ReductionGradientRules.apply(
                    producer,
                    occurrence.outputIndex(),
                    gradient,
                    occurrence.selectedInputs(),
                    constants);
            case NORMALIZATION -> NormalizationGradientRules.apply(
                    producer,
                    occurrence.outputIndex(),
                    gradient,
                    occurrence.selectedInputs(),
                    constants);
            case LINEAR_ALGEBRA -> LinearAlgebraGradientRules.apply(
                    producer, gradient, occurrence.selectedInputs());
            case ATTENTION -> AttentionGradientRules.apply(
                    producer,
                    occurrence.outputIndex(),
                    gradient,
                    occurrence.selectedInputs(),
                    constants);
            case CONVOLUTION -> ConvolutionGradientRules.apply(
                    producer, gradient, occurrence.selectedInputs());
            case POOLING -> PoolingGradientRules.apply(producer, gradient, constants);
            case LOSS -> LossGradientRules.apply(
                    producer, gradient, occurrence.selectedInputs(), constants);
            case LAYOUT -> LayoutGradientRules.apply(
                    producer, gradient, occurrence.selectedInputs(), constants);
            case INDEXING -> IndexingGradientRules.apply(
                    producer, gradient, occurrence.selectedInputs(), constants);
            case ORDERING -> OrderingGradientRules.apply(producer, gradient, constants);
            case STOCHASTIC -> StochasticGradientRules.apply(producer, gradient, constants);
        };
    }

    private static void append(
            IdentityHashMap<Tensor, List<Tensor>> contributions,
            Tensor tensor,
            Tensor contribution) {
        contributions.computeIfAbsent(tensor, ignored -> new ArrayList<>()).add(contribution);
    }

    private static Tensor accumulate(List<Tensor> contributions) {
        Tensor result = contributions.getFirst();
        for (int index = 1; index < contributions.size(); index++) {
            result = result.add(contributions.get(index));
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
     */
    static final class DerivativeConstants {
        private final Map<ScalarValue, Tensor> bases = new HashMap<>();
        private final List<CompileTimeConstantGraph.Binding> bindings = new ArrayList<>();

        /**
         * Creates one empty request-local cache with deterministic first-use binding order.
         */
        DerivativeConstants() {}

        /**
         * Returns an exact typed positive-zero expression with the supplied Tensor's Shape.
         *
         * @param tensor non-null floating Tensor whose exact data type and Shape are reused
         * @return a non-null public expand expression rooted at the request-local base zero
         */
        Tensor zeroLike(Tensor tensor) {
            return zeroBase(tensor.descriptor().dataType())
                    .expand(tensor.descriptor().shape());
        }

        /**
         * Returns an exact typed positive-one expression with the supplied Tensor's Shape.
         *
         * @param tensor non-null floating Tensor whose exact data type and Shape are reused
         * @return a non-null public expand expression rooted at the request-local base one
         */
        Tensor oneLike(Tensor tensor) {
            return oneBase(tensor.descriptor().dataType())
                    .expand(tensor.descriptor().shape());
        }

        /**
         * Returns an exact typed scalar-value expression with the supplied Tensor's Shape.
         *
         * @param value non-null floating scalar value whose exact typed bits identify the base
         * @param tensor non-null Tensor whose exact Shape is reused
         * @return a non-null public expand expression rooted at the request-local exact base
         * @throws IllegalArgumentException if {@code value} is not floating
         */
        Tensor valueLike(ScalarValue value, Tensor tensor) {
            return base(value).expand(tensor.descriptor().shape());
        }

        /**
         * Returns the request-local scalar positive-zero leaf for one floating data type.
         *
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor zeroBase(DataType dataType) {
            return base(scalar(dataType, false));
        }

        /**
         * Returns the request-local scalar positive-one leaf for one floating data type.
         *
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor oneBase(DataType dataType) {
            return base(scalar(dataType, true));
        }

        /**
         * Returns the exact fixed typed scalar-operation value {@code 2}.
         *
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact represented coefficient selected by Compiler 0005A
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        ScalarValue two(DataType dataType) {
            return switch (dataType) {
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x4000);
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
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact represented negative coefficient selected for loss formulas
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        ScalarValue negativeTwo(DataType dataType) {
            return switch (dataType) {
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xC000);
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
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact represented coefficient selected by Compiler 0005A
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        ScalarValue negativeHalf(DataType dataType) {
            return switch (dataType) {
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0xBF00);
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
         * @param value non-null exact BFLOAT16, FLOAT32, or FLOAT64 scalar
         * @return the exact cached leaf, created and explicitly bound on first request
         * @throws IllegalArgumentException if {@code value} is not floating
         */
        Tensor base(ScalarValue value) {
            Objects.requireNonNull(value, "value");
            if (!value.dataType().isFloating()) {
                throw new IllegalArgumentException(
                        "derivative constants require floating data type: " + value.dataType());
            }
            return bases.computeIfAbsent(value, this::create);
        }

        /**
         * Snapshots generated leaf bindings in deterministic creation order.
         *
         * @return a non-null immutable snapshot; later constant creation does not mutate it
         */
        List<CompileTimeConstantGraph.Binding> bindings() {
            return List.copyOf(bindings);
        }

        private Tensor create(ScalarValue value) {
            Tensor tensor = TensorFactory.create(new TensorDescriptor(
                    value.dataType(), Shape.scalar(), Optional.empty(), false));
            bindings.add(new CompileTimeConstantGraph.Binding(
                    tensor, new CompileTimeConstantGraph.Splat(value)));
            return tensor;
        }

        private static ScalarValue scalar(DataType dataType, boolean one) {
            return switch (dataType) {
                case BFLOAT16 ->
                        ScalarValue.bfloat16Bits((short) (one ? 0x3F80 : 0x0000));
                case FLOAT32 -> ScalarValue.float32(one ? 1.0f : 0.0f);
                case FLOAT64 -> ScalarValue.float64(one ? 1.0d : 0.0d);
                case INT32, INT64, BOOL ->
                        throw new IllegalArgumentException(
                                "derivative constants require floating data type: " + dataType);
            };
        }
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
     * @param originalProducers non-null unmodifiable identity set of original forward producers
     * @param ingress non-null caller bindings followed by generated derivative bindings
     */
    record Expansion(
            List<TargetGradient> targetGradients,
            Set<TensorProducer> originalProducers,
            CompileTimeConstantGraph.Ingress ingress) {
        /**
         * Validates and snapshots one complete expansion result.
         *
         * @param targetGradients non-null ordered target-to-gradient roles
         * @param originalProducers non-null identity set of original forward producers
         * @param ingress non-null caller and derivative constant ingress
         * @throws NullPointerException if a required component or role element is {@code null}
         */Expansion {
            targetGradients = List.copyOf(targetGradients);
            Objects.requireNonNull(originalProducers, "originalProducers");
            Objects.requireNonNull(ingress, "ingress");
        }
    }
}
