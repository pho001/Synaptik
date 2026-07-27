package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.ArrayList;
import java.util.EnumMap;
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
 * Contributions are appended in reverse producer-postorder and input-position order, then merged
 * through left-associated public Tensor addition. The only seed is an implicit exact typed
 * positive one for the scalar objective. Request-local BFLOAT16, FLOAT32, or FLOAT64 zero/one
 * leaves are storage-free and are registered explicitly as logical splats.</p>
 *
 * <p>The returned Tensor roles and original-producer identities are an ephemeral handoff to one
 * combined capture. This owner does not retain global state, mutate an original Tensor, infer a
 * constant from storage or provenance, capture or validate a graph, materialize values, lower
 * work, or execute computation.</p>
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
     *     producer identity set, and ingress ordered as caller bindings followed by generated
     *     derivative bindings
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
        for (int index = selected.size() - 1; index >= 0; index--) {
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
        for (CompileTimeConstantGraph.Binding binding : constants.bindings()) {
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

    private static Tensor[] apply(
            AutogradPreflight.SelectedOccurrence occurrence,
            Tensor gradient,
            DerivativeConstants constants) {
        TensorProducer producer = occurrence.producer();
        var kind = producer.operation().kind();
        if (kind instanceof BinaryArithmeticKind
                || kind instanceof ScalarElementwiseKind
                || kind instanceof UnaryElementwiseKind
                || kind == WhereSelectionKind.WHERE
                || kind == CastKind.CAST) {
            return ElementwiseGradientRules.apply(
                    producer,
                    occurrence.outputIndex(),
                    gradient,
                    occurrence.selectedInputs(),
                    constants);
        }
        if (kind == AggregateReductionKind.SUM || kind == CumulativeScanKind.CUM_SUM) {
            return ReductionGradientRules.apply(producer, gradient);
        }
        if (kind == ContiguousKind.CONTIGUOUS
                || kind instanceof ShapeTransformKind
                || kind instanceof AxisTransformKind) {
            return LayoutGradientRules.apply(producer, gradient);
        }
        throw new IllegalStateException("operation was not preflight-approved: " + kind);
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
     * Request-local exact typed zero/one leaf owner used by derivative rules.
     *
     * <p>At most one scalar base zero and one scalar base one are created per floating data type,
     * in deterministic first-use order. BFLOAT16 uses exact bits {@code 0x0000} and
     * {@code 0x3F80}; FLOAT32 and FLOAT64 use their exact positive zero and one values. Each base
     * is provenance-free, storage-free, unlabeled, non-gradient Tensor metadata paired with one
     * explicit logical-splat binding. Shape-specific constants are ordinary public
     * {@link Tensor#expand(io.github.pho001.synaptik.model.shape.Shape) expand} expressions.</p>
     */
    static final class DerivativeConstants {
        private final Map<DataType, Tensor> zeros = new EnumMap<>(DataType.class);
        private final Map<DataType, Tensor> ones = new EnumMap<>(DataType.class);
        private final List<CompileTimeConstantGraph.Binding> bindings = new ArrayList<>();

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
         * Returns the request-local scalar positive-zero leaf for one floating data type.
         *
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor zeroBase(DataType dataType) {
            return zeros.computeIfAbsent(dataType, type -> create(type, false));
        }

        /**
         * Returns the request-local scalar positive-one leaf for one floating data type.
         *
         * @param dataType non-null BFLOAT16, FLOAT32, or FLOAT64 type
         * @return the exact cached leaf, created and bound on first request
         * @throws IllegalArgumentException if {@code dataType} is not floating
         */
        Tensor oneBase(DataType dataType) {
            return ones.computeIfAbsent(dataType, type -> create(type, true));
        }

        /**
         * Snapshots generated leaf bindings in deterministic creation order.
         *
         * @return a non-null immutable snapshot; later constant creation does not mutate it
         */
        List<CompileTimeConstantGraph.Binding> bindings() {
            return List.copyOf(bindings);
        }

        private Tensor create(DataType dataType, boolean one) {
            ScalarValue value = scalar(dataType, one);
            Tensor tensor = TensorFactory.create(new TensorDescriptor(
                    dataType, Shape.scalar(), Optional.empty(), false));
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
