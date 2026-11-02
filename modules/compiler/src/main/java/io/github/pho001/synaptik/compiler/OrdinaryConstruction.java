package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.ToIntFunction;

/**
 * Compile-local construction boundary for the existing ordinary derivative formulas.
 *
 * <p>Real operands are Tensor references themselves. Metadata operands contain descriptors and
 * explicit literal facts, never Tensor identities, producers, storage or graph-local IDs. The
 * family owners contain the only derivative formulas; this adapter only constructs the ordinary
 * Model operations named by those formulas. Metadata completion checks every canonical sibling
 * through Model and independently through the existing Compiler inference dispatch.</p>
 *
 * <p>Source-role operands remain distinct from values returned by a primal resolver. Reading a
 * source descriptor does not resolve a primal. An observer is synchronous and optional; nothing
 * is retained on its behalf, and the unobserved real route allocates no observation objects.
 * There is no runtime consumer, numerical evaluator or Scan-admission decision here.</p>
 *
 * <p>An instance belongs to one synchronous compile-local construction context and is not shared
 * between concurrent requests. Existing family owners are retained only after first use.</p>
 *
 * @param <V> direct Tensor or descriptor-only operand type
 */
abstract class OrdinaryConstruction<V> {
    private ElementwiseGradientRules<V> elementwise;
    private ReductionGradientRules<V> reduction;
    private NormalizationGradientRules<V> normalization;
    private LinearAlgebraGradientRules<V> linearAlgebra;
    private AttentionGradientRules<V> attention;
    private ConvolutionGradientRules<V> convolution;
    private PoolingGradientRules<V> pooling;
    private LossGradientRules<V> loss;
    private LayoutGradientRules<V> layout;
    private IndexingGradientRules<V> indexing;
    private OrderingGradientRules<V> ordering;

    /** Creates one compile-local construction context; every family owner stays lazy. */
    OrdinaryConstruction() {}

    /**
     * Obtains the existing Elementwise formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final ElementwiseGradientRules<V> elementwise() {
        if (elementwise == null) elementwise = new ElementwiseGradientRules<>(this);
        return elementwise;
    }

    /**
     * Obtains the existing Reduction formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final ReductionGradientRules<V> reduction() {
        if (reduction == null) reduction = new ReductionGradientRules<>(this);
        return reduction;
    }

    /**
     * Obtains the existing Normalization formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final NormalizationGradientRules<V> normalization() {
        if (normalization == null) normalization = new NormalizationGradientRules<>(this);
        return normalization;
    }

    /**
     * Obtains the existing LinearAlgebra formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final LinearAlgebraGradientRules<V> linearAlgebra() {
        if (linearAlgebra == null) linearAlgebra = new LinearAlgebraGradientRules<>(this);
        return linearAlgebra;
    }

    /**
     * Obtains the existing Attention formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final AttentionGradientRules<V> attention() {
        if (attention == null) attention = new AttentionGradientRules<>(this);
        return attention;
    }

    /**
     * Obtains the existing Convolution formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final ConvolutionGradientRules<V> convolution() {
        if (convolution == null) convolution = new ConvolutionGradientRules<>(this);
        return convolution;
    }

    /**
     * Obtains the existing Pooling formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final PoolingGradientRules<V> pooling() {
        if (pooling == null) pooling = new PoolingGradientRules<>(this);
        return pooling;
    }

    /**
     * Obtains the existing Loss formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final LossGradientRules<V> loss() {
        if (loss == null) loss = new LossGradientRules<>(this);
        return loss;
    }

    /**
     * Obtains the existing Layout formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final LayoutGradientRules<V> layout() {
        if (layout == null) layout = new LayoutGradientRules<>(this);
        return layout;
    }

    /**
     * Obtains the existing Indexing formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final IndexingGradientRules<V> indexing() {
        if (indexing == null) indexing = new IndexingGradientRules<>(this);
        return indexing;
    }

    /**
     * Obtains the existing Ordering formula owner only when that family is used.
     * @return the request-local owner bound to this construction route
     */
    final OrderingGradientRules<V> ordering() {
        if (ordering == null) ordering = new OrderingGradientRules<>(this);
        return ordering;
    }

    /**
     * Describes an original occurrence without resolving any of its value operands.
     * @param <V> direct operand type
     */
    interface Source<V> {
        /**
         * Reads the source operation without resolving primal values.
         * @return the exact canonical source operation
         */
        Operation operation();
        /**
         * Reads original operand roles without resolving primal values.
         * @return original operands in source position order, including repeated identities
         */
        List<V> inputs();
        /**
         * Reads an original canonical output role without demanding its value.
         * @param slot canonical output slot
         * @return that exact original output role
         */
        V output(int slot);
    }

    /**
     * Receives each completed ordinary operation before its caller continues construction.
     * @param <V> direct operand type
     */
    interface Observer<V> {
        /**
         * Observes one occurrence, including eager results not used by the final cotangent.
         * @param operation complete canonical ordinary operation
         * @param inputs ordered exact operand references
         * @param outputs all canonical siblings in slot order
         * @param deferred independent binding-free obligations, empty for real observation
         */
        void created(Operation operation, List<V> inputs, List<V> outputs,
                List<CapturedGraphInference.ConstraintRequest> deferred);
    }

    /**
     * Adapts one real source occurrence without wrapping its individual Tensor operands.
     * @param producer exact original occurrence
     * @return an occurrence-local source view preserving all operand/output identities
     */
    static Source<Tensor> source(TensorProducer producer) {
        Objects.requireNonNull(producer, "producer");
        return new Source<>() {
            @Override public Operation operation() { return producer.operation(); }
            @Override public List<Tensor> inputs() { return producer.inputs(); }
            @Override public Tensor output(int slot) { return producer.output(slot); }
        };
    }

    /**
     * Creates a descriptor-only source occurrence with complete original sibling roles.
     * @param operation canonical source operation
     * @param inputs original input roles, snapshotted without resolving primals
     * @param outputs complete original canonical output roles, snapshotted
     * @return source view retaining exact role references
     */
    static Source<Value> source(Operation operation, List<Value> inputs, List<Value> outputs) {
        Objects.requireNonNull(operation, "operation");
        List<Value> inputRoles = List.copyOf(inputs);
        List<Value> outputRoles = List.copyOf(outputs);
        List<TensorDescriptor> inputDescriptors = inputRoles.stream().map(Value::descriptor).toList();
        List<TensorDescriptor> outputDescriptors = outputRoles.stream().map(Value::descriptor).toList();
        List<TensorDescriptor> checked = ScanBody.describeOrdinary(
                operation, inputDescriptors, outputRoles.size());
        if (!checked.equals(outputDescriptors)) {
            throw new IllegalArgumentException("source canonical sibling descriptors disagree");
        }
        CapturedGraphInference.checkOrdinary(operation, inputDescriptors, outputDescriptors);
        return new Source<>() {
            @Override public Operation operation() { return operation; }
            @Override public List<Value> inputs() { return inputRoles; }
            @Override public Value output(int slot) { return outputRoles.get(slot); }
        };
    }

    /**
     * Creates the real route for an existing request-local or closed lexical constant owner.
     * @param constants exact existing owner, retaining its binding and first-use order
     * @param observer nullable synchronous operation observer
     * @return compile-local direct-Tensor construction
     */
    static OrdinaryConstruction<Tensor> real(FirstOrderAutograd.DerivativeConstants constants,
            Observer<Tensor> observer) {
        return new Real(constants, observer);
    }

    /**
     * Creates a checked metadata route without constructing any Tensor or identity.
     * @param literal optional scalar-role binder retaining the requested exact literal fact;
     *     null uses exact typed metadata literals
     * @param like optional source-role-aware literal binder retaining the exact literal fact and
     *     source Shape; null results use ordinary EXPAND
     * @param rowRole optional original-source row-role lookup; null denotes ordinary/shared roles
     * @param observer nullable synchronous observer, including primal-only temporary construction
     * @return compile-local metadata construction with exact-bit scalar caching
     */
    static OrdinaryConstruction<Value> metadata(Function<ScalarValue, Value> literal,
            BiFunction<ScalarValue, Value, Value> like, ToIntFunction<Value> rowRole,
            Observer<Value> observer) {
        return new Metadata(literal, like, rowRole, observer);
    }

    /**
     * A descriptor-only operand; literal presence is an explicit exact logical-splat source fact.
     * Scalar bases and declared shape literals retain their represented type and bits explicitly.
     * Identity-sensitive clients must use reference identity, not descriptor or record equality.
     * @param descriptor immutable ordinary descriptor
     * @param literal exact literal role, empty for nonliteral source operands and operation results
     */
    record Value(TensorDescriptor descriptor, Optional<ScalarValue> literal) {
        /**
         * Retains one ordinary descriptor and an optional explicit exact literal-splat fact.
         * @param descriptor non-null immutable ordinary descriptor
         * @param literal non-null exact literal role, or empty for nonliteral operands
         * @throws NullPointerException if either component is null
         * @throws IllegalArgumentException if a literal has the wrong type or requests gradients
         */
        Value {
            Objects.requireNonNull(descriptor, "descriptor");
            Objects.requireNonNull(literal, "literal");
            if (literal.isPresent() && (descriptor.dataType() != literal.orElseThrow().dataType()
                    || descriptor.requiresGrad())) {
                throw new IllegalArgumentException("literal role requires exact type and no gradients");
            }
        }
        /**
         * Creates a nonliteral metadata operand without a Tensor or identity.
         * @param descriptor non-null original ordinary descriptor
         */
        Value(TensorDescriptor descriptor) { this(descriptor, Optional.empty()); }
    }

    /**
     * Reads descriptor metadata without demanding or resolving a primal.
     *
     * @param value non-null exact typed operand or literal value
     * @return the exact immutable descriptor
     */
    abstract TensorDescriptor descriptor(V value);
    /**
     * Adapts complete original occurrence roles for the shared contribution engine.
     * Real identity mapping reuses the producer's operand list without operand wrappers or copies;
     * metadata mapping independently checks every canonical source sibling before formula use.
     *
     * @param producer exact original source occurrence
     * @param role total identity-preserving source Tensor to direct operand mapping
     * @return source view that never resolves a primal
     * @throws IllegalArgumentException if metadata canonical source descriptors disagree
     */
    abstract Source<V> sourceRoles(TensorProducer producer, Function<Tensor, V> role);
    /**
     * Publishes a disconnected positive zero without resolving a primal or changing ordinary
     * recipe identity. Exact scalar targets reuse the scalar base; other shared zeros retain the
     * existing descriptor-keyed cache and one EXPAND site. Explicit lexical zero bindings take
     * precedence and are never inferred from an equal descriptor.
     *
     * @param original exact original floating target role, not a resolved primal
     * @return exact typed cached zero or explicitly declared lexical zero binding
     * @throws IllegalArgumentException if the target type is not floating
     */
    abstract V disconnectedZeroLike(V original);
    /**
     * Allocates positional result storage with its actual operand-array component type.
     *
     * @param size non-negative positional result count
     * @return a new correctly typed array whose elements are initially null
     */
    abstract V[] array(int size);
    /**
     * Obtains an explicitly owned scalar base cached by exact represented type and bits.
     *
     * @param value non-null exact typed operand or literal value
     * @return the exact cached real leaf or declared metadata literal role
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V base(ScalarValue value);
    /**
     * Requests an explicitly declared shape literal without constructing an expansion.
     *
     * @param value non-null exact typed operand or literal value
     * @param original non-null original source role; descriptor access does not resolve its primal
     * @return the bound operand, or null when ordinary scalar expansion is required
     */
    abstract V boundLike(ScalarValue value, V original);
    /**
     * Reads the existing checked source role without resolving a value.
     *
     * @param original non-null original source role; descriptor access does not resolve its primal
     * @return the declared row axis, or -1 for the ordinary/shared route
     */
    abstract int originalRowRole(V original);
    /**
     * Constructs the ordinary unary operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param input non-null direct operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unary(UnaryElementwiseKind kind, V input);
    /**
     * Constructs the ordinary binary operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param left non-null left operand
     * @param right non-null right operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V binary(BinaryArithmeticKind kind, V left, V right);
    /**
     * Constructs the ordinary scalar operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param input non-null direct operand
     * @param value non-null exact typed operand or literal value
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V scalar(ScalarElementwiseKind kind, V input, ScalarValue value);
    /**
     * Constructs the ordinary comparison operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param left non-null left operand
     * @param right non-null right operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V comparison(BinaryComparisonKind kind, V left, V right);
    /**
     * Constructs the ordinary logical operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param left non-null left operand
     * @param right non-null right operand for AND/OR; null only for unary NOT
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V logical(BooleanLogicalKind kind, V left, V right);
    /**
     * Constructs the ordinary classification operation synchronously on the selected route.
     *
     * @param kind existing ordinary operation selected by the formula, not an admission decision
     * @param input non-null direct operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V classification(FloatingClassificationKind kind, V input);
    /**
     * Constructs the ordinary where operation synchronously on the selected route.
     *
     * @param condition non-null exact Boolean selector
     * @param ifTrue non-null selected true branch
     * @param ifFalse non-null selected false branch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V where(V condition, V ifTrue, V ifFalse);
    /**
     * Constructs the ordinary cast operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param dataType exact target data type
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V cast(V input, DataType dataType);
    /**
     * Constructs the ordinary matmul operation synchronously on the selected route.
     *
     * @param left non-null left operand
     * @param right non-null right operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V matmul(V left, V right);
    /**
     * Constructs the ordinary reshape operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param shape non-null exact logical target Shape
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V reshape(V input, Shape shape);
    /**
     * Constructs the ordinary expand operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param shape non-null exact logical target Shape
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V expand(V input, Shape shape);
    /**
     * Constructs the ordinary contiguous operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V contiguous(V input);
    /**
     * Constructs the ordinary expandDims operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V expandDims(V input, int axis);
    /**
     * Constructs the ordinary squeeze operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V squeeze(V input, int axis);
    /**
     * Constructs the ordinary permute operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axes non-null canonical ordered axes, observed but not mutated
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V permute(V input, int... axes);
    /**
     * Constructs the ordinary sum operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sum(V input);
    /**
     * Constructs the ordinary sum operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param keep whether reduced axes retain singleton extents
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sum(V input, int axis, boolean keep);
    /**
     * Constructs the ordinary sum operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axes non-null canonical ordered axes, observed but not mutated
     * @param keep whether reduced axes retain singleton extents
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sum(V input, int[] axes, boolean keep);
    /**
     * Constructs the ordinary mean operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axes non-null canonical ordered axes, observed but not mutated
     * @param keep whether reduced axes retain singleton extents
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V mean(V input, int[] axes, boolean keep);
    /**
     * Constructs the ordinary prod operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param keep whether reduced axes retain singleton extents
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V prod(V input, int axis, boolean keep);
    /**
     * Constructs the ordinary sumToShape operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param shape non-null exact logical target Shape
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sumToShape(V input, Shape shape);
    /**
     * Constructs the ordinary cumSum operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param exclusive existing exclusive-scan attribute
     * @param reverse existing traversal-direction attribute
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V cumSum(V input, int axis, boolean exclusive, boolean reverse);
    /**
     * Constructs the ordinary cumProd operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param exclusive existing exclusive-scan attribute
     * @param reverse existing traversal-direction attribute
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V cumProd(V input, int axis, boolean exclusive, boolean reverse);
    /**
     * Constructs the ordinary softmax operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V softmax(V input, int axis);
    /**
     * Constructs the ordinary logSoftmax operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V logSoftmax(V input, int axis);
    /**
     * Constructs the ordinary gather operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V gather(V input, V indices, int axis);
    /**
     * Constructs the ordinary gatherElements operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V gatherElements(V input, V indices, int axis);
    /**
     * Constructs the ordinary gatherNd operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param batch canonical batch-prefix dimension count
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V gatherNd(V input, V indices, int batch);
    /**
     * Constructs the ordinary scatterAdd operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param updates non-null position-aligned update operand
     * @param axis canonical non-negative axis supplied by the formula
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V scatterAdd(V input, V indices, V updates, int axis);
    /**
     * Constructs the ordinary scatterElements operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param updates non-null position-aligned update operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param reduction existing exact scatter reduction policy
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V scatterElements(V input, V indices, V updates, int axis, ScatterReduction reduction);
    /**
     * Constructs the ordinary scatterNd operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param indices non-null resolved exact discrete routing operand
     * @param updates non-null position-aligned update operand
     * @param reduction existing exact scatter reduction policy
     * @param batch canonical batch-prefix dimension count
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V scatterNd(V input, V indices, V updates, ScatterReduction reduction, int batch);
    /**
     * Constructs the ordinary oneHot operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param depth positive canonical class depth
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V oneHot(V input, long depth);
    /**
     * Constructs the ordinary argsort operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param descending existing source sort direction
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V argsort(V input, int axis, boolean descending);
    /**
     * Constructs the ordinary argMax operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param keep whether reduced axes retain singleton extents
     * @param ties existing exact tie-index policy
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V argMax(V input, int axis, boolean keep, ArgExtremaTiePolicy ties);
    /**
     * Constructs the ordinary select operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param index canonical non-negative selected position
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V select(V input, int axis, long index);
    /**
     * Constructs the ordinary sliceUpdate operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param update non-null update operand
     * @param prefix non-null exact per-axis placement prefix Shape
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sliceUpdate(V input, V update, Shape prefix);
    /**
     * Constructs the ordinary sliceUpdate operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param update non-null update operand
     * @param attrs exact canonical source slice attributes, including empty starts and lengths
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sliceUpdate(V input, V update, SliceAttrs attrs);
    /**
     * Constructs the ordinary sliceByLength operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param attrs exact canonical source slice attributes, including empty starts and lengths
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V sliceByLength(V input, SliceAttrs attrs);
    /**
     * Constructs the ordinary cropToShape operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param target non-null exact crop result Shape
     * @param prefix non-null exact per-axis placement prefix Shape
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V cropToShape(V input, Shape target, Shape prefix);
    /**
     * Constructs the ordinary unfold operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @param step positive logical distance between window starts
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unfold(V input, int axis, long window, long step);
    /**
     * Constructs the ordinary foldAxis operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param axis canonical non-negative axis supplied by the formula
     * @param outputSize non-negative logical folded extent
     * @param step positive logical distance between window starts
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V foldAxis(V input, int axis, long outputSize, long step);
    /**
     * Constructs the ordinary unfold2d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unfold2d(V input, Window2dAttrs window);
    /**
     * Constructs the ordinary unfold2d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @param paddingValue non-null exact represented padding scalar metadata
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unfold2d(V input, Window2dAttrs window, ScalarValue paddingValue);
    /**
     * Constructs the ordinary fold2d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param shape non-null exact logical target Shape
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V fold2d(V input, Shape shape, Window2dAttrs window);
    /**
     * Constructs the ordinary unfold3d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unfold3d(V input, Window3dAttrs window);
    /**
     * Constructs the ordinary unfold3d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @param paddingValue non-null exact represented padding scalar metadata
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V unfold3d(V input, Window3dAttrs window, ScalarValue paddingValue);
    /**
     * Constructs the ordinary fold3d operation synchronously on the selected route.
     *
     * @param input non-null direct operand
     * @param shape non-null exact logical target Shape
     * @param window non-null canonical window geometry, or positive logical axis window size
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    abstract V fold3d(V input, Shape shape, Window3dAttrs window);

    /**
     * Retains selected nullable cotangents in a directly typed positional array.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return a new actual operand array preserving position and null elements
     */
    final V[] result(V a) { V[] result = array(1); result[0] = a; return result; }
    /**
     * Retains selected nullable cotangents in a directly typed positional array.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return a new actual operand array preserving position and null elements
     */
    final V[] result(V a, V b) { V[] result = array(2); result[0] = a; result[1] = b; return result; }
    /**
     * Retains selected nullable cotangents in a directly typed positional array.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @param c third positional result element; may be null
     * @return a new actual operand array preserving position and null elements
     */
    final V[] result(V a, V b, V c) {
        V[] result = array(3); result[0] = a; result[1] = b; result[2] = c; return result;
    }
    /**
     * Constructs an exact typed positive zero using the original role and Shape.
     *
     * @param original non-null original source role; descriptor access does not resolve its primal
     * @return the ordinary EXPAND result from a scalar or explicitly row-bound exact literal
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V zeroLike(V original) {
        return valueLike(FirstOrderAutograd.DerivativeConstants.scalar(
                descriptor(original).dataType(), false), original);
    }
    /**
     * Constructs an exact typed positive one using the original role and Shape.
     *
     * @param original non-null original source role; descriptor access does not resolve its primal
     * @return the ordinary EXPAND result from a scalar or explicitly row-bound exact literal
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V oneLike(V original) {
        return valueLike(FirstOrderAutograd.DerivativeConstants.scalar(
                descriptor(original).dataType(), true), original);
    }
    /**
     * Constructs an explicitly owned exact literal with the source role's Shape.
     * A lexical row binding supplies the coordinate, not the selected VJP's structural presence.
     * Its ordinary expansion therefore remains a represented construction site even when the
     * bound literal already has the requested Shape. Numerical zero is not structural absence.
     *
     * @param value non-null exact typed operand or literal value
     * @param original non-null original source role; descriptor access does not resolve its primal
     * @return the ordinary EXPAND result, retaining the current construction owner's scope
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V valueLike(ScalarValue value, V original) {
        V bound = boundLike(value, original);
        return expand(bound == null ? base(value) : bound, descriptor(original).shape());
    }
    /**
     * Obtains the existing exact typed positive-zero base without reading a primal.
     *
     * @param type non-null existing floating coefficient type
     * @return the request-local scalar literal role
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V zeroBase(DataType type) {
        return base(FirstOrderAutograd.DerivativeConstants.scalar(type, false));
    }
    /**
     * Obtains the existing exact typed positive-one base without reading a primal.
     *
     * @param type non-null existing floating coefficient type
     * @return the request-local scalar literal role
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V oneBase(DataType type) {
        return base(FirstOrderAutograd.DerivativeConstants.scalar(type, true));
    }
    /**
     * Reuses the existing coefficient-bit table for positive two.
     *
     * @param type non-null existing floating coefficient type
     * @return exact scalar-operation metadata, never a Tensor leaf
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final ScalarValue two(DataType type) { return FirstOrderAutograd.DerivativeConstants.two(type); }
    /**
     * Reuses the existing coefficient-bit table for negative two.
     *
     * @param type non-null existing floating coefficient type
     * @return exact scalar-operation metadata, never a Tensor leaf
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final ScalarValue negativeTwo(DataType type) {
        return FirstOrderAutograd.DerivativeConstants.negativeTwo(type);
    }
    /**
     * Reuses the existing coefficient-bit table for negative one half.
     *
     * @param type non-null existing floating coefficient type
     * @return exact scalar-operation metadata, never a Tensor leaf
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final ScalarValue negativeHalf(DataType type) {
        return FirstOrderAutograd.DerivativeConstants.negativeHalf(type);
    }
    /**
     * Constructs the ordinary add operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V add(V a, V b) { return binary(BinaryArithmeticKind.ADD, a, b); }
    /**
     * Constructs the ordinary sub operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V sub(V a, V b) { return binary(BinaryArithmeticKind.SUB, a, b); }
    /**
     * Constructs the ordinary mul operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V mul(V a, V b) { return binary(BinaryArithmeticKind.MUL, a, b); }
    /**
     * Constructs the ordinary div operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V div(V a, V b) { return binary(BinaryArithmeticKind.DIV, a, b); }
    /**
     * Constructs the ordinary pow operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V pow(V a, V b) { return binary(BinaryArithmeticKind.POW, a, b); }
    /**
     * Constructs the ordinary add operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V add(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.ADD, a, b); }
    /**
     * Constructs the ordinary sub operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V sub(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.SUB, a, b); }
    /**
     * Constructs the ordinary mul operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V mul(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.MUL, a, b); }
    /**
     * Constructs the ordinary div operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V div(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.DIV, a, b); }
    /**
     * Constructs the ordinary pow operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V pow(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.POW, a, b); }
    /**
     * Constructs the ordinary maximum operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V maximum(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.MAX, a, b); }
    /**
     * Constructs the ordinary minimum operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V minimum(V a, ScalarValue b) { return scalar(ScalarElementwiseKind.MIN, a, b); }
    /**
     * Constructs the ordinary equalTo operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V equalTo(V a, V b) { return comparison(BinaryComparisonKind.EQUAL, a, b); }
    /**
     * Constructs ordinary represented numeric inequality through the shared comparison owner.
     *
     * @param a non-null left value
     * @param b non-null right value
     * @return the canonical BOOL result with ordinary promotion and NaN comparison behavior
     * @throws IllegalArgumentException if the ordinary Model or independent inference guards reject
     *     the operands
     */
    final V notEqualTo(V a, V b) { return comparison(BinaryComparisonKind.NOT_EQUAL, a, b); }
    /**
     * Constructs the ordinary lessThan operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V lessThan(V a, V b) { return comparison(BinaryComparisonKind.LESS_THAN, a, b); }
    /**
     * Constructs the ordinary greaterThan operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V greaterThan(V a, V b) { return comparison(BinaryComparisonKind.GREATER_THAN, a, b); }
    /**
     * Constructs the ordinary logicalAnd operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V logicalAnd(V a, V b) { return logical(BooleanLogicalKind.AND, a, b); }
    /**
     * Constructs the ordinary logicalOr operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @param b second operand; nullable only for a positional result-array element or unary logical dispatch
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V logicalOr(V a, V b) { return logical(BooleanLogicalKind.OR, a, b); }
    /**
     * Constructs the ordinary logicalNot operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V logicalNot(V a) { return logical(BooleanLogicalKind.NOT, a, null); }
    /**
     * Constructs the ordinary isNaN operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V isNaN(V a) { return classification(FloatingClassificationKind.IS_NAN, a); }
    /**
     * Constructs the ordinary isFinite operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V isFinite(V a) { return classification(FloatingClassificationKind.IS_FINITE, a); }
    /**
     * Constructs the ordinary isInf operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V isInf(V a) { return classification(FloatingClassificationKind.IS_INF, a); }
    /**
     * Constructs the ordinary neg operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V neg(V a) { return unary(UnaryElementwiseKind.NEG, a); }
    /**
     * Constructs the ordinary log operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V log(V a) { return unary(UnaryElementwiseKind.LOG, a); }
    /**
     * Constructs the ordinary exp operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V exp(V a) { return unary(UnaryElementwiseKind.EXP, a); }
    /**
     * Constructs the ordinary sqrt operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V sqrt(V a) { return unary(UnaryElementwiseKind.SQRT, a); }
    /**
     * Constructs the ordinary rsqrt operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V rsqrt(V a) { return unary(UnaryElementwiseKind.RSQRT, a); }
    /**
     * Constructs the ordinary tanh operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V tanh(V a) { return unary(UnaryElementwiseKind.TANH, a); }
    /**
     * Constructs the ordinary erf operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V erf(V a) { return unary(UnaryElementwiseKind.ERF, a); }
    /**
     * Constructs the ordinary sigmoid operation synchronously on the selected route.
     *
     * @param a first operand; nullable only for a positional result-array element
     * @return the canonical operation result, checked by Model and independently by Compiler on the metadata route
     * @throws IllegalArgumentException if the existing ordinary guards reject the request
     */
    final V sigmoid(V a) { return unary(UnaryElementwiseKind.SIGMOID, a); }

    /**
     * Copies canonical integer attributes for the existing public Tensor overload.
     * @param values immutable canonical attribute list
     * @return independently owned primitive argument array in the same order
     */
    private static int[] ints(List<Integer> values) {
        return values.stream().mapToInt(Integer::intValue).toArray();
    }
    /**
     * Copies canonical long attributes without normalizing coordinates or empty regions.
     * @param values immutable canonical attribute list
     * @return independently owned primitive argument array in the same order
     */
    private static long[] longs(List<Long> values) {
        return values.stream().mapToLong(Long::longValue).toArray();
    }
    /**
     * Retains canonical axes in immutable operation attributes without normalizing raw syntax.
     * @param values canonical axis arguments, not mutated
     * @return immutable boxed attribute values in the same order
     */
    private static List<Integer> integers(int[] values) {
        return java.util.Arrays.stream(values).boxed().toList();
    }

    private static final class Real extends OrdinaryConstruction<Tensor> {
        private final FirstOrderAutograd.DerivativeConstants constants;
        private final Observer<Tensor> observer;
        /**
         * Binds the existing literal owner and optional synchronous operation observation.
         * @param constants non-null request-local or lexical constant owner
         * @param observer nullable callback; no observation state is allocated when absent
         */
        Real(FirstOrderAutograd.DerivativeConstants constants, Observer<Tensor> observer) {
            this.constants = Objects.requireNonNull(constants, "constants");
            this.observer = observer;
        }
        /**
         * Publishes one newly created operation and all of its canonical siblings, if requested.
         * @param result exact result returned by an existing public Tensor operation
         * @return the same result reference, without any per-Tensor wrapper
         */
        private Tensor completed(Tensor result) {
            if (observer != null) {
                TensorProducer producer = result.provenance().orElseThrow().producer();
                List<Tensor> outputs = java.util.stream.IntStream.range(0, producer.outputCount())
                        .mapToObj(producer::output).toList();
                observer.created(producer.operation(), producer.inputs(), outputs, List.of());
            }
            return result;
        }
        @Override TensorDescriptor descriptor(Tensor value) { return value.descriptor(); }
        @Override Source<Tensor> sourceRoles(TensorProducer producer, Function<Tensor, Tensor> role) {
            if (role == FirstOrderAutograd.IDENTITY) return source(producer);
            List<Tensor> inputs = producer.inputs().stream().map(role).toList();
            List<Tensor> outputs = java.util.stream.IntStream.range(0, producer.outputCount())
                    .mapToObj(slot -> role.apply(producer.output(slot))).toList();
            return new Source<>() {
                @Override public Operation operation() { return producer.operation(); }
                @Override public List<Tensor> inputs() { return inputs; }
                @Override public Tensor output(int slot) { return outputs.get(slot); }
            };
        }
        @Override Tensor disconnectedZeroLike(Tensor original) {
            return constants.disconnectedZeroLike(original, this);
        }
        @Override Tensor[] array(int size) { return new Tensor[size]; }
        @Override Tensor base(ScalarValue value) { return constants.base(value); }
        @Override Tensor boundLike(ScalarValue value, Tensor original) {
            return constants.boundLike(value, original);
        }
        @Override int originalRowRole(Tensor original) { return constants.originalRowRole(original); }
        @Override Tensor unary(UnaryElementwiseKind kind, Tensor input) {
            return completed(switch (kind) {
                case NEG -> input.neg(); case LOG -> input.log(); case EXP -> input.exp();
                case SQRT -> input.sqrt(); case RSQRT -> input.rsqrt(); case TANH -> input.tanh();
                case ERF -> input.erf(); case SIGMOID -> input.sigmoid();
                default -> throw new IllegalArgumentException("not an ordinary gradient construction: " + kind);
            });
        }
        @Override Tensor binary(BinaryArithmeticKind kind, Tensor a, Tensor b) {
            return completed(switch (kind) {
                case ADD -> a.add(b); case SUB -> a.sub(b); case MUL -> a.mul(b); case DIV -> a.div(b);
                case MIN -> a.minimum(b); case MAX -> a.maximum(b); case POW -> a.pow(b);
            });
        }
        @Override Tensor scalar(ScalarElementwiseKind kind, Tensor a, ScalarValue b) {
            return completed(switch (kind) {
                case ADD -> a.add(b); case SUB -> a.sub(b); case MUL -> a.mul(b); case DIV -> a.div(b);
                case MIN -> a.minimum(b); case MAX -> a.maximum(b); case POW -> a.pow(b);
                case CLAMP -> throw new IllegalArgumentException("gradient construction has no clamp operation");
            });
        }
        @Override Tensor comparison(BinaryComparisonKind kind, Tensor a, Tensor b) {
            return completed(switch (kind) {
                case EQUAL -> a.equalTo(b); case LESS_THAN -> a.lessThan(b); case GREATER_THAN -> a.greaterThan(b);
                case NOT_EQUAL -> a.notEqualTo(b);
                default -> throw new IllegalArgumentException("not an ordinary gradient comparison: " + kind);
            });
        }
        @Override Tensor logical(BooleanLogicalKind kind, Tensor a, Tensor b) {
            return completed(switch (kind) {
                case AND -> a.logicalAnd(b); case OR -> a.logicalOr(b); case NOT -> a.logicalNot();
                default -> throw new IllegalArgumentException("not an ordinary gradient logical operation: " + kind);
            });
        }
        @Override Tensor classification(FloatingClassificationKind kind, Tensor a) {
            return completed(switch (kind) {
                case IS_NAN -> a.isNaN(); case IS_FINITE -> a.isFinite(); case IS_INF -> a.isInf();
            });
        }
        @Override Tensor where(Tensor c, Tensor a, Tensor b) { return completed(Tensor.where(c, a, b)); }
        @Override Tensor cast(Tensor a, DataType type) { return completed(a.cast(type)); }
        @Override Tensor matmul(Tensor a, Tensor b) { return completed(a.matmul(b)); }
        @Override Tensor reshape(Tensor a, Shape shape) { return completed(a.reshape(shape)); }
        @Override Tensor expand(Tensor a, Shape shape) { return completed(a.expand(shape)); }
        @Override Tensor contiguous(Tensor a) { return completed(a.contiguous()); }
        @Override Tensor expandDims(Tensor a, int axis) { return completed(a.expandDims(axis)); }
        @Override Tensor squeeze(Tensor a, int axis) { return completed(a.squeeze(axis)); }
        @Override Tensor permute(Tensor a, int... axes) { return completed(a.permute(axes)); }
        @Override Tensor sum(Tensor a) { return completed(a.sum()); }
        @Override Tensor sum(Tensor a, int axis, boolean keep) { return completed(a.sum(axis, keep)); }
        @Override Tensor sum(Tensor a, int[] axes, boolean keep) { return completed(a.sum(axes, keep)); }
        @Override Tensor mean(Tensor a, int[] axes, boolean keep) { return completed(a.mean(axes, keep)); }
        @Override Tensor prod(Tensor a, int axis, boolean keep) { return completed(a.prod(axis, keep)); }
        @Override Tensor sumToShape(Tensor a, Shape shape) { return completed(a.sumToShape(shape)); }
        @Override Tensor cumSum(Tensor a, int axis, boolean exclusive, boolean reverse) {
            return completed(a.cumSum(axis, exclusive, reverse));
        }
        @Override Tensor cumProd(Tensor a, int axis, boolean exclusive, boolean reverse) {
            return completed(a.cumProd(axis, exclusive, reverse));
        }
        @Override Tensor softmax(Tensor a, int axis) { return completed(a.softmax(axis)); }
        @Override Tensor logSoftmax(Tensor a, int axis) { return completed(a.logSoftmax(axis)); }
        @Override Tensor gather(Tensor a, Tensor indices, int axis) { return completed(a.gather(indices, axis)); }
        @Override Tensor gatherElements(Tensor a, Tensor indices, int axis) {
            return completed(a.gatherElements(indices, axis));
        }
        @Override Tensor gatherNd(Tensor a, Tensor indices, int batch) {
            return completed(a.gatherNd(indices, batch));
        }
        @Override Tensor scatterAdd(Tensor a, Tensor indices, Tensor updates, int axis) {
            return completed(a.scatterAdd(indices, updates, axis));
        }
        @Override Tensor scatterElements(Tensor a, Tensor indices, Tensor updates, int axis, ScatterReduction reduction) {
            return completed(a.scatterElements(indices, updates, axis, reduction));
        }
        @Override Tensor scatterNd(Tensor a, Tensor indices, Tensor updates, ScatterReduction reduction, int batch) {
            return completed(a.scatterNd(indices, updates, reduction, batch));
        }
        @Override Tensor oneHot(Tensor a, long depth) { return completed(a.oneHot(depth)); }
        @Override Tensor argsort(Tensor a, int axis, boolean descending) { return completed(a.argsort(axis, descending)); }
        @Override Tensor argMax(Tensor a, int axis, boolean keep, ArgExtremaTiePolicy ties) {
            return completed(a.argMax(axis, keep, ties));
        }
        @Override Tensor select(Tensor a, int axis, long index) { return completed(a.select(axis, index)); }
        @Override Tensor sliceUpdate(Tensor a, Tensor update, Shape prefix) { return completed(a.sliceUpdate(update, prefix)); }
        @Override Tensor sliceUpdate(Tensor a, Tensor update, SliceAttrs attrs) {
            return completed(a.sliceUpdate(update, longs(attrs.starts()), ints(attrs.axes()), longs(attrs.steps())));
        }
        @Override Tensor sliceByLength(Tensor a, SliceAttrs attrs) {
            return completed(a.sliceByLength(longs(attrs.starts()), longs(attrs.lengths()), ints(attrs.axes()), longs(attrs.steps())));
        }
        @Override Tensor cropToShape(Tensor a, Shape target, Shape prefix) { return completed(a.cropToShape(target, prefix)); }
        @Override Tensor unfold(Tensor a, int axis, long window, long step) { return completed(a.unfold(axis, window, step)); }
        @Override Tensor foldAxis(Tensor a, int axis, long outputSize, long step) { return completed(a.foldAxis(axis, outputSize, step)); }
        @Override Tensor unfold2d(Tensor a, Window2dAttrs window) { return completed(a.unfold2d(window)); }
        @Override Tensor unfold2d(Tensor a, Window2dAttrs window, ScalarValue paddingValue) {
            return completed(a.unfold2d(window, paddingValue));
        }
        @Override Tensor fold2d(Tensor a, Shape shape, Window2dAttrs window) { return completed(a.fold2d(shape, window)); }
        @Override Tensor unfold3d(Tensor a, Window3dAttrs window) { return completed(a.unfold3d(window)); }
        @Override Tensor unfold3d(Tensor a, Window3dAttrs window, ScalarValue paddingValue) {
            return completed(a.unfold3d(window, paddingValue));
        }
        @Override Tensor fold3d(Tensor a, Shape shape, Window3dAttrs window) { return completed(a.fold3d(shape, window)); }
    }

    private static final class Metadata extends OrdinaryConstruction<Value> {
        private final Function<ScalarValue, Value> literal;
        private final BiFunction<ScalarValue, Value, Value> like;
        private final ToIntFunction<Value> rowRole;
        private final Observer<Value> observer;
        private final Map<ScalarValue, Value> bases = new HashMap<>();
        private final Map<DisconnectedZeroKey, Value> disconnectedZeros = new HashMap<>();
        /**
         * Binds explicit metadata literal/row roles without creating real constant leaves.
         * @param literal optional exact scalar-role binder
         * @param like optional source-role-aware shape-literal binder
         * @param rowRole optional original-source row lookup
         * @param observer optional synchronous operation completion callback
         */
        Metadata(Function<ScalarValue, Value> literal, BiFunction<ScalarValue, Value, Value> like,
                ToIntFunction<Value> rowRole, Observer<Value> observer) {
            this.literal = literal; this.like = like; this.rowRole = rowRole; this.observer = observer;
        }
        /**
         * Completes a current single-output generated operation through both semantic owners.
         * Source occurrences with multiple canonical siblings use the outer checked source factory.
         * @param kind existing ordinary kind named by the actual formula
         * @param attrs exact canonical attributes, not raw negative-axis or slice syntax
         * @param inputs ordered direct operands, retaining repeated identities
         * @return the Model-canonical result after independent Compiler descriptor/predicate proof
         * @throws IllegalArgumentException if either semantic owner rejects the construction
         */
        private Value emit(OperationKind kind, OperationAttrs attrs, List<Value> inputs) {
            Operation operation = new Operation(kind, attrs);
            List<TensorDescriptor> inputDescriptors = inputs.stream().map(Value::descriptor).toList();
            List<TensorDescriptor> descriptors = ScanBody.describeOrdinary(operation, inputDescriptors, 1);
            List<CapturedGraphInference.ConstraintRequest> deferred = CapturedGraphInference.checkOrdinary(
                    operation, inputDescriptors, descriptors);
            List<Value> outputs = descriptors.stream().map(Value::new).toList();
            if (observer != null) observer.created(operation, inputs, outputs, deferred);
            return outputs.getFirst();
        }
        @Override TensorDescriptor descriptor(Value value) { return value.descriptor(); }
        @Override Source<Value> sourceRoles(TensorProducer producer, Function<Tensor, Value> role) {
            return source(producer.operation(), producer.inputs().stream().map(role).toList(),
                    java.util.stream.IntStream.range(0, producer.outputCount())
                            .mapToObj(slot -> role.apply(producer.output(slot))).toList());
        }
        @Override Value disconnectedZeroLike(Value original) {
            ScalarValue zero = FirstOrderAutograd.DerivativeConstants.scalar(
                    original.descriptor().dataType(), false);
            Value bound = boundLike(zero, original);
            if (bound != null) return bound;
            DisconnectedZeroKey key = new DisconnectedZeroKey(zero.dataType(), original.descriptor().shape());
            return disconnectedZeros.computeIfAbsent(key, ignored -> key.shape().equals(Shape.scalar())
                    ? zeroBase(key.dataType()) : expand(zeroBase(key.dataType()), key.shape()));
        }

        /** Exact shared-zero cache key; explicit lexical bindings bypass this descriptor cache. */
        private record DisconnectedZeroKey(DataType dataType, Shape shape) {}
        @Override Value[] array(int size) { return new Value[size]; }
        @Override Value base(ScalarValue value) {
            if (!value.dataType().isFloating()) {
                throw new IllegalArgumentException("derivative literal must be floating");
            }
            return bases.computeIfAbsent(value, key -> literal == null
                    ? new Value(new TensorDescriptor(
                            key.dataType(), Shape.scalar(), Optional.empty(), false), Optional.of(key))
                    : requireLiteral(literal.apply(key), key, Shape.scalar()));
        }
        @Override Value boundLike(ScalarValue value, Value original) {
            Value bound = like == null ? null : like.apply(value, original);
            return bound == null ? null : requireLiteral(bound, value, original.descriptor().shape());
        }

        /**
         * Requires an explicitly declared binding, not an inferred constant or prototype primal.
         * @param binding caller-declared scalar or shape-literal metadata role
         * @param value exact required represented type and bits
         * @param shape exact required scalar or original-role Shape
         * @return the same checked binding identity
         * @throws NullPointerException if a scalar binder returned null
         * @throws IllegalArgumentException if the explicit literal fact or Shape disagrees
         */
        private Value requireLiteral(Value binding, ScalarValue value, Shape shape) {
            Objects.requireNonNull(binding, "literal binding");
            if (!binding.literal().equals(Optional.of(value))
                    || !binding.descriptor().shape().equals(shape)) {
                throw new IllegalArgumentException("literal binding must retain exact value and Shape");
            }
            return binding;
        }
        @Override int originalRowRole(Value original) { return rowRole == null ? -1 : rowRole.applyAsInt(original); }
        @Override Value unary(UnaryElementwiseKind kind, Value a) { return emit(kind, NoOperationAttrs.INSTANCE, List.of(a)); }
        @Override Value binary(BinaryArithmeticKind kind, Value a, Value b) { return emit(kind, NoOperationAttrs.INSTANCE, List.of(a, b)); }
        @Override Value scalar(ScalarElementwiseKind kind, Value a, ScalarValue b) { return emit(kind, new ScalarValueAttrs(b), List.of(a)); }
        @Override Value comparison(BinaryComparisonKind kind, Value a, Value b) { return emit(kind, NoOperationAttrs.INSTANCE, List.of(a, b)); }
        @Override Value logical(BooleanLogicalKind kind, Value a, Value b) {
            return emit(kind, NoOperationAttrs.INSTANCE, b == null ? List.of(a) : List.of(a, b));
        }
        @Override Value classification(FloatingClassificationKind kind, Value a) { return emit(kind, NoOperationAttrs.INSTANCE, List.of(a)); }
        @Override Value where(Value c, Value a, Value b) { return emit(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE, List.of(c, a, b)); }
        @Override Value cast(Value a, DataType type) { return emit(CastKind.CAST, new CastAttrs(type), List.of(a)); }
        @Override Value matmul(Value a, Value b) { return emit(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE, List.of(a, b)); }
        @Override Value reshape(Value a, Shape shape) { return emit(ShapeTransformKind.RESHAPE, new TargetShapeAttrs(shape), List.of(a)); }
        @Override Value expand(Value a, Shape shape) { return emit(ShapeTransformKind.EXPAND, new TargetShapeAttrs(shape), List.of(a)); }
        @Override Value contiguous(Value a) { return emit(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE, List.of(a)); }
        @Override Value expandDims(Value a, int axis) { return emit(AxisTransformKind.EXPAND_DIMS, new AxisTransformAttrs(axis), List.of(a)); }
        @Override Value squeeze(Value a, int axis) { return emit(AxisTransformKind.SQUEEZE, new AxisTransformAttrs(axis), List.of(a)); }
        @Override Value permute(Value a, int... axes) { return emit(AxisTransformKind.PERMUTE, new PermutationAttrs(integers(axes)), List.of(a)); }
        @Override Value sum(Value a) { return emit(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE, List.of(a)); }
        @Override Value sum(Value a, int axis, boolean keep) { return emit(AggregateReductionKind.SUM, new AxisReductionAttrs(axis, keep), List.of(a)); }
        @Override Value sum(Value a, int[] axes, boolean keep) { return emit(AggregateReductionKind.SUM, new MultiAxisReductionAttrs(integers(axes), keep), List.of(a)); }
        @Override Value mean(Value a, int[] axes, boolean keep) { return emit(AggregateReductionKind.MEAN, new MultiAxisReductionAttrs(integers(axes), keep), List.of(a)); }
        @Override Value prod(Value a, int axis, boolean keep) { return emit(AggregateReductionKind.PROD, new AxisReductionAttrs(axis, keep), List.of(a)); }
        @Override Value sumToShape(Value a, Shape shape) { return emit(AggregateReductionKind.SUM, new SumToShapeAttrs(shape), List.of(a)); }
        @Override Value cumSum(Value a, int axis, boolean exclusive, boolean reverse) { return emit(CumulativeScanKind.CUM_SUM, new CumulativeScanAttrs(axis, exclusive, reverse), List.of(a)); }
        @Override Value cumProd(Value a, int axis, boolean exclusive, boolean reverse) { return emit(CumulativeScanKind.CUM_PROD, new CumulativeScanAttrs(axis, exclusive, reverse), List.of(a)); }
        @Override Value softmax(Value a, int axis) { return emit(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(axis), List.of(a)); }
        @Override Value logSoftmax(Value a, int axis) { return emit(SoftmaxKind.LOG_SOFTMAX, new SoftmaxAttrs(axis), List.of(a)); }
        @Override Value gather(Value a, Value indices, int axis) { return emit(AxisGatherKind.GATHER, new IndexAxisAttrs(axis), List.of(a, indices)); }
        @Override Value gatherElements(Value a, Value indices, int axis) { return emit(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(axis), List.of(a, indices)); }
        @Override Value gatherNd(Value a, Value indices, int batch) { return emit(GatherNdKind.GATHER_ND, new GatherNdAttrs(batch), List.of(a, indices)); }
        @Override Value scatterAdd(Value a, Value indices, Value updates, int axis) { return emit(AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(axis), List.of(a, indices, updates)); }
        @Override Value scatterElements(Value a, Value indices, Value updates, int axis, ScatterReduction reduction) { return emit(AxisScatterKind.SCATTER_ELEMENTS, new ScatterElementsAttrs(axis, reduction), List.of(a, indices, updates)); }
        @Override Value scatterNd(Value a, Value indices, Value updates, ScatterReduction reduction, int batch) { return emit(ScatterNdKind.SCATTER_ND, new ScatterNdAttrs(batch, reduction), List.of(a, indices, updates)); }
        @Override Value oneHot(Value a, long depth) { return emit(OneHotKind.ONE_HOT, new OneHotAttrs(depth), List.of(a)); }
        @Override Value argsort(Value a, int axis, boolean descending) { return emit(OrderingKind.ARGSORT, new SortAttrs(axis, descending), List.of(a)); }
        @Override Value argMax(Value a, int axis, boolean keep, ArgExtremaTiePolicy ties) { return emit(AggregateReductionKind.ARG_MAX, new ArgExtremaAttrs(axis, keep, ties), List.of(a)); }
        @Override Value select(Value a, int axis, long index) { return emit(SelectKind.SELECT, new SelectAttrs(axis, index), List.of(a)); }
        @Override Value sliceUpdate(Value a, Value update, Shape prefix) { return emit(SliceKind.SLICE_UPDATE, new CropToShapeAttrs(update.descriptor().shape(), prefix), List.of(a, update)); }
        @Override Value sliceUpdate(Value a, Value update, SliceAttrs attrs) { return emit(SliceKind.SLICE_UPDATE, attrs, List.of(a, update)); }
        @Override Value sliceByLength(Value a, SliceAttrs attrs) { return emit(SliceKind.SLICE, attrs, List.of(a)); }
        @Override Value cropToShape(Value a, Shape target, Shape prefix) { return emit(SliceKind.SLICE, new CropToShapeAttrs(target, prefix), List.of(a)); }
        @Override Value unfold(Value a, int axis, long window, long step) { return emit(WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(axis, window, step), List.of(a)); }
        @Override Value foldAxis(Value a, int axis, long outputSize, long step) { return emit(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(axis, outputSize, step), List.of(a)); }
        @Override Value unfold2d(Value a, Window2dAttrs window) {
            return emit(WindowTransformKind.UNFOLD2D, window, List.of(a));
        }
        @Override Value unfold2d(Value a, Window2dAttrs window, ScalarValue paddingValue) {
            return emit(WindowTransformKind.UNFOLD2D,
                    new Unfold2dAttrs(window, paddingValue), List.of(a));
        }
        @Override Value fold2d(Value a, Shape shape, Window2dAttrs window) { return emit(WindowTransformKind.FOLD2D, new Fold2dAttrs(shape, window), List.of(a)); }
        @Override Value unfold3d(Value a, Window3dAttrs window) {
            return emit(WindowTransformKind.UNFOLD3D, window, List.of(a));
        }
        @Override Value unfold3d(Value a, Window3dAttrs window, ScalarValue paddingValue) {
            return emit(WindowTransformKind.UNFOLD3D,
                    new Unfold3dAttrs(window, paddingValue), List.of(a));
        }
        @Override Value fold3d(Value a, Shape shape, Window3dAttrs window) { return emit(WindowTransformKind.FOLD3D, new Fold3dAttrs(shape, window), List.of(a)); }
    }
}
