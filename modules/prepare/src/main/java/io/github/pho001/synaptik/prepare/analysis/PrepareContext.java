package io.github.pho001.synaptik.prepare.analysis;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Provides the complete validated projection for analyzing one planned partition.
 *
 * <p>The projection contains the partition's ordered semantic nodes, projected logical values,
 * matching logical-memory requirements, compile-time logical-splat constants, and one opaque
 * backend input object. It deliberately
 * exposes no Compiler aggregate or implementation type. Every collection is an immutable
 * membership snapshot; list order and map encounter order are deterministic from the supplied
 * containers, while contained immutable references are retained exactly.</p>
 *
 * <p>Construction fails closed unless node order matches the partition, every referenced value is
 * resolved by one unique projected-value entry, every projected value has one descriptor-matching
 * logical requirement, projected shapes are static or bounded single named-B templates
 * with fixed feature dimensions, and each constant is an exact-typed
 * projected graph input. The contract does not separately require every otherwise valid projected
 * value to occur in a node input or output. These checks complete before a context can reach
 * backend analysis. This value does not bind dynamic dimensions, select a route, assign a slot,
 * allocate storage, or contain executable state.</p>
 *
 * <p>General-scan attachments retain their exact independently scoped body graphs and ordered
 * roles. Every local descriptor, including unused formals and sibling results, must be static or
 * use the same single named batch coordinate as its outer occurrence. Symbolic formal/result
 * layouts remain unresolved; this projection never substitutes a sample batch size.
 * Carry/invariant formals match actual dtype, shape and gradient eligibility; formal layouts
 * are canonical when static and unresolved when symbolic. Sequence-step formals remove axis zero,
 * whose size must match the header's fixed time count. Role results match the corresponding
 * static canonical or symbolic unresolved outer descriptors: differentiable outer roles use the OR
 * of carry, sequence and invariant actual flags, including unused actuals and zero time counts; other
 * dtypes use false. Formals keep their positional flags and ordinary local flags are unchanged.
 * Length mode has one separate non-gradient INT64 {@code [B]} actual and no lengths formal.
 * Static or one-named-batch actual, local and result templates are checked structurally before
 * numeric extent access. Every local, actual and result fixed-feature product is checked in both
 * modes, independently of empty time, feature or batch axes.
 * Each value retains its own declared dtype. Declared row-axis extents and compatible tied
 * positions are checked without inferring roles from equal extents or re-proving body algebra.
 * SHARED positions have no inferred batch axis. Actual layouts, including the lengths
 * layout, are retained without importing any concrete backend's read-layout restrictions.
 * Generated reverse scans use these same boundary checks; the restricted Compiler admission
 * for differentiating an original length-aware source is not a projection restriction.
 * Local IDs never resolve through the outer value or logical-requirement inventory. Scan demand
 * must agree with locally visible consumers and supplied publication/external-consumer obligations.
 * Only complete-graph projection can verify those external obligations against actual graph use.
 * Model's ordinary descriptor-only guards independently check every canonical local sibling;
 * this neither executes body arithmetic nor qualifies backend execution.</p>
 *
 * <p>The canonical record shape has six components: partition DAG, values, logical-memory
 * requirements, constants, backend inputs, and scoped Scan activity associations. Activity must
 * classify every numeric carry exactly once as a differentiable carried root or a distinct
 * temporal accumulator; length-mode carried roots cannot occupy SHARED accumulator positions.
 * Each accumulator's actual contribution availability must equal its declared contribution scope.
 * Ordinary constructors supply an empty activity map; copies preserve associations explicitly.</p>
 *
 * @param <I> concrete backend-owned immutable analysis-input role
 * @param partitionDag non-null exact immutable partition topology retained by exact reference
 * @param values non-null ordered projected values to snapshot; elements must be non-null, unique
 *     by value ID, and have static or qualified ordinary single named-B descriptor shapes
 * @param memoryRequirements non-null ordered logical requirements to snapshot; elements must be
 *     non-null and provide exactly one descriptor-matching entry for each projected value
 * @param constants non-null map of projected graph-input IDs to exact-typed logical splat values;
 *     keys and values must be non-null, and membership is copied in supplied encounter order
 * @param backendInputs non-null backend-owned immutable input retained opaquely by exact reference
 * @param scanActivityPlans non-null containing-node associations to exact local body activity;
 *     keys must belong to this projection and plans retain the exact associated body graph
 */
public record PrepareContext<I extends BackendAnalysisInputs>(
        PartitionDag partitionDag,
        List<GraphValue> values,
        List<LogicalMemoryRequirement> memoryRequirements,
        Map<ValueId, ScalarValue> constants,
        I backendInputs,
        Map<NodeId, ScanActivityPlan> scanActivityPlans) {
    /**
     * Validates and snapshots one complete partition-analysis projection.
     *
     * <p>Top-level references are validated in component order. List elements and duplicate value
     * identities are then checked in supplied order, followed by partition order, static-shape,
     * reference-closure, logical-requirement, scan binding/demand, and constant checks. No supplied
     * collection container is retained.</p>
     *
     * @param partitionDag non-null validated partition topology retained exactly
     * @param values non-null ordered projected values to snapshot
     * @param memoryRequirements non-null ordered projected logical requirements to snapshot
     * @param constants non-null projected logical-splat constants to snapshot
     * @param backendInputs non-null immutable backend input to retain opaquely and exactly
     * @param scanActivityPlans non-null exact local-body activity associations to snapshot
     * @throws NullPointerException if a component, list element, constant key, or constant value
     *     is {@code null}; the message identifies the component or indexed entry
     * @throws IllegalArgumentException if node order differs from the partition, a projected value
     *     or requirement ID is duplicated, a descriptor shape is neither static nor in the bounded
     *     ordinary single named-B domain, a node input or
     *     output is absent from the projected values, requirements do not match projected values
     *     exactly, actual/local/result templates have incompatible batch axes, a scan's scoped
     *     descriptors, positional bindings, roles, activity classification/contribution scopes or
     *     locally knowable demand disagree, or a constant is not an exact-typed projected graph input
     * @throws ArithmeticException if checked scan cardinality, element-count or canonical boundary
     *     layout geometry overflows
     */
    public PrepareContext {
        Objects.requireNonNull(partitionDag, "partitionDag");
        Objects.requireNonNull(values, "values");
        Objects.requireNonNull(memoryRequirements, "memoryRequirements");
        Objects.requireNonNull(constants, "constants");
        Objects.requireNonNull(backendInputs, "backendInputs");
        Objects.requireNonNull(scanActivityPlans, "scanActivityPlans");
        var activity = new LinkedHashMap<NodeId, ScanActivityPlan>();
        for (var entry : scanActivityPlans.entrySet()) {
            NodeId id = Objects.requireNonNull(entry.getKey(), "scanActivityPlans.key");
            ScanActivityPlan plan = Objects.requireNonNull(entry.getValue(), "scanActivityPlans.value");
            CompiledNode containing = partitionDag.nodes().stream()
                    .filter(node -> node.id().equals(id)).findFirst().orElseThrow(() ->
                            new IllegalArgumentException("activity key is outside projection: " + id));
            if (containing.scanBody().isEmpty()
                    || containing.scanBody().orElseThrow().graph() != plan.graph()) {
                throw new IllegalArgumentException("activity graph differs from containing Scan: " + id);
            }
            activity.put(id, plan);
        }
        scanActivityPlans = Collections.unmodifiableMap(activity);

        var valuesById = new LinkedHashMap<ValueId, GraphValue>();
        for (int index = 0; index < values.size(); index++) {
            GraphValue value = Objects.requireNonNull(values.get(index), "values[" + index + "]");
            if (valuesById.putIfAbsent(value.id(), value) != null) {
                throw new IllegalArgumentException(
                        "values[" + index + "].id duplicates " + value.id());
            }
        }
        values = List.copyOf(values);
        validateValueShapes(values);

        var requirementsById = new LinkedHashMap<ValueId, LogicalMemoryRequirement>();
        var requirementIndexes = new HashMap<ValueId, Integer>();
        for (int index = 0; index < memoryRequirements.size(); index++) {
            LogicalMemoryRequirement requirement = Objects.requireNonNull(
                    memoryRequirements.get(index), "memoryRequirements[" + index + "]");
            if (requirementsById.putIfAbsent(requirement.valueId(), requirement) != null) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + index
                                + "].valueId duplicates "
                                + requirement.valueId());
            }
            requirementIndexes.put(requirement.valueId(), index);
        }
        memoryRequirements = List.copyOf(memoryRequirements);

        var projectedInputIds = new HashSet<ValueId>();
        for (int nodeIndex = 0; nodeIndex < partitionDag.nodes().size(); nodeIndex++) {
            CompiledNode node = partitionDag.nodes().get(nodeIndex);
            for (int inputIndex = 0; inputIndex < node.inputs().size(); inputIndex++) {
                ValueId input = node.inputs().get(inputIndex);
                if (!valuesById.containsKey(input)) {
                    throw new IllegalArgumentException(
                            "nodes["
                                    + nodeIndex
                                    + "].inputs["
                                    + inputIndex
                                    + "] is absent from values: "
                                    + input);
                }
                projectedInputIds.add(input);
            }
            for (int outputIndex = 0; outputIndex < node.outputs().size(); outputIndex++) {
                ValueId output = node.outputs().get(outputIndex);
                if (!valuesById.containsKey(output)) {
                    throw new IllegalArgumentException(
                            "nodes["
                                    + nodeIndex
                                    + "].outputs["
                                    + outputIndex
                                    + "] is absent from values: "
                                    + output);
                }
            }
        }

        for (int valueIndex = 0; valueIndex < values.size(); valueIndex++) {
            GraphValue value = values.get(valueIndex);
            LogicalMemoryRequirement requirement = requirementsById.get(value.id());
            if (requirement == null) {
                throw new IllegalArgumentException(
                        "memoryRequirements has no entry for values["
                                + valueIndex
                                + "].id "
                                + value.id());
            }
            if (!requirement.descriptor().equals(value.descriptor())) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + requirementIndexes.get(value.id())
                                + "].descriptor does not match values["
                                + valueIndex
                                + "].descriptor for "
                                + value.id());
            }
        }
        for (int requirementIndex = 0;
                requirementIndex < memoryRequirements.size();
                requirementIndex++) {
            LogicalMemoryRequirement requirement = memoryRequirements.get(requirementIndex);
            if (!valuesById.containsKey(requirement.valueId())) {
                throw new IllegalArgumentException(
                        "memoryRequirements["
                                + requirementIndex
                                + "].valueId is absent from values: "
                                + requirement.valueId());
            }
        }

        for (CompiledNode node : partitionDag.nodes()) {
            if (node.scanBody().isPresent()) {
                validateScan(node, partitionDag, valuesById, requirementsById);
                ScanActivityPlan plan = scanActivityPlans.get(node.id());
                if (plan != null) validateScanActivity(node, plan);
            }
        }

        var copiedConstants = new LinkedHashMap<ValueId, ScalarValue>();
        for (Map.Entry<ValueId, ScalarValue> entry : constants.entrySet()) {
            ValueId valueId = Objects.requireNonNull(entry.getKey(), "constants key");
            ScalarValue constant =
                    Objects.requireNonNull(entry.getValue(), "constants[" + valueId + "]");
            GraphValue value = valuesById.get(valueId);
            if (value == null) {
                throw new IllegalArgumentException(
                        "constants key is absent from values: " + valueId);
            }
            LogicalMemoryRequirement requirement = requirementsById.get(valueId);
            if (!projectedInputIds.contains(valueId)
                    || requirement.producerPartition().isPresent()) {
                throw new IllegalArgumentException(
                        "constants key is not a projected graph input: " + valueId);
            }
            if (constant.dataType() != value.descriptor().dataType()) {
                throw new IllegalArgumentException(
                        "constants["
                                + valueId
                                + "] data type "
                                + constant.dataType()
                                + " does not match descriptor data type "
                                + value.descriptor().dataType());
            }
            copiedConstants.put(valueId, constant);
        }
        constants = Collections.unmodifiableMap(copiedConstants);
    }

    /**
     * Reports explicit symbolic execution mode, independently of eventual byte independence.
     * @return true when any projected descriptor contains the qualified named batch dimension
     */
    public boolean requiresBatchSize() {
        return values.stream().anyMatch(value -> !value.descriptor().shape().isFullyStatic());
    }

    /**
     * Validates the shared fixed-or-sole-named-B vocabulary and common-symbol proof.
     *
     * <p>This synchronous cold validation is shared by partition projections and complete graph
     * sources. It reads immutable descriptors only, neither mutates nor snapshots the list, and
     * retains no list or value reference. It does not resolve layouts or actual batch sizes,
     * establish caller-axis or ROW authority, or qualify a backend route.</p>
     *
     * @param values non-null caller-owned ordered values with non-null elements, inspected only
     * @return the sole named batch symbol, or empty for entirely static descriptors
     * @throws NullPointerException if the list or an element is null
     * @throws IllegalArgumentException if a shape is outside the bounded vocabulary or symbols
     *     disagree across values
     */
    public static Optional<String> validateValueShapes(List<GraphValue> values) {
        Objects.requireNonNull(values, "values");
        String symbol = null;
        for (int index = 0; index < values.size(); index++) {
            GraphValue value = Objects.requireNonNull(values.get(index), "values[" + index + "]");
            Shape shape = value.descriptor().shape();
            if (!shape.isFullyStatic() && !ordinaryBatchShape(shape)) {
                throw new IllegalArgumentException(
                        "values[" + index + "].descriptor.shape must be fully static: " + shape);
            }
            for (Dimension dimension : shape.dimensions()) {
                if (dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension named) {
                    if (symbol != null && !symbol.equals(named.symbol())) {
                        throw new IllegalArgumentException("multiple batch symbols in projection");
                    }
                    symbol = named.symbol();
                }
            }
        }
        return Optional.ofNullable(symbol);
    }

    /**
     * Checks the bounded projection vocabulary without resolving numeric geometry or proving
     * backend support. Cross-value agreement on the single named symbol is checked separately.
     *
     * @param shape non-null projected shape
     * @return whether every axis is fixed or the sole named axis
     */
    private static boolean ordinaryBatchShape(Shape shape) {
        int named = 0;
        for (Dimension dimension : shape.dimensions()) {
            if (dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension) named++;
            else if (!(dimension instanceof StaticDimension)) return false;
        }
        return named == 1;
    }

    /**
     * Checks boundaries and ordinary canonical descriptors without executing body algebra.
     * Lengths participates in outer use but never indexes a formal or contributes to gradient OR.
     * Declared row roles constrain extents and tied positions; Model's shared descriptor guards
     * check every known PURE occurrence before backend-specific numerical/dataflow admission.
     * The sole declared named coordinate is shared with Model's deferred ordinary proof;
     * all-valid bodies retain whole-Tensor semantics rather than length-mode ROW restrictions.
     * Length-mode batch identity is a Dimension, not a sampled extent. Every local declaration
     * and latent port is checked before analysis, including fixed features behind empty axes.
     * Empty carry and zero-ROW length forms use lengths as the batch authority, independently
     * of numerical carry count or dtype.
     *
     * @param node non-null structurally validated body-bearing outer node
     * @param dag non-null outer partition topology containing {@code node}
     * @param outer non-null complete projected outer-value index
     * @param requirements non-null descriptor-matching outer logical-requirement index
     * @throws IllegalArgumentException if a local descriptor has an unbound or malformed symbol, a positional descriptor
     *     disagrees, any length-mode actual/local/result template is unbound, a length-mode
     *     dtype/lengths/row boundary is invalid, or demand contradicts
     *     locally visible uses or supplied logical obligations
     * @throws ArithmeticException if checked cardinality, element-count or boundary layout
     *     geometry overflows
     */
    private static void validateScan(
            CompiledNode node,
            PartitionDag dag,
            Map<ValueId, GraphValue> outer,
            Map<ValueId, LogicalMemoryRequirement> requirements) {
        CompiledScanBody body = node.scanBody().orElseThrow();
        ScanAttrs attrs = (ScanAttrs) node.operation().attrs();
        boolean lengthMode = attrs.lengthAware();
        int sequenceEnd = Math.addExact(attrs.carryCount(), attrs.sequenceCount());
        int formalCount = Math.addExact(sequenceEnd, attrs.invariantCount());
        int actualCount = Math.addExact(formalCount, lengthMode ? 1 : 0);
        Dimension batch = new StaticDimension(0);
        if (lengthMode) {
            TensorDescriptor lengths = outer.get(node.inputs().get(formalCount)).descriptor();
            if (lengths.dataType() != DataType.INT64 || lengths.requiresGrad()
                    || lengths.shape().rank() != 1) {
                throw new IllegalArgumentException("scan " + node.id() + " invalid lengths descriptor");
            }
            batch = lengths.shape().dimension(0);
        }
        var local = new HashMap<ValueId, TensorDescriptor>();
        Optional<io.github.pho001.synaptik.model.shape.DynamicDimension> declaredBatch = Optional.empty();
        for (ValueId id : node.inputs()) {
            for (Dimension dimension : outer.get(id).descriptor().shape().dimensions()) {
                if (dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension named) {
                    if (declaredBatch.isPresent() && !declaredBatch.orElseThrow().equals(named)) {
                        throw new IllegalArgumentException(
                                "scan " + node.id() + " has multiple named coordinates");
                    }
                    declaredBatch = Optional.of(named);
                }
            }
        }
        final var occurrenceBatch = declaredBatch;
        for (GraphValue value : body.graph().values()) {
            if (!value.descriptor().shape().isFullyStatic()) {
                if (!ordinaryBatchShape(value.descriptor().shape())
                        || value.descriptor().layout().isPresent()
                        || value.descriptor().shape().dimensions().stream()
                                .filter(d -> d instanceof io.github.pho001.synaptik.model.shape.DynamicDimension)
                                .anyMatch(d -> !d.equals(occurrenceBatch.orElse(null))))
                    throw new IllegalArgumentException(
                            "scan " + node.id() + " local descriptor must be fully static or bound to the occurrence batch: " + value.id());
            }
            validateFixedFeatures(value.descriptor().shape());
            local.put(value.id(), value.descriptor());
        }
        for (CompiledNode ordinary : body.graph().nodes()) {
            var described = io.github.pho001.synaptik.model.tensor.ScanBody.describeOrdinary(
                    ordinary.operation(), ordinary.inputs().stream().map(local::get).toList(),
                    ordinary.outputs().size(), occurrenceBatch);
            for (int port = 0; port < described.size(); port++) {
                TensorDescriptor expected = described.get(port);
                TensorDescriptor declared = local.get(ordinary.outputs().get(port));
                if (declared.dataType() != expected.dataType()
                        || !declared.shape().equals(expected.shape())
                        || declared.requiresGrad() != expected.requiresGrad()) {
                    throw new IllegalArgumentException("scan " + node.id()
                            + " invalid ordinary declaration at " + ordinary.id() + " port " + port);
                }
            }
        }
        boolean gradient = false;
        var actualRowAxes = new HashMap<ValueId, Integer>();
        for (int index = 0; index < actualCount; index++) {
            ValueId input = node.inputs().get(index);
            LogicalMemoryRequirement requirement = requirements.get(input);
            boolean producedHere = dag.producer(input).isPresent();
            if (!requirement.consumerPartitions().contains(dag.partition())
                    || requirement.producerPartition().equals(Optional.of(dag.partition())) != producedHere) {
                throw new IllegalArgumentException(
                        "scan " + node.id() + " logical use mismatch at input " + index);
            }
            TensorDescriptor actual = outer.get(input).descriptor();
            validateFixedFeatures(actual.shape());
            if (index == formalCount) continue;
            gradient |= actual.requiresGrad();
            Shape shape = actual.shape();
            if (index >= attrs.carryCount()
                    && index < sequenceEnd) {
                if (shape.rank() == 0
                        || !shape.dimension(0).equals(new StaticDimension(attrs.timeCount()))) {
                    throw new IllegalArgumentException(
                            "scan " + node.id() + " sequence time mismatch at input " + index);
                }
                shape = Shape.ofDimensions(
                        shape.dimensions().subList(1, shape.rank()).toArray(Dimension[]::new));
            }
            TensorDescriptor expected = canonical(actual, shape, actual.requiresGrad());
            if (!expected.equals(local.get(body.graph().inputs().get(index)))) {
                throw new IllegalArgumentException(
                        "scan " + node.id() + " formal descriptor mismatch at input " + index);
            }
            if (lengthMode) {
                int rowAxis = -1;
                int outerRowAxis = -1;
                if (index < attrs.rowCarryCount()) {
                    rowAxis = outerRowAxis = 0;
                } else if (index >= attrs.carryCount() && index < sequenceEnd) {
                    rowAxis = attrs.sequenceRowAxes().get(index - attrs.carryCount()).orElse(-1);
                    outerRowAxis = rowAxis < 0 ? -1 : rowAxis + 1;
                }
                validateRowShape(shape, rowAxis, batch);
                Integer previous = actualRowAxes.putIfAbsent(input, outerRowAxis);
                if (previous != null && previous != outerRowAxis) {
                    throw new IllegalArgumentException("scan " + node.id() + " incompatible tied row roles");
                }
            }
        }
        for (int index = 0; index < attrs.carryCount(); index++) {
            TensorDescriptor before = local.get(body.graph().inputs().get(index));
            TensorDescriptor after = local.get(body.nextCarry().get(index));
            if (before.dataType() != after.dataType() || !before.shape().equals(after.shape())) {
                throw new IllegalArgumentException(
                        "scan " + node.id() + " next-carry descriptor mismatch at role " + index);
            }
        }
        var roles = new ArrayList<>(body.nextCarry());
        roles.addAll(body.emissions());
        roles.addAll(body.residuals());
        var demand = new ArrayList<Integer>();
        for (int index = 0; index < roles.size(); index++) {
            TensorDescriptor source = local.get(roles.get(index));
            Shape shape = source.shape();
            if (lengthMode && (index < attrs.rowCarryCount()
                    || index >= attrs.carryCount() && index < attrs.carryCount() + attrs.emissionCount())) {
                validateRowShape(shape, 0, batch);
            }
            if (index >= attrs.carryCount()) {
                var dimensions = new ArrayList<Dimension>();
                dimensions.add(new StaticDimension(attrs.timeCount()));
                dimensions.addAll(shape.dimensions());
                shape = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
            }
            ValueId output = node.outputs().get(index);
            validateFixedFeatures(outer.get(output).descriptor().shape());
            TensorDescriptor expected = canonical(
                    source, shape, gradient && source.dataType().isDifferentiable());
            if (!expected.equals(outer.get(output).descriptor())) {
                throw new IllegalArgumentException(
                        "scan " + node.id() + " outer descriptor mismatch at output " + index);
            }
            LogicalMemoryRequirement requirement = requirements.get(output);
            boolean consumedHere = !dag.consumers(output).isEmpty();
            if (!requirement.producerPartition().equals(Optional.of(dag.partition()))
                    || requirement.consumerPartitions().contains(dag.partition()) != consumedHere) {
                throw new IllegalArgumentException(
                        "scan " + node.id() + " logical use mismatch at output " + index);
            }
            if (requirement.graphOutput() || !requirement.consumerPartitions().isEmpty()) {
                demand.add(index);
            }
        }
        if (!demand.equals(body.demandedOutputIndices())) {
            throw new IllegalArgumentException("scan " + node.id() + " demand disagrees with logical uses");
        }
    }

    /**
     * Authenticates activity declaration positions and dataflow independently of the carrier.
     * Numeric carries have a complete disjoint root/accumulator classification. Length-mode
     * SHARED positions are accumulators, never carried roots. An accumulator's actual contribution
     * has exactly its declared availability, shape and dtype, not merely ordinary ADD
     * broadcast/promotion compatibility. Controlled operands and carried next values require
     * explicit availability bindings: an unconditional formal is not structural absence.
     * Optional raw-identity merges likewise require each operand's exact result shape and dtype;
     * ordinary broadcasting or promotion cannot stand in for a sole raw value.
     * Presence is structural availability, never a numerical-zero or tensor-extent test.
     *
     * @param containing exact body-bearing containing Scan
     * @param plan exact-local-graph activity
     * @throws IllegalArgumentException if roots, scopes, merges or temporal ADD bindings disagree
     */
    private static void validateScanActivity(CompiledNode containing, ScanActivityPlan plan) {
        var body = containing.scanBody().orElseThrow();
        var attrs = (ScanAttrs) containing.operation().attrs();
        if (plan.graph() != body.graph()) throw activityFailure("local graph identity");
        var values = new HashMap<ValueId, TensorDescriptor>();
        body.graph().values().forEach(value -> values.put(value.id(), value.descriptor()));
        var availability = new HashMap<ValueId, List<Integer>>();
        var carryPositions = new HashSet<Integer>();
        int root = 0;
        for (var carry : plan.carries()) {
            int position = carry.numericCarryPosition();
            if (position < 0 || position >= attrs.carryCount()
                    || attrs.lengthAware() && position >= attrs.rowCarryCount()
                    || !carryPositions.add(position)
                    || !values.get(body.graph().inputs().get(position)).dataType().isDifferentiable()) {
                throw activityFailure("carried root declaration");
            }
            availability.put(body.graph().inputs().get(position), List.of(root++));
        }
        var seeds = new HashSet<Integer>();
        for (int formal : plan.seedFormals()) {
            if (formal < attrs.carryCount()
                    || formal >= attrs.carryCount() + attrs.sequenceCount() || !seeds.add(formal)
                    || !values.get(body.graph().inputs().get(formal)).dataType().isDifferentiable()) {
                throw activityFailure("selected seed declaration");
            }
            availability.put(body.graph().inputs().get(formal), List.of(root++));
        }
        var nodes = new HashMap<NodeId, CompiledNode>();
        var producers = new HashMap<ValueId, CompiledNode>();
        for (CompiledNode node : body.graph().nodes()) {
            nodes.put(node.id(), node);
            node.outputs().forEach(value -> producers.put(value, node));
        }
        if (!plan.nodeScopes().keySet().equals(nodes.keySet())) throw activityFailure("scope coverage");
        for (CompiledNode node : body.graph().nodes()) {
            Integer scope = plan.nodeScopes().get(node.id());
            if (scope == null || scope < 0 || scope >= plan.scopes().size()) {
                throw activityFailure("node scope index");
            }
            List<Integer> required = plan.scopes().get(scope);
            var merge = plan.merges().get(node.id());
            if (merge != null) {
                if (node.operation().kind()
                                != io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD
                        || node.inputs().size() != 2 || node.outputs().size() != 1) {
                    throw activityFailure("optional merge operation");
                }
                var result = values.get(node.outputs().getFirst());
                for (ValueId input : node.inputs()) {
                    var operand = values.get(input);
                    if (operand.dataType() != result.dataType() || !operand.shape().equals(result.shape())) {
                        throw activityFailure("optional merge operand descriptor");
                    }
                }
                List<Integer> left = plan.scopes().get(merge.leftScope());
                List<Integer> right = plan.scopes().get(merge.rightScope());
                var union = new java.util.TreeSet<Integer>(left);
                union.addAll(right);
                if (!required.equals(List.copyOf(union))
                        || !left.equals(availability.get(node.inputs().get(0)))
                        || !right.equals(availability.get(node.inputs().get(1)))) {
                    throw activityFailure("optional merge operand scopes");
                }
            } else {
                for (ValueId input : node.inputs()) {
                    List<Integer> supplied = availability.get(input);
                    if (supplied != null && !supplied.containsAll(required)) {
                        throw activityFailure("input unavailable in owner scope");
                    }
                }
            }
            node.outputs().forEach(value -> availability.put(value, required));
        }
        for (var carry : plan.carries()) {
            List<Integer> next = availability.get(
                    body.nextCarry().get(carry.numericCarryPosition()));
            if (next == null || !next.equals(plan.scopes().get(carry.nextScope()))) {
                throw activityFailure("next carried root scope");
            }
        }
        var accumulators = new HashSet<Integer>();
        for (var accumulator : plan.accumulators()) {
            int position = accumulator.numericSharedCarryPosition();
            if (position < attrs.rowCarryCount() || position >= attrs.carryCount()
                    || carryPositions.contains(position) || !accumulators.add(position)) {
                throw activityFailure("shared accumulator declaration");
            }
            CompiledNode addition = producers.get(body.nextCarry().get(position));
            if (addition == null || addition.operation().kind()
                            != io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD
                    || addition.inputs().size() != 2
                    || !addition.inputs().get(0).equals(body.graph().inputs().get(position))
                    || plan.merges().containsKey(addition.id())
                    || !Objects.equals(plan.nodeScopes().get(addition.id()), accumulator.contributionScope())
                    || !Objects.equals(availability.get(addition.inputs().get(1)),
                            plan.scopes().get(accumulator.contributionScope()))
                    || !values.get(body.graph().inputs().get(position)).dataType().isDifferentiable()) {
                throw activityFailure("distinct temporal accumulator ADD");
            }
            TensorDescriptor state = values.get(body.graph().inputs().get(position));
            TensorDescriptor contribution = values.get(addition.inputs().get(1));
            if (state.dataType() != contribution.dataType()
                    || !state.shape().equals(contribution.shape())) {
                throw activityFailure("temporal accumulator contribution descriptor");
            }
        }
        if (carryPositions.size() + accumulators.size() != attrs.carryCount()) {
            throw activityFailure("numeric carry classification coverage");
        }
    }

    /**
     * Creates a fail-closed association error without inspecting invocation storage.
     * @param detail failed structural boundary
     * @return non-null preparation exception
     */
    private static IllegalArgumentException activityFailure(String detail) {
        return new IllegalArgumentException("invalid Scan activity: " + detail);
    }

    /**
     * Checks an explicitly declared row boundary, never deriving a role from its shape.
     *
     * @param shape non-null static or one-named-batch local boundary shape
     * @param axis minus one for SHARED, otherwise the proved row axis
     * @param batch exact static or named batch dimension from the lengths declaration
     * @throws IllegalArgumentException if a declared row axis is absent or has a different extent
     */
    private static void validateRowShape(Shape shape, int axis, Dimension batch) {
        if (axis >= 0 && (shape.rank() <= axis || !shape.dimension(axis).equals(batch))) {
            throw new IllegalArgumentException("scan declared row shape mismatch");
        }
        for (int i = 0; i < shape.rank(); i++) if (i != axis && !shape.dimension(i).isStatic())
            throw new IllegalArgumentException("scan SHARED/features must be static");
    }

    /**
     * Checks nonzero fixed factors independently of a zero time, feature or batch extent.
     * @param shape non-null declared template, never evaluated at a sample batch
     * @throws ArithmeticException if the fixed-feature product overflows
     */
    private static void validateFixedFeatures(Shape shape) {
        long product = 1;
        for (Dimension dimension : shape.dimensions()) if (dimension instanceof StaticDimension s && s.size() != 0)
            product = Math.multiplyExact(product, s.size());
    }

    /**
     * Describes a canonical formal or outer result, leaving ordinary local descriptors untouched.
     *
     * @param source non-null descriptor supplying the unchanged dtype
     * @param shape non-null static or single named-B formal or result shape
     * @param gradient required gradient eligibility at this boundary
     * @return non-null immutable descriptor, canonical when static and unresolved when symbolic
     * @throws ArithmeticException if canonical layout geometry overflows
     */
    private static TensorDescriptor canonical(
            TensorDescriptor source, Shape shape, boolean gradient) {
        return new TensorDescriptor(source.dataType(), shape,
                shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), gradient);
    }

    /**
     * Constructs an ordinary projection without derivative activity.
     *
     * @param partitionDag exact non-null partition topology
     * @param values non-null projected values
     * @param memoryRequirements non-null matching logical requirements
     * @param constants non-null projected logical splats
     * @param backendInputs non-null immutable backend inputs
     */
    public PrepareContext(
            PartitionDag partitionDag, List<GraphValue> values,
            List<LogicalMemoryRequirement> memoryRequirements,
            Map<ValueId, ScalarValue> constants, I backendInputs) {
        this(partitionDag, values, memoryRequirements, constants, backendInputs, Map.of());
    }

    /**
     * Preserves the original source-level construction form while deriving one authoritative DAG.
     *
     * @param partition non-null planned partition retained by the constructed projection
     * @param nodes non-null ordered nodes used to construct exactly one validated projection
     * @param values non-null ordered projected values to snapshot
     * @param memoryRequirements non-null ordered projected logical requirements to snapshot
     * @param constants non-null projected logical-splat constants to snapshot
     * @param backendInputs non-null immutable backend input retained opaquely and exactly
     * @throws NullPointerException if a supplied component is {@code null} or the compatibility
     *     node list contains a null element
     * @throws IllegalArgumentException if the constructed partition DAG or the remaining
     *     projected facts fail the canonical validation
     * @throws ArithmeticException if checked scan cardinality, element-count or canonical boundary
     *     layout geometry overflows
     */
    public PrepareContext(
            PlannedPartition partition,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> memoryRequirements,
            Map<ValueId, ScalarValue> constants,
            I backendInputs) {
        this(
                new PartitionDag(partition, nodes),
                values,
                memoryRequirements,
                constants,
                backendInputs);
    }

    /**
     * Constructs a partition projection retaining explicit scoped derivative activity.
     *
     * @param partition non-null planned partition
     * @param nodes non-null exact ordered nodes
     * @param values non-null projected values
     * @param memoryRequirements non-null matching logical requirements
     * @param constants non-null projected logical splats
     * @param backendInputs non-null immutable backend inputs
     * @param scanActivityPlans non-null exact containing-Scan activity associations
     */
    public PrepareContext(
            PlannedPartition partition, List<CompiledNode> nodes, List<GraphValue> values,
            List<LogicalMemoryRequirement> memoryRequirements, Map<ValueId, ScalarValue> constants,
            I backendInputs, Map<NodeId, ScanActivityPlan> scanActivityPlans) {
        this(new PartitionDag(partition, nodes), values, memoryRequirements, constants,
                backendInputs, scanActivityPlans);
    }

    /**
     * Returns the sole retained node and topology projection.
     *
     * @return the exact non-null immutable partition DAG supplied to the canonical constructor or
     *     constructed once by the list-taking compatibility constructor
     */
    @Override
    public PartitionDag partitionDag() {
        return partitionDag;
    }

    /**
     * Returns the analyzed partition recipe.
     *
     * @return the exact non-null immutable partition reference supplied at construction
     */
    public PlannedPartition partition() {
        return partitionDag.partition();
    }

    /**
     * Returns the validated nodes in exact partition order.
     *
     * @return non-null immutable ordered snapshot containing the exact supplied node references
     */
    public List<CompiledNode> nodes() {
        return partitionDag.nodes();
    }

    /**
     * Returns the projected logical graph values in supplied deterministic order.
     *
     * @return non-null immutable ordered snapshot of unique scoped graph values
     */
    @Override
    public List<GraphValue> values() {
        return values;
    }

    /**
     * Returns the projected logical-memory requirements in supplied deterministic order.
     *
     * @return non-null immutable ordered snapshot with exactly one matching entry per value
     */
    @Override
    public List<LogicalMemoryRequirement> memoryRequirements() {
        return memoryRequirements;
    }

    /**
     * Returns projected exact-typed compile-time logical splats.
     *
     * @return non-null immutable map in supplied encounter order; keys are projected graph inputs
     *     and values are the exact supplied immutable scalar references
     */
    @Override
    public Map<ValueId, ScalarValue> constants() {
        return constants;
    }

    /**
     * Returns the backend's opaque immutable analysis input.
     *
     * @return exact non-null backend-owned input reference supplied at construction
     */
    @Override
    public I backendInputs() {
        return backendInputs;
    }
}
