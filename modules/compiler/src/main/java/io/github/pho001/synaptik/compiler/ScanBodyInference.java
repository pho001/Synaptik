package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.*;
import io.github.pho001.synaptik.model.operation.elementwise.logical.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;

/**
 * Captures and independently validates lexical scan bodies, positional roles and ordinary-use
 * demand. Local IDs belong only to their containing body; no source identity survives capture.
 * Bodies retain all declared inputs and canonical saved sites, without optimizing across them.
 * Inference admits one declared named coordinate independently of the narrower AD
 * inventory. Every local declaration, PURE family, shape attribute and outer port is checked;
 * numeric-domain predicates remain typed and containing-occurrence-owned for fresh binding.
 */
final class ScanBodyInference {
    /** Prevents instances of this stateless lexical validation boundary. */
    private ScanBodyInference() {}

    /**
     * Rejects escape and sharing between different lexical templates, permitting template reuse.
     * @param roots non-null outer traversal roots, including generated gradients
     * @throws IllegalArgumentException if a body value escapes or belongs to different templates
     */
    static void validateScopes(List<Tensor> roots) {
        Set<Tensor> outer = Collections.newSetFromMap(new IdentityHashMap<>());
        Set<TensorProducer> producers = Collections.newSetFromMap(new IdentityHashMap<>());
        var owners = new IdentityHashMap<Tensor, ScanBody>();
        var pending = new ArrayDeque<>(roots);
        while (!pending.isEmpty()) {
            Tensor value = pending.removeLast();
            if (!outer.add(value)) continue;
            value.provenance().ifPresent(p -> {
                TensorProducer producer = p.producer();
                if (!producers.add(producer)) return;
                for (int i = 0; i < producer.outputCount(); i++) outer.add(producer.output(i));
                pending.addAll(producer.inputs());
                producer.scanBody().ifPresent(body -> {
                    var locals = new ArrayList<>(formals(body));
                    locals.addAll(body.retainedValues());
                    for (Tensor local : locals) {
                        if (local.hostStorage().isPresent()) {
                            throw new IllegalArgumentException("scan body contains host-backed value");
                        }
                        ScanBody previous = owners.putIfAbsent(local, body);
                        if (previous != null && previous != body) {
                            throw new IllegalArgumentException("value shared across different scan templates");
                        }
                    }
                });
            });
        }
        for (Tensor value : outer) {
            if (owners.containsKey(value)) throw new IllegalArgumentException("scan body value escapes its scope");
        }
    }

    /**
     * Captures one occurrence with declaration-ordered inputs and unique local graph outputs.
     * @param body non-null validated source template
     * @return independently captured body with empty initial outer-port demand
     */
    static CompiledScanBody capture(ScanBody body) {
        return capture(body, null);
    }

    /**
     * Captures one exact occurrence with optional ephemeral source/local assignment observation.
     * @param body validated lexical source template
     * @param observer nullable synchronous assignment observer, never retained in compiled output
     * @return independently captured and inferred local body
     */
    static CompiledScanBody capture(ScanBody body, GraphCapture.AssignmentObserver observer) {
        var roles = new ArrayList<>(body.nextCarry());
        roles.addAll(body.emissions());
        roles.addAll(body.retainedValues());
        var unique = new ArrayList<Tensor>();
        var positions = new IdentityHashMap<Tensor, Integer>();
        for (Tensor role : roles) {
            if (!positions.containsKey(role)) {
                positions.put(role, unique.size());
                unique.add(role);
            }
        }
        CompiledGraphModel graph = GraphCapture.captureBody(unique, formals(body), observer);
        CapturedGraphInference.inferAndValidate(graph);
        List<ValueId> bindings = roles.stream().map(t -> graph.outputs().get(positions.get(t))).toList();
        int carry = body.nextCarry().size();
        int emissions = carry + body.emissions().size();
        return new CompiledScanBody(graph, bindings.subList(0, carry),
                bindings.subList(carry, emissions), bindings.subList(emissions, bindings.size()), List.of());
    }

    /**
     * Returns all declarations without identity deduplication or removal of unused inputs.
     * @param body non-null source template
     * @return immutable carry/step/invariant declaration order
     */
    static List<Tensor> formals(ScanBody body) {
        var result = new ArrayList<>(body.carry());
        result.addAll(body.step());
        result.addAll(body.invariants());
        return List.copyOf(result);
    }

    /**
     * Re-derives every local and outer descriptor without binding a representative batch. Named
     * layouts remain unresolved; private local predicates retain the containing occurrence's
     * ownership. Lengths is a separate discrete actual, with the exact static or named row
     * Dimension and no corresponding local formal.
     * Empty carry and explicit length mode are independent of numeric/ROW carry count.
     * Each canonical value retains its own ordinary floating or discrete descriptor.
     * @param node non-null body-bearing general scan
     * @param inputs non-null ordered outer input descriptors
     * @return exact canonical outer descriptors with conservative OR gradient eligibility
     * @throws IllegalArgumentException if body, formals, header or bindings disagree
     * @throws ArithmeticException if checked element-count, feature-count or layout arithmetic overflows
     */
    static CapturedGraphInference.InferenceResult infer(CompiledNode node, List<TensorDescriptor> inputs) {
        ScanAttrs attrs = (ScanAttrs) node.operation().attrs();
        CompiledScanBody body = node.scanBody().orElseThrow();
        if (inputs.size() != attrs.inputCount() || node.inputs().size() != attrs.inputCount()
                || node.outputs().size() != attrs.outputCount()
                || body.graph().inputs().size() != Math.addExact(Math.addExact(attrs.carryCount(),
                        attrs.sequenceCount()), attrs.invariantCount())
                || body.nextCarry().size() != attrs.carryCount()
                || body.emissions().size() != attrs.emissionCount()
                || body.residuals().size() != attrs.residualCount()) {
            throw new IllegalArgumentException("scan inference header/body cardinality mismatch");
        }
        validateResiduals(body, attrs.carryCount());
        var validated = CapturedGraphInference.inferAndValidate(body.graph());
        Map<ValueId, TensorDescriptor> local = new HashMap<>();
        for (GraphValue value : body.graph().values()) local.put(value.id(), value.descriptor());
        DynamicDimension batch = null;
        for (ValueId id : body.graph().inputs()) {
            var descriptor = local.get(id);
            for (Dimension dimension : descriptor.shape().dimensions()) if (dimension instanceof DynamicDimension named) {
                if (batch != null && !batch.equals(named)) throw new IllegalArgumentException("multiple scan batch symbols");
                batch = named;
            }
        }
        for (GraphValue value : body.graph().values()) {
            validateNamedShape(value.descriptor().shape(), batch);
            if (!value.descriptor().shape().isFullyStatic() && value.descriptor().layout().isPresent()) {
                throw new IllegalArgumentException("symbolic scan body layout must remain unresolved");
            }
            value.descriptor().shape().knownElementCount();
        }
        for (CompiledNode ordinary : body.graph().nodes()) validateNamedTransform(ordinary, local, batch);
        for (var constraint : validated.constraints()) validateNamedPredicate(constraint.predicate());
        int formalCount = Math.addExact(Math.addExact(attrs.carryCount(), attrs.sequenceCount()), attrs.invariantCount());
        for (int i = 0; i < formalCount; i++) {
            TensorDescriptor actual = inputs.get(i);
            Shape shape = actual.shape();
            validateNamedShape(shape, batch);
            if (!shape.isFullyStatic() && actual.layout().isPresent()) throw new IllegalArgumentException("resolved symbolic scan actual");
            shape.knownElementCount();
            if (i >= attrs.carryCount() && i < attrs.carryCount() + attrs.sequenceCount()) {
                if (shape.rank() == 0 || !shape.dimension(0).equals(new StaticDimension(attrs.timeCount()))) {
                    throw new IllegalArgumentException("scan sequence time mismatch");
                }
                shape = Shape.ofDimensions(shape.dimensions().subList(1, shape.rank()).toArray(Dimension[]::new));
            }
            TensorDescriptor formal = local.get(body.graph().inputs().get(i));
            TensorDescriptor expected = new TensorDescriptor(actual.dataType(), shape,
                    shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), actual.requiresGrad());
            if (!expected.equals(formal)) throw new IllegalArgumentException("scan formal descriptor mismatch at " + i);
        }
        for (int i = 0; i < attrs.carryCount(); i++) {
            TensorDescriptor before = local.get(body.graph().inputs().get(i));
            TensorDescriptor after = local.get(body.nextCarry().get(i));
            if (before.dataType() != after.dataType() || !before.shape().equals(after.shape())) {
                throw new IllegalArgumentException("scan next carry descriptor mismatch");
            }
        }
        if (attrs.lengthAware()) validateRows(node, inputs, local);
        boolean grad = inputs.subList(0, formalCount).stream().anyMatch(TensorDescriptor::requiresGrad);
        var roles = new ArrayList<>(body.nextCarry());
        roles.addAll(body.emissions());
        roles.addAll(body.residuals());
        var outputs = new ArrayList<TensorDescriptor>();
        for (int i = 0; i < roles.size(); i++) {
            TensorDescriptor source = local.get(roles.get(i));
            Shape shape = source.shape();
            if (i >= attrs.carryCount()) {
                var dims = new ArrayList<Dimension>();
                dims.add(new StaticDimension(attrs.timeCount()));
                dims.addAll(shape.dimensions());
                shape = Shape.ofDimensions(dims.toArray(Dimension[]::new));
            }
            outputs.add(new TensorDescriptor(source.dataType(), shape,
                    shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), grad && source.dataType().isDifferentiable()));
        }
        var constraints = validated.constraints().stream().map(p -> new CapturedGraphInference.ConstraintRequest(
                "body " + p.nodeId() + ": " + p.subject(), p.predicate())).toList();
        return new CapturedGraphInference.InferenceResult(outputs, constraints);
    }

    /**
     * Checks every declaration/result/attribute against one lexical named-coordinate vocabulary.
     * @param shape complete shape, including unused domains
     * @param batch declaration-owned symbol, or null for a static body
     * @throws IllegalArgumentException if a foreign, repeated, anonymous or derived axis occurs
     */
    private static void validateNamedShape(Shape shape, DynamicDimension batch) {
        int named = 0;
        for (Dimension d : shape.dimensions()) if (!(d instanceof StaticDimension)) {
            if (!(d instanceof DynamicDimension) || !d.equals(batch) || ++named > 1) {
                throw new IllegalArgumentException("scan requires fixed extents or one declared named batch axis");
            }
        }
    }

    /**
     * Checks the known PURE inventory, every shape attribute, complete-coordinate extractions
     * and reshape preservation independently of element-count equality. Other ordinary family
     * guards and numeric-domain predicates remain owned by captured inference.
     * @param node local ordinary occurrence
     * @param local exact scope-local descriptor table
     * @param batch sole declared symbol, or null
     * @throws IllegalArgumentException if an attribute invents a symbol or reshape mixes B
     * @throws ArithmeticException if fixed feature products overflow
     */
    private static void validateNamedTransform(CompiledNode node, Map<ValueId, TensorDescriptor> local,
            DynamicDimension batch) {
        if (node.scanBody().isPresent()) throw new IllegalArgumentException("nested scan body family");
        validateNamedTransform(node.operation(), node.inputs().stream().map(local::get).toList(),
                node.outputs().stream().map(local::get).toList(), batch);
    }

    /**
     * Applies the same lexical vocabulary and deferred-shape proof to pre-ID construction.
     * @param operation exact ordinary operation
     * @param inputs independently validated input descriptors
     * @param outputs complete independently validated canonical sibling descriptors
     * @param deferred independent family obligations, retaining numeric-domain binding predicates
     * @throws IllegalArgumentException if symbolic geometry would require a sampled binding
     */
    static void validateConstruction(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, List<CapturedGraphInference.ConstraintRequest> deferred) {
        DynamicDimension batch = null;
        var descriptors = new ArrayList<>(inputs); descriptors.addAll(outputs);
        for (TensorDescriptor descriptor : descriptors) for (Dimension d : descriptor.shape().dimensions()) {
            if (d instanceof DynamicDimension named) {
                if (batch != null && !batch.equals(named)) throw new IllegalArgumentException("multiple scan batch symbols");
                batch = named;
            }
        }
        validateConstruction(operation, inputs, outputs, deferred, Optional.ofNullable(batch));
    }

    /**
     * Checks pre-ID construction against the containing source's exact declared symbolic scope.
     * Model owns ordinary descriptor and symbolic-transform guards; independent Compiler
     * descriptors and deferred numeric-domain predicates remain checked at this boundary.
     * @param operation exact ordinary operation and attributes
     * @param inputs independently validated ordered input descriptors
     * @param outputs complete independently validated canonical sibling descriptors
     * @param deferred independent family obligations to validate without sampled bindings
     * @param namedBatch sole declared named batch dimension, or empty for a static scope
     * @throws IllegalArgumentException if vocabulary, descriptors or symbolic proof are invalid
     * @throws ArithmeticException if checked static feature or shape arithmetic overflows
     */
    static void validateConstruction(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, List<CapturedGraphInference.ConstraintRequest> deferred,
            Optional<DynamicDimension> namedBatch) {
        DynamicDimension batch = namedBatch.orElse(null);
        var descriptors = new ArrayList<>(inputs); descriptors.addAll(outputs);
        for (TensorDescriptor descriptor : descriptors) {
            validateNamedShape(descriptor.shape(), batch);
            if (!descriptor.shape().isFullyStatic() && descriptor.layout().isPresent()) {
                throw new IllegalArgumentException("symbolic scan construction layout must remain unresolved");
            }
            descriptor.shape().knownElementCount();
        }
        validateNamedTransform(operation, inputs, outputs, batch);
        deferred.forEach(request -> validateNamedPredicate(request.predicate()));
    }

    private static void validateNamedTransform(Operation operation, List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs, DynamicDimension batch) {
        // The ordinary Model guard owns the known PURE inventory. Independent Compiler
        // inference has already checked the same complete canonical descriptor signature.
        List<TensorDescriptor> described;
        try {
            described = ScanBody.describeOrdinary(operation, inputs, outputs.size(), Optional.ofNullable(batch));
        } catch (IllegalArgumentException failure) {
            throw new IllegalArgumentException("unsupported non-PURE or invalid ordinary scan body occurrence", failure);
        }
        if (!described.equals(outputs)) {
            throw new IllegalArgumentException("ordinary scan body descriptors disagree with Model construction");
        }
        var attrs = operation.attrs();
        List<Shape> shapes = switch (attrs) {
            case TargetShapeAttrs a -> List.of(a.targetShape());
            case SumToShapeAttrs a -> List.of(a.targetShape());
            case CropToShapeAttrs a -> List.of(a.targetShape(), a.prefixShape());
            case LayerNormAttrs a -> List.of(a.normalizedShape());
            case AffineLayerNormAttrs a -> List.of(a.normalizedShape());
            case RmsNormAttrs a -> List.of(a.normalizedShape());
            case Fold2dAttrs a -> List.of(a.outputShape());
            case Fold3dAttrs a -> List.of(a.outputShape());
            default -> List.of();
        };
        for (Shape shape : shapes) validateNamedShape(shape, batch);
    }

    /**
     * Rejects binding-dependent shape equality/broadcast guesses while retaining numeric-domain
     * obligations (for example a minimum contributor count) for fresh scoped binding.
     * @param predicate exact family-derived private predicate
     * @throws IllegalArgumentException if equality would specialize the named coordinate
     */
    private static void validateNamedPredicate(GraphPredicate predicate) {
        if (predicate instanceof DimensionEqual && GraphPredicateProof.evaluate(predicate) == ProofStatus.DEFERRED
                || predicate instanceof AnyOf any && any.predicates().stream().allMatch(DimensionEqual.class::isInstance)
                && GraphPredicateProof.evaluate(predicate) == ProofStatus.DEFERRED) {
            throw new IllegalArgumentException("unproved scan named-coordinate equality/broadcast");
        }
        if (predicate instanceof AllOf all) all.predicates().forEach(ScanBodyInference::validateNamedPredicate);
    }


    /**
     * Checks the complete length header/dataflow independently of Model source construction.
     * Independent ordinary inference precedes shared per-output coordinate recomputation.
     * @param node structurally checked body-bearing length occurrence
     * @param inputs ordered actual descriptors, including final INT64 lengths
     * @param local complete descriptor table owned by this body scope only
     * @throws IllegalArgumentException if positional bindings, row flow or shared carry use fail
     * @throws ArithmeticException if a nonempty fixed-feature product overflows, even at T=0/B=0
     */
    private static void validateRows(CompiledNode node, List<TensorDescriptor> inputs,
            Map<ValueId, TensorDescriptor> local) {
        ScanAttrs a = (ScanAttrs) node.operation().attrs();
        CompiledScanBody b = node.scanBody().orElseThrow();
        TensorDescriptor lengths = inputs.getLast();
        if (lengths.dataType() != DataType.INT64 || lengths.requiresGrad()
                || lengths.shape().rank() != 1
                || !lengths.shape().isFullyStatic() && lengths.layout().isPresent()) throw rowFailure();
        Dimension batch = lengths.shape().dimension(0);
        if (!(batch instanceof StaticDimension) && !(batch instanceof DynamicDimension)) throw rowFailure();
        lengths.shape().knownElementCount();
        Map<ValueId, Scan.ValueSemantics> facts = new HashMap<>();
        Map<ValueId, Integer> actualRoles = new HashMap<>();
        Map<ValueId, ValueId> sharedRoots = new HashMap<>();
        for (int i = 0; i < b.graph().inputs().size(); i++) {
            ValueId id = b.graph().inputs().get(i);
            int role = -1, outerRole = -1;
            if (i < a.carryCount()) {
                if (i < a.rowCarryCount()) role = outerRole = 0;
                else sharedRoots.put(id, b.nextCarry().get(i));
            } else if (i < a.carryCount() + a.sequenceCount()) {
                role = a.sequenceRowAxes().get(i - a.carryCount()).orElse(-1);
                outerRole = role < 0 ? -1 : role + 1;
            }
            facts.put(id, Scan.declaredSemantics(local.get(id),
                    role < 0 ? OptionalInt.empty() : OptionalInt.of(role), batch));
            Integer previous = actualRoles.put(node.inputs().get(i), outerRole);
            if (previous != null && previous != outerRole) throw rowFailure();
        }
        Set<ValueId> dependent = new HashSet<>(sharedRoots.keySet());
        Map<ValueId, CompiledNode> producers = new HashMap<>();
        Set<NodeId> accumulatorSites = new HashSet<>();
        for (CompiledNode p : b.graph().nodes()) {
            // Every syntactic use is checked before per-output numeric/control routes.
            for (int i = 0; i < p.inputs().size(); i++) {
                ValueId id = p.inputs().get(i);
                if (!facts.containsKey(id)) throw rowFailure();
                if (dependent.contains(id) && (i != 0 || p.operation().kind() != BinaryArithmeticKind.ADD
                        || !p.outputs().getFirst().equals(sharedRoots.get(id)))) throw rowFailure();
            }
            List<Scan.ValueSemantics> results = Scan.ordinarySemantics(p.operation(),
                    p.inputs().stream().map(local::get).toList(),
                    p.inputs().stream().map(facts::get).toList(),
                    p.outputs().stream().map(local::get).toList(), batch);
            for (int slot = 0; slot < p.outputs().size(); slot++) {
                ValueId id = p.outputs().get(slot);
                Scan.ValueSemantics fact = results.get(slot);
                facts.put(id, fact);
                producers.put(id, p);
                if (fact.directInputs().stream().anyMatch(i -> dependent.contains(p.inputs().get(i)))) dependent.add(id);
            }
        }
        for (int i = 0; i < a.carryCount(); i++) {
            ValueId root = b.nextCarry().get(i);
            if (i < a.rowCarryCount()) {
                if (facts.get(root).rowAxis().orElse(-1) != 0 || facts.get(root).terminal()) throw rowFailure();
            } else {
                CompiledNode p = producers.get(root);
                if (p == null || p.operation().kind() != BinaryArithmeticKind.ADD || p.inputs().size() != 2
                        || !p.inputs().getFirst().equals(b.graph().inputs().get(i))
                        || facts.get(root).rowAxis().isPresent() || facts.get(p.inputs().get(1)).rowAxis().isPresent()
                        || !local.get(p.inputs().get(1)).shape().equals(local.get(root).shape())
                        || local.get(p.inputs().get(1)).dataType() != local.get(root).dataType()
                        || dependent.contains(p.inputs().get(1)) || !accumulatorSites.add(p.id())) throw rowFailure();
            }
        }
        for (ValueId id : b.emissions())
            if (facts.get(id).rowAxis().orElse(-1) != 0 || facts.get(id).terminal()) throw rowFailure();
    }


    /**
     * Creates the diagnostic for an unsupported or unproved length-mode coordinate mapping.
     * @return a fresh fail-closed exception, never null
     */
    private static IllegalArgumentException rowFailure() {
        return new IllegalArgumentException("invalid or unproved length-aware scan row semantics");
    }

    /**
     * Verifies the canonical original-primal inventory without comparing IDs across scopes.
     * @param body non-null complete local graph and ordered roles
     * @param carryCount nonnegative number of leading declared carried formals
     * @throws IllegalArgumentException if retained roles are missing, duplicated or reordered
     */
    private static void validateResiduals(CompiledScanBody body, int carryCount) {
        Map<ValueId, CompiledNode> producers = new HashMap<>();
        body.graph().nodes().forEach(n -> n.outputs().forEach(v -> producers.put(v, n)));
        Set<ValueId> carry = new HashSet<>(body.graph().inputs().subList(0, carryCount));
        Set<ValueId> seen = new HashSet<>();
        Set<NodeId> visited = new HashSet<>();
        var retained = new ArrayList<ValueId>();
        for (ValueId root : body.nextCarry()) retain(root, producers, carry, seen, visited, retained);
        for (ValueId root : body.emissions()) retain(root, producers, carry, seen, visited, retained);
        for (ValueId input : body.graph().inputs().subList(0, carryCount)) {
            if (seen.add(input)) retained.add(input);
        }
        if (!retained.equals(body.residuals()) || visited.size() != body.graph().nodes().size()) {
            throw new IllegalArgumentException("scan residual roles must preserve canonical saved-value inventory/order");
        }
    }

    /**
     * Visits one local root in source producer postorder and ascending canonical sibling order.
     * Explicit entry/completion events preserve depth-first encounter order without recursion.
     * @param value non-null root in the current body scope
     * @param producers local output-to-producer lookup
     * @param carry local carried-input set; other inputs do not need residual ports
     * @param seen mutable unique retained-value set
     * @param visited mutable visited producer set
     * @param retained mutable ordered canonical inventory
     */
    private static void retain(ValueId value, Map<ValueId, CompiledNode> producers,
            Set<ValueId> carry, Set<ValueId> seen, Set<NodeId> visited, List<ValueId> retained) {
        var pending = new ArrayDeque<ResidualVisit>();
        pending.addLast(new ResidualVisit(value, false));
        while (!pending.isEmpty()) {
            ResidualVisit visit = pending.removeLast();
            CompiledNode producer = producers.get(visit.value());
            if (producer == null) {
                if (carry.contains(visit.value()) && seen.add(visit.value())) retained.add(visit.value());
            } else if (visit.complete()) {
                for (ValueId output : producer.outputs()) if (seen.add(output)) retained.add(output);
            } else if (visited.add(producer.id())) {
                pending.addLast(new ResidualVisit(visit.value(), true));
                for (int i = producer.inputs().size() - 1; i >= 0; i--) {
                    pending.addLast(new ResidualVisit(producer.inputs().get(i), false));
                }
            }
        }
    }

    /**
     * Scope-local postorder work; IDs are compared only inside their containing body.
     * @param value root or operand local ID
     * @param complete whether its producer's ordered operands have completed
     */
    private record ResidualVisit(ValueId value, boolean complete) {}

    /**
     * Refreshes demand from final outer publications and actual input edges, never sibling liveness.
     * @param validated non-null graph and aligned compile-local sidecars
     * @return same result if unchanged, otherwise revalidated graph with identical IDs and roles
     */
    static ValidatedGraph refreshDemand(ValidatedGraph validated) {
        CompiledGraphModel graph = validated.graph();
        Set<ValueId> used = new HashSet<>(graph.outputs());
        graph.nodes().forEach(n -> used.addAll(n.inputs()));
        var nodes = new ArrayList<CompiledNode>();
        boolean changed = false;
        for (CompiledNode node : graph.nodes()) {
            if (node.scanBody().isPresent()) {
                CompiledScanBody body = node.scanBody().orElseThrow();
                var demand = new ArrayList<Integer>();
                for (int i = 0; i < node.outputs().size(); i++) if (used.contains(node.outputs().get(i))) demand.add(i);
                if (!demand.equals(body.demandedOutputIndices())) {
                    body = new CompiledScanBody(body.graph(), body.nextCarry(), body.emissions(), body.residuals(), demand);
                    node = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(body));
                    changed = true;
                }
            }
            nodes.add(node);
        }
        if (!changed) return validated;
        var rebuilt = new CompiledGraphModel(graph.values(), nodes, graph.inputs(), graph.outputs(), graph.nodePhases());
        return CapturedGraphInference.inferAndValidate(
                validated.constantGraph().replaceGraphPreservingInputRoles(rebuilt),
                new DerivativeGraphMetadata(rebuilt, validated.derivatives().derivativeOrderByNode(),
                        validated.derivatives().scanActivityByNode()));
    }
}
