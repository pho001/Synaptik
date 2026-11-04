package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;

/**
 * Builds a bounded native binder around the actual neutral compiled body and ordinary Metal
 * lowering. No source Tensor, derivative formula, Runtime graph or invocation value is consulted.
 * Static F32/BF16/F16 computation uses the same ordinary owners and original typed storage.
 */
final class MetalScanLowering {
    private MetalScanLowering() {}

    /**
     * Independently proves the full captured occurrence without reading any tensor contents.
     * @param query containing operation, ordered boundaries, exact body and optional activity
     * @return true only when every declaration and ordinary body instruction is qualified
     */
    static boolean supports(OperationCapabilityQuery query) {
        try {
            plan(query);
            return true;
        } catch (IllegalArgumentException | ArithmeticException | java.util.NoSuchElementException unsupported) {
            return false;
        }
    }

    /**
     * Lowers one containing Scan to its authenticated once-encoded native body/binder payload.
     * @param query independently revalidated containing occurrence
     * @param inputs ordered outer program input ordinals, including final lengths in length mode
     * @param outputs demanded outer result ordinals, in increasing demanded-role order
     * @return one native GENERAL_SCAN instruction, never a time-expanded image
     * @throws IllegalArgumentException if the occurrence or boundary cardinalities are unsupported
     * @throws ArithmeticException if checked geometry or encoding size overflows
     */
    static MetalMpsGraphProgram.Node lower(OperationCapabilityQuery query, int[] inputs, int[] outputs) {
        MetalScanPlan plan = plan(query);
        if (inputs.length != plan.header().inputCount() || outputs.length != plan.demanded().size()) {
            throw invalid("outer ordinal cardinality");
        }
        return MetalMpsGraphProgram.Node.generic(MetalMpsGraphProgram.NodeKind.GENERAL_SCAN,
                inputs, outputs, MetalMpsGraphProgram.AttributeKind.GENERAL_SCAN, MetalScanEncoding.encode(plan));
    }

    /**
     * Independently validates raw program metadata, without trusting a provider/preparer plan.
     * @param node raw GENERAL_SCAN node whose result references name demanded roles only
     * @param values complete outer canonical physical descriptor table
     * @throws IllegalArgumentException if any body, binding, resource or outer fact is malformed
     */
    static void validate(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> values) {
        try {
            MetalScanDecoding.validateOuter(node, values);
        } catch (ArithmeticException | IndexOutOfBoundsException invalid) {
            throw new IllegalArgumentException("Malformed Metal Scan physical occurrence", invalid);
        }
    }

    /**
     * Resolves local physical layouts and freezes binding, demand, control and resource facts.
     * @param query exact containing occurrence; no symbolic dimensions or FLOAT64 arithmetic
     * @return immutable bounded plan retaining the actual compiled graph and body association
     * @throws IllegalArgumentException if any binding, local instruction or physical fact is invalid
     * @throws ArithmeticException if checked resource arithmetic overflows
     */
    static MetalScanPlan plan(OperationCapabilityQuery query) {
        if (query.operation().kind() != ScanKind.SCAN || !(query.operation().attrs() instanceof ScanAttrs attrs)) {
            throw invalid("not a general Scan");
        }
        var body = query.scanBody().orElseThrow(() -> invalid("missing captured body"));
        var graph = body.graph();
        if (graph.values().size() > MetalScanEncoding.MAX_VALUES || graph.nodes().size() > MetalScanEncoding.MAX_UNITS
                || query.inputs().size() > MetalScanEncoding.MAX_VALUES || query.outputs().size() > MetalScanEncoding.MAX_VALUES
                || query.inputs().size() != attrs.inputCount() || query.outputs().size() != attrs.outputCount()
                || body.nextCarry().size() != attrs.carryCount() || body.emissions().size() != attrs.emissionCount()
                || body.residuals().size() != attrs.residualCount()
                || graph.inputs().size() != attrs.carryCount() + attrs.sequenceCount() + attrs.invariantCount()) {
            throw invalid("header/body cardinality");
        }
        query.scanActivityPlan().ifPresent(activity -> {
            long roots = (long) activity.carries().size() + activity.seedFormals().size();
            if (roots > MetalScanEncoding.MAX_VALUES || activity.scopes().size() > MetalScanEncoding.MAX_VALUES + MetalScanEncoding.MAX_UNITS
                    || Math.multiplyExact((roots + 63) / 64, activity.scopes().size() + roots) > MetalScanEncoding.MAX_WORDS) {
                throw invalid("bounded structural activity table");
            }
        });
        var projected = new LinkedHashMap<ValueId, GraphValue>();
        for (var value : graph.values()) {
            checkTypeShape(value.descriptor());
            projected.put(value.id(), value);
        }
        query.inputs().forEach(MetalScanLowering::checkInputBoundary);
        query.outputs().forEach(MetalScanLowering::checkBoundary);
        var roles = new ArrayList<>(body.nextCarry());
        roles.addAll(body.emissions());
        roles.addAll(body.residuals());
        boolean gradient = query.inputs().subList(0, graph.inputs().size()).stream().anyMatch(TensorDescriptor::requiresGrad);
        for (int f = 0; f < graph.inputs().size(); f++) {
            var formal = projected.get(graph.inputs().get(f)).descriptor();
            var actual = query.inputs().get(f);
            var shape = actual.shape().dimensions();
            if (f >= attrs.carryCount() && f < attrs.carryCount() + attrs.sequenceCount()) {
                if (shape.isEmpty() || actual.shape().dimension(0).staticSize().orElseThrow() != attrs.timeCount()) {
                    throw invalid("sequence time extent");
                }
                shape = shape.subList(1, shape.size());
            }
            if (actual.dataType() != formal.dataType() || actual.requiresGrad() != formal.requiresGrad()
                    || !shape.equals(formal.shape().dimensions()) || !canonical(formal)) {
                throw invalid("formal/actual descriptor");
            }
        }
        for (int p = 0; p < roles.size(); p++) {
            var local = projected.get(roles.get(p)).descriptor();
            var outer = query.outputs().get(p);
            var shape = outer.shape().dimensions();
            if (p >= attrs.carryCount()) {
                if (shape.isEmpty() || outer.shape().dimension(0).staticSize().orElseThrow() != attrs.timeCount()) {
                    throw invalid("history time extent");
                }
                shape = shape.subList(1, shape.size());
            }
            if (outer.dataType() != local.dataType() || !shape.equals(local.shape().dimensions())
                    || outer.requiresGrad() != (gradient && local.dataType().isDifferentiable())) {
                throw invalid("canonical role descriptor");
            }
            if (p < attrs.carryCount()) {
                var formal = projected.get(graph.inputs().get(p)).descriptor();
                if (local.dataType() != formal.dataType() || !local.shape().equals(formal.shape())) {
                    throw invalid("next carry declaration");
                }
            }
        }
        long batch = 0;
        Dimension batchDimension = null;
        if (attrs.lengthAware()) {
            var lengths = query.inputs().getLast();
            if (lengths.dataType() != DataType.INT64 || lengths.requiresGrad() || lengths.shape().rank() != 1) {
                throw invalid("lengths must be non-gradient INT64[B]");
            }
            batchDimension = lengths.shape().dimension(0);
            batch = batchDimension.staticSize().orElseThrow();
        }
        var semantics = new LinkedHashMap<ValueId, Scan.ValueSemantics>();
        for (int f = 0; f < graph.inputs().size(); f++) {
            int axis = !attrs.lengthAware() ? -1 : f < attrs.carryCount() ? f < attrs.rowCarryCount() ? 0 : -1
                    : f < attrs.carryCount() + attrs.sequenceCount()
                            ? attrs.sequenceRowAxes().get(f - attrs.carryCount()).orElse(-1) : -1;
            semantics.put(graph.inputs().get(f), attrs.lengthAware()
                    ? Scan.declaredSemantics(projected.get(graph.inputs().get(f)).descriptor(),
                            axis < 0 ? OptionalInt.empty() : OptionalInt.of(axis), batchDimension)
                    : new Scan.ValueSemantics(OptionalInt.empty(), false, List.of()));
        }
        var ordinary = new ArrayList<MetalMpsGraphProgram.Node>();
        var ordinals = new LinkedHashMap<ValueId, Integer>();
        int ordinal = 0;
        for (ValueId id : projected.keySet()) ordinals.put(id, ordinal++);
        for (var node : graph.nodes()) {
            var operands = node.inputs().stream().map(id -> projected.get(id).descriptor()).toList();
            var declared = node.outputs().stream().map(id -> projected.get(id).descriptor()).toList();
            if (java.util.stream.Stream.concat(operands.stream(), declared.stream())
                    .anyMatch(value -> value.dataType() == DataType.FLOAT64)) {
                throw invalid("FLOAT64 Scan storage is source-only; no ordinary F64 body instruction");
            }
            var inferred = ScanBody.describeOrdinary(node.operation(), operands, declared.size(), Optional.empty());
            for (int port = 0; port < declared.size(); port++) {
                var expected = inferred.get(port); var actual = declared.get(port);
                if (expected.dataType() != actual.dataType() || !expected.shape().equals(actual.shape())
                        || expected.requiresGrad() != actual.requiresGrad()) throw invalid("ordinary result declaration");
            }
            var outputs = ScanBody.closeOrdinaryLayouts(node.operation(), operands, declared);
            for (int port = 0; port < outputs.size(); port++) {
                projected.put(node.outputs().get(port), new GraphValue(node.outputs().get(port), outputs.get(port)));
            }
            if (!MetalCapabilityProvider.supportsScanBody(new OperationCapabilityQuery(node.operation(), operands, outputs))) {
                throw invalid("unsupported ordinary body instruction: " + node.operation().kind());
            }
            var facts = attrs.lengthAware() ? Scan.ordinarySemantics(node.operation(), operands,
                    node.inputs().stream().map(semantics::get).toList(), outputs, batchDimension)
                    : outputs.stream().map(unused -> new Scan.ValueSemantics(OptionalInt.empty(), false, List.of())).toList();
            for (int port = 0; port < outputs.size(); port++) semantics.put(node.outputs().get(port), facts.get(port));
            ordinary.add(MetalNegPartitionPreparer.lowerScanBody(node.operation(), indexes(node.inputs(), ordinals),
                    indexes(node.outputs(), ordinals), operands, outputs));
        }
        validateLengthFeedback(attrs, body.graph().nodes(), body.graph().inputs(), body.nextCarry(), body.emissions(), semantics);
        var protectedValues = new HashSet<>(roles);
        // Reuse Prepare's exact local graph projection and ordinary logical-memory requirements.
        if (!graph.nodes().isEmpty()) {
            var partition = new PlannedPartition(MetalCapabilityProvider.METAL_BACKEND_ID,
                    graph.nodes().stream().map(CompiledNode::id).toList());
            var logical = LogicalMemoryPlanning.plan(graph, List.of(partition));
            var requirements = new ArrayList<LogicalMemoryRequirement>();
            for (var r : logical.requirements()) requirements.add(new LogicalMemoryRequirement(r.valueId(),
                    projected.get(r.valueId()).descriptor(), r.producerPartition(), r.consumerPartitions(),
                    protectedValues.contains(r.valueId())));
            new PrepareContext<>(new PartitionDag(partition, graph.nodes()), List.copyOf(projected.values()),
                    requirements, Map.of(), new AnalysisInputs(), Map.of());
        }
        long cursor = 0;
        var externalFormals = new HashSet<>(graph.inputs().subList(attrs.carryCount(), graph.inputs().size()));
        var values = new ArrayList<MetalScanPlan.Value>();
        for (var entry : projected.entrySet()) {
            var logicalDescriptor = entry.getValue().descriptor();
            var descriptor = new TensorDescriptor(logicalDescriptor.dataType(), logicalDescriptor.shape(),
                    Optional.of(LayoutDescriptor.contiguous(logicalDescriptor.shape())), logicalDescriptor.requiresGrad());
            var facts = semantics.get(entry.getKey());
            if (descriptor.layout().isEmpty()) throw invalid("unresolved local physical layout");
            long bytes = Math.multiplyExact(descriptor.layout().orElseThrow().referencedElementSpan(), descriptor.dataType().byteWidth());
            cursor = aligned(cursor);
            long offset = externalFormals.contains(entry.getKey()) ? -1 : cursor;
            if (offset >= 0) cursor = Math.addExact(cursor, bytes);
            int row = facts.rowAxis().orElse(-1);
            long rowInner = 0;
            if (row >= 0) {
                rowInner = 1;
                for (int a = row + 1; a < descriptor.shape().rank(); a++) {
                    rowInner = Math.multiplyExact(rowInner, descriptor.shape().dimension(a).staticSize().orElseThrow());
                }
            }
            values.add(new MetalScanPlan.Value(descriptor, row, facts.terminal(), offset, bytes, rowInner));
        }
        // Unique computed next-carry writers already own an alternate bank. Only formal or
        // repeated next bindings require extra storage to preserve simultaneous transitions.
        var nextUses = new HashMap<ValueId, Integer>();
        body.nextCarry().forEach(id -> nextUses.merge(id, 1, Integer::sum));
        var formalSet = new HashSet<>(graph.inputs());
        for (int c = 0; c < attrs.carryCount(); c++) {
            ValueId next = body.nextCarry().get(c);
            if (formalSet.contains(next) || nextUses.get(next) != 1) {
                cursor = Math.addExact(aligned(cursor), values.get(ordinals.get(graph.inputs().get(c))).byteSize());
            }
        }
        var physicalDescriptors = values.stream().map(v -> MetalMpsGraphProgram.ValueDescriptor.from(v.descriptor())).toList();
        var units = new ArrayList<MetalScanPlan.Unit>();
        for (int n = 0; n < graph.nodes().size(); n++) {
            var node = graph.nodes().get(n);
            var activity = query.scanActivityPlan();
            int scope = activity.map(a -> a.nodeScopes().get(node.id())).orElse(-1);
            var merge = activity.map(a -> a.merges().get(node.id())).orElse(null);
            int accumulator = -1;
            if (activity.isPresent()) for (var a : activity.orElseThrow().accumulators()) {
                if (node.outputs().contains(body.nextCarry().get(a.numericSharedCarryPosition()))) accumulator = a.numericSharedCarryPosition();
            }
            int row = semantics.get(node.outputs().getFirst()).rowAxis().orElse(-1);
            boolean contraction = row < 0 && node.inputs().stream().anyMatch(id -> semantics.get(id).rowAxis().isPresent());
            long scratchBytes = MetalProgramWorkspace.scratchBytes(ordinary.get(n), physicalDescriptors);
            long scratchOffset = scratchBytes == 0 ? -1 : aligned(cursor);
            if (scratchBytes != 0) cursor = Math.addExact(scratchOffset, scratchBytes);
            units.add(new MetalScanPlan.Unit(ordinary.get(n), scope, merge == null ? -1 : merge.leftScope(),
                    merge == null ? -1 : merge.rightScope(), row, contraction, accumulator, scratchOffset, scratchBytes));
        }
        long controlOffset = aligned(cursor);
        long roots = query.scanActivityPlan().map(a -> (long) a.carries().size() + a.seedFormals().size()).orElse(0L);
        long scopes = query.scanActivityPlan().map(a -> (long) a.scopes().size()).orElse(0L);
        long rootWords = Math.addExact(roots, 63) / 64;
        long rows = attrs.lengthAware() ? batch : 1;
        long controlWords = Math.addExact(10, Math.addExact(Math.multiplyExact(Math.addExact(scopes, roots), rootWords),
                Math.addExact(Math.multiplyExact(Math.multiplyExact(rows, rootWords), 2), attrs.lengthAware() ? batch : 0)));
        long workspace = Math.addExact(controlOffset, Math.multiplyExact(controlWords, 8));
        var plan = new MetalScanPlan(attrs, body, values, units, graph.inputs().stream().map(ordinals::get).toList(),
                roles.stream().map(ordinals::get).toList(), body.demandedOutputIndices(), query.scanActivityPlan(),
                batch, workspace, controlOffset, controlWords);
        MetalScanEncoding.wordCount(plan);
        return plan;
    }

    private static void validateLengthFeedback(ScanAttrs attrs, List<CompiledNode> nodes, List<ValueId> inputs,
            List<ValueId> next, List<ValueId> emissions, Map<ValueId, Scan.ValueSemantics> semantics) {
        if (!attrs.lengthAware()) return;
        var dependent = new HashSet<ValueId>();
        var allowed = new HashMap<ValueId, ValueId>();
        for (int c = attrs.rowCarryCount(); c < attrs.carryCount(); c++) {
            dependent.add(inputs.get(c)); allowed.put(inputs.get(c), next.get(c));
        }
        for (var node : nodes) for (int i = 0; i < node.inputs().size(); i++) {
            var input = node.inputs().get(i);
            if (dependent.contains(input)) {
                if (node.operation().kind() != BinaryArithmeticKind.ADD || i != 0 || node.outputs().size() != 1
                        || !node.outputs().getFirst().equals(allowed.get(input))) throw invalid("shared-state feedback");
                dependent.add(node.outputs().getFirst());
            }
        }
        for (int c = 0; c < attrs.carryCount(); c++) {
            var facts = semantics.get(next.get(c));
            if (c < attrs.rowCarryCount()) {
                if (facts.rowAxis().orElse(-1) != 0 || facts.terminal()) throw invalid("ROW carry provenance");
            } else {
                final int carry = c;
                var producer = nodes.stream().filter(n -> n.outputs().contains(next.get(carry))).findFirst()
                        .orElseThrow(() -> invalid("missing shared accumulation"));
                if (producer.operation().kind() != BinaryArithmeticKind.ADD || producer.inputs().size() != 2
                        || !producer.inputs().getFirst().equals(inputs.get(c)) || dependent.contains(producer.inputs().get(1))
                        || facts.rowAxis().isPresent() || semantics.get(producer.inputs().get(1)).rowAxis().isPresent()) {
                    throw invalid("separate ordered shared carry ADD");
                }
            }
        }
        for (ValueId id : emissions) if (semantics.get(id).rowAxis().orElse(-1) != 0 || semantics.get(id).terminal()) {
            throw invalid("ROW emission provenance");
        }
    }

    private static void checkTypeShape(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic() || descriptor.shape().rank() > 16
                || (!numericCarrier(descriptor.dataType()) && descriptor.dataType() != DataType.BOOL
                        && descriptor.dataType() != DataType.INT32 && descriptor.dataType() != DataType.INT64
                        && descriptor.dataType() != DataType.FLOAT64)
                || descriptor.dataType() == DataType.FLOAT64 && descriptor.requiresGrad()) {
            throw invalid("static numeric/typed-control or non-gradient raw F64 domain");
        }
        long elements = 1;
        for (var dimension : descriptor.shape().dimensions()) elements = Math.multiplyExact(elements, dimension.staticSize().orElseThrow());
        Math.multiplyExact(elements, descriptor.dataType().byteWidth());
    }

    static boolean numericCarrier(DataType type) {
        return type == DataType.FLOAT32 || type == DataType.BFLOAT16 || type == DataType.FLOAT16;
    }

    /**
     * Checks logical input geometry without mistaking a produced view for represented storage.
     * Authenticated external numeric layouts are mapped once by the ordinary CONTIGUOUS owner;
     * produced views retain their independently proved canonical representation.
     */
    private static void checkInputBoundary(TensorDescriptor descriptor) {
        checkTypeShape(descriptor);
        if (descriptor.dataType() == DataType.FLOAT64 && !canonical(descriptor)) {
            throw invalid("raw F64 Scan input requires canonical storage");
        }
        var physical = MetalMpsGraphProgram.ValueDescriptor.from(descriptor);
        boolean emptyView = physical.elementCount() == 0 && descriptor.layout()
                .filter(layout -> layout.isView()
                        && java.util.Arrays.stream(layout.strides()).allMatch(stride -> stride >= 0)).isPresent();
        if (!physical.supportsStorageLayout(0) && !emptyView) {
            throw invalid("resolved Scan input layout required");
        }
    }

    private static void checkBoundary(TensorDescriptor descriptor) {
        checkTypeShape(descriptor);
        if (!canonical(descriptor)) throw invalid("canonical Scan boundary required");
    }

    private static boolean canonical(TensorDescriptor descriptor) {
        return descriptor.layout().filter(layout -> layout.equals(LayoutDescriptor.contiguous(descriptor.shape()))).isPresent();
    }

    private static int[] indexes(List<ValueId> ids, Map<ValueId, Integer> ordinals) {
        return ids.stream().mapToInt(id -> {
            Integer value = ordinals.get(id);
            if (value == null) throw invalid("nonlocal value reference");
            return value;
        }).toArray();
    }

    private static long aligned(long bytes) { return Math.addExact(bytes, 7) & ~7L; }
    private static IllegalArgumentException invalid(String reason) { return new IllegalArgumentException("Metal Scan: " + reason); }

    /** Immutable local analysis role; cold proof needs no device context or native state. */
    private record AnalysisInputs() implements io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs {}
}
