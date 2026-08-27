package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionUnitPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Stateless bounded recognizer for the closed CPU 0008C family matrix. It scans the shared DAG's
 * stable partition order, uses its exact producer and consumer occurrences for privacy and
 * single-use gates, never rewrites graph state, and associates retained supported facts only
 * after the independently selected 0008B baseline is available. Family attempts, member
 * positions, baseline-unit associations, and recognition dispositions remain CPU-owned.
 * Recognition is cold metadata only: it does not change capability, lowering, generated artifact
 * identity, finalization, or hot dispatch.
 */
public final class CpuSpecializedSubgraphRecognizer {
    /** Maximum topology/family attempts made by one recognition scan. */
    static final int MAX_ATTEMPTS = 24;
    /** Maximum facts retained by one partition. */
    static final int MAX_FACTS = 8;
    /** Maximum members in one fact. */
    static final int MAX_MEMBERS = 6;
    /** Maximum referenced semantic input/result positions in one fact. */
    static final int MAX_BOUNDARIES = 10;
    /** Maximum baseline units associated with one fact. */
    static final int MAX_UNITS = 2;

    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates a stateless cold recognizer. */
    public CpuSpecializedSubgraphRecognizer() { }

    /**
     * Recognizes supported facts and validates their exact baseline-unit association.
     *
     * @param context complete non-null stable CPU partition projection whose shared DAG supplies
     *     exact producer and consumer occurrences
     * @param baseline non-null successful 0008B execution-unit plans whose stable IR and exact
     *     resource topology are snapshotted into every retained fact
     * @return immutable facts in increasing anchor order
     * @throws NullPointerException if an argument or required fact is null
     * @throws IllegalArgumentException if projection or baseline association is malformed, or an
     *     existing-specialized candidate disagrees with its established IR
     * @throws ArithmeticException if checked recognition arithmetic overflows
     */
    public List<CpuSpecializedSubgraph> recognize(
            PrepareContext<CpuPartitionAnalysisInputs> context,
            List<ExecutionUnitPlan> baseline) {
        Objects.requireNonNull(context, "context");
        List<ExecutionUnitPlan> baselineUnits = List.copyOf(baseline);
        Map<ValueId, GraphValue> values = values(context);
        List<CompiledNode> nodes = context.partitionDag().nodes();
        var claimed = new BitSet(nodes.size());
        var facts = new ArrayList<CpuSpecializedSubgraph>();
        var attempts = new Attempts();
        for (int ordinal = 0; ordinal < nodes.size() && facts.size() < MAX_FACTS; ordinal++) {
            if (claimed.get(ordinal)) continue;
            Candidate selected = null;
            for (RecognizerFamily family : familiesFor(nodes, ordinal)) {
                if (!attempts.take()) return List.copyOf(facts);
                Candidate candidate = candidate(context, values, ordinal, family, attempts);
                if (attempts.exhausted()) return List.copyOf(facts);
                if (candidate == null) continue;
                if (candidate.members().stream().anyMatch(claimed::get)) continue;
                selected = candidate;
                break;
            }
            if (selected == null) continue;
            List<Integer> units = associate(selected.members(), baselineUnits);
            if (units.isEmpty() || facts.size() >= MAX_FACTS
                    || !candidateWithinBudgets(selected.members().size(),
                        selected.epilogue().operationCount(), selected.accesses().size(),
                        units.size())) continue;
            List<BaselineUnitFact> baselineFacts = units.stream()
                    .map(index -> baselineFact(baselineUnits.get(index))).toList();
            CpuSpecializedSubgraph fact = selected.toFact(units, baselineFacts);
            validateExistingSpecialized(fact, baselineUnits, units);
            facts.add(fact);
            selected.members().forEach(claimed::set);
        }
        return List.copyOf(facts);
    }

    /**
     * Performs focused MATMUL recognition without admitting the unsupported anchor to a plan.
     * This method is diagnostic cold analysis only and returns an empty baseline association.
     *
     * @param context complete non-null projection whose shared DAG supplies exact producer and
     *     consumer occurrences and whose anchor may be unsupported by CPU
     * @return immutable zero-or-one MATMUL fact list
     * @throws NullPointerException if {@code context} or a required fact is null
     * @throws IllegalArgumentException if projected graph facts are malformed
     */
    public List<MatmulEpilogue> recognizeUnsupportedMatmul(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        Map<ValueId, GraphValue> values = values(context);
        var attempts = new Attempts();
        for (int ordinal = 0; ordinal < context.partitionDag().nodes().size(); ordinal++) {
            if (!attempts.take()) return List.of();
            Candidate candidate = candidate(context, values, ordinal, RecognizerFamily.MATMUL,
                    attempts);
            if (candidate != null) return List.of((MatmulEpilogue) candidate.toFact(
                    List.of(), List.of()));
        }
        return List.of();
    }

    private Candidate candidate(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int ordinal, RecognizerFamily family,
            Attempts attempts) {
        return switch (family) {
            case CONV1D_COMPOSITION -> convolution(context, values, ordinal, 1, attempts);
            case CONV2D -> convolution(context, values, ordinal, 2, attempts);
            case CONV3D -> convolution(context, values, ordinal, 3, attempts);
            case MATMUL -> matmul(context, values, ordinal, attempts);
            case REDUCTION -> reduction(context, values, ordinal, attempts);
            case EXPLICIT_SEMANTIC_KERNEL -> explicit(context, values, ordinal);
        };
    }

    private Candidate convolution(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int ordinal, int dimensions, Attempts attempts) {
        List<CompiledNode> nodes = context.partitionDag().nodes();
        int anchorEnd = ordinal;
        Form form;
        ConvolutionAttributes attributes;
        CompiledNode anchor;
        if (dimensions == 1) {
            if (ordinal + 3 >= nodes.size()) return null;
            List<CompiledNode> four = nodes.subList(ordinal, ordinal + 4);
            if (!conv1dShape(four)) return null;
            anchor = four.get(2); anchorEnd = ordinal + 3; form = Form.CONV1D_COMPOSITION;
            Conv2dAttrs attrs = (Conv2dAttrs) anchor.operation().attrs();
            attributes = new ConvolutionAttributes(1, List.of(attrs.strideWidth()),
                    List.of(attrs.paddingWidth()), List.of(attrs.dilationWidth()), attrs.groups(),
                    anchor.inputs().size() == 3);
        } else if (dimensions == 2) {
            anchor = nodes.get(ordinal);
            if (anchor.operation().kind() != Conv2dKind.CONV2D
                    || !(anchor.operation().attrs() instanceof Conv2dAttrs attrs)) return null;
            form = Form.CONV2D;
            attributes = new ConvolutionAttributes(2,
                    List.of(attrs.strideHeight(), attrs.strideWidth()),
                    List.of(attrs.paddingHeight(), attrs.paddingWidth()),
                    List.of(attrs.dilationHeight(), attrs.dilationWidth()), attrs.groups(),
                    anchor.inputs().size() == 3);
        } else {
            anchor = nodes.get(ordinal);
            if (anchor.operation().kind() != Conv3dKind.CONV3D
                    || !(anchor.operation().attrs() instanceof Conv3dAttrs attrs)) return null;
            form = Form.CONV3D;
            attributes = new ConvolutionAttributes(3,
                    List.of(attrs.strideDepth(), attrs.strideHeight(), attrs.strideWidth()),
                    List.of(attrs.paddingDepth(), attrs.paddingHeight(), attrs.paddingWidth()),
                    List.of(attrs.dilationDepth(), attrs.dilationHeight(), attrs.dilationWidth()),
                    attrs.groups(), anchor.inputs().size() == 3);
        }
        if (!supportedOccurrence(anchor, values)) return null;
        ValueId output = nodes.get(anchorEnd).outputs().getFirst();
        DataType resultType = require(values, output).descriptor().dataType();
        Suffix longest = floating(resultType) ? suffix(context, values, anchorEnd, output)
                : Suffix.none(anchorEnd);
        Suffix suffix = selectSuffix(context, values, ordinal, anchorEnd, longest, attempts);
        if (suffix == null) return null;
        List<Integer> members = range(ordinal, suffix.end());
        ExecutionDisposition disposition = dimensions == 2
                && suffix.epilogue().addInputOrder() != AddInputOrder.NONE
                && (suffix.epilogue().terminal() == Terminal.NONE
                    || suffix.epilogue().terminal() == Terminal.RELU)
                ? ExecutionDisposition.EXISTING_SPECIALIZED : ExecutionDisposition.ORDINARY_SPLIT;
        return build(Family.CONVOLUTION, form, attributes, members, anchor.inputs(),
                nodes.get(suffix.end()).outputs(), values, suffix.epilogue(), disposition);
    }

    private Candidate matmul(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int ordinal, Attempts attempts) {
        CompiledNode node = context.partitionDag().nodes().get(ordinal);
        if (node.operation().kind() != MatmulKind.MATMUL
                || node.operation().attrs() != NoOperationAttrs.INSTANCE
                || node.inputs().size() != 2 || node.outputs().size() != 1) return null;
        DataType resultType = require(values, node.outputs().getFirst()).descriptor().dataType();
        if (resultType == DataType.BOOL) return null;
        MatmulInputForm inputForm = transposedWeight(context, values, node.inputs().get(1))
                ? MatmulInputForm.TRANSPOSED_WEIGHT : MatmulInputForm.ORDINARY;
        Suffix suffix = floating(resultType) ? suffix(context, values, ordinal, node.outputs().getFirst())
                : Suffix.none(ordinal);
        if (suffix.epilogue().addInputOrder() != AddInputOrder.NONE) {
            GraphValue external = require(values, suffix.externalAdd());
            long[] result = require(values, node.outputs().getFirst()).descriptor().shape().toLongArray();
            long[] bias = external.descriptor().shape().toLongArray();
            if (bias.length != 1 || result.length == 0 || bias[0] != result[result.length - 1])
                suffix = Suffix.none(ordinal);
        }
        suffix = selectSuffix(context, values, ordinal, ordinal, suffix, attempts);
        if (suffix == null) return null;
        List<Integer> members = range(ordinal, suffix.end());
        return build(Family.MATMUL, Form.MATMUL, new MatmulAttributes(inputForm), members,
                node.inputs(), context.partitionDag().nodes().get(suffix.end()).outputs(), values,
                suffix.epilogue(), ExecutionDisposition.UNSUPPORTED_ANCHOR);
    }

    private Candidate reduction(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int ordinal, Attempts attempts) {
        CompiledNode node = context.partitionDag().nodes().get(ordinal);
        if (!(node.operation().kind() instanceof AggregateReductionKind kind)
                || !reductionKind(kind) || node.inputs().size() != 1 || node.outputs().size() != 1)
            return null;
        DataType resultType = require(values, node.outputs().getFirst()).descriptor().dataType();
        if (!floating(resultType)) return null;
        ReductionAttributes attributes = reductionAttributes(kind, node.operation().attrs(),
                require(values, node.inputs().getFirst()).descriptor().shape().rank());
        if (attributes == null || !supportedOccurrence(node, values)) return null;
        Suffix suffix = selectSuffix(context, values, ordinal, ordinal,
                suffix(context, values, ordinal, node.outputs().getFirst()), attempts);
        if (suffix == null) return null;
        List<Integer> members = range(ordinal, suffix.end());
        return build(Family.REDUCTION, reductionForm(kind), attributes, members, node.inputs(),
                context.partitionDag().nodes().get(suffix.end()).outputs(), values, suffix.epilogue(),
                ExecutionDisposition.ORDINARY_SPLIT);
    }

    private Candidate explicit(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int ordinal) {
        CompiledNode node = context.partitionDag().nodes().get(ordinal);
        Form form; ExplicitAttributes attrs;
        Object kind = node.operation().kind(); Object raw = node.operation().attrs();
        if (kind instanceof SoftmaxKind softmax && raw instanceof SoftmaxAttrs value) {
            form = softmax == SoftmaxKind.SOFTMAX ? Form.SOFTMAX : Form.LOG_SOFTMAX;
            attrs = new ExplicitAttributes(softmax == SoftmaxKind.SOFTMAX
                    ? ExplicitForm.SOFTMAX : ExplicitForm.LOG_SOFTMAX, value.axis(), Shape.scalar(),
                    Optional.empty(), Optional.empty());
        } else if (kind == LayerNormKind.LAYER_NORM && raw instanceof LayerNormAttrs value) {
            form = Form.LAYER_NORM; attrs = new ExplicitAttributes(ExplicitForm.LAYER, -1,
                    value.normalizedShape(), Optional.of(value.epsilon()), Optional.empty());
        } else if (kind == LayerNormKind.LAYER_NORM && raw instanceof AffineLayerNormAttrs value) {
            form = Form.LAYER_NORM; attrs = new ExplicitAttributes(ExplicitForm.LAYER_AFFINE, -1,
                    value.normalizedShape(), Optional.of(value.epsilon()), Optional.empty());
        } else if (kind == RmsNormKind.RMS_NORM && raw instanceof RmsNormAttrs value) {
            form = Form.RMS_NORM; attrs = new ExplicitAttributes(node.inputs().size() == 1
                    ? ExplicitForm.RMS : ExplicitForm.RMS_SCALED, -1, value.normalizedShape(),
                    Optional.of(value.epsilon()), Optional.empty());
        } else if (kind == BatchNormKind.BATCH_NORM_INFERENCE
                && raw instanceof BatchNormInferenceAttrs value) {
            form = Form.BATCH_NORM_INFERENCE; attrs = new ExplicitAttributes(
                    ExplicitForm.BATCH_INFERENCE, value.channelAxis(), Shape.scalar(),
                    Optional.of(value.epsilon()), Optional.empty());
        } else if (kind == BatchNormKind.BATCH_NORM_TRAINING
                && raw instanceof BatchNormTrainingAttrs value) {
            form = Form.BATCH_NORM_TRAINING; attrs = new ExplicitAttributes(
                    ExplicitForm.BATCH_TRAINING, value.channelAxis(), Shape.scalar(),
                    Optional.of(value.momentum()), Optional.of(value.epsilon()));
        } else return null;
        if (!supportedOccurrence(node, values) || !candidateEligible(context, values,
                List.of(ordinal), Suffix.none(ordinal))) return null;
        return build(Family.EXPLICIT_SEMANTIC_KERNEL, form, attrs, List.of(ordinal), node.inputs(),
                node.outputs(), values, Epilogue.none(), ExecutionDisposition.EXISTING_SPECIALIZED);
    }

    private Candidate build(Family family, Form form, AnchorAttributes attributes,
            List<Integer> members, List<ValueId> inputs, List<ValueId> results,
            Map<ValueId, GraphValue> values, Epilogue epilogue,
            ExecutionDisposition disposition) {
        List<DataType> inputTypes = inputs.stream().map(id -> require(values, id).descriptor().dataType()).toList();
        List<DataType> resultTypes = results.stream().map(id -> require(values, id).descriptor().dataType()).toList();
        var accesses = new ArrayList<AccessFact>(); inputs.forEach(id -> accesses.add(access(require(values, id))));
        results.forEach(id -> accesses.add(access(require(values, id))));
        if (!candidateWithinBudgets(members.size(), epilogue.operationCount(),
                accesses.size(), 0))
            return null;
        var identity = new StructuralIdentity(family, form, inputTypes, resultTypes, accesses,
                attributes, epilogue, List.of());
        return new Candidate(family, form, members, inputTypes, resultTypes, accesses, epilogue,
                disposition, identity);
    }

    private boolean supportedOccurrence(CompiledNode node, Map<ValueId, GraphValue> values) {
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(id -> require(values, id).descriptor()).toList(), node.outputs().stream()
                .map(id -> require(values, id).descriptor()).toList());
        return capabilities.supports(query);
    }

    private static Suffix suffix(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int anchorEnd, ValueId anchorOutput) {
        List<CompiledNode> nodes = context.partitionDag().nodes();
        if (anchorEnd + 1 >= nodes.size()) return Suffix.none(anchorEnd);
        CompiledNode next = nodes.get(anchorEnd + 1);
        boolean add = next.operation().kind() == BinaryArithmeticKind.ADD
                && next.operation().attrs() == NoOperationAttrs.INSTANCE
                && next.inputs().size() == 2 && next.outputs().size() == 1
                && context.partitionDag().consumers(anchorOutput).stream()
                        .filter(occurrence -> occurrence.node() == next).count() == 1;
        if (add) {
            ValueId external = next.inputs().getFirst().equals(anchorOutput)
                    ? next.inputs().get(1) : next.inputs().getFirst();
            AddInputOrder order = next.inputs().getFirst().equals(anchorOutput)
                    ? AddInputOrder.PRECEDING_LEFT : AddInputOrder.PRECEDING_RIGHT;
            if (sameResult(values, anchorOutput, next.outputs().getFirst())
                    && broadcasts(require(values, external).descriptor().shape(),
                        require(values, anchorOutput).descriptor().shape())) {
                TerminalFact terminal = anchorEnd + 2 < nodes.size()
                        ? terminal(nodes.get(anchorEnd + 2), next.outputs().getFirst(), values) : null;
                if (terminal != null) return new Suffix(anchorEnd + 2,
                        new Epilogue(order, terminal.terminal(), terminal.clamp()), external);
                return new Suffix(anchorEnd + 1,
                        new Epilogue(order, Terminal.NONE, Optional.empty()), external);
            }
        }
        TerminalFact terminal = terminal(next, anchorOutput, values);
        return terminal == null ? Suffix.none(anchorEnd) : new Suffix(anchorEnd + 1,
                new Epilogue(AddInputOrder.NONE, terminal.terminal(), terminal.clamp()), null);
    }

    private static Suffix selectSuffix(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, int anchorStart, int anchorEnd, Suffix longest,
            Attempts attempts) {
        var candidates = new ArrayList<Suffix>();
        candidates.add(longest);
        if (longest.epilogue().operationCount() == 2) {
            candidates.add(new Suffix(longest.end() - 1,
                    new Epilogue(longest.epilogue().addInputOrder(), Terminal.NONE,
                            Optional.empty()), longest.externalAdd()));
        }
        if (longest.end() != anchorEnd) candidates.add(Suffix.none(anchorEnd));
        for (Suffix candidate : candidates) {
            if (!attempts.take()) return null;
            if (candidateEligible(context, values, range(anchorStart, candidate.end()), candidate))
                return candidate;
        }
        return null;
    }

    private static TerminalFact terminal(CompiledNode node, ValueId input,
            Map<ValueId, GraphValue> values) {
        if (!node.inputs().equals(List.of(input)) || node.outputs().size() != 1
                || !sameResult(values, input, node.outputs().getFirst())) return null;
        if (node.operation().kind() instanceof UnaryElementwiseKind kind
                && node.operation().attrs() == NoOperationAttrs.INSTANCE) {
            Terminal terminal = switch (kind) {
                case RELU -> Terminal.RELU; case SIGMOID -> Terminal.SIGMOID;
                case TANH -> Terminal.TANH; case GELU -> Terminal.GELU;
                case GELU_TANH_APPROXIMATION -> Terminal.GELU_TANH_APPROXIMATION;
                case SILU -> Terminal.SILU; default -> null;
            };
            return terminal == null ? null : new TerminalFact(terminal, Optional.empty());
        }
        if (node.operation().kind() == ScalarElementwiseKind.CLAMP
                && node.operation().attrs() instanceof ClampRangeAttrs attrs)
            return new TerminalFact(Terminal.CLAMP, Optional.of(attrs));
        return null;
    }

    private static boolean candidateEligible(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, List<Integer> members, Suffix suffix) {
        if (members.size() > MAX_MEMBERS || suffix.epilogue().operationCount() > 2) return false;
        Set<ValueId> internal = new HashSet<>();
        for (int i = 0; i < members.size() - 1; i++) {
            CompiledNode node = context.partitionDag().nodes().get(members.get(i));
            if (node.outputs().size() != 1) return false;
            ValueId output = node.outputs().getFirst(); internal.add(output);
            int uses = context.partitionDag().consumers(output).size();
            LogicalMemoryRequirement memory = context.memoryRequirements().stream()
                    .filter(v -> v.valueId().equals(output)).findFirst().orElse(null);
            if (uses != 1 || memory == null || memory.graphOutput()
                    || memory.producerPartition().isEmpty()
                    || !memory.producerPartition().orElseThrow().equals(context.partition())
                    || !memory.consumerPartitions().equals(List.of(context.partition()))) return false;
        }
        for (int ordinal : members) {
            CompiledNode node = context.partitionDag().nodes().get(ordinal);
            for (ValueId id : node.outputs()) if (!eligibleOutput(require(values, id))) return false;
        }
        return true;
    }

    private static boolean candidateWithinBudgets(int members, int epilogueOperations,
            int boundaries, int units) {
        return members <= MAX_MEMBERS && epilogueOperations <= 2
                && boundaries <= MAX_BOUNDARIES && units <= MAX_UNITS;
    }

    private static boolean eligibleOutput(GraphValue value) {
        var descriptor = value.descriptor();
        if (descriptor.layout().isEmpty()) return false;
        var layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        return layout.storageOffset() >= 0 && Arrays.stream(strides).noneMatch(v -> v < 0)
                && injective(descriptor.shape().toLongArray(), strides);
    }

    private static AccessFact access(GraphValue value) {
        var descriptor = value.descriptor(); var layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides(); long[] extents = descriptor.shape().toLongArray();
        if (layout.storageOffset() < 0 || Arrays.stream(strides).anyMatch(v -> v < 0))
            throw new IllegalArgumentException("recognition requires resolved non-negative layout");
        int suffix = 0; long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        CpuAccessPlan.Regime regime = suffix == extents.length ? CpuAccessPlan.Regime.DENSE_LINEAR
                : Arrays.stream(strides).allMatch(v -> v == 0) ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        return new AccessFact(descriptor.dataType(), descriptor.shape(), layout.storageOffset(),
                Arrays.stream(strides).boxed().toList(), regime, injective(extents, strides));
    }

    private static boolean injective(long[] extents, long[] strides) {
        long count = 1;
        for (long extent : extents) { if (extent == 0) return true; count = Math.multiplyExact(count, extent); }
        if (count > 1_000_000) {
            long span = 1; var axes = new ArrayList<Integer>();
            for (int i = 0; i < extents.length; i++) axes.add(i);
            axes.sort(Comparator.comparingLong(i -> strides[i]));
            for (int axis : axes) { if (extents[axis] > 1 && strides[axis] < span) return false;
                span = Math.addExact(span, Math.multiplyExact(extents[axis] - 1, strides[axis])); }
            return true;
        }
        var addresses = new HashSet<Long>(); long[] coordinates = new long[extents.length];
        for (long i = 0; i < count; i++) {
            long address = 0; for (int axis = 0; axis < extents.length; axis++)
                address = Math.addExact(address, Math.multiplyExact(coordinates[axis], strides[axis]));
            if (!addresses.add(address)) return false;
            for (int axis = extents.length - 1; axis >= 0; axis--)
                if (++coordinates[axis] < extents[axis]) break; else coordinates[axis] = 0;
        }
        return true;
    }

    private static BaselineUnitFact baselineFact(ExecutionUnitPlan unit) {
        CpuKernelIr encoded = unit.portablePlan().kernelIr();
        List<CpuKernelIr.Value> materialized = encoded.values().stream()
                .filter(value -> value.kind() != CpuKernelIr.Value.Kind.VIRTUAL).toList();
        var boundaries = new ArrayList<BoundaryResourceFact>(materialized.size());
        for (int index = 0; index < materialized.size(); index++) {
            CpuKernelIr.Value value = materialized.get(index);
            CpuAccessPlan.Binding binding = unit.accessBindings().get(index);
            boundaries.add(new BoundaryResourceFact(value.dataType(), value.kind(),
                    value.accessPlan(), binding.extents(), binding.baseElementOffset(),
                    binding.effectiveStrides(), binding.elementCount(), binding.start(),
                    binding.end(), binding.referencedElementSpan(), binding.startCoordinates(),
                    binding.startAddress(), binding.accessedElementStart(),
                    binding.accessedElementEnd(),
                    unit.carrierPattern().get(index), unit.generatedCarrierPattern().get(index)));
        }
        var declaration = unit.runtimeFacts().workspaceDeclaration();
        WorkspaceRole role = switch (unit.runtimeFacts().workspaceUse()) {
            case NONE -> WorkspaceRole.NONE;
            case MATERIALIZATION -> WorkspaceRole.MATERIALIZATION;
            case SCATTER_PRODUCT -> WorkspaceRole.SCATTER_PRODUCT;
            case ORDERING_INDICES -> WorkspaceRole.ORDERING_INDICES;
            case AGGREGATE_EXACT_STATE -> WorkspaceRole.AGGREGATE_EXACT_STATE;
        };
        WorkspaceResourceFact workspace = declaration
                .map(value -> new WorkspaceResourceFact(role, value.byteSize(),
                        value.byteAlignment()))
                .orElseGet(() -> new WorkspaceResourceFact(WorkspaceRole.NONE, 0, 0));
        var strategy = unit.executionStrategy();
        var materialization = unit.runtimeFacts().materialization().map(value ->
                new MaterializationFact(value.sourceBoundaryIndex(), value.sourceBinding(),
                        value.consumerBinding(), value.elementCount(), value.byteCount(),
                        value.byteAlignment(), value.useCount(), value.expectedRunCount(),
                        value.directCost(), value.copyCost(), value.contiguousCost(),
                        value.copiedTotalCost(), value.netBenefit(), value.benefitBasisPoints(),
                        value.selectionReason()));
        PackedTopology topology = packedTopology(unit, materialized.size());
        var execution = new BaselineExecutionFact(BaselineRoute.PORTABLE,
                unit.portablePlan().specialization(),
                BaselineCompute.valueOf(strategy.compute().name()),
                BaselineOrchestration.valueOf(strategy.orchestration().name()),
                box(unit.extents()), unit.elementCount(), unit.selectedRangeCount(),
                unit.minimumElementsPerWorker(), unit.vectorSpeciesBitSize(),
                box(unit.runtimeFacts().affineAddressPairs()), materialization, topology.kind(),
                topology.geometry(), unit.fusionReason());
        return new BaselineUnitFact(unit.portablePlan().portableKernelIr().structuralKey(),
                execution, unit.dependencies(), boundaries, unit.outputCount(), workspace);
    }

    private static PackedTopology packedTopology(ExecutionUnitPlan unit, int boundaryCount) {
        long[] bases = new long[boundaryCount];
        var runtime = unit.runtimeFacts();
        if (unit.conv3dGeometry().isPresent()) return packed(RuntimeTopology.CONV3D,
                unit.conv3dGeometry().orElseThrow().pack(bases));
        if (unit.conv2dGeometry().isPresent()) return packed(RuntimeTopology.CONV2D,
                unit.conv2dGeometry().orElseThrow().pack(bases));
        if (runtime.batchNormTrainingGeometry().isPresent()) {
            return packed(RuntimeTopology.BATCH_NORM_TRAINING,
                    runtime.batchNormTrainingGeometry().orElseThrow().pack(bases, 0));
        }
        if (runtime.batchNormInferenceGeometry().isPresent()) {
            return packed(RuntimeTopology.BATCH_NORM_INFERENCE,
                    runtime.batchNormInferenceGeometry().orElseThrow().pack(bases));
        }
        if (runtime.trailingNormalizationGeometry().isPresent()) {
            return packed(RuntimeTopology.TRAILING_NORMALIZATION,
                    runtime.trailingNormalizationGeometry().orElseThrow().pack(bases));
        }
        if (runtime.softmaxGeometry().isPresent()) return packed(RuntimeTopology.SOFTMAX,
                runtime.softmaxGeometry().orElseThrow().pack(bases));
        if (runtime.advancedReductionGeometry().isPresent()) {
            return packed(RuntimeTopology.ADVANCED_REDUCTION,
                    runtime.advancedReductionGeometry().orElseThrow().pack(bases));
        }
        if (runtime.aggregateGeometry().isPresent()) return packed(RuntimeTopology.AGGREGATE,
                runtime.aggregateGeometry().orElseThrow().pack(bases, 0));
        return new PackedTopology(RuntimeTopology.POINTWISE, List.of());
    }

    private static PackedTopology packed(RuntimeTopology kind, long[] geometry) {
        return new PackedTopology(kind, box(geometry));
    }

    private static List<Long> box(long[] values) {
        return Arrays.stream(values).boxed().toList();
    }

    private static List<Integer> associate(List<Integer> members,
            List<ExecutionUnitPlan> baseline) {
        var result = new ArrayList<Integer>(); var covered = new ArrayList<Integer>();
        for (int i = 0; i < baseline.size(); i++) {
            List<Integer> unitMembers = baseline.get(i).memberNodeOrdinals();
            if (members.containsAll(unitMembers)) { result.add(i); covered.addAll(unitMembers); }
        }
        return covered.equals(members) ? List.copyOf(result) : List.of();
    }

    private static void validateExistingSpecialized(CpuSpecializedSubgraph fact,
            List<ExecutionUnitPlan> baseline, List<Integer> units) {
        if (fact.disposition() != ExecutionDisposition.EXISTING_SPECIALIZED) return;
        if (units.size() != 1) throw new IllegalArgumentException("existing specialized fact must map to one unit");
        Object ir = baseline.get(units.getFirst()).portablePlan().portableKernelIr();
        if (fact instanceof ConvolutionEpilogue convolution) {
            if (!(ir instanceof CpuConv2dIr conv) || convolution.form() != Form.CONV2D
                    || conv.epilogue() != (convolution.epilogue().terminal() == Terminal.RELU
                        ? CpuConv2dIr.Epilogue.ADD_RELU : CpuConv2dIr.Epilogue.ADD))
                throw new IllegalArgumentException("recognized Conv2d form disagrees with existing IR");
        } else if (fact instanceof ExplicitSemanticKernel explicit) {
            String simple = ir.getClass().getSimpleName();
            boolean agrees = switch (explicit.form()) {
                case SOFTMAX, LOG_SOFTMAX -> simple.equals("CpuSoftmaxIr");
                case LAYER_NORM, RMS_NORM -> simple.equals("CpuTrailingNormalizationIr");
                case BATCH_NORM_INFERENCE -> simple.equals("CpuBatchNormInferenceIr");
                case BATCH_NORM_TRAINING -> simple.equals("CpuBatchNormTrainingIr");
                default -> false;
            };
            if (!agrees) throw new IllegalArgumentException("explicit semantic fact disagrees with existing IR");
        }
    }

    private static boolean conv1dShape(List<CompiledNode> nodes) {
        if (nodes.size() != 4) return false;
        CompiledNode a = nodes.get(0), b = nodes.get(1), conv = nodes.get(2), squeeze = nodes.get(3);
        return expand2(a) && expand2(b) && conv.operation().kind() == Conv2dKind.CONV2D
                && conv.operation().attrs() instanceof Conv2dAttrs attrs
                && attrs.strideHeight() == 1 && attrs.paddingHeight() == 0 && attrs.dilationHeight() == 1
                && squeeze.operation().kind() == AxisTransformKind.SQUEEZE
                && squeeze.operation().attrs() instanceof AxisTransformAttrs sa && sa.axis() == 2
                && squeeze.inputs().equals(conv.outputs());
    }
    private static boolean expand2(CompiledNode node) { return node.operation().kind() == AxisTransformKind.EXPAND_DIMS
            && node.operation().attrs() instanceof AxisTransformAttrs attrs && attrs.axis() == 2; }
    private static boolean transposedWeight(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, GraphValue> values, ValueId id) {
        PartitionDag.ProducerOccurrence occurrence = context.partitionDag().producer(id)
                .orElse(null);
        CompiledNode producer = occurrence == null ? null : occurrence.node();
        if (producer == null || occurrence.outputPosition() != 0
                || producer.operation().kind() != AxisTransformKind.PERMUTE
                || producer.outputs().size() != 1 || producer.inputs().size() != 1
                || require(values, id).descriptor().shape().rank() != 2) return false;
        Object attrs = producer.operation().attrs();
        return attrs instanceof PermutationAttrs permutation
                && permutation.axes().equals(List.of(1, 0))
                && context.partitionDag().consumers(id).size() == 1;
    }
    private static ReductionAttributes reductionAttributes(AggregateReductionKind kind, Object attrs, int rank) {
        if (attrs == NoOperationAttrs.INSTANCE) return new ReductionAttributes(kind, ReductionForm.FULL,
                java.util.stream.IntStream.range(0, rank).boxed().toList(), false, 0);
        if (attrs instanceof AxisReductionAttrs a) return new ReductionAttributes(kind,
                ReductionForm.SINGLE_AXIS, List.of(a.axis()), a.keepDimensions(), 0);
        if (attrs instanceof MultiAxisReductionAttrs a) return new ReductionAttributes(kind,
                ReductionForm.MULTI_AXIS, a.axes(), a.keepDimensions(), 0);
        if (attrs instanceof StatisticalReductionAttrs a) return new ReductionAttributes(kind,
                ReductionForm.STATISTICAL, a.axes(), a.keepDimensions(), a.correction());
        return null;
    }
    private static boolean reductionKind(AggregateReductionKind kind) { return switch (kind) {
        case SUM, MEAN, PROD, MIN, MAX, LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM -> true;
        default -> false; }; }
    private static Form reductionForm(AggregateReductionKind kind) { return Form.valueOf(kind.name()); }
    private static boolean floating(DataType type) { return type == DataType.FLOAT32 || type == DataType.FLOAT64; }
    private static boolean sameResult(Map<ValueId, GraphValue> values, ValueId a, ValueId b) {
        var da = require(values, a).descriptor(); var db = require(values, b).descriptor();
        return da.dataType() == db.dataType() && da.shape().equals(db.shape()) && db.layout().isPresent();
    }
    private static boolean broadcasts(Shape from, Shape to) {
        long[] a = from.toLongArray(), b = to.toLongArray(); if (a.length > b.length) return false;
        for (int i = 1; i <= a.length; i++) if (a[a.length - i] != 1 && a[a.length - i] != b[b.length - i]) return false;
        return true;
    }
    private static List<Integer> range(int start, int end) { return java.util.stream.IntStream.rangeClosed(start, end).boxed().toList(); }
    private static Map<ValueId, GraphValue> values(PrepareContext<?> context) { var result = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(value -> result.put(value.id(), value)); return result; }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) { GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("recognition value is not projected: " + id); return value; }

    private static List<RecognizerFamily> familiesFor(List<CompiledNode> nodes, int ordinal) {
        Object kind = nodes.get(ordinal).operation().kind();
        if (kind == AxisTransformKind.EXPAND_DIMS) return List.of(RecognizerFamily.CONV1D_COMPOSITION);
        if (kind == Conv2dKind.CONV2D) return List.of(RecognizerFamily.CONV2D);
        if (kind == Conv3dKind.CONV3D) return List.of(RecognizerFamily.CONV3D);
        if (kind == MatmulKind.MATMUL) return List.of(RecognizerFamily.MATMUL);
        if (kind instanceof AggregateReductionKind) return List.of(RecognizerFamily.REDUCTION);
        if (kind instanceof SoftmaxKind || kind == LayerNormKind.LAYER_NORM
                || kind == RmsNormKind.RMS_NORM
                || kind == BatchNormKind.BATCH_NORM_INFERENCE
                || kind == BatchNormKind.BATCH_NORM_TRAINING) {
            return List.of(RecognizerFamily.EXPLICIT_SEMANTIC_KERNEL);
        }
        return List.of(RecognizerFamily.values());
    }

    private enum RecognizerFamily { CONV1D_COMPOSITION, CONV2D, CONV3D, MATMUL, REDUCTION, EXPLICIT_SEMANTIC_KERNEL }
    private record Suffix(int end, Epilogue epilogue, ValueId externalAdd) { static Suffix none(int end) {
        return new Suffix(end, Epilogue.none(), null); } }
    private record TerminalFact(Terminal terminal, Optional<ClampRangeAttrs> clamp) { }
    private record PackedTopology(RuntimeTopology kind, List<Long> geometry) { }
    private static final class Attempts {
        private int value;
        private boolean exhausted;
        boolean take() {
            if (value >= MAX_ATTEMPTS) { exhausted = true; return false; }
            value = Math.addExact(value, 1); return true;
        }
        int value() { return value; }
        boolean exhausted() { return exhausted; }
    }
    private record Candidate(Family family, Form form, List<Integer> members,
            List<DataType> inputTypes, List<DataType> resultTypes, List<AccessFact> accesses,
            Epilogue epilogue, ExecutionDisposition disposition, StructuralIdentity identity) {
        CpuSpecializedSubgraph toFact(List<Integer> units, List<BaselineUnitFact> baselineFacts) {
            var exactIdentity = new StructuralIdentity(identity.family(), identity.form(),
                    identity.inputDataTypes(), identity.resultDataTypes(), identity.accessFacts(),
                    identity.attributes(), identity.epilogue(), baselineFacts);
            return switch (family) {
            case MATMUL -> new MatmulEpilogue(members, units, inputTypes, resultTypes, accesses, epilogue, exactIdentity);
            case CONVOLUTION -> new ConvolutionEpilogue(form, members, units, inputTypes, resultTypes,
                    accesses, epilogue, disposition, exactIdentity);
            case REDUCTION -> new ReductionEpilogue(form, members, units, inputTypes, resultTypes,
                    accesses, epilogue, exactIdentity);
            case EXPLICIT_SEMANTIC_KERNEL -> new ExplicitSemanticKernel(form, members, units,
                    inputTypes, resultTypes, accesses, exactIdentity); }; }
    }
}
