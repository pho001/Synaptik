package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one bounded supported CPU unit into one route-neutral portable representation.
 *
 * <p>For ordinary pointwise directed acyclic graphs (DAGs), lowering consumes the validated
 * partition-local producer and consumer occurrences carried by {@link PrepareContext}, derives
 * external boundaries in deterministic first-use order, retains eligible unpublished internal
 * results as typed virtual values (including BOOL masks), and emits one ordered store for every
 * published or otherwise unit-boundary output. Port occurrences determine graph topology, while
 * semantic input positions and candidate-local IR use counts remain CPU-owned lowering facts.
 * All materialized pointwise outputs share one checked iteration domain. It consumes
 * Model operation, shape, and layout contracts during analysis and maps every admitted unary kind
 * to one distinct CPU opcode without decomposition. Generated and Runtime code see only the
 * resulting CPU-private IR and cold bindings. Exact one-node movement, indexing, scatter, fold,
 * ordering, explicit-state random, cumulative-scan, aggregate, softmax, and trailing Layer/RMS
 * normalization families are delegated to focused lowerers; the movement family includes
 * functional slice update.</p>
 */
public final class CpuPartitionLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    private final CpuScalarPowerAnalysis scalarPowerAnalysis = new CpuScalarPowerAnalysis();
    private final CpuAffineLayoutLowering affineLowering = new CpuAffineLayoutLowering();
    private final CpuNonAffineMovementLowering movementLowering = new CpuNonAffineMovementLowering();
    private final CpuIndexingLowering indexingLowering = new CpuIndexingLowering();
    private final CpuScatterLowering scatterLowering = new CpuScatterLowering();
    private final CpuFoldLowering foldLowering = new CpuFoldLowering();
    private final CpuOrderingLowering orderingLowering = new CpuOrderingLowering();
    private final CpuRandomLowering randomLowering = new CpuRandomLowering();
    private final CpuScanLowering scanLowering = new CpuScanLowering();
    private final CpuAggregateLowering aggregateLowering = new CpuAggregateLowering();
    private final CpuArgExtremaLowering argExtremaLowering = new CpuArgExtremaLowering();
    private final CpuMaskedReductionLowering maskedReductionLowering =
            new CpuMaskedReductionLowering();
    private final CpuAdvancedReductionLowering advancedReductionLowering =
            new CpuAdvancedReductionLowering();
    private final CpuSoftmaxLowering softmaxLowering = new CpuSoftmaxLowering();
    private final CpuTrailingNormalizationLowering trailingNormalizationLowering =
            new CpuTrailingNormalizationLowering();
    private final CpuBatchNormInferenceLowering batchNormInferenceLowering =
            new CpuBatchNormInferenceLowering();
    private final CpuBatchNormTrainingLowering batchNormTrainingLowering =
            new CpuBatchNormTrainingLowering();
    private final CpuConv2dLowering conv2dLowering = new CpuConv2dLowering();
    private final CpuConv3dLowering conv3dLowering = new CpuConv3dLowering();
    private final CpuConv1dCompositionLowering conv1dCompositionLowering =
            new CpuConv1dCompositionLowering();

    /** Creates a stateless lowering boundary with the current CPU capability and power analysis. */
    public CpuPartitionLowering() { }

    /**
     * Lowers one supported specialized-family seed, affine chain, or one-through-eight-node
     * ordinary pointwise DAG, and rejects every unsupported unit shape.
     *
     * @param context non-null complete validated CPU partition projection whose shared DAG
     *     supplies exact stable node, producer, consumer, and external-input occurrences
     * @return one immutable single-unit lowering with deterministic materialized boundaries and
     *     ordered stores; never {@code null}
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if ownership, cardinality, dataflow, or an occurrence is
     *     outside the implemented CPU matrix
     * @throws ArithmeticException if exact Shape, layout, or address arithmetic overflows
     */
    public LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        List<CompiledNode> nodes = context.partitionDag().nodes();
        if (!context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        if (nodes.isEmpty() || nodes.size() > 8) {
            throw new IllegalArgumentException("CPU pointwise partition requires one through eight nodes");
        }
        if (nodes.size() == 4 && nodes.get(2).operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D) {
            return conv1dCompositionLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D) {
            return conv2dLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv3dKind.CONV3D) {
            return conv3dLowering.lower(context);
        }
        if (nodes.size() == 1) {
            Object kind = nodes.getFirst().operation().kind();
            if (kind instanceof io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind) {
                return softmaxLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.normalization.LayerNormKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.normalization.RmsNormKind) {
                return trailingNormalizationLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.normalization.BatchNormKind
                    .BATCH_NORM_INFERENCE) {
                return batchNormInferenceLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.normalization.BatchNormKind
                    .BATCH_NORM_TRAINING) {
                return batchNormTrainingLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind) {
                if (kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.LOG_SUM_EXP
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.VARIANCE
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.STANDARD_DEVIATION
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.L1_NORM
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.L2_NORM) {
                    return advancedReductionLowering.lower(context);
                }
                if (nodes.getFirst().operation().attrs()
                        instanceof io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs) {
                    return maskedReductionLowering.lower(context);
                }
                if (kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.ARG_MIN
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.ARG_MAX) {
                    return argExtremaLowering.lower(context);
                }
                return aggregateLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind) {
                return scanLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.random.GraphRngKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.random.DropoutKind) {
                return randomLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.ordering.OrderingKind
                    || kind == io.github.pho001.synaptik.model.operation.ordering.TopKKind.TOP_K) {
                return orderingLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD_AXIS
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD2D) {
                return foldLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.index.AxisScatterKind
                    || kind == io.github.pho001.synaptik.model.operation.index.ScatterNdKind.SCATTER_ND) {
                return scatterLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.index.AxisGatherKind
                    || kind == io.github.pho001.synaptik.model.operation.index.GatherNdKind.GATHER_ND
                    || kind == io.github.pho001.synaptik.model.operation.index.OneHotKind.ONE_HOT) {
                return indexingLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.PadKind.PAD
                    || kind == io.github.pho001.synaptik.model.operation.layout.TileKind.TILE
                    || kind == io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.CONCAT
                    || kind == io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.STACK
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.UNFOLD_AXIS
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.UNFOLD2D
                    || kind == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE_UPDATE) {
                return movementLowering.lower(context);
            }
        }
        if (nodes.stream().allMatch(node -> {
            Object kind = node.operation().kind();
            return kind instanceof io.github.pho001.synaptik.model.operation.layout.ContiguousKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.index.SelectKind
                    || kind == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE;
        })) return affineLowering.lower(context);
        Map<ValueId, GraphValue> values = indexValues(context.values());
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(value -> memory.put(value.valueId(), value));

        var external = new LinkedHashMap<ValueId, Integer>();
        var producedOrdinals = new LinkedHashMap<ValueId, Integer>();
        var virtualOutputs = new ArrayList<ValueId>();
        var materializedOutputs = new ArrayList<ValueId>();
        var instructions = new ArrayList<PendingInstruction>();
        var produced = new java.util.LinkedHashSet<ValueId>();
        PartitionDag dag = context.partitionDag();
        dag.externalInputs().forEach(occurrence ->
                external.putIfAbsent(occurrence.valueId(), -1));
        for (CompiledNode node : dag.nodes()) {
            if (node.outputs().size() != 1) {
                throw new IllegalArgumentException("pointwise node must have exactly one output");
            }
            assertOccurrence(node, values);
            ValueId output = node.outputs().getFirst();
            PartitionDag.ProducerOccurrence producer = dag.producer(output).orElseThrow();
            if (producer.node() != node || producer.outputPosition() != 0
                    || external.containsKey(output) || !produced.add(output)) {
                throw new IllegalArgumentException("partition dataflow must be acyclic and non-aliasing");
            }
            instructions.add(new PendingInstruction(node, output));
            producedOrdinals.put(output, -1);
        }
        for (CompiledNode node : dag.nodes()) {
            ValueId output = node.outputs().getFirst();
            LogicalMemoryRequirement requirement = memory.get(output);
            if (requirement == null) {
                throw new IllegalArgumentException("pointwise output memory fact is not projected");
            }
            boolean consumedInside = !dag.consumers(output).isEmpty();
            if (consumedInside && !requirement.graphOutput()) {
                requireVirtual(requirement, context, output);
                virtualOutputs.add(output);
            } else {
                requireMaterializedIntermediate(requirement, context, output);
                materializedOutputs.add(output);
            }
        }
        if (materializedOutputs.isEmpty()) {
            throw new IllegalArgumentException("pointwise unit must have a materialized output");
        }
        Shape iterationShape = require(values, materializedOutputs.getFirst()).descriptor().shape();
        for (ValueId output : materializedOutputs) {
            if (!require(values, output).descriptor().shape().equals(iterationShape)) {
                throw new IllegalArgumentException("pointwise outputs require one iteration domain");
            }
        }
        long elementCount = iterationShape.knownElementCount().orElseThrow();

        var irValues = new ArrayList<CpuKernelIr.Value>();
        var boundaryValues = new ArrayList<ValueId>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        var spans = new ArrayList<Long>();
        var boundaryTypes = new ArrayList<DataType>();
        int ordinal = 0;
        for (ValueId id : external.keySet()) {
            GraphValue value = require(values, id);
            Normalized normalized = normalize(value.descriptor().shape(),
                    value.descriptor().layout().orElseThrow(), iterationShape,
                    CpuAccessPlan.AccessKind.READ);
            external.put(id, ordinal);
            irValues.add(new CpuKernelIr.Value(ordinal++, value.descriptor().dataType(),
                    CpuKernelIr.Value.Kind.INPUT, normalized.plan()));
            boundaryValues.add(id);
            bindings.add(normalized.binding());
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
            boundaryTypes.add(value.descriptor().dataType());
        }
        for (ValueId id : dag.nodes().stream().map(node -> node.outputs().getFirst()).toList()) {
            GraphValue value = require(values, id);
            boolean materialized = materializedOutputs.contains(id);
            Normalized normalized = normalize(value.descriptor().shape(), materialized
                    ? value.descriptor().layout().orElseThrow()
                    : LayoutDescriptor.contiguous(value.descriptor().shape()), iterationShape,
                    materialized ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ);
            producedOrdinals.put(id, ordinal);
            irValues.add(new CpuKernelIr.Value(ordinal++, value.descriptor().dataType(),
                    materialized ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.VIRTUAL,
                    normalized.plan()));
            if (materialized) {
                boundaryValues.add(id);
                bindings.add(normalized.binding());
                spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
                boundaryTypes.add(value.descriptor().dataType());
            }
        }
        var irInstructions = new ArrayList<CpuKernelIr.Instruction>();
        for (PendingInstruction pending : instructions) {
            var inputs = pending.node().inputs().stream().map(id -> {
                Integer value = producedOrdinals.get(id);
                return value != null && value >= 0 ? value : external.get(id);
            }).toList();
            CpuPointwiseOpcode opcode = opcode(pending.node());
            CpuKernelIr.ScalarImmediate immediate = scalarImmediate(pending.node());
            CpuKernelIr.ClampImmediate clamp = clampImmediate(pending.node());
            CpuKernelIr.PowerRealization realization = opcode == CpuPointwiseOpcode.SCALAR_POW
                    ? scalarPowerAnalysis.analyze(immediate) : null;
            irInstructions.add(new CpuKernelIr.Instruction(opcode, inputs,
                    producedOrdinals.get(pending.output()), immediate, realization, clamp));
        }
        var stores = new ArrayList<CpuKernelIr.Store>();
        int outputOrdinal = 0;
        for (ValueId id : materializedOutputs) {
            stores.add(new CpuKernelIr.Store(producedOrdinals.get(id), outputOrdinal++));
        }
        var ir = new CpuKernelIr(irValues, irInstructions,
                new CpuKernelIr.Loop("start", "end"),
                stores);
        return new LoweredPartition(ir, boundaryValues, bindings, spans, boundaryTypes,
                virtualOutputs, iterationShape.toLongArray(), elementCount,
                materializedOutputs.size() == 1
                        ? "legal: bounded pointwise unit"
                        : "legal: bounded multi-store pointwise DAG unit",
                new long[0]);
    }

    private void assertOccurrence(CompiledNode node, Map<ValueId, GraphValue> values) {
        var query = new OperationCapabilityQuery(node.operation(),
                node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                node.outputs().stream().map(id -> require(values, id).descriptor()).toList());
        if (!capabilities.supports(query)) {
            throw new IllegalArgumentException("partition contains an unsupported CPU occurrence");
        }
    }

    /**
     * Maps one already-admitted pointwise occurrence to its CPU-private generated opcode.
     *
     * @param node non-null compiled occurrence whose kind belongs to the bounded pointwise family
     * @return the non-null exact generated opcode
     * @throws NullPointerException if {@code node} or its operation is {@code null}
     * @throws IllegalArgumentException if the operation kind has no admitted CPU opcode
     */
    static CpuPointwiseOpcode opcode(CompiledNode node) {
        Object kind = node.operation().kind();
        if (kind instanceof BinaryArithmeticKind value) return switch (value) {
            case ADD -> CpuPointwiseOpcode.ADD; case SUB -> CpuPointwiseOpcode.SUB;
            case MUL -> CpuPointwiseOpcode.MUL; case DIV -> CpuPointwiseOpcode.DIV;
            case MIN -> CpuPointwiseOpcode.MIN; case MAX -> CpuPointwiseOpcode.MAX;
            case POW -> CpuPointwiseOpcode.POW;
            default -> throw unsupported();
        };
        if (kind instanceof ScalarElementwiseKind value) return switch (value) {
            case ADD -> CpuPointwiseOpcode.SCALAR_ADD; case SUB -> CpuPointwiseOpcode.SCALAR_SUB;
            case MUL -> CpuPointwiseOpcode.SCALAR_MUL;
            case DIV -> CpuPointwiseOpcode.SCALAR_DIV;
            case POW -> CpuPointwiseOpcode.SCALAR_POW;
            case MIN -> CpuPointwiseOpcode.SCALAR_MIN;
            case MAX -> CpuPointwiseOpcode.SCALAR_MAX;
            case CLAMP -> CpuPointwiseOpcode.SCALAR_CLAMP;
            default -> throw unsupported();
        };
        if (kind instanceof UnaryElementwiseKind value) return switch (value) {
            case ABS -> CpuPointwiseOpcode.ABS; case NEG -> CpuPointwiseOpcode.NEG;
            case RECIPROCAL -> CpuPointwiseOpcode.RECIPROCAL; case LOG -> CpuPointwiseOpcode.LOG;
            case LOG1P -> CpuPointwiseOpcode.LOG1P; case EXP -> CpuPointwiseOpcode.EXP;
            case EXPM1 -> CpuPointwiseOpcode.EXPM1; case ERF -> CpuPointwiseOpcode.ERF;
            case SQRT -> CpuPointwiseOpcode.SQRT; case RSQRT -> CpuPointwiseOpcode.RSQRT;
            case FLOOR -> CpuPointwiseOpcode.FLOOR; case CEIL -> CpuPointwiseOpcode.CEIL;
            case SIGN -> CpuPointwiseOpcode.SIGN; case RELU -> CpuPointwiseOpcode.RELU;
            case SIGMOID -> CpuPointwiseOpcode.SIGMOID; case TANH -> CpuPointwiseOpcode.TANH;
            case GELU -> CpuPointwiseOpcode.GELU_EXACT;
            case GELU_TANH_APPROXIMATION -> CpuPointwiseOpcode.GELU_TANH_APPROXIMATION;
            case SILU -> CpuPointwiseOpcode.SILU;
        };
        if (kind instanceof FloatingClassificationKind value) return switch (value) {
            case IS_FINITE -> CpuPointwiseOpcode.IS_FINITE; case IS_NAN -> CpuPointwiseOpcode.IS_NAN;
            case IS_INF -> CpuPointwiseOpcode.IS_INF;
        };
        if (kind instanceof BinaryComparisonKind value) return switch (value) {
            case GREATER_THAN -> CpuPointwiseOpcode.GREATER_THAN;
            case GREATER_OR_EQUAL -> CpuPointwiseOpcode.GREATER_OR_EQUAL;
            case LESS_THAN -> CpuPointwiseOpcode.LESS_THAN;
            case LESS_OR_EQUAL -> CpuPointwiseOpcode.LESS_OR_EQUAL;
            case EQUAL -> CpuPointwiseOpcode.EQUAL; case NOT_EQUAL -> CpuPointwiseOpcode.NOT_EQUAL;
        };
        if (kind instanceof BooleanLogicalKind value) return switch (value) {
            case AND -> CpuPointwiseOpcode.LOGICAL_AND;
            case OR -> CpuPointwiseOpcode.LOGICAL_OR;
            case NOT -> CpuPointwiseOpcode.LOGICAL_NOT;
        };
        if (kind == WhereSelectionKind.WHERE) return CpuPointwiseOpcode.WHERE;
        if (kind == CastKind.CAST) return CpuPointwiseOpcode.CAST;
        throw unsupported();
    }

    private static IllegalArgumentException unsupported() {
        return new IllegalArgumentException("unsupported CPU pointwise opcode");
    }

    /**
     * Encodes a scalar-elementwise attribute without numerical conversion.
     *
     * @param node non-null compiled occurrence to inspect
     * @return the exact typed immediate, or {@code null} when the occurrence has no scalar-value
     *     attributes
     * @throws NullPointerException if {@code node} or its operation is {@code null}
     * @throws IllegalArgumentException if the scalar type has no admitted immediate encoding
     */
    static CpuKernelIr.ScalarImmediate scalarImmediate(CompiledNode node) {
        if (!(node.operation().attrs() instanceof ScalarValueAttrs attrs)) return null;
        ScalarValue value = attrs.value();
        long bits = switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(value.float32Value()) & 0xffff_ffffL;
            case INT32 -> value.int32Value() & 0xffff_ffffL;
            case INT64 -> value.int64Value();
            default -> throw new IllegalArgumentException("unsupported scalar immediate type");
        };
        return new CpuKernelIr.ScalarImmediate(value.dataType(), bits);
    }

    /**
     * Encodes both ordered bounds of a scalar clamp occurrence.
     *
     * @param node non-null compiled occurrence to inspect
     * @return the exact two-bound immediate, or {@code null} when the occurrence has no clamp
     *     attributes
     * @throws NullPointerException if {@code node} or its operation is {@code null}
     * @throws IllegalArgumentException if either bound type has no admitted immediate encoding
     */
    static CpuKernelIr.ClampImmediate clampImmediate(CompiledNode node) {
        if (!(node.operation().attrs() instanceof ClampRangeAttrs attrs)) return null;
        return new CpuKernelIr.ClampImmediate(scalarImmediate(attrs.minValue()),
                scalarImmediate(attrs.maxValue()));
    }

    private static CpuKernelIr.ScalarImmediate scalarImmediate(ScalarValue value) {
        long bits = switch (value.dataType()) {
            case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value());
            case FLOAT32 -> Float.floatToRawIntBits(value.float32Value()) & 0xffff_ffffL;
            case INT32 -> value.int32Value() & 0xffff_ffffL;
            case INT64 -> value.int64Value();
            default -> throw new IllegalArgumentException("unsupported scalar immediate type");
        };
        return new CpuKernelIr.ScalarImmediate(value.dataType(), bits);
    }

    private static Map<ValueId, GraphValue> indexValues(List<GraphValue> source) {
        var result = new LinkedHashMap<ValueId, GraphValue>();
        source.forEach(value -> result.put(value.id(), value));
        return result;
    }

    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }

    private static void requireVirtual(LogicalMemoryRequirement requirement,
            PrepareContext<?> context, ValueId id) {
        if (requirement == null || requirement.graphOutput()
                || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())
                || !requirement.consumerPartitions().equals(List.of(context.partition()))) {
            throw new IllegalArgumentException("internal result must be private and single-partition: " + id);
        }
    }

    private static void requireMaterializedIntermediate(LogicalMemoryRequirement requirement,
            PrepareContext<?> context, ValueId id) {
        if (requirement == null || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())) {
            throw new IllegalArgumentException(
                    "materialized output must be produced by the analyzed partition: " + id);
        }
    }

    private static Normalized normalize(Shape source, LayoutDescriptor layout, Shape iteration,
            CpuAccessPlan.AccessKind kind) {
        long[] extents = iteration.toLongArray();
        long[] sourceExtents = source.toLongArray();
        long[] sourceStrides = layout.strides();
        long[] strides = new long[extents.length];
        int offset = extents.length - sourceExtents.length;
        if (offset < 0) throw new IllegalArgumentException("source rank exceeds iteration rank");
        for (int axis = 0; axis < extents.length; axis++) {
            if (axis < offset) strides[axis] = 0;
            else {
                long sourceExtent = sourceExtents[axis - offset];
                long targetExtent = extents[axis];
                if (sourceExtent != targetExtent && sourceExtent != 1) {
                    throw new IllegalArgumentException("shape does not right-broadcast to iteration shape");
                }
                strides[axis] = sourceExtent == 1 && targetExtent != 1
                        ? 0 : sourceStrides[axis - offset];
            }
        }
        if (kind == CpuAccessPlan.AccessKind.WRITE) validateDistinctWrites(extents, strides);
        int suffix = 0;
        long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++;
            expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        var roles = new ArrayList<CpuAccessPlan.AxisRole>(extents.length);
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        boolean allZero = java.util.Arrays.stream(strides).allMatch(value -> value == 0);
        boolean dense = suffix == extents.length;
        boolean bias = extents.length > 0 && suffix == 1
                && roles.subList(0, extents.length - 1).stream()
                .allMatch(role -> role == CpuAccessPlan.AxisRole.BROADCAST);
        CpuAccessPlan.Regime regime = allZero && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : dense ? CpuAccessPlan.Regime.DENSE_LINEAR
                : bias ? CpuAccessPlan.Regime.LAST_AXIS_BIAS
                : suffix > 0 ? CpuAccessPlan.Regime.BLOCK_OUTER
                : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        var plan = new CpuAccessPlan(kind, regime, extents.length, roles, suffix);
        long count = iteration.knownElementCount().orElseThrow();
        return new Normalized(plan, CpuAccessPlan.Binding.create(plan, extents,
                layout.storageOffset(), strides, count, 0, count, layout.referencedElementSpan()));
    }

    private static void validateDistinctWrites(long[] extents, long[] strides) {
        if (java.util.Arrays.stream(extents).anyMatch(extent -> extent == 0)) return;
        var seen = new java.util.HashSet<Long>();
        long count = 1;
        for (long extent : extents) count = Math.multiplyExact(count, extent);
        if (count > 1_000_000) {
            var axes = new ArrayList<Integer>();
            for (int i = 0; i < extents.length; i++) if (extents[i] > 1) axes.add(i);
            axes.sort(java.util.Comparator.comparingLong(i -> strides[i]));
            long covered = 1;
            for (int axis : axes) {
                if (strides[axis] < covered) throw new IllegalArgumentException(
                        "output geometry repeats a write address");
                covered = Math.addExact(covered,
                        Math.multiplyExact(extents[axis] - 1, strides[axis]));
            }
            return;
        }
        long[] coordinates = new long[extents.length];
        for (long logical = 0; logical < count; logical++) {
            long address = 0;
            for (int axis = 0; axis < extents.length; axis++) address = Math.addExact(address,
                    Math.multiplyExact(coordinates[axis], strides[axis]));
            if (!seen.add(address)) throw new IllegalArgumentException(
                    "output geometry repeats a write address");
            for (int axis = extents.length - 1; axis >= 0; axis--) {
                if (++coordinates[axis] < extents[axis]) break;
                coordinates[axis] = 0;
            }
        }
    }

    private record PendingInstruction(CompiledNode node, ValueId output) { }
    private record Normalized(CpuAccessPlan plan, CpuAccessPlan.Binding binding) { }

    /**
     * Immutable lowering result consumed by route-neutral CPU analysis.
     *
     * @param portableKernelIr non-null route-independent pointwise, affine, movement, indexing,
     *     scatter, fold, ordering, random, or cumulative-scan
     *     representation
     * @param boundaryValues non-null deterministic external-read values followed by the sole
     *     final materialized output; copied defensively
     * @param accessBindings non-null normalized cold bindings in the same boundary order; copied
     *     defensively
     * @param referencedElementSpans non-null exact per-boundary layout spans used for resource
     *     declarations; copied defensively
     * @param boundaryDataTypes non-null exact logical types in boundary order; copied defensively
     * @param virtualValues non-null ordered internal single-use results with no declaration or
     *     Runtime slot; copied defensively
     * @param extents non-null final iteration extents; copied defensively
     * @param elementCount checked non-negative product of {@code extents}
     * @param fusionReason non-null cold diagnostic explanation
     * @param affineAddressPairs alternating source/result addresses for affine copying, or an
     *     empty array for pointwise lowering; copied defensively
     * @param movementGeometry compact cold movement geometry, present only for data movement
     * @param indexingGeometry compact cold indexing geometry, present only for gather or one-hot
     * @param scatterGeometry compact cold functional-scatter geometry, present only for scatter
     * @param foldGeometry compact cold overlap-fold geometry, present only for fold
     * @param orderingGeometry compact cold stable-ordering geometry, present only for SORT,
     *     ARGSORT, or TOP_K
     * @param randomGeometry compact cold explicit-state geometry, present only for INITIAL_STATE
     *     or DROPOUT
     * @param scanGeometry compact cold cumulative-scan geometry, present only for CUM_SUM or
     *     CUM_PROD
     * @param aggregateGeometry compact cold ordinary extrema/Boolean output-cell geometry
     * @param argExtremaGeometry compact cold one-axis logical-index geometry
     * @param maskedReductionGeometry compact cold directional masked-reduction geometry
     * @param advancedReductionGeometry compact cold logarithmic/statistical/norm geometry
     * @param softmaxGeometry compact cold shape-preserving normalization geometry
     * @param trailingNormalizationGeometry compact cold Layer/RMS trailing-slice geometry
     * @param batchNormInferenceGeometry compact cold zero-workspace arbitrary-axis inference
     *     geometry
     * @param batchNormTrainingGeometry compact cold complete-channel training geometry with one
     *     exact-state slice per simultaneous non-empty range
     * @param conv2dGeometry compact cold grouped NCHW Conv2d boundary geometry
     * @param conv3dGeometry compact cold grouped NCDHW Conv3d boundary geometry
     */
    public record LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
            List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
            List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
            long elementCount, String fusionReason, long[] affineAddressPairs,
            Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
            Optional<CpuIndexingLowering.Geometry> indexingGeometry,
            Optional<CpuScatterLowering.Geometry> scatterGeometry,
            Optional<CpuFoldLowering.Geometry> foldGeometry,
            Optional<CpuOrderingLowering.Geometry> orderingGeometry,
            Optional<CpuRandomLowering.Geometry> randomGeometry,
            Optional<CpuScanLowering.Geometry> scanGeometry,
            Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
            Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
            Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
            Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
            Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
            Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
            Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
            Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry) {

        /**
         * Creates an established-family lowering with no Conv3d geometry.
         *
         * @param portableKernelIr non-null route-independent canonical kernel IR
         * @param boundaryValues external reads followed by materialized outputs
         * @param accessBindings normalized cold boundary bindings
         * @param referencedElementSpans exact declaration spans in boundary order
         * @param boundaryDataTypes exact represented types in boundary order
         * @param virtualValues internal single-use results without Runtime slots
         * @param extents final logical range extents
         * @param elementCount checked non-negative logical range count
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs affine addresses or an empty array
         * @param movementGeometry optional movement geometry
         * @param indexingGeometry optional indexing geometry
         * @param scatterGeometry optional scatter geometry
         * @param foldGeometry optional fold geometry
         * @param orderingGeometry optional stable-ordering geometry
         * @param randomGeometry optional explicit-state geometry
         * @param scanGeometry optional cumulative-scan geometry
         * @param aggregateGeometry optional ordinary-aggregate geometry
         * @param argExtremaGeometry optional arg-extrema geometry
         * @param maskedReductionGeometry optional masked-reduction geometry
         * @param advancedReductionGeometry optional logarithmic/statistical/norm geometry
         * @param softmaxGeometry optional softmax geometry
         * @param trailingNormalizationGeometry optional Layer/RMS geometry
         * @param batchNormInferenceGeometry optional batch-normalization inference geometry
         * @param batchNormTrainingGeometry optional batch-normalization training geometry
         * @param conv2dGeometry optional grouped NCHW Conv2d geometry
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if boundary, range, or specialized-family facts
         *     disagree
         * @throws ArithmeticException if checked range or geometry arithmetic overflows
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry,
                Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
                Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
                Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
                Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
                Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
                Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
                Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
                Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry,
                Optional<CpuConv2dLowering.Geometry> conv2dGeometry) {
            this(portableKernelIr,boundaryValues,accessBindings,referencedElementSpans,
                    boundaryDataTypes,virtualValues,extents,elementCount,fusionReason,
                    affineAddressPairs,movementGeometry,indexingGeometry,scatterGeometry,
                    foldGeometry,orderingGeometry,randomGeometry,scanGeometry,aggregateGeometry,
                    argExtremaGeometry,maskedReductionGeometry,advancedReductionGeometry,
                    softmaxGeometry,trailingNormalizationGeometry,batchNormInferenceGeometry,
                    batchNormTrainingGeometry,conv2dGeometry,Optional.empty());
        }

        /**
         * Creates an established-family lowering with no Conv2d geometry.
         *
         * @param portableKernelIr route-independent canonical kernel IR
         * @param boundaryValues external reads followed by materialized outputs
         * @param accessBindings normalized cold boundary bindings
         * @param referencedElementSpans exact declaration spans in boundary order
         * @param boundaryDataTypes exact represented types in boundary order
         * @param virtualValues internal single-use results without Runtime slots
         * @param extents final logical range extents
         * @param elementCount checked logical range count
         * @param fusionReason cold diagnostic explanation
         * @param affineAddressPairs affine addresses or an empty array
         * @param movementGeometry optional movement geometry
         * @param indexingGeometry optional indexing geometry
         * @param scatterGeometry optional scatter geometry
         * @param foldGeometry optional fold geometry
         * @param orderingGeometry optional ordering geometry
         * @param randomGeometry optional explicit-state geometry
         * @param scanGeometry optional scan geometry
         * @param aggregateGeometry optional aggregate geometry
         * @param argExtremaGeometry optional arg-extrema geometry
         * @param maskedReductionGeometry optional masked-reduction geometry
         * @param advancedReductionGeometry optional advanced-reduction geometry
         * @param softmaxGeometry optional softmax geometry
         * @param trailingNormalizationGeometry optional Layer/RMS geometry
         * @param batchNormInferenceGeometry optional batch-normalization inference geometry
         * @param batchNormTrainingGeometry optional batch-normalization training geometry
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if boundary, range, or specialized-family facts
         *     disagree
         * @throws ArithmeticException if checked range or geometry arithmetic overflows
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry,
                Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
                Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
                Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
                Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
                Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
                Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry,
                Optional<CpuBatchNormInferenceLowering.Geometry> batchNormInferenceGeometry,
                Optional<CpuBatchNormTrainingLowering.Geometry> batchNormTrainingGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                    argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                    softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                    batchNormTrainingGeometry, Optional.empty(), Optional.empty());
        }

        /**
         * Creates an existing-family lowering with no batch-normalization geometry.
         *
         * <p>Parameters have the ownership, nullability, and cardinality contracts of the
         * corresponding record components.</p>
         *
         * @param portableKernelIr route-independent family identity
         * @param boundaryValues ordered materialized boundaries
         * @param accessBindings ordered cold bindings
         * @param referencedElementSpans exact declaration spans
         * @param boundaryDataTypes exact boundary types
         * @param virtualValues internal unmaterialized values
         * @param extents iteration extents
         * @param elementCount logical work-item count
         * @param fusionReason lowering explanation
         * @param affineAddressPairs affine addresses, or empty
         * @param movementGeometry optional movement geometry
         * @param indexingGeometry optional indexing geometry
         * @param scatterGeometry optional scatter geometry
         * @param foldGeometry optional fold geometry
         * @param orderingGeometry optional ordering geometry
         * @param randomGeometry optional explicit-state random geometry
         * @param scanGeometry optional cumulative-scan geometry
         * @param aggregateGeometry optional ordinary-aggregate geometry
         * @param argExtremaGeometry optional arg-extrema geometry
         * @param maskedReductionGeometry optional masked-reduction geometry
         * @param advancedReductionGeometry optional advanced-reduction geometry
         * @param softmaxGeometry optional softmax geometry
         * @param trailingNormalizationGeometry optional trailing-normalization geometry
         * @throws NullPointerException if a required reference or list element is null
         * @throws IllegalArgumentException if boundary, family, or geometry facts disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry,
                Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
                Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
                Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
                Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
                Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry,
                Optional<CpuTrailingNormalizationLowering.Geometry> trailingNormalizationGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                    argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                    softmaxGeometry, trailingNormalizationGeometry, Optional.empty(), Optional.empty());
        }

        /**
         * Creates an existing-family lowering with no trailing-normalization geometry.
         *
         * @param portableKernelIr non-null route-independent family identity
         * @param boundaryValues non-null ordered materialized boundaries; copied defensively
         * @param accessBindings non-null ordered cold bindings; copied defensively
         * @param referencedElementSpans non-null exact declaration spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal unmaterialized values; copied defensively
         * @param extents non-null iteration extents; copied defensively
         * @param elementCount non-negative logical work-item count
         * @param fusionReason non-null lowering explanation
         * @param affineAddressPairs non-null affine address pairs, or empty; copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional explicit-state random geometry
         * @param scanGeometry non-null optional cumulative-scan geometry
         * @param aggregateGeometry non-null optional ordinary-aggregate geometry
         * @param argExtremaGeometry non-null optional arg-extrema geometry
         * @param maskedReductionGeometry non-null optional masked-reduction geometry
         * @param advancedReductionGeometry non-null optional advanced-reduction geometry
         * @param softmaxGeometry non-null optional stable-softmax geometry
         * @throws NullPointerException if a required reference or list element is null
         * @throws IllegalArgumentException if boundary, family, or geometry facts disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry,
                Optional<CpuAggregateLowering.Geometry> aggregateGeometry,
                Optional<CpuArgExtremaLowering.Geometry> argExtremaGeometry,
                Optional<CpuMaskedReductionLowering.Geometry> maskedReductionGeometry,
                Optional<CpuAdvancedReductionLowering.Geometry> advancedReductionGeometry,
                Optional<CpuSoftmaxLowering.Geometry> softmaxGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                    argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                    softmaxGeometry, Optional.empty());
        }

        /**
         * Creates an existing-family lowering with optional aggregate geometry and no
         * arg-extrema geometry.
         *
         * @param portableKernelIr non-null existing portable representation
         * @param boundaryValues non-null ordered materialized boundaries; copied defensively
         * @param accessBindings non-null ordered cold access bindings; copied defensively
         * @param referencedElementSpans non-null exact declaration spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null unmaterialized internal values; copied defensively
         * @param extents non-null iteration extents; copied defensively
         * @param elementCount checked non-negative iteration count
         * @param fusionReason non-null lowering explanation
         * @param affineAddressPairs non-null affine addresses, empty otherwise; copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional random geometry
         * @param scanGeometry non-null optional cumulative-scan geometry
         * @param aggregateGeometry non-null optional ordinary aggregate geometry
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if boundary collections or family geometry disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry,
                Optional<CpuAggregateLowering.Geometry> aggregateGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry,
                    aggregateGeometry, Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty());
        }

        /**
         * Creates an existing-family lowering with optional scan geometry and no aggregate
         * geometry.
         *
         * @param portableKernelIr non-null existing portable representation
         * @param boundaryValues non-null ordered materialized boundaries
         * @param accessBindings non-null ordered cold access bindings
         * @param referencedElementSpans non-null exact declaration spans
         * @param boundaryDataTypes non-null exact boundary types
         * @param virtualValues non-null unmaterialized internal values
         * @param extents non-null iteration extents
         * @param elementCount non-negative iteration count
         * @param fusionReason non-null lowering explanation
         * @param affineAddressPairs non-null affine addresses, empty otherwise
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional random geometry
         * @param scanGeometry non-null optional scan geometry
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if boundary or geometry facts disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry,
                Optional<CpuScanLowering.Geometry> scanGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }

        /**
         * Creates an existing-family lowering without cumulative-scan geometry.
         *
         * @param portableKernelIr non-null route-independent portable representation
         * @param boundaryValues non-null ordered external boundaries; copied defensively
         * @param accessBindings non-null ordered normalized accesses; copied defensively
         * @param referencedElementSpans non-null exact boundary spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null iteration extents; copied defensively
         * @param elementCount checked non-negative iteration-domain size
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs non-null alternating affine addresses, empty otherwise;
         *     copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional explicit-state random geometry
         * @throws NullPointerException if a required component or list element is null
         * @throws IllegalArgumentException if boundary collections or family geometry disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry,
                Optional<CpuRandomLowering.Geometry> randomGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }

        /**
         * Creates a lowering without random geometry for existing portable families.
         *
         * @param portableKernelIr non-null route-independent portable representation
         * @param boundaryValues non-null ordered external boundary values; copied defensively
         * @param accessBindings non-null ordered normalized accesses; copied defensively
         * @param referencedElementSpans non-null exact boundary spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null iteration extents; copied defensively
         * @param elementCount checked non-negative product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs non-null alternating affine addresses, empty otherwise;
         *     copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @throws NullPointerException if a required component or list element is null
         * @throws IllegalArgumentException if boundary collections or family geometry disagree
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry,
                Optional<CpuScatterLowering.Geometry> scatterGeometry,
                Optional<CpuFoldLowering.Geometry> foldGeometry,
                Optional<CpuOrderingLowering.Geometry> orderingGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }
        /**
         * Creates a pointwise or affine lowering without movement geometry.
         *
         * @param portableKernelIr non-null pointwise or affine portable representation
         * @param boundaryValues non-null external inputs followed by the final output; copied
         *     defensively
         * @param accessBindings non-null boundary access bindings; copied defensively
         * @param referencedElementSpans non-null exact boundary spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null final iteration extents; copied defensively
         * @param elementCount checked non-negative product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs alternating source/result addresses for affine copying, or an
         *     empty array for pointwise lowering; copied defensively
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if boundary collections have inconsistent cardinality
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }

        /**
         * Creates a lowering with optional compact movement geometry and no indexing geometry.
         *
         * @param portableKernelIr non-null route-independent portable representation
         * @param boundaryValues non-null external inputs followed by the output; copied defensively
         * @param accessBindings non-null boundary access bindings; copied defensively
         * @param referencedElementSpans non-null exact boundary spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null output iteration extents; copied defensively
         * @param elementCount checked non-negative product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs non-null alternating affine addresses, empty otherwise;
         *     copied defensively
         * @param movementGeometry non-null optional compact movement geometry
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }

        /**
         * Creates a lowering with movement or indexing geometry and no scatter geometry.
         *
         * @param portableKernelIr non-null route-independent portable representation
         * @param boundaryValues non-null external inputs followed by the output; copied defensively
         * @param accessBindings non-null boundary access bindings; copied defensively
         * @param referencedElementSpans non-null exact boundary spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null output iteration extents; copied defensively
         * @param elementCount checked non-negative product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs non-null alternating affine addresses, empty otherwise; copied
         *     defensively
         * @param movementGeometry non-null optional compact movement geometry
         * @param indexingGeometry non-null optional compact indexing geometry
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if boundary collections have inconsistent cardinality
         */
        public LoweredPartition(CpuPortableKernelIr portableKernelIr, List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings, List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes, List<ValueId> virtualValues, long[] extents,
                long elementCount, String fusionReason, long[] affineAddressPairs,
                Optional<CpuNonAffineMovementLowering.Geometry> movementGeometry,
                Optional<CpuIndexingLowering.Geometry> indexingGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty());
        }
        /**
         * Validates matching boundary facts and snapshots every mutable collection or array.
         *
         * @param portableKernelIr non-null route-independent portable representation
         * @param boundaryValues non-null ordered materialized boundaries; copied defensively
         * @param accessBindings non-null access geometry in boundary order; copied defensively
         * @param referencedElementSpans non-null exact declaration spans; copied defensively
         * @param boundaryDataTypes non-null exact boundary types; copied defensively
         * @param virtualValues non-null internal values without Runtime slots; copied defensively
         * @param extents non-null final iteration or slice-domain extents; copied defensively
         * @param elementCount checked non-negative execution-domain count
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs non-null alternating affine addresses, empty otherwise;
         *     copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional functional-scatter geometry
         * @param foldGeometry non-null optional overlap-fold geometry
         * @param orderingGeometry non-null optional stable-ordering geometry
         * @param randomGeometry non-null optional explicit-state random geometry
         * @param scanGeometry non-null optional cumulative-scan geometry
         * @param aggregateGeometry non-null optional ordinary-aggregate geometry
         * @param argExtremaGeometry non-null optional arg-extrema geometry
         * @param maskedReductionGeometry non-null optional masked-reduction geometry
         * @param advancedReductionGeometry non-null optional advanced-reduction geometry
         * @param softmaxGeometry non-null optional stable-softmax geometry
         * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
         * @param batchNormInferenceGeometry non-null optional batch-normalization inference geometry
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if fewer than one input plus one output are present or
         *     boundary collections have different cardinalities
         */
        public LoweredPartition {
            Objects.requireNonNull(portableKernelIr, "portableKernelIr");
            boundaryValues = List.copyOf(boundaryValues);
            accessBindings = List.copyOf(accessBindings);
            referencedElementSpans = List.copyOf(referencedElementSpans);
            boundaryDataTypes = List.copyOf(boundaryDataTypes);
            virtualValues = List.copyOf(virtualValues);
            extents = extents.clone();
            affineAddressPairs = affineAddressPairs.clone();
            movementGeometry = Objects.requireNonNull(movementGeometry, "movementGeometry");
            indexingGeometry = Objects.requireNonNull(indexingGeometry, "indexingGeometry");
            scatterGeometry = Objects.requireNonNull(scatterGeometry, "scatterGeometry");
            foldGeometry = Objects.requireNonNull(foldGeometry, "foldGeometry");
            orderingGeometry = Objects.requireNonNull(orderingGeometry, "orderingGeometry");
            randomGeometry = Objects.requireNonNull(randomGeometry, "randomGeometry");
            scanGeometry = Objects.requireNonNull(scanGeometry, "scanGeometry");
            aggregateGeometry = Objects.requireNonNull(aggregateGeometry, "aggregateGeometry");
            argExtremaGeometry = Objects.requireNonNull(argExtremaGeometry, "argExtremaGeometry");
            maskedReductionGeometry = Objects.requireNonNull(maskedReductionGeometry,
                    "maskedReductionGeometry");
            advancedReductionGeometry = Objects.requireNonNull(advancedReductionGeometry,
                    "advancedReductionGeometry");
            softmaxGeometry = Objects.requireNonNull(softmaxGeometry, "softmaxGeometry");
            trailingNormalizationGeometry = Objects.requireNonNull(trailingNormalizationGeometry,
                    "trailingNormalizationGeometry");
            batchNormInferenceGeometry = Objects.requireNonNull(batchNormInferenceGeometry,
                    "batchNormInferenceGeometry");
            batchNormTrainingGeometry = Objects.requireNonNull(batchNormTrainingGeometry,
                    "batchNormTrainingGeometry");
            conv2dGeometry = Objects.requireNonNull(conv2dGeometry, "conv2dGeometry");
            Objects.requireNonNull(fusionReason, "fusionReason");
            int size = boundaryValues.size();
            if (size < 1 || accessBindings.size() != size || referencedElementSpans.size() != size
                    || boundaryDataTypes.size() != size) {
                throw new IllegalArgumentException("lowering boundary facts must agree");
            }
        }
        /**
         * Returns final iteration extents without exposing the retained array.
         *
         * @return a new defensive copy of the non-null extents
         */
        @Override public long[] extents() { return extents.clone(); }
        /**
         * Returns cold-composed affine source/result addresses.
         *
         * @return a defensive copy of alternating source/result address pairs, empty for
         *     pointwise IR
         */
        @Override public long[] affineAddressPairs() { return affineAddressPairs.clone(); }
        /**
         * Returns the cache-compatible generated form.
         *
         * @return the retained pointwise IR or a newly encoded instruction-free affine IR;
         *     never {@code null}
         */
        public CpuKernelIr kernelIr() {
            return portableKernelIr instanceof CpuKernelIr pointwise ? pointwise
                    : portableKernelIr instanceof CpuAffineCopyIr affine
                        ? affine.encodedKernelIr()
                        : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr movement
                            ? movement.encodedKernelIr()
                            : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr indexing
                                ? indexing.encodedKernelIr()
                                : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr scatter
                                    ? scatter.encodedKernelIr()
                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFoldIr fold
                                        ? fold.encodedKernelIr()
                                        : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuOrderingIr ordering
                                            ? ordering.encodedKernelIr()
                                            : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr random
                                                ? random.encodedKernelIr()
                                                : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr scan
                                                    ? scan.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuArgExtremaIr argExtrema
                                                        ? argExtrema.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMaskedReductionIr masked
                                                        ? masked.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAdvancedReductionIr advanced
                                                        ? advanced.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxIr softmax
                                                        ? softmax.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuTrailingNormalizationIr normalization
                                                        ? normalization.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormInferenceIr batch
                                                        ? batch.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuBatchNormTrainingIr training
                                                        ? training.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv2dIr conv2d
                                                        ? conv2d.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr conv3d
                                                        ? conv3d.encodedKernelIr()
                                                    : ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr)
                                                        portableKernelIr).encodedKernelIr();
        }
    }
}
