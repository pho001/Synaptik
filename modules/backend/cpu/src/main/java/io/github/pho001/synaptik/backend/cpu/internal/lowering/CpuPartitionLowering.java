package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
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
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Lowers one bounded CPU unit into generated portable IR or mutually exclusive direct scan facts.
 * A proved explicit static typed-one SUM is recognized before ordinary family dispatch and
 * lowers to output-only constant stores, never a new shared compile-constant fact.
 * Static homogeneous FLOAT32/FLOAT64 private producer/SUM-or-MEAN/result-tail units compose
 * the same pointwise sites and exact aggregate facts after that count proof, within eight nodes.
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
 * ordering, explicit-state random, cumulative-scan, aggregate, softmax, trailing Layer/RMS
 * normalization, and atomic first-class loss families are delegated to focused lowerers; the
 * movement family includes functional slice update. A loss lowerer preserves its two semantic
 * input positions even when they share one physical read boundary. A general scan delegates its
 * complete body and binder to {@link CpuGeneralScanLowering}; it is one generated unit rather
 * than an ordinary pointwise fusion candidate or direct fixed recurrence. Proved private
 * rank-two transpose prefixes delegate directly to MATMUL's existing strided reads.</p>
 * <p>The bounded common pointwise facts and aggregate axis/target normalization are also used
 * by Scan. Ordinary broader-family admission stays independent; shared validation changes no
 * route, layout mapping, fusion boundary or exact-state reduction policy. A separate bounded
 * ordinary row candidate may interleave scalar sites and feature contractions, retain exact
 * outside/SUM stores, and publish several completed terminals. The full-partition selector
 * proves its complete uses and convexity before accepting full-body work count one.</p>
 */
public final class CpuPartitionLowering {
    /**
     * Checks CPU's temporary low-arithmetic admission without invoking capability or lowering.
     * Exact represented-value operations, casts and storage/control remain available under
     * their independently checked structural predicates.
     *
     * @param query non-null exact semantic occurrence
     * @return whether no low floating operand occupies a numerical role
     * @throws NullPointerException if the query is null
     */
    public static boolean permitsLowPrecisionUse(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        boolean lowOperand = false;
        for (var input : query.inputs()) lowOperand |= low(input.dataType());
        if (!lowOperand) return true;
        var kind = query.operation().kind();
        if (kind instanceof BinaryArithmeticKind || kind instanceof ScalarElementwiseKind
                || kind instanceof UnaryElementwiseKind || kind instanceof FloatingClassificationKind
                || kind instanceof BinaryComparisonKind || kind instanceof BooleanLogicalKind
                || kind == WhereSelectionKind.WHERE || kind == CastKind.CAST)
            return !opcode(kind).usesFloatingArithmetic();
        if (kind instanceof AggregateReductionKind reduction) {
            return switch (reduction) {
                case MIN, MAX, ALL, ANY, ARG_MIN, ARG_MAX -> true;
                default -> false;
            };
        }
        if (kind instanceof Pool2dKind pool) return pool == Pool2dKind.MAX_POOL2D;
        if (kind instanceof Pool3dKind pool) return pool == Pool3dKind.MAX_POOL3D;
        if (query.operation().attrs() instanceof ScatterElementsAttrs scatter)
            return scatter.reduction() != ScatterReduction.ADD && scatter.reduction() != ScatterReduction.MUL;
        if (query.operation().attrs() instanceof ScatterNdAttrs scatter)
            return scatter.reduction() != ScatterReduction.ADD && scatter.reduction() != ScatterReduction.MUL;
        return !(kind == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL
                || kind instanceof io.github.pho001.synaptik.model.operation.convolution.Conv2dKind
                || kind instanceof io.github.pho001.synaptik.model.operation.convolution.Conv3dKind
                || kind instanceof io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind
                || kind instanceof io.github.pho001.synaptik.model.operation.normalization.LayerNormKind
                || kind instanceof io.github.pho001.synaptik.model.operation.normalization.RmsNormKind
                || kind instanceof io.github.pho001.synaptik.model.operation.normalization.BatchNormKind
                || kind instanceof io.github.pho001.synaptik.model.operation.loss.LossKind
                || kind instanceof io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind
                || kind instanceof io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind
                || kind instanceof io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind
                || kind == io.github.pho001.synaptik.model.operation.random.DropoutKind.DROPOUT
                || kind == io.github.pho001.synaptik.model.operation.index.AxisScatterKind.SCATTER_ADD
                || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD_AXIS
                || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD2D
                || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD3D);
    }

    private static boolean low(DataType type) {
        return type == DataType.BFLOAT16 || type == DataType.FLOAT16;
    }
    /**
     * Authenticates retained typed portable facts before resource or realization work.
     * Low storage alone is not arithmetic; count-write is a SUM-derived physical family.
     *
     * @param ir non-null exact portable facts
     * @throws NullPointerException if the facts are null
     * @throws IllegalArgumentException if low floating arithmetic is requested
     */
    public static void requirePermittedLowPrecisionUse(CpuPortableKernelIr ir) {
        Objects.requireNonNull(ir, "ir");
        switch (ir) {
            case CpuKernelIr kernel -> requirePermittedEnvelope(kernel);
            case CpuAffineCopyIr ignored -> { }
            case CpuDataMovementIr ignored -> { }
            case CpuIndexingIr ignored -> { }
            case CpuOrderingIr ignored -> { }
            case CpuArgExtremaIr ignored -> { }
            case CpuConstantWriteIr count -> requireNumericalType(count.immediate().dataType());
            case CpuAggregateIr aggregate -> {
                if (aggregate.kind() != CpuAggregateIr.Kind.MIN && aggregate.kind() != CpuAggregateIr.Kind.MAX)
                    requireNumericalType(aggregate.dataType());
            }
            case CpuPool2dIr pool -> {
                if (pool.kind() != CpuPool2dIr.Kind.MAX) requireNumericalType(pool.dataType());
            }
            case CpuPool3dIr pool -> {
                if (pool.kind() != CpuPool3dIr.Kind.MAX) requireNumericalType(pool.dataType());
            }
            case CpuScatterIr scatter -> {
                if (scatter.reduction() == ScatterReduction.ADD || scatter.reduction() == ScatterReduction.MUL)
                    requireNumericalType(scatter.boundaryTypes().get(scatter.occurrenceToBoundary().getFirst()));
            }
            case CpuRandomIr random -> {
                if (random.family() == CpuRandomIr.Family.DROPOUT) requireNumericalType(random.valueType());
            }
            case CpuAggregateFusionIr fusion -> {
                requirePermittedLowPrecisionUse(fusion.aggregate());
                fusion.producer().ifPresent(CpuPartitionLowering::requirePermittedLowPrecisionUse);
                fusion.epilogue().ifPresent(CpuPartitionLowering::requirePermittedLowPrecisionUse);
            }
            case CpuGeneralScanIr scan -> {
                requirePermittedLowPrecisionUse(scan.arithmetic());
                scan.composition().ifPresent(composition -> composition.units().forEach(unit -> {
                    requirePermittedLowPrecisionUse(unit.ir());
                    requirePermittedEnvelope(unit.kernel());
                }));
            }
            case CpuRowMapReductionIr row -> requirePermittedLowPrecisionUse(row.arithmetic());
            case CpuFoldIr fold -> requireNumericalType(fold.dataType());
            case CpuScanIr scan -> requireNumericalType(scan.dataType());
            case CpuMaskedReductionIr reduction -> requireNumericalType(reduction.dataType());
            case CpuAdvancedReductionIr reduction -> requireNumericalType(reduction.dataType());
            case CpuSoftmaxIr softmax -> requireNumericalType(softmax.dataType());
            case CpuSoftmaxBackwardIr softmax -> requirePermittedLowPrecisionUse(softmax.aggregate());
            case CpuTrailingNormalizationIr normalization -> normalization.inputTypes().forEach(CpuPartitionLowering::requireNumericalType);
            case CpuBatchNormInferenceIr normalization -> normalization.inputTypes().forEach(CpuPartitionLowering::requireNumericalType);
            case CpuBatchNormTrainingIr normalization -> normalization.inputTypes().forEach(CpuPartitionLowering::requireNumericalType);
            case CpuConv2dIr convolution -> convolution.inputTypes().forEach(CpuPartitionLowering::requireNumericalType);
            case CpuConv3dIr convolution -> convolution.inputTypes().forEach(CpuPartitionLowering::requireNumericalType);
            case CpuAttentionIr attention -> {
                requireNumericalType(attention.queryType());
                requireNumericalType(attention.keyType());
                requireNumericalType(attention.valueType());
            }
            case CpuLossIr loss -> {
                requireNumericalType(loss.predictionType());
                requireNumericalType(loss.targetType());
            }
        }
    }

    /**
     * Authenticates the current envelope and the typed facts consumed by generated emission,
     * before cache lookup, fresh generation or direct definition.
     *
     * @param specialization non-null exact generated specialization
     * @param ir non-null actual represented envelope
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if typed MATMUL disagrees or low arithmetic is requested
     */
    public static void requirePermittedLowPrecisionUse(CpuKernelSpecialization specialization, CpuKernelIr ir) {
        Objects.requireNonNull(specialization, "specialization");
        Objects.requireNonNull(ir, "ir");
        if (specialization.matmulIr().isPresent()) {
            var matrix = specialization.matmulIr().orElseThrow();
            String matrixKey = matrix.structuralKey();
            if (!matrixKey.equals(ir.structuralKey())
                    || !matrixKey.equals(specialization.loweringFingerprint().hex()))
                throw new IllegalArgumentException("typed MATMUL facts disagree with represented envelope");
            requireNumericalType(matrix.leftType());
            requireNumericalType(matrix.rightType());
        }
        specialization.generalScanIr().ifPresent(CpuPartitionLowering::requirePermittedLowPrecisionUse);
        specialization.rowMapReductionIr().ifPresent(CpuPartitionLowering::requirePermittedLowPrecisionUse);
        specialization.aggregateFusionIr().ifPresent(CpuPartitionLowering::requirePermittedLowPrecisionUse);
        for (int ordinal = 0; ordinal < specialization.compositionUnits().size(); ordinal++) {
            var helper = specialization.compositionUnits().get(ordinal).specialization();
            var physical = specialization.generalScanIr().orElseThrow().composition()
                    .orElseThrow().units().get(ordinal);
            requirePermittedLowPrecisionUse(physical.ir());
            requirePermittedLowPrecisionUse(helper, physical.kernel());
        }
        requirePermittedEnvelope(ir);
    }

    private static void requirePermittedEnvelope(CpuKernelIr ir) {
        for (var instruction : ir.instructions()) if (instruction.opcode().usesFloatingArithmetic())
            for (int input : instruction.inputs()) requireNumericalType(ir.values().get(input).dataType());
        if (!ir.instructions().isEmpty() || ir.values().stream().noneMatch(v -> low(v.dataType()))) return;
        String family = ir.familyIdentity();
        boolean numerical;
        if (family.startsWith("aggregate:")) {
            var kind = CpuAggregateIr.encodedKind(ir);
            numerical = kind != CpuAggregateIr.Kind.MIN && kind != CpuAggregateIr.Kind.MAX;
        } else if (family.startsWith("pool2d:")) numerical = CpuPool2dIr.encodedKind(ir) != CpuPool2dIr.Kind.MAX;
        else if (family.startsWith("pool3d:")) numerical = CpuPool3dIr.encodedKind(ir) != CpuPool3dIr.Kind.MAX;
        else if (family.startsWith("scatter:")) {
            var reduction = CpuScatterIr.encodedReduction(ir);
            numerical = reduction == ScatterReduction.ADD || reduction == ScatterReduction.MUL;
        } else if (family.startsWith("random:")) numerical = CpuRandomIr.encodedFamily(ir) == CpuRandomIr.Family.DROPOUT;
        else numerical = family.startsWith("constant-write-v1:") || family.startsWith("matmul:")
                || family.startsWith("conv2d:") || family.startsWith("conv3d:")
                || family.startsWith("fold:") || family.startsWith("scan:")
                || family.startsWith("masked-reduction:") || family.startsWith("advanced-reduction:")
                || family.startsWith("softmax:") || family.startsWith("softmax-backward:")
                || family.startsWith("trailing-normalization:") || family.startsWith("batch-normalization-")
                || family.startsWith("attention:") || family.startsWith("loss:");
        if (numerical) throw lowArithmetic();
    }

    private static void requireNumericalType(DataType type) {
        if (low(type)) throw lowArithmetic();
    }

    private static IllegalArgumentException lowArithmetic() {
        return new IllegalArgumentException("CPU BFLOAT16/FLOAT16 numerical computation is temporarily unsupported");
    }

    /**
     * Rejects source operations that use BFLOAT16 or FLOAT16 in unsupported numerical roles.
     * Layout-only and other admitted low-precision uses remain eligible for lowering.
     * @param context non-null source projection with complete value descriptors and scan plans
     * @throws IllegalArgumentException if an operation requires unsupported low-precision arithmetic
     */
    static void requirePermittedSource(PrepareContext<? extends BackendAnalysisInputs> context) {
        boolean lowValue = false;
        for (var value : context.values()) lowValue |= low(value.descriptor().dataType());
        if (!lowValue) return;
        var values = indexValues(context.values());
        for (var node : context.nodes()) {
            var query = new OperationCapabilityQuery(node.operation(),
                    node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                    node.outputs().stream().map(id -> require(values, id).descriptor()).toList(), node.scanBody(),
                    Optional.ofNullable(context.scanActivityPlans().get(node.id())));
            if (!permitsLowPrecisionUse(query)) throw lowArithmetic();
        }
    }

    /**
     * Checks original ordinary semantics and the sole named batch without sampling geometry.
     * @param context exact local ordinary projection
     * @return non-null sole named batch dimension
     * @throws IllegalArgumentException if dimensions, purity or ordinary descriptors disagree
     */
    static io.github.pho001.synaptik.model.shape.DynamicDimension requireBatchContext(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        var values = indexValues(context.values());
        io.github.pho001.synaptik.model.shape.DynamicDimension batch = null;
        for (GraphValue value : context.values()) {
            int named = 0;
            for (var dimension : value.descriptor().shape().dimensions()) {
                if (dimension instanceof io.github.pho001.synaptik.model.shape.DynamicDimension candidate) {
                    if (++named > 1 || batch != null && !batch.equals(candidate))
                        throw new IllegalArgumentException("independent ordinary batch geometry");
                    batch = candidate;
                } else if (!(dimension instanceof io.github.pho001.synaptik.model.shape.StaticDimension)) {
                    throw new IllegalArgumentException("unproved ordinary dimension expression");
                }
            }
        }
        if (batch == null) throw new IllegalArgumentException("missing ordinary batch coordinate");
        for (CompiledNode node : context.nodes()) {
            var outputs = io.github.pho001.synaptik.model.tensor.ScanBody.describeOrdinary(
                    node.operation(), node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                    node.outputs().size());
            for (int port = 0; port < outputs.size(); port++) {
                var expected = outputs.get(port);
                var actual = require(values, node.outputs().get(port)).descriptor();
                if (actual.dataType() != expected.dataType() || !actual.shape().equals(expected.shape())
                        || actual.requiresGrad() != expected.requiresGrad())
                    throw new IllegalArgumentException("ordinary canonical descriptor mismatch");
            }
        }
        requirePermittedSource(context);
        return batch;
    }

    /**
     * Derives an affine logical count from exact numeric boundary geometry.
     * @param binding proved one-B boundary geometry
     * @return constant-plus-B logical count with unit alignment
     * @throws ArithmeticException if products or coefficients overflow
     * @throws IllegalArgumentException if the count is not affine
     */
    static io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize batchElements(
            CpuAccessPlan.BatchBinding binding) {
        long constant = 1, factor = 0;
        for (int axis = binding.plan().iterationRank() - 1; axis >= 0; axis--) {
            long c = binding.extentConstants().get(axis), f = binding.extentBatchFactors().get(axis);
            if (factor != 0 && f != 0) throw new IllegalArgumentException("non-affine ordinary count");
            factor = Math.addExact(Math.multiplyExact(factor, c), Math.multiplyExact(constant, f));
            constant = Math.multiplyExact(constant, c);
        }
        return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                constant, factor, 1);
    }


    /**
     * Derives bounded ordinary/Scan symbolic boundary layout intent once from projected semantics.
     * Authenticated transpose, expand and singleton-preserving rank/reshape mappings retain
     * actual strides, including sequence time strides, separately from nominal Scan frames.
     * No dimension is substituted with a representative batch size. Returned maps are cold,
     * immutable numeric recipes; no graph is retained by the eventual invocation.
     * @param context non-null exact CPU partition projection
     * @return immutable value-to-read-mapping recipes, including fixed closure helpers
     * @throws IllegalArgumentException if dimensions, layout or operation is outside this slice
     * @throws ArithmeticException if structural coefficient arithmetic overflows
     */
    public static Map<ValueId, CpuAccessPlan.BatchBinding> batchLayouts(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        var layouts = new LinkedHashMap<ValueId, CpuAccessPlan.BatchBinding>();
        for (var value : context.values()) {
            var canonical = batchCanonical(value.descriptor(), CpuAccessPlan.AccessKind.READ);
            var layout = value.descriptor().layout();
            if (layout.isPresent()) {
                var constants = java.util.Arrays.stream(layout.orElseThrow().strides()).boxed().toList();
                canonical = batchAccess(CpuAccessPlan.AccessKind.READ, canonical.extentConstants(),
                        canonical.extentBatchFactors(), constants, java.util.Collections.nCopies(constants.size(), 0L));
                canonical = new CpuAccessPlan.BatchBinding(canonical.plan(), canonical.extentConstants(),
                        canonical.extentBatchFactors(), layout.orElseThrow().storageOffset(),
                        canonical.strideConstants(), canonical.strideBatchFactors());
            }
            layouts.put(value.id(), canonical);
        }
        for (var node : context.nodes()) {
            Object kind = node.operation().kind();
            if (!(kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind)
                    && kind != io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND
                    && kind != io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE) continue;
            var source = layouts.get(node.inputs().getFirst());
            var target = layouts.get(node.outputs().getFirst());
            var constants = new ArrayList<Long>(); var factors = new ArrayList<Long>();
            if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE) {
                var axes = ((io.github.pho001.synaptik.model.operation.layout.PermutationAttrs) node.operation().attrs()).axes();
                for (int axis : axes) { constants.add(source.strideConstants().get(axis)); factors.add(source.strideBatchFactors().get(axis)); }
            } else if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE) {
                constants.addAll(target.strideConstants()); factors.addAll(target.strideBatchFactors());
            } else if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND) {
                int shift = target.plan().iterationRank() - source.plan().iterationRank();
                for (int axis = 0; axis < target.plan().iterationRank(); axis++) {
                    int from = axis - shift;
                    boolean broadcast = from < 0 || source.extentConstants().get(from) == 1
                            && source.extentBatchFactors().get(from) == 0
                            && (target.extentConstants().get(axis) != 1
                                || target.extentBatchFactors().get(axis) != 0);
                    constants.add(broadcast ? 0 : source.strideConstants().get(from));
                    factors.add(broadcast ? 0 : source.strideBatchFactors().get(from));
                }
            } else {
                int changed = ((io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs) node.operation().attrs()).axis();
                boolean insert = kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.EXPAND_DIMS;
                for (int axis = 0; axis < target.plan().iterationRank(); axis++) {
                    if (insert && axis == changed) {
                        if (axis == source.plan().iterationRank()) { constants.add(1L); factors.add(0L); }
                        else {
                            long ec = source.extentConstants().get(axis), ef = source.extentBatchFactors().get(axis);
                            long sc = source.strideConstants().get(axis), sf = source.strideBatchFactors().get(axis);
                            if (ef != 0 && sf != 0) throw new IllegalArgumentException("non-affine inserted stride");
                            constants.add(Math.multiplyExact(ec, sc));
                            factors.add(Math.addExact(Math.multiplyExact(ec, sf), Math.multiplyExact(ef, sc)));
                        }
                    } else {
                        int from = insert ? axis < changed ? axis : axis - 1 : axis < changed ? axis : axis + 1;
                        constants.add(source.strideConstants().get(from)); factors.add(source.strideBatchFactors().get(from));
                    }
                }
            }
            var mapped = batchAccess(CpuAccessPlan.AccessKind.READ,
                    target.extentConstants(), target.extentBatchFactors(), constants, factors);
            long origin = kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE
                    && source.plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                    ? target.baseElementOffset() : source.baseElementOffset();
            layouts.put(node.outputs().getFirst(), new CpuAccessPlan.BatchBinding(mapped.plan(),
                    mapped.extentConstants(), mapped.extentBatchFactors(), origin,
                    mapped.strideConstants(), mapped.strideBatchFactors()));
        }
        return java.util.Collections.unmodifiableMap(layouts);
    }

    /**
     * Declares exact backing spans for the prepared canonical/transpose/expand domain.
     * View copies preserve their source backing, including a scalar backing for an empty expand;
     * materialized results, Scan ports and explicit splats allocate their actual logical element
     * count. These are storage-free byte recipes, not an evaluation at a representative B.
     * Each value retains its represented byte width; discrete INT64 control is not a body float.
     * @param context non-null exact projection
     * @return immutable complete numeric byte declarations, with no recipe evaluation
     * @throws IllegalArgumentException if a shape is outside the bounded affine domain
     * @throws ArithmeticException if coefficient arithmetic overflows
     */
    public static Map<ValueId, io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize>
            batchByteSizes(PrepareContext<? extends BackendAnalysisInputs> context) {
        var sizes = new LinkedHashMap<ValueId, io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize>();
        for (var value : context.values()) {
            if (value.descriptor().shape().isFullyStatic() && value.descriptor().layout().isPresent()) {
                int width = value.descriptor().dataType().byteWidth();
                sizes.put(value.id(), new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                        Math.multiplyExact(value.descriptor().layout().orElseThrow().referencedElementSpan(), width), 0, width));
                continue;
            }
            var binding = batchCanonical(value.descriptor(), CpuAccessPlan.AccessKind.READ);
            long constant = 1, factor = 0;
            for (int axis = binding.plan().iterationRank() - 1; axis >= 0; axis--) {
                long c = binding.extentConstants().get(axis), f = binding.extentBatchFactors().get(axis);
                if (factor != 0 && f != 0) throw new IllegalArgumentException("non-affine batch count");
                factor = Math.addExact(Math.multiplyExact(factor, c), Math.multiplyExact(constant, f));
                constant = Math.multiplyExact(constant, c);
            }
            int width = value.descriptor().dataType().byteWidth();
            sizes.put(value.id(), new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(
                    Math.multiplyExact(constant, width), Math.multiplyExact(factor, width), width));
        }
        var layouts = batchLayouts(context);
        for (var node : context.nodes()) if (batchView(node.operation().kind())) {
            if (node.operation().kind()
                            == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE
                    && layouts.get(node.inputs().getFirst()).plan().regime()
                            != CpuAccessPlan.Regime.DENSE_LINEAR) continue;
            sizes.put(node.outputs().getFirst(), sizes.get(node.inputs().getFirst()));
        }
        return java.util.Collections.unmodifiableMap(sizes);
    }
    /**
     * Declares the exact backing bytes for one producerless CPU source descriptor.
     *
     * <p>Resolved static affine layouts use Model's referenced element span, preserving origins,
     * holes, broadcasts and the zero span of empty layouts. Their byte declaration is genuinely
     * fixed. An unresolved sole-named-batch template instead retains the existing canonical read
     * binding and resolves its checked span only at the actual invocation batch. Even a template
     * with permanently zero bytes remains structurally batch-bound. This source-only factory
     * neither admits arithmetic nor changes ordinary partition byte recipes or identities.</p>
     *
     * @param descriptor non-null exact source descriptor; static layouts must be resolved, while
     *     named templates use the existing unresolved canonical actual-input contract
     * @return non-null checked fixed size or an unevaluated canonical batch-layout size recipe
     * @throws NullPointerException if {@code descriptor} is null
     * @throws IllegalArgumentException if static layout is unresolved or the named geometry is
     *     outside the existing sole-named-axis affine CPU domain
     * @throws ArithmeticException if static byte multiplication or structural CPU limits overflow
     */
    public static PreparedMemoryPlan.ByteSize descriptorByteSize(TensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor");
        if (descriptor.shape().isFullyStatic()) {
            LayoutDescriptor layout = descriptor.layout().orElseThrow(() ->
                    new IllegalArgumentException("CPU source-only published value has unresolved layout"));
            return new PreparedMemoryPlan.FixedByteSize(Math.multiplyExact(
                    layout.referencedElementSpan(), descriptor.dataType().byteWidth()));
        }
        return new CpuRepresentationRecipes.BatchLayoutByteSize(
                batchCanonical(descriptor, CpuAccessPlan.AccessKind.READ), descriptor.dataType());
    }


    /**
     * Lowers one qualified node to the existing portable route with B-only numeric recipes.
     * This slice fixes generated scalar, one range and typed carriers before any run exists.
     * Retained F32 occurrences keep ordinary-batch-v1. Added typed/control/ROW(1) occurrences
     * bind exact mapping coefficients and typed reduction state in ordinary-batch-typed-v1;
     * their checked-segment route is independent of Scan origin and actual B/L.
     * @param node non-null exact projected node
     * @param values non-null projected descriptors by value identity
     * @param layouts non-null prepared semantic read mappings
     * @param sizes non-null prepared backing declarations
     * @param carriers non-null requested ordered unique-input-then-output carrier pattern
     * @return immutable numeric portable route; no graph or Model lowerer enters Runtime
     * @throws IllegalArgumentException if the occurrence or carriers are unsupported
     * @throws ArithmeticException if structural geometry overflows
     */
    public static io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan lowerOrdinaryBatch(
            CompiledNode node, Map<ValueId, GraphValue> values,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts,
            Map<ValueId, io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize> sizes,
            List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess> carriers) {
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(id -> values.get(id).descriptor()).toList(),
                node.outputs().stream().map(id -> values.get(id).descriptor()).toList());
        if (!supportsOrdinaryBatch(query)) throw new IllegalArgumentException("unsupported ordinary batch occurrence: " + node.operation());
        boolean typed = !supportsOrdinaryBatch(query,false);
        DataType type = query.inputs().getFirst().dataType(), resultType = query.outputs().getFirst().dataType();
        if (typed && carriers.stream().anyMatch(c -> c != io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT))
            throw new IllegalArgumentException("typed ordinary batch requires checked segments");
        var ids = new ArrayList<ValueId>(); for (var id : node.inputs()) if (!ids.contains(id)) ids.add(id);
        ids.add(node.outputs().getFirst());
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int i = 0; i < ids.size() - 1; i++) bindings.add(layouts.get(ids.get(i)));
        var output = batchCanonical(values.get(ids.getLast()).descriptor(), CpuAccessPlan.AccessKind.WRITE);
        bindings.add(output);
        Object kind = node.operation().kind(); var attrs = node.operation().attrs();
        Optional<CpuMatmulLowering.BatchGeometry> matrix = Optional.empty();
        Optional<CpuAggregateLowering.BatchGeometry> aggregate = Optional.empty();
        Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr> matrixIr = Optional.empty();
        CpuKernelIr ir;
        if (kind == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
            matrix = Optional.of(new CpuMatmulLowering.BatchGeometry(DataType.FLOAT32, bindings.get(0), bindings.get(1), output));
            var typedMatrix = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
                    io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Realization.DIRECT_SCALAR,
                    io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.Epilogue.none(), 0,
                    io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr.NumericalForm.SEQUENTIAL,
                    List.of(bindings.get(0).plan(), bindings.get(1).plan()), output.plan(), matrix.orElseThrow());
            matrixIr = Optional.of(typedMatrix); ir = typedMatrix.encodedKernelIr();
        } else if (kind instanceof io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind reduction) {
            if (bindings.getFirst().plan().regime() != CpuAccessPlan.Regime.DENSE_LINEAR
                    && bindings.getFirst().plan().regime() != CpuAccessPlan.Regime.SCALAR_ALL_ZERO) {
                throw new IllegalArgumentException("ordinary batch aggregate requires canonical or scalar-broadcast reads");
            }
            var form = CpuAggregateLowering.reductionForm(attrs);
            boolean keep = CpuAggregateLowering.keepDimensions(attrs);
            var axes = CpuAggregateLowering.normalizedAxes(query.inputs().getFirst().shape(),query.outputs().getFirst().shape(),attrs);
            var aggregateKind = io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr.Kind.valueOf(reduction.name());
            aggregate = Optional.of(new CpuAggregateLowering.BatchGeometry(aggregateKind, type, form, axes, keep, bindings.getFirst(), output));
            ir = new CpuKernelIr(List.of(new CpuKernelIr.Value(0, type, CpuKernelIr.Value.Kind.INPUT, bindings.getFirst().plan()),
                    new CpuKernelIr.Value(1, type, CpuKernelIr.Value.Kind.OUTPUT, output.plan())), List.of(),
                    new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)),
                    "aggregate:" + reduction + ":" + form + ":axes=" + axes + ":keep=" + keep
                            + ":source=[]:target=[]:domain=run:limbs="+(type == DataType.FLOAT64 ? 34 : 5)
                            +":slice="+(type == DataType.FLOAT64 ? 280 : 48)+":floating=1:range=1:workspace=1");
        } else {
            boolean view = batchView(kind);
            if (view) {
                var bytes = (io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize) sizes.get(ids.getLast());
                List<Long> ec = List.of(bytes.constantBytes() / type.byteWidth()), ef = List.of(bytes.bytesPerBatch() / type.byteWidth());
                bindings.set(0, batchAccess(CpuAccessPlan.AccessKind.READ, ec, ef, List.of(1L), List.of(0L)));
                output = batchAccess(CpuAccessPlan.AccessKind.WRITE, ec, ef, List.of(1L), List.of(0L)); bindings.set(1, output);
            } else for (int i = 0; i < bindings.size() - 1; i++) bindings.set(i, batchBroadcast(bindings.get(i), output));
            var irValues = new ArrayList<CpuKernelIr.Value>();
            for (int i = 0; i < bindings.size(); i++) irValues.add(new CpuKernelIr.Value(i, i == bindings.size()-1 ? resultType : type,
                    i == bindings.size() - 1 ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.INPUT, bindings.get(i).plan()));
            boolean pointwise = kind instanceof BinaryArithmeticKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
            if (pointwise) validatePointwise(node,query.inputs(),query.outputs().getFirst());
            var opcode = pointwise ? opcode(node.operation().kind()) : CpuPointwiseOpcode.CAST;
            List<Integer> operands = pointwise ? node.inputs().stream().map(ids::indexOf).toList() : List.of(0);
            ir = new CpuKernelIr(irValues, List.of(new CpuKernelIr.Instruction(opcode, operands, irValues.size() - 1, pointwise ? scalarImmediate(node) : null)),
                    new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(irValues.size() - 1, 0)));
        }
        if (matrixIr.isEmpty()) {
            ir = new CpuKernelIr(ir.values(), ir.instructions(), ir.loop(), ir.stores(), ir.familyIdentity()
                    + (typed ? ":mapping="+bindings+":ordinary-batch-typed-v1" : ":ordinary-batch-v1"));
        }
        var specialization = new io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                ids.stream().map(id -> values.get(id).descriptor().dataType()).toList(), carriers, 0, -1, List.of(), aggregate.isPresent(),
                matrix.isPresent() ? 54 : type != resultType ? 60 : 52, matrixIr, Optional.empty(), true,typed);
        return new io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuPortableRoutePlan(ir, specialization, bindings, matrix, aggregate);
    }

    /**
     * Qualifies the retained ordinary F32 domain or the bounded additive F32/F64 outer closure
     * and non-gradient [B] floating-to-INT64 control CAST. Answers depend on the occurrence only,
     * never on Scan/AD origin. F64 MATMUL and arbitrary rank-three algebra remain closed.
     * @param query non-null complete semantic occurrence
     * @return whether exact shapes, attributes, types and symbolic dimensions qualify
     */
    public static boolean supportsOrdinaryBatch(OperationCapabilityQuery query) {
        return supportsOrdinaryBatch(query,false) || supportsOrdinaryBatch(query,true);
    }

    /**
     * Checks one compatibility domain without consulting graph provenance or runtime values.
     * @param query complete occurrence, including exact attributes and layouts
     * @param typed whether the additive typed/control/ROW(1) domain is under proof
     * @return whether every descriptor, family, attribute and mapping predicate qualifies
     */
    private static boolean supportsOrdinaryBatch(OperationCapabilityQuery query, boolean typed) {
        try {
            if (query.outputs().size() != 1 || query.inputs().isEmpty() || query.scanBody().isPresent()) return false;
            var input = query.inputs().getFirst(); var output = query.outputs().getFirst();
            Object kind = query.operation().kind(), attrs = query.operation().attrs();
            DataType type = input.dataType();
            boolean control = typed && kind == CastKind.CAST && type != output.dataType();
            if (typed && type != DataType.FLOAT32 && type != DataType.FLOAT64) return false;
            if (control && (query.inputs().size() != 1 || input.requiresGrad() || output.requiresGrad()
                    || input.shape().rank() != 1 || !(input.shape().dimension(0) instanceof io.github.pho001.synaptik.model.shape.DynamicDimension)
                    || output.dataType() != DataType.INT64 || !input.shape().equals(output.shape())
                    || !(attrs instanceof io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs c) || c.targetDataType() != DataType.INT64)) return false;
            if (typed && !control && output.dataType() != type) return false;
            boolean rowOneStack = java.util.stream.Stream.concat(query.inputs().stream(),query.outputs().stream())
                    .anyMatch(d -> d.shape().rank() == 3 && d.shape().dimension(2) instanceof io.github.pho001.synaptik.model.shape.DynamicDimension);
            if (typed && rowOneStack && kind != io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND
                    && kind != io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE
                    && !(kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind)
                    && kind != io.github.pho001.synaptik.model.operation.layout.ContiguousKind.CONTIGUOUS && kind != CastKind.CAST)
                return false;
            if (output.requiresGrad() != (!control && query.inputs().stream().anyMatch(TensorDescriptor::requiresGrad))) return false;
            var all = new ArrayList<TensorDescriptor>(); all.addAll(query.inputs()); all.addAll(query.outputs());
            var symbols = new java.util.HashSet<String>();
            for (var d : all) {
                if (d.dataType() != (typed ? control && d == output ? DataType.INT64 : type : DataType.FLOAT32) || d.shape().rank() > 3) return false;
                if (!d.shape().isFullyStatic() && d.shape().rank() == 3
                        && !(d.shape().dimension(1) instanceof io.github.pho001.synaptik.model.shape.DynamicDimension)
                        && !(typed && d.shape().dimension(2) instanceof io.github.pho001.synaptik.model.shape.DynamicDimension)
                        && d.shape().dimensions().stream().noneMatch(dimension -> dimension.equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1)))) return false;
                int named = 0;
                for (var extent : d.shape().dimensions()) {
                    if (extent instanceof io.github.pho001.synaptik.model.shape.DynamicDimension n) { symbols.add(n.symbol()); named++; }
                    else if (!(extent instanceof io.github.pho001.synaptik.model.shape.StaticDimension s) || s.size() > Integer.MAX_VALUE) return false;
                }
                if (named > 1 || d.layout().isPresent() && d.layout().orElseThrow().storageOffset() != 0) return false;
                batchCanonical(d, CpuAccessPlan.AccessKind.READ);
                if (typed && d.layout().isPresent()) {
                    var l = d.layout().orElseThrow();
                    if (!d.shape().isFullyStatic()) return false;
                    long[] dense = LayoutDescriptor.contiguous(d.shape()).strides(), actual = l.strides();
                    boolean admitted = java.util.Arrays.equals(dense,actual) || java.util.Arrays.stream(actual).allMatch(s -> s == 0);
                    if (d.shape().rank() == 2) admitted |= java.util.Arrays.equals(actual,new long[]{1,d.shape().toLongArray()[0]});
                    if (!admitted) return false;
                }
            }
            if (symbols.size() > 1) return false;
            if (control) return true;
            if (kind == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
                if (typed) return false;
                if (query.inputs().size() != 2 || attrs != io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE) return false;
                var right = query.inputs().get(1);
                return input.shape().rank() == 2 && right.shape().rank() == 2
                        && input.shape().dimension(1).equals(right.shape().dimension(0))
                        && output.shape().equals(Shape.ofDimensions(input.shape().dimension(0), right.shape().dimension(1)));
            }
            if (kind == BinaryArithmeticKind.ADD || !typed && kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV) return query.inputs().size() == 2
                    && attrs == io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE
                    && io.github.pho001.synaptik.model.shape.ShapeBroadcast.broadcast(input.shape(), query.inputs().get(1).shape()).equals(output.shape());
            if (query.inputs().size() != 1) return false;
            if (kind instanceof io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind s)
                return (typed ? java.util.Set.of("ADD","MUL","DIV") : java.util.Set.of("ADD","SUB","MUL","DIV")).contains(s.name())
                        && attrs instanceof io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs a
                        && a.value().dataType() == type && input.shape().equals(output.shape());
            if (kind instanceof io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind u)
                return !typed && java.util.Set.of("NEG","TANH","SIGMOID").contains(u.name())
                        && attrs == io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE
                        && input.shape().equals(output.shape());
            if (kind == io.github.pho001.synaptik.model.operation.layout.ContiguousKind.CONTIGUOUS) return attrs == io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE && input.shape().equals(output.shape());
            if (kind == io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST
                    && attrs instanceof io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs c)
                return c.targetDataType() == type && input.shape().equals(output.shape());
            if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE
                    && attrs instanceof io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs t) {
                return t.targetShape().equals(output.shape()) && input.shape().dimensions().stream()
                        .filter(d -> !d.equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1))).toList()
                        .equals(output.shape().dimensions().stream().filter(d -> !d.equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1))).toList());
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE
                    && attrs instanceof io.github.pho001.synaptik.model.operation.layout.PermutationAttrs p) {
                return p.axes().size() == input.shape().rank() && new java.util.HashSet<>(p.axes()).size() == p.axes().size()
                        && p.axes().stream().allMatch(a -> a >= 0 && a < input.shape().rank()) && output.shape().equals(Shape.ofDimensions(p.axes().stream()
                        .map(input.shape()::dimension).toArray(io.github.pho001.synaptik.model.shape.Dimension[]::new)));
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.EXPAND_DIMS
                    && attrs instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs a) {
                var dimensions = new ArrayList<>(input.shape().dimensions());
                if (a.axis() < 0 || a.axis() > dimensions.size()) return false;
                dimensions.add(a.axis(), new io.github.pho001.synaptik.model.shape.StaticDimension(1));
                return output.shape().dimensions().equals(dimensions);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.SQUEEZE
                    && attrs instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs a) {
                var dimensions = new ArrayList<>(input.shape().dimensions());
                if (a.axis() < 0 || a.axis() >= dimensions.size() || !dimensions.get(a.axis()).equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1))) return false;
                dimensions.remove(a.axis()); return output.shape().dimensions().equals(dimensions);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND
                    && attrs instanceof io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs t) {
                return t.targetShape().equals(output.shape()) && io.github.pho001.synaptik.model.shape.ShapeBroadcast.broadcast(input.shape(), output.shape()).equals(output.shape());
            }
            if (kind != io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM
                    && kind != io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.MEAN) return false;
            if (attrs instanceof io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs
                    && kind != io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.SUM) return false;
            CpuAggregateLowering.normalizedAxes(input.shape(),output.shape(),attrs);
            return true;
        } catch (IllegalArgumentException | ArithmeticException unsupported) { return false; }
    }

    private static boolean batchView(Object kind) {
        return kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind
                || kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND
                || kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE;
    }

    /**
     * Derives canonical affine strides without resolving the single named batch coordinate.
     * Shared by ordinary and Scan lowering; a zero coefficient is not a sampled empty extent.
     * Nonzero fixed factors are checked independently of all empty axes before a recipe exists.
     * @param descriptor non-null bounded static/single-named-axis descriptor
     * @param kind non-null access permission for the resulting recipe
     * @return immutable zero-offset canonical coefficient binding
     * @throws IllegalArgumentException if dimensions are unsupported or strides are non-affine
     * @throws ArithmeticException if a structural coefficient overflows
     */
    static CpuAccessPlan.BatchBinding batchCanonical(TensorDescriptor descriptor, CpuAccessPlan.AccessKind kind) {
        var constants = new ArrayList<Long>(); var factors = new ArrayList<Long>();
        long fixedFeatures = 1;
        for (var d : descriptor.shape().dimensions()) {
            if (d instanceof io.github.pho001.synaptik.model.shape.StaticDimension s) {
                if (s.size() != 0) fixedFeatures = Math.multiplyExact(fixedFeatures, s.size());
                constants.add(s.size()); factors.add(0L);
            }
            else if (d instanceof io.github.pho001.synaptik.model.shape.DynamicDimension) { constants.add(0L); factors.add(1L); }
            else throw new IllegalArgumentException("unsupported ordinary batch expression");
        }
        var sc = new ArrayList<Long>(java.util.Collections.nCopies(constants.size(), 0L));
        var sf = new ArrayList<Long>(sc); long c = 1, f = 0;
        for (int axis = constants.size() - 1; axis >= 0; axis--) {
            sc.set(axis, c); sf.set(axis, f);
            if (f != 0 && factors.get(axis) != 0) throw new IllegalArgumentException("non-affine batch geometry");
            f = Math.addExact(Math.multiplyExact(f, constants.get(axis)), Math.multiplyExact(c, factors.get(axis)));
            c = Math.multiplyExact(c, constants.get(axis));
        }
        return batchAccess(kind, constants, factors, sc, sf);
    }

    private static CpuAccessPlan.BatchBinding batchBroadcast(CpuAccessPlan.BatchBinding input, CpuAccessPlan.BatchBinding output) {
        int rank = output.plan().iterationRank(), shift = rank - input.plan().iterationRank();
        var sc = new ArrayList<Long>(); var sf = new ArrayList<Long>();
        for (int axis = 0; axis < rank; axis++) {
            int from = axis - shift;
            boolean broadcast = from < 0 || input.extentConstants().get(from) == 1 && input.extentBatchFactors().get(from) == 0;
            sc.add(broadcast ? 0 : input.strideConstants().get(from)); sf.add(broadcast ? 0 : input.strideBatchFactors().get(from));
        }
        var mapped = batchAccess(CpuAccessPlan.AccessKind.READ,
                output.extentConstants(), output.extentBatchFactors(), sc, sf);
        return new CpuAccessPlan.BatchBinding(mapped.plan(), mapped.extentConstants(),
                mapped.extentBatchFactors(), input.baseElementOffset(),
                mapped.strideConstants(), mapped.strideBatchFactors());
    }

    /**
     * Classifies exact affine axis recipes once at lowering, preserving coordinate order.
     * No size-based inference or invocation binding is performed.
     * @param kind non-null read/write permission
     * @param ec non-null non-negative extent constants, in logical axis order
     * @param ef non-null extent batch coefficients, one per axis
     * @param sc non-null element-stride constants, one per axis
     * @param sf non-null element-stride batch coefficients, one per axis
     * @return immutable zero-offset access with coefficient-proved contiguous suffix
     * @throws IllegalArgumentException if axis recipes disagree or are unsupported
     * @throws ArithmeticException if a structural coefficient product overflows
     */
    static CpuAccessPlan.BatchBinding batchAccess(CpuAccessPlan.AccessKind kind, List<Long> ec,
            List<Long> ef, List<Long> sc, List<Long> sf) {
        int rank = ec.size(), suffix = 0; long c = 1, f = 0;
        for (int axis = rank - 1; axis >= 0; axis--) {
            if (sc.get(axis) != c || sf.get(axis) != f) break;
            suffix++; f = Math.addExact(Math.multiplyExact(f, ec.get(axis)), Math.multiplyExact(c, ef.get(axis)));
            c = Math.multiplyExact(c, ec.get(axis));
        }
        boolean allZero = sc.stream().allMatch(v -> v == 0) && sf.stream().allMatch(v -> v == 0);
        var regime = suffix == rank ? CpuAccessPlan.Regime.DENSE_LINEAR
                : allZero ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        for (int axis = 0; axis < rank; axis++) roles.add(sc.get(axis) == 0 && sf.get(axis) == 0 ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= rank - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        return new CpuAccessPlan.BatchBinding(new CpuAccessPlan(kind, regime, rank, roles, suffix), ec, ef, 0, sc, sf);
    }
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    private final Map<ValueId, CpuAccessPlan.BatchBinding> scanBatchLayouts;
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
    private final CpuMatmulLowering matmulLowering = new CpuMatmulLowering();
    private final CpuPool2dLowering pool2dLowering = new CpuPool2dLowering();
    private final CpuPool3dLowering pool3dLowering = new CpuPool3dLowering();
    private final CpuAttentionLowering attentionLowering = new CpuAttentionLowering();
    private final CpuLossLowering lossLowering = new CpuLossLowering();
    private final CpuPool1dCompositionLowering pool1dCompositionLowering =
            new CpuPool1dCompositionLowering();
    private final CpuConv1dCompositionLowering conv1dCompositionLowering =
            new CpuConv1dCompositionLowering();

    /** Creates a stateless lowering boundary with the current CPU capability and power analysis. */
    public CpuPartitionLowering() { this(Map.of()); }

    /**
     * Constructs a cold lowering owner retaining exact containing-body numeric layouts.
     * The same owner is used for decomposition candidates and final unit lowering, so a subset
     * cannot silently canonicalize a view produced outside that subset.
     *
     * @param scanBatchLayouts exact local physical mappings, empty on unchanged ordinary routes
     */
    public CpuPartitionLowering(Map<ValueId, CpuAccessPlan.BatchBinding> scanBatchLayouts) {
        this.scanBatchLayouts = Map.copyOf(scanBatchLayouts);
    }

    /**
     * Lowers with explicitly scoped containing-body numeric layout facts.
     * @param context exact projected unit
     * @param layouts complete containing-body physical mapping
     * @return independently qualified ordinary or Scan physical facts
     */
    public LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        return new CpuPartitionLowering(layouts).lower(context);
    }

    /**
     * Lowers one proved compact-input fold or static typed-one count, specialized-family seed, affine chain,
     * or one-through-eight-node ordinary pointwise DAG, rejecting unsupported unit shapes.
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
        requirePermittedSource(context);
        List<CompiledNode> nodes = context.partitionDag().nodes();
        boolean symbolic = context.requiresBatchSize();
        if (!context.partition().owner().equals(CpuCapabilityProvider.CPU_BACKEND_ID)) {
            throw new IllegalArgumentException("partition owner must be CPU");
        }
        if (scanBatchLayouts.isEmpty() && nodes.size() > 1 && nodes.stream().anyMatch(n -> n.operation().kind()
                == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL)) {
            try { return lowerRowRegion(context); }
            catch (IllegalArgumentException notARegion) { /* Preserve every existing fallback. */ }
        }
        if (symbolic) {
            if (nodes.size() == 1 && nodes.getFirst().scanBody().isPresent())
                return new CpuGeneralScanLowering().lower(context,
                        scanBatchLayouts.isEmpty() ? batchLayouts(context) : scanBatchLayouts);
            if (scanBatchLayouts.isEmpty())
                throw new IllegalArgumentException("symbolic unit is not a proved row region");
        }
        if (!symbolic) {
            var compactFold = foldLowering.kernelBroadcast(context);
            if (compactFold.isPresent()) return compactFold.orElseThrow();
            var constantCount = aggregateLowering.foldStaticTypedOne(context);
            if (constantCount.isPresent()) return constantCount.orElseThrow();
            var softmaxBackward = aggregateLowering.fuseSoftmaxBackward(context, this);
            if (softmaxBackward.isPresent()) return softmaxBackward.orElseThrow();
            var aggregateFusion = aggregateLowering.fuse(context, this);
            if (aggregateFusion.isPresent()) return aggregateFusion.orElseThrow();
        }
        if (nodes.isEmpty() || nodes.size() > 8) {
            throw new IllegalArgumentException("CPU pointwise partition requires one through eight nodes");
        }
        if (!symbolic && nodes.size() == 4 && nodes.get(2).operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D) {
            return conv1dCompositionLowering.lower(context);
        }
        if (!symbolic && nodes.size() == 3 && nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.EXPAND_DIMS
                && nodes.get(1).operation().kind()
                        instanceof io.github.pho001.synaptik.model.operation.pooling.Pool2dKind) {
            return pool1dCompositionLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv2dKind.CONV2D) {
            return symbolic ? conv2dLowering.lowerBatch(context, scanBatchLayouts) : conv2dLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.convolution.Conv3dKind.CONV3D) {
            return symbolic ? conv3dLowering.lowerBatch(context, scanBatchLayouts) : conv3dLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL
                || nodes.size() > 1 && nodes.getFirst().operation().kind()
                        == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE
                    && nodes.getLast().operation().kind()
                        == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
            return symbolic ? matmulLowering.lowerBatch(context, scanBatchLayouts) : matmulLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                instanceof io.github.pho001.synaptik.model.operation.pooling.Pool2dKind) {
            return symbolic ? pool2dLowering.lowerBatch(context, scanBatchLayouts) : pool2dLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                instanceof io.github.pho001.synaptik.model.operation.pooling.Pool3dKind) {
            return symbolic ? pool3dLowering.lowerBatch(context, scanBatchLayouts) : pool3dLowering.lower(context);
        }
        if (nodes.getFirst().operation().kind()
                == io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind
                        .SCALED_DOT_PRODUCT_ATTENTION) {
            return symbolic ? attentionLowering.lowerBatch(context, scanBatchLayouts) : attentionLowering.lower(context);
        }
        if (nodes.size() == 1) {
            Object kind = nodes.getFirst().operation().kind();
            if (kind == io.github.pho001.synaptik.model.operation.scan.ScanKind.SCAN) {
                return new CpuGeneralScanLowering().lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind) {
                return new CpuRecurrentScanLowering().lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.loss.LossKind) {
                return symbolic ? lossLowering.lowerBatch(context, scanBatchLayouts) : lossLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind) {
                return symbolic ? softmaxLowering.lowerBatch(context, scanBatchLayouts) : softmaxLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.normalization.LayerNormKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.normalization.RmsNormKind) {
                return symbolic ? trailingNormalizationLowering.lowerBatch(context, scanBatchLayouts) : trailingNormalizationLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.normalization.BatchNormKind
                    .BATCH_NORM_INFERENCE) {
                return symbolic ? batchNormInferenceLowering.lowerBatch(context, scanBatchLayouts) : batchNormInferenceLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.normalization.BatchNormKind
                    .BATCH_NORM_TRAINING) {
                return symbolic ? batchNormTrainingLowering.lowerBatch(context, scanBatchLayouts) : batchNormTrainingLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind) {
                if (kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.LOG_SUM_EXP
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.VARIANCE
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.STANDARD_DEVIATION
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.L1_NORM
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.L2_NORM) {
                return symbolic ? advancedReductionLowering.lowerBatch(context, scanBatchLayouts) : advancedReductionLowering.lower(context);
                }
                if (nodes.getFirst().operation().attrs()
                        instanceof io.github.pho001.synaptik.model.operation.reduction.MaskedReductionAttrs) {
                return symbolic ? maskedReductionLowering.lowerBatch(context, scanBatchLayouts) : maskedReductionLowering.lower(context);
                }
                if (kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.ARG_MIN
                        || kind == io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind.ARG_MAX) {
                return symbolic ? argExtremaLowering.lowerBatch(context, scanBatchLayouts) : argExtremaLowering.lower(context);
                }
                return symbolic ? aggregateLowering.lowerBatch(context, scanBatchLayouts) : aggregateLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind) {
                return symbolic ? scanLowering.lowerBatch(context, scanBatchLayouts) : scanLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.random.GraphRngKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.random.DropoutKind) {
                return randomLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.ordering.OrderingKind
                    || kind == io.github.pho001.synaptik.model.operation.ordering.TopKKind.TOP_K) {
                return symbolic ? orderingLowering.lowerBatch(context, scanBatchLayouts) : orderingLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD_AXIS
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD2D
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.FOLD3D) {
                return symbolic ? foldLowering.lowerBatch(context, scanBatchLayouts) : foldLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.index.AxisScatterKind
                    || kind == io.github.pho001.synaptik.model.operation.index.ScatterNdKind.SCATTER_ND) {
                return symbolic ? scatterLowering.lowerBatch(context, scanBatchLayouts) : scatterLowering.lower(context);
            }
            if (kind instanceof io.github.pho001.synaptik.model.operation.index.AxisGatherKind
                    || kind == io.github.pho001.synaptik.model.operation.index.GatherNdKind.GATHER_ND
                    || kind == io.github.pho001.synaptik.model.operation.index.OneHotKind.ONE_HOT) {
                return symbolic ? indexingLowering.lowerBatch(context, scanBatchLayouts) : indexingLowering.lower(context);
            }
            if (kind == io.github.pho001.synaptik.model.operation.layout.PadKind.PAD
                    || kind == io.github.pho001.synaptik.model.operation.layout.TileKind.TILE
                    || kind == io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.CONCAT
                    || kind == io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind.STACK
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.UNFOLD_AXIS
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.UNFOLD2D
                    || kind == io.github.pho001.synaptik.model.operation.layout.WindowTransformKind.UNFOLD3D
                    || kind == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE_UPDATE) {
                return symbolic ? movementLowering.lowerBatch(context, scanBatchLayouts) : movementLowering.lower(context);
            }
        }
        if (nodes.stream().allMatch(node -> {
            Object kind = node.operation().kind();
            return kind instanceof io.github.pho001.synaptik.model.operation.layout.ContiguousKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind
                    || kind instanceof io.github.pho001.synaptik.model.operation.index.SelectKind
                    || kind == io.github.pho001.synaptik.model.operation.layout.SliceKind.SLICE;
        })) return symbolic ? affineLowering.lowerBatch(context, scanBatchLayouts)
                : affineLowering.lower(context, !scanBatchLayouts.isEmpty());
        if (symbolic) return lowerPointwiseBatch(context, scanBatchLayouts);
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
            assertOccurrence(context, node, values);
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
            CpuPointwiseOpcode opcode = opcode(pending.node().operation().kind());
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

    /**
     * Builds the ordinary pointwise DAG with unresolved numeric geometry. The same opcodes,
     * immediate decoding and scalar-power owner used by resolved lowering supply arithmetic;
     * this method binds only physical membership and coefficient-proved addressing.
     *
     * @param context exact local ordinary unit with complete outside-use requirements
     * @param layouts complete containing-body numeric physical mappings
     * @return unresolved family-owned ordinary geometry with no sampled B
     * @throws IllegalArgumentException if the DAG or geometry is not ordinary pointwise
     */
    private LoweredPartition lowerPointwiseBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        requireBatchContext(context);
        var values = indexValues(context.values());
        var requirements = new HashMap<ValueId, LogicalMemoryRequirement>();
        context.memoryRequirements().forEach(requirement -> requirements.put(requirement.valueId(), requirement));
        var dag = context.partitionDag();
        var external = new LinkedHashMap<ValueId, Integer>();
        dag.externalInputs().forEach(occurrence -> external.putIfAbsent(occurrence.valueId(), -1));
        var materialized = new ArrayList<ValueId>();
        var virtual = new ArrayList<ValueId>();
        for (CompiledNode node : context.nodes()) {
            if (node.outputs().size() != 1) throw new IllegalArgumentException("pointwise canonical output arity");
            opcode(node.operation().kind());
            ValueId output = node.outputs().getFirst();
            var requirement = Objects.requireNonNull(requirements.get(output), "pointwise output requirement");
            if (dag.consumers(output).isEmpty() || requirement.graphOutput()) materialized.add(output);
            else virtual.add(output);
        }
        if (materialized.isEmpty()) throw new IllegalArgumentException("pointwise output is not materialized");
        var iterationShape = values.get(materialized.getFirst()).descriptor().shape();
        for (ValueId output : materialized)
            if (!values.get(output).descriptor().shape().equals(iterationShape))
                throw new IllegalArgumentException("pointwise outputs have incompatible iteration domains");
        var iteration = Objects.requireNonNull(layouts.get(materialized.getFirst()), "pointwise output layout");
        var irValues = new ArrayList<CpuKernelIr.Value>();
        var boundaries = new ArrayList<ValueId>();
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>();
        var types = new ArrayList<DataType>();
        var ordinals = new HashMap<ValueId, Integer>();
        for (ValueId input : external.keySet()) {
            var access = batchBroadcast(Objects.requireNonNull(layouts.get(input), "pointwise input layout"), iteration);
            int ordinal = irValues.size();
            ordinals.put(input, ordinal);
            DataType type = values.get(input).descriptor().dataType();
            irValues.add(new CpuKernelIr.Value(ordinal, type, CpuKernelIr.Value.Kind.INPUT, access.plan()));
            boundaries.add(input); bindings.add(access); types.add(type);
        }
        for (CompiledNode node : context.nodes()) {
            ValueId output = node.outputs().getFirst();
            boolean store = materialized.contains(output);
            var access = batchBroadcast(Objects.requireNonNull(layouts.get(output), "pointwise result layout"), iteration);
            if (store) {
                var read = access.plan();
                var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, read.regime(),
                        read.iterationRank(), read.axisRoles(), read.contiguousSuffix());
                access = new CpuAccessPlan.BatchBinding(write, access.extentConstants(),
                        access.extentBatchFactors(), access.baseElementOffset(),
                        access.strideConstants(), access.strideBatchFactors());
            }
            int ordinal = irValues.size();
            ordinals.put(output, ordinal);
            DataType type = values.get(output).descriptor().dataType();
            irValues.add(new CpuKernelIr.Value(ordinal, type,
                    store ? CpuKernelIr.Value.Kind.OUTPUT : CpuKernelIr.Value.Kind.VIRTUAL, access.plan()));
            if (store) { boundaries.add(output); bindings.add(access); types.add(type); }
        }
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        for (CompiledNode node : context.nodes()) {
            CpuPointwiseOpcode op = opcode(node.operation().kind());
            var immediate = scalarImmediate(node);
            instructions.add(new CpuKernelIr.Instruction(op,
                    node.inputs().stream().map(ordinals::get).toList(),
                    ordinals.get(node.outputs().getFirst()), immediate,
                    op == CpuPointwiseOpcode.SCALAR_POW ? scalarPowerAnalysis.analyze(immediate) : null,
                    clampImmediate(node)));
        }
        var stores = new ArrayList<CpuKernelIr.Store>();
        for (int port = 0; port < materialized.size(); port++)
            stores.add(new CpuKernelIr.Store(ordinals.get(materialized.get(port)), port));
        var ir = new CpuKernelIr(irValues, instructions, new CpuKernelIr.Loop("start", "end"), stores);
        requirePermittedLowPrecisionUse(ir);
        return new LoweredPartition(new BatchPointwiseGeometry(ir, boundaries, types, bindings,
                virtual, batchElements(iteration)));
    }

    /**
     * Numeric pointwise geometry retaining each real typed physical boundary.
     *
     * @param ir ordinary pointwise instructions
     * @param boundaries exact boundary declaration order
     * @param boundaryTypes represented boundary types
     * @param bindings exact B-only physical mappings
     * @param virtualValues unmaterialized ordinary values
     * @param iterationCount checked logical count recipe
     */
    private record BatchPointwiseGeometry(CpuKernelIr ir, List<ValueId> boundaries,
            List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
            List<ValueId> virtualValues,
            io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterationCount)
            implements BatchUnitGeometry {
        /**
         * Snapshots the ordinary numeric pointwise recipe without invocation state.
         * @param ir immutable ordinary code-shaping facts
         * @param boundaries exact boundary order
         * @param boundaryTypes each boundary's own type
         * @param bindings full-local affine physical mappings
         * @param virtualValues unmaterialized ordinary values
         * @param iterationCount affine logical count
         */
        private BatchPointwiseGeometry {
            boundaries = List.copyOf(boundaries);
            boundaryTypes = List.copyOf(boundaryTypes);
            bindings = List.copyOf(bindings);
            virtualValues = List.copyOf(virtualValues);
        }
        @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes() {
            return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(0, 0, 1);
        }
        @Override public List<CpuKernelSpecialization.PackReset> resets() { return List.of(); }
        @Override public String identity() { return "pointwise-batch-v1:" + bindings; }
        @Override public LoweredPartition bind(long batch) {
            var resolved = bindings.stream().map(binding -> binding.bind(batch)).toList();
            long count = iterationCount.byteSize(batch);
            long[] shape = resolved.getLast().extents().stream().mapToLong(Long::longValue).toArray();
            return new LoweredPartition(ir, boundaries, resolved,
                    resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                    boundaryTypes, virtualValues, shape, count,
                    "legal: numeric ordinary pointwise geometry", new long[0]);
        }
    }

    private void assertOccurrence(PrepareContext<? extends BackendAnalysisInputs> context,
            CompiledNode node, Map<ValueId, GraphValue> values) {
        var query = new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
                node.outputs().stream().map(id -> require(values, id).descriptor()).toList(), node.scanBody(),
                Optional.ofNullable(context.scanActivityPlans().get(node.id())));
        if (!capabilities.supports(query)) {
            throw new IllegalArgumentException("partition contains an unsupported CPU occurrence");
        }
        Object kind = node.operation().kind();
        if (List.of(BinaryArithmeticKind.ADD,BinaryArithmeticKind.SUB,BinaryArithmeticKind.MUL,BinaryArithmeticKind.DIV,
                ScalarElementwiseKind.ADD,ScalarElementwiseKind.SUB,ScalarElementwiseKind.MUL,ScalarElementwiseKind.DIV,
                UnaryElementwiseKind.NEG,UnaryElementwiseKind.TANH,UnaryElementwiseKind.SIGMOID).contains(kind)
                || kind == CastKind.CAST && query.inputs().getFirst().dataType() == query.outputs().getFirst().dataType()) {
            validatePointwise(node,query.inputs(),query.outputs().getFirst());
        }
    }

    /**
     * Constructs only an ordinary candidate; the full-partition selector must additionally
     * prove outside-path convexity before accepting it. The compact inventory contains ordered
     * external inputs followed by selected producing/view sites. The selector appends every
     * outside value when normalizing complete-partition uses, so no leave/reentry edge is lost.
     * @param context exact candidate projection retaining the full value inventory
     * @return complete candidate lowering with exact retained-store obligations
     * @throws IllegalArgumentException if this is not the bounded ordinary row pattern
     */
    private LoweredPartition lowerRowRegion(PrepareContext<? extends BackendAnalysisInputs> context) {
        var nodes = context.nodes();
        if (nodes.size() > 64) throw new IllegalArgumentException("row candidate attempt is bounded");
        var values = indexValues(context.values()); var ordinal = new LinkedHashMap<ValueId,Integer>();
        for (int v = 0; v < context.values().size(); v++) ordinal.put(context.values().get(v).id(),v);
        var producer = new LinkedHashMap<ValueId,CompiledNode>();
        for (var node : nodes) {
            if (node.outputs().size() != 1 || node.scanBody().isPresent()) throw new IllegalArgumentException("row node arity");
            producer.put(node.outputs().getFirst(),node);
        }
        var terminals = new ArrayList<Integer>(); var contractions = new ArrayList<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction>();
        Shape rowShape = null;
        for (var node : nodes) if (node.operation().kind() == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
            var leftProducer = producer.get(node.inputs().getFirst());
            if (leftProducer != null && leftProducer.operation().kind()
                    == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE
                    && leftProducer.operation().attrs() instanceof io.github.pho001.synaptik.model.operation.layout.PermutationAttrs permutation
                    && permutation.axes().equals(List.of(1,0))) {
                terminals.add(ordinal.get(node.outputs().getFirst()));
                Shape right = values.get(node.inputs().get(1)).descriptor().shape();
                if (right.rank() != 2 || rowShape != null && !rowShape.dimension(0).equals(right.dimension(0)))
                    throw new IllegalArgumentException("different row axes");
                rowShape = right;
            }
        }
        if (terminals.isEmpty() || rowShape == null) throw new IllegalArgumentException("no row contraction terminal");
        var rowDimension = rowShape.dimension(0);
        var mappings = batchLayouts(context);
        var roles = new ArrayList<>(java.util.Collections.nCopies(context.values().size(),-1));
        var origins = new ArrayList<Integer>(); var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        var rowSites = new ArrayList<Integer>(); var instructions = new ArrayList<CpuKernelIr.Instruction>();
        var produced = new java.util.HashSet<ValueId>(producer.keySet());
        var external = new java.util.LinkedHashSet<ValueId>();
        nodes.forEach(n -> n.inputs().stream().filter(id -> !produced.contains(id)).forEach(external::add));
        var retained = new ArrayList<Integer>(); var outputs = new ArrayList<ValueId>(); var virtual = new ArrayList<ValueId>();
        var memory = new LinkedHashMap<ValueId,LogicalMemoryRequirement>();
        context.memoryRequirements().forEach(m -> memory.put(m.valueId(),m));
        for (var value : context.values()) {
            int v = ordinal.get(value.id()); origins.add(v); accesses.add(mappings.get(value.id()));
            if (external.contains(value.id()) && value.descriptor().shape().rank() == 2
                    && value.descriptor().shape().dimension(0).equals(rowDimension)) roles.set(v,0);
        }
        for (var node : nodes) {
            ValueId id = node.outputs().getFirst(); int out = ordinal.get(id); var desc = values.get(id).descriptor();
            Object kind = node.operation().kind(); int first = ordinal.get(node.inputs().getFirst());
            boolean view = batchView(kind) && kind != io.github.pho001.synaptik.model.operation.layout.ContiguousKind.CONTIGUOUS;
            if (view) {
                origins.set(out,origins.get(first));
                roles.set(out,rowViewAxis(node,values.get(node.inputs().getFirst()).descriptor().shape(),
                        desc.shape(),roles.get(first),rowDimension));
                if (memory.get(id).graphOutput()) throw new IllegalArgumentException("external row view requires a separate copy");
                virtual.add(id); continue;
            }
            if (kind == io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
                CpuMatmulLowering.validateRankTwo(values.get(node.inputs().get(0)).descriptor().shape(),
                        values.get(node.inputs().get(1)).descriptor().shape(),desc.shape());
                contractions.add(new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction(out,
                        first,ordinal.get(node.inputs().get(1))));
                if (!terminals.contains(out)) {
                    if (roles.get(first) != 0) throw new IllegalArgumentException("unproved feature contraction row");
                    int right = ordinal.get(node.inputs().get(1));
                    roles.set(right,-1); roles.set(origins.get(right),-1); roles.set(out,0); rowSites.add(out);
                }
            } else {
                validatePointwise(node,node.inputs().stream().map(i -> values.get(i).descriptor()).toList(),desc);
                int row = -1;
                for (var input : node.inputs()) {
                    int source = ordinal.get(input), role = roles.get(source);
                    if (role < 0) continue;
                    int shifted = role+desc.shape().rank()-values.get(input).descriptor().shape().rank();
                    if (row >= 0 && row != shifted) throw new IllegalArgumentException("pointwise row provenance disagrees");
                    row = shifted;
                }
                if (row < 0 || row >= desc.shape().rank() || !desc.shape().dimension(row).equals(rowDimension))
                    throw new IllegalArgumentException("shared/non-row pointwise site");
                roles.set(out,row); rowSites.add(out);
                instructions.add(new CpuKernelIr.Instruction(opcode(node.operation().kind()),node.inputs().stream().map(ordinal::get).toList(),
                        out,scalarImmediate(node)));
            }
            if (terminals.contains(out) || memory.get(id).graphOutput()
                    || context.partitionDag().consumers(id).isEmpty()) { outputs.add(id); retained.add(out); }
            else virtual.add(id);
        }
        var irValues = new ArrayList<CpuKernelIr.Value>(); var stores = new ArrayList<CpuKernelIr.Store>();
        for (var value : context.values()) {
            int v = ordinal.get(value.id());
            var access = accesses.get(v);
            if (outputs.contains(value.id())) access = batchCanonical(value.descriptor(),CpuAccessPlan.AccessKind.WRITE);
            accesses.set(v,access);
            irValues.add(new CpuKernelIr.Value(v,value.descriptor().dataType(), external.contains(value.id())
                    ? CpuKernelIr.Value.Kind.INPUT : outputs.contains(value.id()) ? CpuKernelIr.Value.Kind.OUTPUT
                    : CpuKernelIr.Value.Kind.VIRTUAL,access.plan()));
        }
        for (int p = 0; p < outputs.size(); p++) stores.add(new CpuKernelIr.Store(ordinal.get(outputs.get(p)),p));
        var selectedIds = new ArrayList<>(external); nodes.forEach(n -> selectedIds.add(n.outputs().getFirst()));
        var remap = new LinkedHashMap<Integer,Integer>();
        for (var id : selectedIds) remap.put(ordinal.get(id),remap.size());
        var selectedValues = selectedIds.stream().map(id -> {
            var old = irValues.get(ordinal.get(id));
            return new CpuKernelIr.Value(remap.get(old.ordinal()),old.dataType(),old.kind(),old.accessPlan());
        }).toList();
        var selectedInstructions = instructions.stream().map(i -> new CpuKernelIr.Instruction(i.opcode(),
                i.inputs().stream().map(remap::get).toList(),remap.get(i.output()),i.scalarImmediate())).toList();
        var selectedStores = stores.stream().map(s -> new CpuKernelIr.Store(remap.get(s.value()),s.outputOrdinal())).toList();
        var arithmetic = new CpuKernelIr(selectedValues,selectedInstructions,new CpuKernelIr.Loop("start","end"),selectedStores);
        var selectedAccesses = selectedIds.stream().map(id -> accesses.get(ordinal.get(id))).toList();
        var selectedOrigins = selectedIds.stream().map(id -> remap.get(origins.get(ordinal.get(id)))).toList();
        var selectedRoles = selectedIds.stream().map(id -> roles.get(ordinal.get(id))).toList();
        var selectedContractions = contractions.stream().map(c -> new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction(
                remap.get(c.output()),remap.get(c.left()),remap.get(c.right()))).toList();
        var region = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr(arithmetic,rowSites.stream().map(remap::get).toList(),
                selectedContractions,List.of(),terminals.stream().map(remap::get).toList(),selectedAccesses,selectedOrigins,
                selectedRoles,retained.stream().map(remap::get).toList());
        var uses = new ArrayList<List<Integer>>();
        for (int v = 0; v < selectedValues.size(); v++) uses.add(new ArrayList<>());
        for (var node : nodes) for (var input : node.inputs()) uses.get(remap.get(ordinal.get(input))).add(remap.get(ordinal.get(node.outputs().getFirst())));
        // Views are transparent storage origins but remain real dependency edges.
        for (int v = 0; v < selectedOrigins.size(); v++) if (selectedOrigins.get(v) != v && !uses.get(selectedOrigins.get(v)).contains(v)) uses.get(selectedOrigins.get(v)).add(v);
        var emptyOrder = java.util.stream.IntStream.range(0,selectedValues.size()).mapToObj(v -> List.<Integer>of()).toList();
        if (CpuRowMapReductionLowering.prove(region,uses,emptyOrder).isEmpty()) throw new IllegalArgumentException("row-region local proof failed");
        var boundaries = new ArrayList<>(external); boundaries.addAll(outputs);
        var bindings = context.requiresBatchSize() ? List.<CpuAccessPlan.Binding>of()
                : boundaries.stream().map(id -> accesses.get(ordinal.get(id)).bind(0)).toList();
        var spans = context.requiresBatchSize() ? List.<Long>of() : bindings.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList();
        return new LoweredPartition(region,boundaries,bindings,spans,boundaries.stream().map(id -> values.get(id).descriptor().dataType()).toList(),
                virtual,new long[]{1},1,"legal: full-body row map with completed terminal contractions",new long[0]);
    }

    /**
     * Propagates an authenticated row role through explicit axis operations. Reshape is limited
     * to singleton insertions/removals; equal feature and row extents never justify swapping them.
     * @param node one existing admitted layout operation
     * @param source exact input Dimensions
     * @param target exact result Dimensions
     * @param row authenticated source row, or minus one for a shared input
     * @param rowDimension the contraction's exact ordered row Dimension
     * @return result row role, or minus one for a wholly shared view
     * @throws IllegalArgumentException if the transformation cannot preserve row provenance
     */
    private static int rowViewAxis(CompiledNode node,Shape source,Shape target,int row,
            io.github.pho001.synaptik.model.shape.Dimension rowDimension) {
        Object kind = node.operation().kind();
        if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.PERMUTE)
            return row < 0 ? -1 : ((io.github.pho001.synaptik.model.operation.layout.PermutationAttrs)node.operation().attrs()).axes().indexOf(row);
        if (kind instanceof io.github.pho001.synaptik.model.operation.layout.AxisTransformKind) {
            int axis = ((io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs)node.operation().attrs()).axis();
            if (row < 0) return -1;
            if (kind == io.github.pho001.synaptik.model.operation.layout.AxisTransformKind.EXPAND_DIMS) return row < axis ? row : row+1;
            if (axis == row) throw new IllegalArgumentException("cannot remove the row axis");
            return row < axis ? row : row-1;
        }
        if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.EXPAND) {
            int shift = target.rank()-source.rank();
            if (row >= 0) {
                if (!target.dimension(row+shift).equals(source.dimension(row))) throw new IllegalArgumentException("cannot expand the ordered row");
                return row+shift;
            }
            // A newly introduced/broadcast leading row is proved by the explicit EXPAND,
            // not by choosing among equal non-singleton extents.
            if (target.rank() >= 2 && target.dimension(0).equals(rowDimension)
                    && (shift > 0 || source.dimension(0).equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1)))) return 0;
            return -1;
        }
        if (kind == io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE) {
            var sourceAxes = java.util.stream.IntStream.range(0,source.rank()).filter(a -> !source.dimension(a).equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1))).boxed().toList();
            var targetAxes = java.util.stream.IntStream.range(0,target.rank()).filter(a -> !target.dimension(a).equals(new io.github.pho001.synaptik.model.shape.StaticDimension(1))).boxed().toList();
            if (!sourceAxes.stream().map(source::dimension).toList().equals(targetAxes.stream().map(target::dimension).toList()))
                throw new IllegalArgumentException("reshape merges or reorders row/feature provenance");
            if (row < 0) return -1;
            if (!sourceAxes.contains(row)) {
                if (source.equals(target)) return row;
                throw new IllegalArgumentException("singleton row reshape is ambiguous");
            }
            return targetAxes.get(sourceAxes.indexOf(row));
        }
        throw new IllegalArgumentException("unproved row view operation");
    }

    /**
     * Validates shared pointwise facts without admitting a backend route or binding layout.
     * This bounded inventory preserves exact Dimensions, including named batch coordinates;
     * ordinary callers keep their independent broader-family provider checks.
     * @param node non-null compiled tensor/scalar arithmetic, unary or raw-copy occurrence
     * @param inputs non-null ordered descriptors at semantic input positions
     * @param output non-null sole result descriptor
     * @return exact result Shape, never a sampled symbolic geometry
     * @throws NullPointerException if a required node, descriptor or input list is null
     * @throws IllegalArgumentException if arity, attributes, dtype or broadcast facts disagree
     */
    static Shape validatePointwise(CompiledNode node, List<TensorDescriptor> inputs, TensorDescriptor output) {
        Object kind = node.operation().kind(), attrs = node.operation().attrs();
        boolean binary = kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB
                || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV;
        boolean scalar = kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB
                || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV;
        boolean unary = kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH
                || kind == UnaryElementwiseKind.SIGMOID;
        boolean cast = kind == CastKind.CAST;
        boolean contiguous = kind == io.github.pho001.synaptik.model.operation.layout.ContiguousKind.CONTIGUOUS;
        if (!binary && !scalar && !unary && !cast && !contiguous || inputs.size() != (binary ? 2 : 1))
            throw new IllegalArgumentException("unsupported shared pointwise facts");
        DataType type = output.dataType();
        if (inputs.stream().anyMatch(input -> input.dataType() != type)
                || scalar && (!(attrs instanceof ScalarValueAttrs a) || a.value().dataType() != type)
                || cast && (!(attrs instanceof io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs a)
                    || a.targetDataType() != type)
                || !scalar && !cast && attrs != io.github.pho001.synaptik.model.operation.NoOperationAttrs.INSTANCE)
            throw new IllegalArgumentException("pointwise dtype or attributes disagree");
        Shape expected = binary ? io.github.pho001.synaptik.model.shape.ShapeBroadcast.broadcast(
                inputs.get(0).shape(),inputs.get(1).shape()) : inputs.getFirst().shape();
        if (!output.shape().equals(expected)) throw new IllegalArgumentException("pointwise result Shape disagrees");
        return expected;
    }

    /**
     * Maps one already-admitted pointwise occurrence to its CPU-private generated opcode.
     *
     * @param kind non-null operation kind belonging to the bounded pointwise family
     * @return the non-null exact generated opcode
     * @throws NullPointerException if {@code kind} is {@code null}
     * @throws IllegalArgumentException if the operation kind has no admitted CPU opcode
     */
    static CpuPointwiseOpcode opcode(OperationKind kind) {
        Objects.requireNonNull(kind, "kind");
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
            case BFLOAT16 -> Short.toUnsignedLong(value.bfloat16Bits());
            case FLOAT16 -> Short.toUnsignedLong(value.float16Bits());
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
            case BFLOAT16 -> Short.toUnsignedLong(value.bfloat16Bits());
            case FLOAT16 -> Short.toUnsignedLong(value.float16Bits());
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
     * Family-owned numeric invocation geometry for an ordinary Scan helper.
     * Binding resolves numeric coefficients, never a semantic graph or generated class.
     */
    public interface BatchUnitGeometry {
        /** Supplies the batch-independent generated representation, without a semantic graph.
         * @return non-null immutable portable ordinary code-shaping facts
         */
        CpuPortableKernelIr ir();
        /** Supplies the physical boundary identities in declaration order.
         * @return non-null ordered boundary list
         */
        List<ValueId> boundaries();
        /** Identifies ordinary intermediates that need no physical boundary or Runtime slot.
         * @return non-null ordered virtual values, empty when none are represented
         */
        default List<ValueId> virtualValues() { return List.of(); }
        /** Supplies represented storage types without substituting the helper's arithmetic type.
         * @return non-null types in physical boundary order
         */
        List<DataType> boundaryTypes();
        /** Supplies the boundary access coefficients to resolve against the invocation's actual B.
         * @return non-null ordered affine B-only access recipes
         */
        List<CpuAccessPlan.BatchBinding> bindings();
        /** Supplies the affine logical work capacity rather than a byte allocation.
         * @return non-null iteration-count recipe; numeric binding determines the exact invocation end
         */
        io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterationCount();
        /** Supplies the helper's scratch capacity as a constant-plus-B byte recipe.
         * @return non-null exact scratch-byte recipe
         */
        io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes();
        /** Identifies mutable packed cells whose original bound seeds must be restored.
         * @return non-null reset descriptors; original bound values are snapshotted at bind
         */
        List<CpuKernelSpecialization.PackReset> resets();
        /**
         * Binds immutable numeric geometry once per invocation.
         * @param batch actual non-negative canonical B
         * @return resolved facts retaining the exact portable code-shaping identity
         * @throws IllegalArgumentException if batch or bound geometry is invalid
         * @throws ArithmeticException if numeric geometry overflows
         */
        LoweredPartition bind(long batch);
        /** Exposes typed matrix code-shaping facts only for a matrix helper.
         * @return non-null optional typed matrix identity, empty for other ordinary families
         */
        default Optional<CpuMatmulIr> matrixIr() { return Optional.empty(); }
        /** Identifies the numeric coefficient recipe without specializing on actual B.
         * @return non-null coefficient-only recipe identity
         */
        String identity();
    }

    /**
     * Immutable lowering result consumed by route-neutral CPU analysis.
     *
     * @param portableKernelIr generated route-independent IR, or null only for a direct scan; pointwise, affine, movement, indexing,
     *     scatter, fold, ordering, random, or cumulative-scan
     *     representation
     * @param boundaryValues non-null deterministic unique external reads followed by every
     *     materialized output, including all canonical scan siblings; copied defensively
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
     * @param affineAddressPairs ordinary source/result pairs, controlled distinct-write
     *     source/result/original-ordinal triples, or empty for non-affine lowering
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
     * @param matmulFacts typed code identity and compact cold normalized full-K MATMUL geometry
     * @param pool2dGeometry compact cold NCHW Pool2d boundary and window geometry
     * @param pool3dGeometry compact cold NCDHW Pool3d boundary and window geometry
     * @param attentionGeometry compact cold scaled-dot-product-attention row geometry
     * @param recurrentScan non-null optional direct scan facts, present exactly when portableKernelIr is null
     * @param generalScan non-null optional generated scan facts, separate from direct recurrence
     * @param batchGeometry non-null optional explicit one-B numeric geometry, absent on resolved routes
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
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<MatmulFacts> matmulFacts,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
            Optional<CpuAttentionLowering.Geometry> attentionGeometry,
            Optional<CpuRecurrentScanLowering.Facts> recurrentScan,
            Optional<CpuGeneralScanLowering.Facts> generalScan,
            Optional<BatchUnitGeometry> batchGeometry) {
        /**
         * Constructs resolved ordinary facts without symbolic invocation geometry.
         * Static family constructors retain their exact numerical and identity meaning.
         * @param portableKernelIr resolved portableKernelIr facts retained under canonical validation
         * @param boundaryValues resolved boundaryValues facts retained under canonical validation
         * @param accessBindings resolved accessBindings facts retained under canonical validation
         * @param referencedElementSpans resolved referencedElementSpans facts retained under canonical validation
         * @param boundaryDataTypes resolved boundaryDataTypes facts retained under canonical validation
         * @param virtualValues resolved virtualValues facts retained under canonical validation
         * @param extents resolved extents facts retained under canonical validation
         * @param elementCount resolved elementCount facts retained under canonical validation
         * @param fusionReason resolved fusionReason facts retained under canonical validation
         * @param affineAddressPairs resolved affineAddressPairs facts retained under canonical validation
         * @param movementGeometry resolved movementGeometry facts retained under canonical validation
         * @param indexingGeometry resolved indexingGeometry facts retained under canonical validation
         * @param scatterGeometry resolved scatterGeometry facts retained under canonical validation
         * @param foldGeometry resolved foldGeometry facts retained under canonical validation
         * @param orderingGeometry resolved orderingGeometry facts retained under canonical validation
         * @param randomGeometry resolved randomGeometry facts retained under canonical validation
         * @param scanGeometry resolved scanGeometry facts retained under canonical validation
         * @param aggregateGeometry resolved aggregateGeometry facts retained under canonical validation
         * @param argExtremaGeometry resolved argExtremaGeometry facts retained under canonical validation
         * @param maskedReductionGeometry resolved maskedReductionGeometry facts retained under canonical validation
         * @param advancedReductionGeometry resolved advancedReductionGeometry facts retained under canonical validation
         * @param softmaxGeometry resolved softmaxGeometry facts retained under canonical validation
         * @param trailingNormalizationGeometry resolved trailingNormalizationGeometry facts retained under canonical validation
         * @param batchNormInferenceGeometry resolved batchNormInferenceGeometry facts retained under canonical validation
         * @param batchNormTrainingGeometry resolved batchNormTrainingGeometry facts retained under canonical validation
         * @param conv2dGeometry resolved conv2dGeometry facts retained under canonical validation
         * @param conv3dGeometry resolved conv3dGeometry facts retained under canonical validation
         * @param matmulFacts resolved matmulFacts facts retained under canonical validation
         * @param pool2dGeometry resolved pool2dGeometry facts retained under canonical validation
         * @param pool3dGeometry resolved pool3dGeometry facts retained under canonical validation
         * @param attentionGeometry resolved attentionGeometry facts retained under canonical validation
         * @param recurrentScan resolved recurrentScan facts retained under canonical validation
         * @param generalScan resolved generalScan facts retained under canonical validation
         */
        public LoweredPartition(
                CpuPortableKernelIr portableKernelIr,
                List<ValueId> boundaryValues,
                List<CpuAccessPlan.Binding> accessBindings,
                List<Long> referencedElementSpans,
                List<DataType> boundaryDataTypes,
                List<ValueId> virtualValues,
                long[] extents,
                long elementCount,
                String fusionReason,
                long[] affineAddressPairs,
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
                Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
                Optional<MatmulFacts> matmulFacts,
                Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
                Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
                Optional<CpuAttentionLowering.Geometry> attentionGeometry,
                Optional<CpuRecurrentScanLowering.Facts> recurrentScan,
                Optional<CpuGeneralScanLowering.Facts> generalScan) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans, boundaryDataTypes, virtualValues, extents, elementCount, fusionReason, affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulFacts, pool2dGeometry, pool3dGeometry, attentionGeometry, recurrentScan, generalScan, Optional.empty());
        }

        /**
         * Retains an explicitly unresolved ordinary numeric recipe without sampling B.
         * @param geometry non-null numeric family-owned geometry
         */
        public LoweredPartition(BatchUnitGeometry geometry) {
            this(geometry.ir(), geometry.boundaries(), List.of(), List.of(),
                    geometry.boundaryTypes(), geometry.virtualValues(), new long[0], 0,
                    "legal: general Scan symbolic ordinary unit", new long[0],
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
        }

        /**
         * Validates and snapshots the generated lowering without direct scan facts.
         * @param portableKernelIr non-null generated portable IR; this compatibility constructor
         *     does not accept direct scan facts; pointwise, affine, movement, indexing,
         *     scatter, fold, ordering, random, or cumulative-scan
         *     representation
         * @param boundaryValues non-null unique external reads followed by every materialized
         *     output in canonical port order; copied defensively
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
         * @param matmulFacts typed code identity and compact cold normalized full-K MATMUL geometry
         * @param pool2dGeometry compact cold NCHW Pool2d boundary and window geometry
         * @param pool3dGeometry compact cold NCDHW Pool3d boundary and window geometry
         * @param attentionGeometry compact cold scaled-dot-product-attention row geometry
         *
         * @throws NullPointerException if a required component is null
         * @throws IllegalArgumentException if family, boundary, strategy or resource facts disagree
         * @throws ArithmeticException if exact geometry arithmetic overflows
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
            Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
            Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
            Optional<MatmulFacts> matmulFacts,
            Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
            Optional<CpuPool3dLowering.Geometry> pool3dGeometry,
            Optional<CpuAttentionLowering.Geometry> attentionGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans, boundaryDataTypes, virtualValues, extents, elementCount, fusionReason, affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry, foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry, argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulFacts, pool2dGeometry, pool3dGeometry, attentionGeometry, Optional.empty(), Optional.empty());
        }

        /**
         * Creates direct scan lowering, never a dummy generated IR.
         * @param facts non-null exact scan facts with canonical reads and all sibling outputs
         */
        public LoweredPartition(CpuRecurrentScanLowering.Facts facts) {
            this(null, facts.boundaryValues(), facts.accessBindings(), facts.spans(),
                    facts.boundaryTypes(), List.of(), new long[] {facts.geometry().batch()},
                    facts.geometry().batch(), "direct fixed recurrent scan", new long[0],
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                            Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                            Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                            Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                            Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                            Optional.of(facts), Optional.empty());
        }

        /**
         * Retains one generated homogeneous FLOAT32/FLOAT64 scan with exact resources, never a
         * direct fixed recurrence. Every selected boundary dtype comes from the proved scan IR,
         * not the supplied carrier or a default FLOAT32 descriptor.
         * Static length mode transports its separate INT64 control dtype without inventing a
         * floating formal or changing the established unique-read/first-write boundary order.
         * Fixed FLOAT32/FLOAT64 carries resolved access spans; symbolic FLOAT32 instead retains
         * recipe bindings, empty resolved arrays and explicit run-bound mode. Neither form
         * discovers storage or changes topology at invocation.
         * @param facts non-null proved body/binder and fixed or symbolic resource facts
         * @param boundaries non-null unique reads then distinct demanded outer wrappers; copied
         * @throws NullPointerException if facts, boundaries or a boundary element is null
         * @throws IllegalArgumentException if boundary, access or resource cardinalities disagree
         */
        public LoweredPartition(CpuGeneralScanLowering.Facts facts, List<ValueId> boundaries) {
            this(facts.ir(), boundaries, facts.accessBindings(), facts.spans(),
                    facts.ir().boundaryTypes(), List.of(),
                    facts.ir().symbolic() ? new long[0] : new long[] {facts.geometry().elements()},
                    facts.ir().symbolic() ? 0 : facts.geometry().elements(),
                    facts.ir().dataType() == DataType.FLOAT32 ? "generated FLOAT32 pointwise general scan"
                            : "generated FLOAT64 general scan", new long[0],
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                    Optional.empty(), Optional.empty(), Optional.of(facts));
        }

        /**
         * Snapshots a generated lowering with no attention or direct-scan facts.
         * All optional family facts are non-null and empty outside their owning family.
         * @param portableKernelIr non-null route-independent generated representation
         * @param boundaryValues unique external reads then materialized outputs; defensively copied
         * @param accessBindings normalized access geometry in boundary order; defensively copied
         * @param referencedElementSpans exact layout spans in elements per boundary; defensively copied
         * @param boundaryDataTypes actual represented types in boundary order; defensively copied
         * @param virtualValues internal values without physical declarations or Runtime slots; copied
         * @param extents logical execution-domain extents; defensively copied
         * @param elementCount checked non-negative execution-domain element count
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs alternating source/result element addresses, empty otherwise; copied
         * @param movementGeometry optional compact data-movement geometry
         * @param indexingGeometry optional gather or one-hot geometry
         * @param scatterGeometry optional functional-scatter geometry
         * @param foldGeometry optional overlap-fold geometry
         * @param orderingGeometry optional stable SORT, ARGSORT or TOP_K geometry
         * @param randomGeometry optional explicit-state initialization or dropout geometry
         * @param scanGeometry optional cumulative-sum or cumulative-product geometry
         * @param aggregateGeometry optional ordinary extrema or Boolean aggregate geometry
         * @param argExtremaGeometry optional one-axis logical-index reduction geometry
         * @param maskedReductionGeometry optional directional masked-reduction geometry
         * @param advancedReductionGeometry optional logarithmic, statistical or norm reduction geometry
         * @param softmaxGeometry optional stable shape-preserving softmax geometry
         * @param trailingNormalizationGeometry optional Layer/RMS trailing-slice geometry
         * @param batchNormInferenceGeometry optional arbitrary-axis batch-normalization inference geometry
         * @param batchNormTrainingGeometry optional complete-channel training and exact-state geometry
         * @param conv2dGeometry optional grouped NCHW convolution geometry
         * @param conv3dGeometry optional grouped NCDHW convolution geometry
         * @param matmulFacts optional typed MATMUL code identity and separate normalized full-K geometry
         * @param pool2dGeometry optional NCHW pooling boundary and window geometry
         * @param pool3dGeometry optional NCDHW pooling boundary and window geometry
         * @throws NullPointerException if a required component or collection element is null
         * @throws IllegalArgumentException if boundary, family or resource facts disagree
         * @throws ArithmeticException if checked geometry arithmetic overflows
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
                Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
                Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
                Optional<MatmulFacts> matmulFacts,
                Optional<CpuPool2dLowering.Geometry> pool2dGeometry,
                Optional<CpuPool3dLowering.Geometry> pool3dGeometry) {
            this(portableKernelIr,boundaryValues,accessBindings,referencedElementSpans,
                    boundaryDataTypes,virtualValues,extents,elementCount,fusionReason,
                    affineAddressPairs,movementGeometry,indexingGeometry,scatterGeometry,
                    foldGeometry,orderingGeometry,randomGeometry,scanGeometry,aggregateGeometry,
                    argExtremaGeometry,maskedReductionGeometry,advancedReductionGeometry,
                    softmaxGeometry,trailingNormalizationGeometry,batchNormInferenceGeometry,
                    batchNormTrainingGeometry,conv2dGeometry,conv3dGeometry,matmulFacts,
                    pool2dGeometry,pool3dGeometry,Optional.empty());
        }

        /**
         * Preserves the schema-55 and earlier constructor surface with no Pool3d geometry.
         *
         * @param portableKernelIr non-null route-independent portable kernel IR
         * @param boundaryValues external reads followed by outputs; copied defensively
         * @param accessBindings normalized boundary bindings; copied defensively
         * @param referencedElementSpans declaration spans in boundary order; copied defensively
         * @param boundaryDataTypes represented boundary types; copied defensively
         * @param virtualValues same-unit values without physical boundaries; copied defensively
         * @param extents logical work-domain extents; copied defensively
         * @param elementCount checked product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs alternating source/result addresses; copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional random geometry
         * @param scanGeometry non-null optional scan geometry
         * @param aggregateGeometry non-null optional aggregate geometry
         * @param argExtremaGeometry non-null optional arg-extrema geometry
         * @param maskedReductionGeometry non-null optional masked-reduction geometry
         * @param advancedReductionGeometry non-null optional advanced-reduction geometry
         * @param softmaxGeometry non-null optional softmax geometry
         * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
         * @param batchNormInferenceGeometry non-null optional inference geometry
         * @param batchNormTrainingGeometry non-null optional training geometry
         * @param conv2dGeometry non-null optional Conv2d geometry
         * @param conv3dGeometry non-null optional Conv3d geometry
         * @param matmulFacts non-null optional MATMUL identity and geometry
         * @param pool2dGeometry non-null optional Pool2d geometry
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if boundary or family facts disagree
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
                Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
                Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
                Optional<MatmulFacts> matmulFacts,
                Optional<CpuPool2dLowering.Geometry> pool2dGeometry) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                    argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                    softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                    batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulFacts,
                    pool2dGeometry, Optional.empty());
        }

        /**
         * Preserves the established lowering constructor with no Pool2d geometry.
         *
         * @param portableKernelIr non-null route-independent portable kernel IR
         * @param boundaryValues external reads followed by outputs; copied defensively
         * @param accessBindings normalized boundary bindings; copied defensively
         * @param referencedElementSpans declaration spans in boundary order; copied defensively
         * @param boundaryDataTypes represented boundary types; copied defensively
         * @param virtualValues same-unit values without physical boundaries; copied defensively
         * @param extents logical work-domain extents; copied defensively
         * @param elementCount checked product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs alternating source/result addresses; copied defensively
         * @param movementGeometry non-null optional movement geometry
         * @param indexingGeometry non-null optional indexing geometry
         * @param scatterGeometry non-null optional scatter geometry
         * @param foldGeometry non-null optional fold geometry
         * @param orderingGeometry non-null optional ordering geometry
         * @param randomGeometry non-null optional random geometry
         * @param scanGeometry non-null optional scan geometry
         * @param aggregateGeometry non-null optional aggregate geometry
         * @param argExtremaGeometry non-null optional arg-extrema geometry
         * @param maskedReductionGeometry non-null optional masked-reduction geometry
         * @param advancedReductionGeometry non-null optional advanced-reduction geometry
         * @param softmaxGeometry non-null optional softmax geometry
         * @param trailingNormalizationGeometry non-null optional trailing-normalization geometry
         * @param batchNormInferenceGeometry non-null optional inference geometry
         * @param batchNormTrainingGeometry non-null optional training geometry
         * @param conv2dGeometry non-null optional Conv2d geometry
         * @param conv3dGeometry non-null optional Conv3d geometry
         * @param matmulFacts non-null optional MATMUL identity and geometry
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if boundary or family facts disagree
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
                Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
                Optional<CpuConv3dLowering.Geometry> conv3dGeometry,
                Optional<MatmulFacts> matmulFacts) {
            this(portableKernelIr, boundaryValues, accessBindings, referencedElementSpans,
                    boundaryDataTypes, virtualValues, extents, elementCount, fusionReason,
                    affineAddressPairs, movementGeometry, indexingGeometry, scatterGeometry,
                    foldGeometry, orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry,
                    argExtremaGeometry, maskedReductionGeometry, advancedReductionGeometry,
                    softmaxGeometry, trailingNormalizationGeometry, batchNormInferenceGeometry,
                    batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry, matmulFacts,
                    Optional.empty());
        }

        /** Typed MATMUL code identity paired with its separate cold geometry.
         * @param ir exact non-null code-shaping MATMUL identity
         * @param geometry exact non-null normalized full-K geometry
         */
        public record MatmulFacts(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr ir,
                CpuMatmulLowering.Geometry geometry) {
            /**
             * Retains the typed matrix identity separately from its cold invocation geometry.
             * Neither product transfers a semantic graph or invocation storage into this pair.
             * @param ir non-null immutable MATMUL code-shaping facts
             * @param geometry non-null normalized full-K numeric invocation geometry
             * @throws NullPointerException if either analysis product is null
             */
            public MatmulFacts {Objects.requireNonNull(ir,"ir");Objects.requireNonNull(geometry,"geometry");}
        }

        /**
         * Creates an established-family lowering with no MATMUL geometry.
         *
         * @param portableKernelIr non-null route-independent portable kernel IR
         * @param boundaryValues external reads followed by materialized outputs; copied
         * @param accessBindings normalized cold boundary bindings; copied
         * @param referencedElementSpans exact declaration spans in boundary order; copied
         * @param boundaryDataTypes exact represented boundary types; copied
         * @param virtualValues same-unit values that require no physical boundary; copied
         * @param extents final logical work-domain extents; copied defensively
         * @param elementCount checked product of {@code extents}
         * @param fusionReason non-null cold diagnostic explanation
         * @param affineAddressPairs alternating affine source/result addresses; copied
         * @param movementGeometry optional static movement geometry
         * @param indexingGeometry optional indexing geometry
         * @param scatterGeometry optional functional-scatter geometry
         * @param foldGeometry optional overlap-fold geometry
         * @param orderingGeometry optional stable-ordering geometry
         * @param randomGeometry optional explicit-state geometry
         * @param scanGeometry optional cumulative-scan geometry
         * @param aggregateGeometry optional ordinary-aggregate geometry
         * @param argExtremaGeometry optional arg-extrema geometry
         * @param maskedReductionGeometry optional masked-reduction geometry
         * @param advancedReductionGeometry optional logarithmic/statistical/norm geometry
         * @param softmaxGeometry optional stable-softmax geometry
         * @param trailingNormalizationGeometry optional Layer/RMS geometry
         * @param batchNormInferenceGeometry optional batch-normalization inference geometry
         * @param batchNormTrainingGeometry optional batch-normalization training geometry
         * @param conv2dGeometry optional grouped NCHW Conv2d geometry
         * @param conv3dGeometry optional grouped NCDHW Conv3d geometry
         * @throws NullPointerException if a required reference or list element is {@code null}
         * @throws IllegalArgumentException if boundary, range, output, or family facts disagree
         * @throws ArithmeticException if checked extent or address arithmetic overflows
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
                Optional<CpuConv2dLowering.Geometry> conv2dGeometry,
                Optional<CpuConv3dLowering.Geometry> conv3dGeometry) {
            this(portableKernelIr,boundaryValues,accessBindings,referencedElementSpans,
                    boundaryDataTypes,virtualValues,extents,elementCount,fusionReason,
                    affineAddressPairs,movementGeometry,indexingGeometry,scatterGeometry,
                    foldGeometry,orderingGeometry,randomGeometry,scanGeometry,aggregateGeometry,
                    argExtremaGeometry,maskedReductionGeometry,advancedReductionGeometry,
                    softmaxGeometry,trailingNormalizationGeometry,batchNormInferenceGeometry,
                    batchNormTrainingGeometry,conv2dGeometry,conv3dGeometry,Optional.empty());
        }

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
         * @param portableKernelIr generated portable representation, or null exactly when direct
         *     recurrentScan facts are present; a direct scan has no generated IR
         * @param boundaryValues unique reads then all materialized outputs; copied defensively
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
         * @param batchNormTrainingGeometry non-null optional training-normalization geometry
         * @param conv2dGeometry non-null optional grouped NCHW convolution geometry
         * @param conv3dGeometry non-null optional grouped NCDHW convolution geometry
         * @param matmulFacts non-null optional normalized matrix-product facts
         * @param pool2dGeometry non-null optional NCHW pooling geometry
         * @param pool3dGeometry non-null optional NCDHW pooling geometry
         * @param attentionGeometry non-null optional attention row geometry
         * @param recurrentScan non-null optional direct facts, mutually exclusive with generated IR
         * @param generalScan non-null optional generated body/binder and exact geometry
         * @param batchGeometry non-null optional unresolved numeric family geometry
         * @throws NullPointerException if a required component or element is {@code null}
         * @throws IllegalArgumentException if fewer than one input plus one output are present or
         *     boundary collections have different cardinalities
         */
        public LoweredPartition {
            batchGeometry = Objects.requireNonNull(batchGeometry, "batchGeometry");
            Objects.requireNonNull(generalScan, "generalScan");
            if ((portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr)
                    != generalScan.isPresent()) throw new IllegalArgumentException("general scan facts disagree");
            Objects.requireNonNull(recurrentScan, "recurrentScan");
            if ((portableKernelIr == null) != recurrentScan.isPresent())
                throw new IllegalArgumentException("exactly one generated or recurrent lowering required");
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
            conv3dGeometry = Objects.requireNonNull(conv3dGeometry, "conv3dGeometry");
            matmulFacts = Objects.requireNonNull(matmulFacts, "matmulFacts");
            pool2dGeometry = Objects.requireNonNull(pool2dGeometry, "pool2dGeometry");
            pool3dGeometry = Objects.requireNonNull(pool3dGeometry, "pool3dGeometry");
            attentionGeometry = Objects.requireNonNull(attentionGeometry, "attentionGeometry");
            Objects.requireNonNull(fusionReason, "fusionReason");
            int size = boundaryValues.size();
            if (recurrentScan.isPresent()) {
                var scan = recurrentScan.orElseThrow();
                if (!boundaryValues.equals(scan.boundaryValues()) || !accessBindings.equals(scan.accessBindings())
                        || !referencedElementSpans.equals(scan.spans()) || !boundaryDataTypes.equals(scan.boundaryTypes())
                        || !virtualValues.isEmpty() || affineAddressPairs.length != 0
                        || !java.util.Arrays.equals(extents, new long[] {scan.geometry().batch()})
                        || elementCount != scan.geometry().batch()
                        || java.util.stream.Stream.of(movementGeometry, indexingGeometry, scatterGeometry, foldGeometry,
                            orderingGeometry, randomGeometry, scanGeometry, aggregateGeometry, argExtremaGeometry,
                            maskedReductionGeometry, advancedReductionGeometry, softmaxGeometry, trailingNormalizationGeometry,
                            batchNormInferenceGeometry, batchNormTrainingGeometry, conv2dGeometry, conv3dGeometry,
                            matmulFacts, pool2dGeometry, pool3dGeometry, attentionGeometry).anyMatch(Optional::isPresent))
                    throw new IllegalArgumentException("direct recurrent lowering facts disagree");
            }
            boolean pool2d = portableKernelIr
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr;
            boolean pool3d = portableKernelIr
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr;
            boolean attention = portableKernelIr
                    instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
            boolean symbolicScan = generalScan.map(f -> f.ir().symbolic()).orElse(false);
            boolean symbolicRegion = portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr r
                    && r.accesses().stream().anyMatch(a -> a.extentBatchFactors().stream().anyMatch(f -> f != 0));
            boolean symbolicOrdinary = batchGeometry.isPresent();
            if (symbolicOrdinary) {
                var recipe = batchGeometry.orElseThrow();
                if (!portableKernelIr.structuralKey().equals(recipe.ir().structuralKey())
                        || !boundaryValues.equals(recipe.boundaries())
                        || !boundaryDataTypes.equals(recipe.boundaryTypes())
                        || recipe.bindings().size() != size || !accessBindings.isEmpty()
                        || !referencedElementSpans.isEmpty() || extents.length != 0
                        || recurrentScan.isPresent() || generalScan.isPresent()) {
                    throw new IllegalArgumentException("symbolic ordinary recipe disagrees");
                }
            }
            if (!symbolicOrdinary && (size < 1 || !symbolicScan && !symbolicRegion && (accessBindings.size() != size || referencedElementSpans.size() != size)
                    || symbolicScan && (!accessBindings.isEmpty() || !referencedElementSpans.isEmpty()
                        || generalScan.orElseThrow().batchBindings().size() != size)
                    || boundaryDataTypes.size() != size
                    || pool2d != pool2dGeometry.isPresent()
                    || pool3d != pool3dGeometry.isPresent()
                    || attention != attentionGeometry.isPresent()
                    || pool2d && (matmulFacts.isPresent() || size != 2
                        || boundaryDataTypes.getFirst() != boundaryDataTypes.getLast()
                        || pool2dGeometry.orElseThrow().dataType() != boundaryDataTypes.getFirst()
                        || pool2dGeometry.orElseThrow().outputCount() != elementCount)
                    || pool3d && (matmulFacts.isPresent() || pool2d || attention || size != 2
                        || boundaryDataTypes.getFirst() != boundaryDataTypes.getLast()
                        || pool3dGeometry.orElseThrow().dataType() != boundaryDataTypes.getFirst()
                        || pool3dGeometry.orElseThrow().outputCount() != elementCount)
                    || attention && (matmulFacts.isPresent() || pool2d || pool3d
                        || attentionGeometry.orElseThrow().rowCount() != elementCount
                        || attentionGeometry.orElseThrow().uniqueInputCount()
                            + attentionGeometry.orElseThrow().outputCount() != size))) {
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
         * Returns the typed MATMUL code identity when this is a MATMUL lowering.
         *
         * @return the retained immutable identity, or empty for every other family
         */
        public Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr> matmulIr(){
            return matmulFacts.map(MatmulFacts::ir).or(() -> batchGeometry.flatMap(BatchUnitGeometry::matrixIr));
        }
        /**
         * Returns the separate cold MATMUL invocation geometry when present.
         *
         * @return the retained immutable geometry, or empty for every other family
         */
        public Optional<CpuMatmulLowering.Geometry> matmulGeometry(){
            return matmulFacts.map(MatmulFacts::geometry);
        }
        /**
         * Returns the cache-compatible generated form.
         *
         * @throws IllegalStateException if this is a direct scan with no generated form
         * @return the retained pointwise IR or a newly encoded instruction-free family/row-region
         *     or output-only constant-write IR;
         *     never {@code null}
         */
        public CpuKernelIr kernelIr() {
            if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSoftmaxBackwardIr backward)
                return backward.encodedKernelIr();
            if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateFusionIr fusion)
                return fusion.encodedKernelIr();
            if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConstantWriteIr constant)
                return constant.encodedKernelIr();
            if (portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr region)
                return region.encodedKernelIr();
            if (recurrentScan.isPresent()) throw new IllegalStateException("direct scan has no generated IR");
            if (generalScan.isPresent()) return generalScan.orElseThrow().ir().encodedKernelIr();
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
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool2dIr pool2d
                                                        ? pool2d.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPool3dIr pool3d
                                                        ? pool3d.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr attention
                                                        ? attention.encodedKernelIr()
                                                    : portableKernelIr instanceof io.github.pho001.synaptik.backend.cpu.internal.ir.CpuLossIr loss
                                                        ? loss.encodedKernelIr()
                                                    : ((io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAggregateIr)
                                                        portableKernelIr).encodedKernelIr();
        }
    }
}
