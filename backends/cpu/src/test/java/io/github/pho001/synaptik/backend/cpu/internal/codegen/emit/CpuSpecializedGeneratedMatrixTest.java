package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.RmsNormKind;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/** Contract-derived generated coverage for CPU first-class specialized families. */
class CpuSpecializedGeneratedMatrixTest {
    private static final List<DataType> FLOATING = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16);
    private static final List<DataType> ARG_INPUTS = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.INT64, DataType.INT32);
    private static final CpuPartitionAnalysisInputs.MaterializationPolicy MATERIALIZATION =
            new CpuPartitionAnalysisInputs.MaterializationPolicy(true, 0, 1, 20, 1, 3, 1_000_000, 1, 1);

    @Test void everyAdmittedArgMaskedAdvancedAndSoftmaxFormGeneratesAcrossRequests() {
        var cases = new ArrayList<Case>();
        for (var kind : List.of(AggregateReductionKind.ARG_MIN, AggregateReductionKind.ARG_MAX)) for (var type : ARG_INPUTS)
            for (var tie : ArgExtremaTiePolicy.values()) for (boolean keep : List.of(false, true)) for (int axis : List.of(0, 1))
                cases.add(row("arg/" + kind + '/' + type + '/' + tie + '/' + keep + '/' + axis,
                        CpuArgExtremaLoweringTest.context(kind, type, Shape.of(2, 3), axis, keep, tie)));
        for (var kind : List.of(AggregateReductionKind.SUM, AggregateReductionKind.MEAN)) for (var type : FLOATING)
            for (var mask : List.of(Shape.scalar(), Shape.of(3), Shape.of(1, 3))) for (int axis : List.of(0, 1))
                cases.add(row("masked/" + kind + '/' + type + '/' + mask + '/' + axis,
                        CpuMaskedReductionLoweringTest.context(kind, type, Shape.of(2, 3), mask, axis)));
        for (var kind : List.of(AggregateReductionKind.LOG_SUM_EXP, AggregateReductionKind.L1_NORM, AggregateReductionKind.L2_NORM,
                AggregateReductionKind.VARIANCE, AggregateReductionKind.STANDARD_DEVIATION)) for (var type : FLOATING)
            for (var axes : List.of(List.of(0), List.of(1), List.of(0, 1))) for (boolean keep : List.of(false, true)) {
                var attrs = kind == AggregateReductionKind.VARIANCE || kind == AggregateReductionKind.STANDARD_DEVIATION
                        ? new StatisticalReductionAttrs(axes, keep, axes.size() == 2 ? 1 : 0) : new MultiAxisReductionAttrs(axes, keep);
                cases.add(row("advanced/" + kind + '/' + type + '/' + axes + '/' + keep,
                        CpuAggregateLoweringTest.context(kind, type, Shape.of(2, 3), attrs, reductionShape(Shape.of(2, 3), axes, keep))));
            }
        for (var kind : SoftmaxKind.values()) for (var type : FLOATING) for (int axis : List.of(0, 1))
            cases.add(row("softmax/" + kind + '/' + type + '/' + axis, CpuSoftmaxLoweringTest.context(kind, type, Shape.of(2, 3), axis)));
        assertGenerated(cases);
    }

    @Test void everyAdmittedTrailingAndBatchNormRoleCombinationGeneratesAcrossRequests() {
        var cases = new ArrayList<Case>();
        for (var input : FLOATING) {
            cases.add(row("layer/plain/" + input, trailingContext(true, List.of(input))));
            cases.add(row("rms/plain/" + input, trailingContext(false, List.of(input))));
        }
        for (var input : FLOATING) for (var scale : FLOATING) {
            cases.add(row("rms/scaled/" + input + '/' + scale, trailingContext(false, List.of(input, scale))));
            for (var bias : FLOATING) cases.add(row("layer/affine/" + input + '/' + scale + '/' + bias,
                    trailingContext(true, List.of(input, scale, bias))));
        }
        for (var types : floatingRoleCombinations()) for (int axis : List.of(0, 1)) {
            cases.add(row("batch-inference/" + types + '/' + axis, CpuBatchNormInferenceLoweringTest.context(types, Shape.of(2, 3, 4), axis, List.of(0, 1, 2, 3, 4))));
            cases.add(row("batch-training/" + types + '/' + axis, CpuBatchNormTrainingLoweringTest.context(types, Shape.of(2, 3, 4), axis,
                    List.of(0, 1, 2, 3, 4), trainingLayouts(Shape.of(2, 3, 4), axis))));
        }
        assertGenerated(cases);
    }

    @Test void everyCurrentDirectMatmulFormGeneratesAcrossRequests() {
        var cases = new ArrayList<Case>();
        var forms = List.of(new Shape[] {Shape.of(3), Shape.of(3)},
                new Shape[] {Shape.of(2, 3), Shape.of(3)},
                new Shape[] {Shape.of(3), Shape.of(3, 4)},
                new Shape[] {Shape.of(2, 3), Shape.of(3, 4)},
                new Shape[] {Shape.of(2, 2, 3), Shape.of(2, 3, 4)},
                new Shape[] {Shape.of(1, 2, 3), Shape.of(2, 3, 4)});
        for (var left : numeric()) for (var right : numeric()) if (left.category() == right.category()) for (var form : forms) {
            Shape output = matmulOutput(form[0], form[1]);
            cases.add(row("matmul/" + left + '/' + right + '/' + form[0] + '/' + form[1],
                    matmulContext(left, right, form[0], form[1], output)));
        }
        assertGenerated(cases);
    }

    @Test void everyCurrentDirectConvolutionAndPoolingFormGeneratesAcrossRequests() {
        var cases = new ArrayList<Case>();
        var conv2d = List.of(new Conv2dCase("ordinary", new Conv2dAttrs(1, 2, 1, 0, 2, 1, 1), Shape.of(1, 2, 7, 7), Shape.of(4, 2, 2, 2), Shape.of(1, 4, 7, 3)),
                new Conv2dCase("grouped", new Conv2dAttrs(1, 2, 1, 0, 2, 1, 2), Shape.of(1, 2, 7, 7), Shape.of(4, 1, 2, 2), Shape.of(1, 4, 7, 3)),
                new Conv2dCase("depthwise", new Conv2dAttrs(1, 2, 1, 0, 2, 1, 2), Shape.of(1, 2, 7, 7), Shape.of(2, 1, 2, 2), Shape.of(1, 2, 7, 3)));
        var conv3d = List.of(new Conv3dCase("ordinary", new Conv3dAttrs(1, 2, 1, 1, 0, 1, 2, 1, 1, 1), Shape.of(1, 2, 7, 7, 7), Shape.of(4, 2, 2, 2, 2), Shape.of(1, 4, 7, 3, 8)),
                new Conv3dCase("grouped", new Conv3dAttrs(1, 2, 1, 1, 0, 1, 2, 1, 1, 2), Shape.of(1, 2, 7, 7, 7), Shape.of(4, 1, 2, 2, 2), Shape.of(1, 4, 7, 3, 8)),
                new Conv3dCase("depthwise", new Conv3dAttrs(1, 2, 1, 1, 0, 1, 2, 1, 1, 2), Shape.of(1, 2, 7, 7, 7), Shape.of(2, 1, 2, 2, 2), Shape.of(1, 2, 7, 3, 8)));
        for (var geometry : conv2d) for (var input : FLOATING) for (var weight : FLOATING) {
            cases.add(row("conv2d/" + geometry.id + "/no-bias/" + input + '/' + weight,
                    CpuConv2dLoweringTest.context(List.of(input, weight), geometry.input, geometry.weight, geometry.output, geometry.attrs, null)));
            for (var bias : FLOATING) cases.add(row("conv2d/" + geometry.id + "/bias/" + input + '/' + weight + '/' + bias,
                    CpuConv2dLoweringTest.context(List.of(input, weight, bias), geometry.input, geometry.weight, geometry.output, geometry.attrs, null)));
        }
        for (var geometry : conv3d) for (var input : FLOATING) for (var weight : FLOATING) {
            cases.add(row("conv3d/" + geometry.id + "/no-bias/" + input + '/' + weight,
                    CpuConv3dLoweringTest.context(List.of(input, weight), geometry.input, geometry.weight, geometry.output, geometry.attrs, null)));
            for (var bias : FLOATING) cases.add(row("conv3d/" + geometry.id + "/bias/" + input + '/' + weight + '/' + bias,
                    CpuConv3dLoweringTest.context(List.of(input, weight, bias), geometry.input, geometry.weight, geometry.output, geometry.attrs, null)));
        }
        for (var type : FLOATING) for (boolean ceil : List.of(false, true)) {
            cases.add(row("max-pool2d/" + type + '/' + ceil, CpuPool2dLoweringTest.context(Pool2dKind.MAX_POOL2D,
                    new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 2, 1, ceil), type, Shape.of(1, 2, 7, 8), Shape.of(1, 2, 4, ceil ? 5 : 4))));
            cases.add(row("average-pool2d/" + type + '/' + ceil, CpuPool2dLoweringTest.context(Pool2dKind.AVERAGE_POOL2D,
                    new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 2, 1, ceil), type, Shape.of(1, 2, 7, 8), Shape.of(1, 2, 4, ceil ? 5 : 4))));
            cases.add(row("max-pool3d/" + type + '/' + ceil, CpuPool3dLoweringTest.context(Pool3dKind.MAX_POOL3D,
                    new MaxPool3dAttrs(2, 2, 3, 2, 2, 2, 1, 1, 1, 2, 1, 1, ceil), type, Shape.of(1, 2, 7, 7, 8), Shape.of(1, 2, 4, ceil ? 5 : 4, ceil ? 5 : 4))));
            cases.add(row("average-pool3d/" + type + '/' + ceil, CpuPool3dLoweringTest.context(Pool3dKind.AVERAGE_POOL3D,
                    new AveragePool3dAttrs(2, 2, 3, 2, 2, 2, 1, 1, 1, 2, 1, 1, ceil), type, Shape.of(1, 2, 7, 7, 8), Shape.of(1, 2, 4, ceil ? 5 : 4, ceil ? 5 : 4))));
        }
        assertGenerated(cases);
    }

    /** Reconstructs one exact direct Pool2d or Pool3d semantic fixture from its inventory facts. */
    static PrepareContext<CpuPartitionAnalysisInputs> semanticPoolContext(
            String form, DataType type, boolean ceilMode) {
        return switch (form) {
            case "MAX_POOL2D" -> CpuPool2dLoweringTest.context(Pool2dKind.MAX_POOL2D,
                    new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 2, 1, ceilMode), type,
                    Shape.of(1, 2, 7, 8), Shape.of(1, 2, 4, ceilMode ? 5 : 4));
            case "AVERAGE_POOL2D" -> CpuPool2dLoweringTest.context(Pool2dKind.AVERAGE_POOL2D,
                    new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 2, 1, ceilMode), type,
                    Shape.of(1, 2, 7, 8), Shape.of(1, 2, 4, ceilMode ? 5 : 4));
            case "MAX_POOL3D" -> CpuPool3dLoweringTest.context(Pool3dKind.MAX_POOL3D,
                    new MaxPool3dAttrs(2, 2, 3, 2, 2, 2, 1, 1, 1, 2, 1, 1, ceilMode), type,
                    Shape.of(1, 2, 7, 7, 8), Shape.of(1, 2, 4, ceilMode ? 5 : 4,
                            ceilMode ? 5 : 4));
            case "AVERAGE_POOL3D" -> CpuPool3dLoweringTest.context(Pool3dKind.AVERAGE_POOL3D,
                    new AveragePool3dAttrs(2, 2, 3, 2, 2, 2, 1, 1, 1, 2, 1, 1, ceilMode), type,
                    Shape.of(1, 2, 7, 7, 8), Shape.of(1, 2, 4, ceilMode ? 5 : 4,
                            ceilMode ? 5 : 4));
            default -> throw new IllegalArgumentException("not a direct pool form: " + form);
        };
    }

    /** Applies one exact inventory carrier, layout, materialization, and strategy request. */
    static PrepareContext<CpuPartitionAnalysisInputs> semanticConfigure(
            PrepareContext<CpuPartitionAnalysisInputs> context, String request) {
        return configure(context, Request.valueOf(request));
    }

    @Test void everyCurrentDirectAttentionAndLossRoleFormGeneratesAcrossRequests() {
        var cases = new ArrayList<Case>();
        for (var query : FLOATING) for (var key : FLOATING) for (var value : FLOATING)
            for (boolean mask : List.of(false, true)) for (boolean causal : List.of(false, true))
                for (boolean weights : List.of(false, true)) for (boolean explicitScale : List.of(false, true))
                    for (List<Integer> roles : attentionRoles(mask, query, key, value)) {
                        String id = "attention/" + query + '/' + key + '/' + value + "/mask=" + mask
                                + "/causal=" + causal + "/weights=" + weights + "/scale="
                                + explicitScale + "/roles=" + roles;
                        cases.add(row(id, attentionContext(query, key, value, mask, causal, weights,
                                explicitScale, roles)));
                    }
        for (var left : FLOATING) for (var right : FLOATING) for (var reduction : LossReduction.values()) {
            cases.add(row("loss/mse/" + left + '/' + right + '/' + reduction + "/ordered",
                    lossContext(LossKind.MEAN_SQUARED_ERROR, left, right, reduction, false, List.of(0, 1))));
            cases.add(row("loss/dense/" + left + '/' + right + '/' + reduction + "/ordered",
                    lossContext(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, left, right, reduction, false, List.of(0, 1))));
            if (left == right) {
                cases.add(row("loss/mse/" + left + '/' + reduction + "/aliased",
                        lossContext(LossKind.MEAN_SQUARED_ERROR, left, right, reduction, false, List.of(0, 0))));
                cases.add(row("loss/dense/" + left + '/' + reduction + "/aliased",
                        lossContext(LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, left, right, reduction, false, List.of(0, 0))));
            }
        }
        for (var logits : FLOATING) for (var index : List.of(DataType.INT64, DataType.INT32))
            for (var reduction : LossReduction.values()) for (boolean ignored : List.of(false, true))
                cases.add(row("loss/index/" + logits + '/' + index + '/' + reduction + "/ignored=" + ignored,
                        lossContext(LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, logits, index, reduction, ignored, List.of(0, 1))));
        assertGenerated(cases);
    }

    @Test void modelValidButCpuInapplicableSpecializedRowsAreExplicitlyChecked() {
        var rejected = rejectedProviderFixtures();
        assertEquals(3, rejected.size());
        for (var value : rejected) assertFalse(new CpuCapabilityProvider().supports(
                new OperationCapabilityQuery(value.operation(), value.inputs(), value.outputs())), value.id());
        assertEquals(0, List.of().size(), "Model/CPU exposes group or instance normalization only as non-direct composition");
    }

    /** The exact specialized provider boundary rows also owned by the canonical inventory. */
    static List<CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture> rejectedProviderFixtures() {
        var rejected = List.of(row("arg/bool", CpuArgExtremaLoweringTest.context(AggregateReductionKind.ARG_MIN, DataType.BOOL,
                        Shape.of(2, 3), 1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                row("masked/int32", CpuMaskedReductionLoweringTest.context(AggregateReductionKind.SUM, DataType.INT32,
                        Shape.of(2, 3), Shape.of(3), 1)),
                row("advanced/int64", CpuAggregateLoweringTest.context(AggregateReductionKind.L1_NORM, DataType.INT64,
                        Shape.of(2, 3), new MultiAxisReductionAttrs(List.of(1), false), Shape.of(2))));
        return rejected.stream().map(value -> {
            var node = value.context().nodes().getFirst();
            return new CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture(value.id(),
                    node.operation().kind().name(), node.operation(),
                    CpuGeneratedDirectEvidenceClosureTest.descriptors(value.context(), node.inputs()),
                    CpuGeneratedDirectEvidenceClosureTest.descriptors(value.context(), node.outputs()),
                    value.id().replace('/', '_').toUpperCase(java.util.Locale.ROOT) + "_INAPPLICABLE");
        }).toList();
    }

    private static void assertGenerated(List<Case> cases) {
        assertFalse(cases.isEmpty()); var counts = new TreeMap<String, Integer>();
        for (Case value : cases) for (Request request : Request.values()) {
            var context = configure(value.context(), request); assertTrue(supports(context), value.id() + '/' + request);
            var plan = new CpuPartitionPreparer().analyze(context).plan(); assertEquals(1, plan.units().size(), value.id());
            var node = context.nodes().getFirst();
            var inputs = CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.inputs());
            var outputs = CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.outputs());
            assertEquals(node.inputs().size(), inputs.size(), value.id() + " input-role arity");
            assertEquals(node.outputs().size(), outputs.size(), value.id() + " output-role arity");
            var route = plan.units().getFirst().portablePlan();
            var boundaryIds = new java.util.LinkedHashSet<ValueId>();
            boundaryIds.addAll(node.inputs()); boundaryIds.addAll(node.outputs());
            assertEquals(node.inputs().stream().distinct().count() + node.outputs().size(),
                    boundaryIds.size(), value.id() + " descriptor-boundary alias deduplication");
            var expectedBoundaryTypes = boundaryIds.stream().map(id -> context.values().stream()
                    .filter(graphValue -> graphValue.id().equals(id)).findFirst().orElseThrow()
                    .descriptor().dataType()).toList();
            assertEquals(expectedBoundaryTypes, route.specialization().boundaryDataTypes(),
                    value.id() + " descriptor roles must reach specialization");
            assertEquals(List.copyOf(boundaryIds), plan.units().getFirst().boundaryValues(),
                    value.id() + " prepared boundary order must deduplicate actual aliases");
            if (route.portableKernelIr() instanceof CpuAttentionIr attention) {
                assertEquals(node.inputs().stream().map(id -> List.copyOf(boundaryIds).indexOf(id)).toList(),
                        attention.roleBoundaryPositions(), value.id()
                                + " prepared attention IR must map Q/K/V/mask roles to aliases");
                assertEquals(expectedBoundaryTypes, attention.boundaryTypes(), value.id()
                        + " prepared attention IR must retain the deduplicated boundary types");
                assertEquals(node.inputs().stream().limit(3).map(id -> context.values().stream()
                        .filter(graphValue -> graphValue.id().equals(id)).findFirst().orElseThrow()
                        .descriptor().dataType()).toList(),
                        List.of(attention.queryType(), attention.keyType(), attention.valueType())
                                .subList(0, 3), value.id()
                                + " Q/K/V IR roles must describe their actual aliased descriptors");
            }
            assertEquals(plan.generatedCarrierPattern(), route.specialization().carrierPattern(),
                    value.id() + " selected carrier pattern");
            assertEquals(0, plan.materializations().size(), value.id()
                    + " materialization-enabled is a candidate request, not proof a copy was selected");
            byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            CpuGeneratedDirectEvidenceClosureTest.assertClosedSpecializedClass(bytes, route.specialization());
            CpuGeneratedCoverageEvidenceRegistry.generated("specialized:" + value.id() + '/' + request,
                    context, plan);
            counts.merge(value.id().substring(0, value.id().indexOf('/')) + '/' + request + "->" + plan.executionStrategy()
                    + "/m=" + plan.materializations().size(), 1, Integer::sum);
        }
        assertEquals(cases.size() * Request.values().length, counts.values().stream().mapToInt(Integer::intValue).sum(), counts.toString());
    }

    private static boolean supports(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var node = context.nodes().getFirst(); return new CpuCapabilityProvider().supports(new OperationCapabilityQuery(node.operation(),
                CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.inputs()), CpuGeneratedDirectEvidenceClosureTest.descriptors(context, node.outputs())));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> configure(PrepareContext<CpuPartitionAnalysisInputs> base, Request request) {
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < base.values().size(); i++) { var old = base.values().get(i); var descriptor = old.descriptor();
            if (request.general && descriptor.shape().rank() != 0) { long[] strides = descriptor.layout().orElseThrow().strides();
                for (int axis = 0; axis < strides.length; axis++) strides[axis] *= 2;
                descriptor = new TensorDescriptor(descriptor.dataType(), descriptor.shape(), Optional.of(LayoutDescriptor.of(descriptor.shape(), strides, 1, true)), descriptor.requiresGrad()); }
            values.add(new GraphValue(old.id(), descriptor)); var requirement = base.memoryRequirements().get(i);
            memory.add(new LogicalMemoryRequirement(requirement.valueId(), descriptor, requirement.producerPartition(), requirement.consumerPartitions(), requirement.graphOutput())); }
        var carriers = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
        for (int i = 0; i < values.size(); i++) carriers.add(request.segment || request.mixed && i % 2 == 1 ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                : CpuGeneratedDirectEvidenceClosureTest.heapCarrier(values.get(i).descriptor().dataType()));
        return new PrepareContext<>(base.partition(), base.nodes(), values, memory, base.constants(), new CpuPartitionAnalysisInputs(false, carriers,
                request.execution, request.materialization ? MATERIALIZATION : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
    }

    private static Shape reductionShape(Shape input, List<Integer> axes, boolean keep) { var result = new ArrayList<Long>(); long[] source = input.toLongArray();
        for (int axis = 0; axis < source.length; axis++) if (axes.contains(axis)) { if (keep) result.add(1L); } else result.add(source[axis]);
        return Shape.of(result.stream().mapToLong(Long::longValue).toArray()); }
    private static List<List<DataType>> floatingRoleCombinations() { var result = new ArrayList<List<DataType>>();
        for (var a : FLOATING) for (var b : FLOATING) for (var c : FLOATING) for (var d : FLOATING) for (var e : FLOATING) result.add(List.of(a,b,c,d,e)); return result; }
    private static List<DataType> numeric() { return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.INT64, DataType.INT32); }
    private static Shape matmulOutput(Shape left, Shape right) { var dims = new ArrayList<Long>(); long[] a = left.toLongArray(), b = right.toLongArray();
        int ar = a.length, br = b.length, batch = Math.max(0, Math.max(ar - 2, br - 2));
        for (int i = 0; i < batch; i++) { long x = i < batch - Math.max(0, ar - 2) ? 1 : a[i - (batch - Math.max(0, ar - 2))]; long y = i < batch - Math.max(0, br - 2) ? 1 : b[i - (batch - Math.max(0, br - 2))]; dims.add(Math.max(x, y)); }
        if (ar > 1) dims.add(a[ar - 2]); if (br > 1) dims.add(b[br - 1]); return Shape.of(dims.stream().mapToLong(Long::longValue).toArray()); }
    private static PrepareContext<CpuPartitionAnalysisInputs> matmulContext(DataType left, DataType right, Shape leftShape, Shape rightShape, Shape outputShape) {
        return CpuScatterLoweringTest.context(new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE), List.of(0, 1),
                List.of(descriptor(left, leftShape), descriptor(right, rightShape)), descriptor(DataTypePromotion.promoteNumeric(left, right), outputShape)); }
    static PrepareContext<CpuPartitionAnalysisInputs> attentionContext(DataType query, DataType key,
            DataType value, boolean mask, boolean causal, boolean weights, boolean explicitScale,
            List<Integer> roles) {
        // A repeated ValueId is admitted only when one descriptor is valid for every role.  The
        // partial partitions below therefore alter sequence/feature geometry, rather than merely
        // relabelling a distinct-descriptor fixture as aliased.
        boolean queryKey = roles.get(0).equals(roles.get(1));
        boolean keyValue = roles.get(1).equals(roles.get(2));
        boolean queryValue = roles.get(0).equals(roles.get(2));
        Shape queryShape = Shape.of(2, 3);
        Shape keyShape = queryKey || queryValue ? Shape.of(2, 3) : Shape.of(4, 3);
        Shape valueShape = queryValue ? queryShape : keyValue ? keyShape
                : Shape.of(keyShape.toLongArray()[0], 5);
        Shape scoreShape = Shape.of(queryShape.toLongArray()[0], keyShape.toLongArray()[0]);
        Shape outputShape = Shape.of(queryShape.toLongArray()[0], valueShape.toLongArray()[1]);
        var inputShapes = new ArrayList<Shape>();
        inputShapes.add(queryShape); inputShapes.add(keyShape); inputShapes.add(valueShape);
        if (mask) inputShapes.add(scoreShape);
        var inputTypes = new ArrayList<DataType>();
        inputTypes.add(query); inputTypes.add(key); inputTypes.add(value);
        if (mask) inputTypes.add(DataType.BOOL);
        DataType output = DataTypePromotion.promoteFloating(DataTypePromotion.promoteFloating(query, key), value);
        Optional<ScalarValue> scale = explicitScale ? Optional.of(scale(output)) : Optional.empty();
        var inputIds = new ArrayList<ValueId>();
        int uniqueInputs = roles.stream().mapToInt(Integer::intValue).max().orElseThrow() + 1;
        for (int index = 0; index < inputTypes.size(); index++) inputIds.add(new ValueId(roles.get(index)));
        var outputIds = new ArrayList<ValueId>();
        outputIds.add(new ValueId(uniqueInputs));
        if (weights) outputIds.add(new ValueId(uniqueInputs + 1));
        var operation = new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                new ScaledDotProductAttentionAttrs(scale, causal));
        var node = new CompiledNode(new NodeId(0), operation, inputIds, outputIds);
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int id = 0; id < uniqueInputs; id++) {
            DataType type = inputTypes.get(roles.indexOf(id));
            var descriptor = descriptor(type, inputShapes.get(roles.indexOf(id)));
            var valueId = new ValueId(id); values.add(new GraphValue(valueId, descriptor));
            memory.add(new LogicalMemoryRequirement(valueId, descriptor, Optional.empty(), List.of(partition), false));
        }
        var outputDescriptor = descriptor(output, outputShape);
        values.add(new GraphValue(outputIds.getFirst(), outputDescriptor));
        memory.add(new LogicalMemoryRequirement(outputIds.getFirst(), outputDescriptor, Optional.of(partition), List.of(), true));
        if (weights) {
            var weightsDescriptor = descriptor(output, scoreShape);
            values.add(new GraphValue(outputIds.get(1), weightsDescriptor));
            memory.add(new LogicalMemoryRequirement(outputIds.get(1), weightsDescriptor, Optional.of(partition), List.of(), true));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT);
    }
    static List<List<Integer>> attentionRoles(boolean mask, DataType query, DataType key,
            DataType value) {
        var result = new ArrayList<List<Integer>>();
        result.add(mask ? List.of(0, 1, 2, 3) : List.of(0, 1, 2));
        // A repeated ValueId is a real alias only when its descriptor can satisfy every role.
        // Do not label a mixed-type descriptor as an alias topology.
        if (query == key) result.add(mask ? List.of(0, 0, 1, 2) : List.of(0, 0, 1));
        if (key == value) result.add(mask ? List.of(0, 1, 1, 2) : List.of(0, 1, 1));
        if (query == value) result.add(mask ? List.of(0, 1, 0, 2) : List.of(0, 1, 0));
        if (query == key && key == value) result.add(mask ? List.of(0, 0, 0, 1) : List.of(0, 0, 0));
        return List.copyOf(result);
    }
    private static ScalarValue scale(DataType type) { return switch (type) {
        case FLOAT64 -> ScalarValue.float64(0.5d);
        case FLOAT32 -> ScalarValue.float32(0.5f);
        case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3f00);
        default -> throw new AssertionError(type);
    }; }
    private static PrepareContext<CpuPartitionAnalysisInputs> lossContext(LossKind kind, DataType left, DataType right, LossReduction reduction, boolean ignored, List<Integer> roles) {
        Shape logits = Shape.of(2, 3, 4); Shape categoricalTarget = Shape.of(2, 4); Shape target = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? categoricalTarget : logits;
        Operation operation = switch (kind) {
            case MEAN_SQUARED_ERROR -> new Operation(kind, new MeanSquaredErrorAttrs(reduction));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind, new DenseCategoricalCrossEntropyWithLogitsAttrs(1, reduction));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> new Operation(kind, new IndexCategoricalCrossEntropyWithLogitsAttrs(1, reduction, ignored ? Optional.of(right == DataType.INT32 ? ScalarValue.int32(-1) : ScalarValue.int64(-1)) : Optional.empty()));
        };
        Shape output = reduction == LossReduction.NONE ? kind == LossKind.MEAN_SQUARED_ERROR ? logits : categoricalTarget : Shape.scalar();
        DataType result = kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS ? left : DataTypePromotion.promoteFloating(left, right);
        List<TensorDescriptor> inputs = roles.equals(List.of(0, 0)) ? List.of(descriptor(left, logits)) : List.of(descriptor(left, logits), descriptor(right, target));
        return CpuScatterLoweringTest.context(operation, roles, inputs, descriptor(result, output)); }
    private static PrepareContext<CpuPartitionAnalysisInputs> trailingContext(boolean layer, List<DataType> roles) {
        Shape input = Shape.of(2, 3), normalized = Shape.of(3); DataType result = roles.getFirst();
        for (int index = 1; index < roles.size(); index++) result = DataTypePromotion.promoteFloating(result, roles.get(index));
        ScalarValue epsilon = epsilon(result);
        var attrs = layer ? roles.size() == 1 ? new LayerNormAttrs(normalized, epsilon) : new AffineLayerNormAttrs(normalized, epsilon)
                : new RmsNormAttrs(normalized, epsilon);
        var descriptors = new ArrayList<TensorDescriptor>(); descriptors.add(descriptor(roles.getFirst(), input));
        for (int index = 1; index < roles.size(); index++) descriptors.add(descriptor(roles.get(index), normalized));
        return io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest.context(
                new Operation(layer ? LayerNormKind.LAYER_NORM : RmsNormKind.RMS_NORM, attrs),
                java.util.stream.IntStream.range(0, roles.size()).boxed().toList(), descriptors, descriptor(result, input));
    }
    private static ScalarValue epsilon(DataType type) { return type == DataType.FLOAT64 ? ScalarValue.float64(1e-5)
            : type == DataType.FLOAT32 ? ScalarValue.float32(1e-5f) : ScalarValue.bfloat16Bits((short) 0x3728); }
    private static TensorDescriptor descriptor(DataType type, Shape shape) { return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private static List<LayoutDescriptor> trainingLayouts(Shape shape, int axis) { var result = new ArrayList<LayoutDescriptor>(); Shape vector = Shape.of(shape.toLongArray()[axis]); result.add(LayoutDescriptor.contiguous(shape));
        for (int i = 1; i < 5; i++) result.add(LayoutDescriptor.contiguous(vector)); result.add(LayoutDescriptor.contiguous(shape)); for (int i = 1; i < 5; i++) result.add(LayoutDescriptor.contiguous(vector)); return result; }
    private static Case row(String id, PrepareContext<CpuPartitionAnalysisInputs> context) { return new Case(id, context); }
    private record Case(String id, PrepareContext<CpuPartitionAnalysisInputs> context) { }
    private record Conv2dCase(String id, Conv2dAttrs attrs, Shape input, Shape weight, Shape output) { }
    private record Conv3dCase(String id, Conv3dAttrs attrs, Shape input, Shape weight, Shape output) { }
    private enum Request {
        HEAP_CONTIGUOUS_SCALAR(false,false,false,false,CpuGeneratedDirectEvidenceClosureTest.scalar(1)), SEGMENT_CONTIGUOUS_VECTOR(true,false,false,false,CpuGeneratedDirectEvidenceClosureTest.vector(1)),
        MIXED_GENERAL_PARALLEL_VECTOR(false,true,true,false,CpuGeneratedDirectEvidenceClosureTest.vector(4)), HEAP_GENERAL_PARALLEL_SCALAR(false,false,true,false,CpuGeneratedDirectEvidenceClosureTest.scalar(4)),
        HEAP_GENERAL_MATERIALIZATION(false,false,true,true,CpuGeneratedDirectEvidenceClosureTest.scalar(1));
        final boolean segment,mixed,general,materialization; final CpuPartitionAnalysisInputs.PortableExecutionConfig execution;
        Request(boolean segment,boolean mixed,boolean general,boolean materialization,CpuPartitionAnalysisInputs.PortableExecutionConfig execution) { this.segment=segment;this.mixed=mixed;this.general=general;this.materialization=materialization;this.execution=execution; }
    }
}
