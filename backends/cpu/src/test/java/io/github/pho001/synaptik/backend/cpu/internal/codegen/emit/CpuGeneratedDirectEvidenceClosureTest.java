package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuArgExtremaLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuConv3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuOrderingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaskedReductionLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMatmulLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterNdAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterNdKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.SortAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.ArgExtremaTiePolicy;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.StatisticalReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.loss.DenseCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.IndexCategoricalCrossEntropyWithLogitsAttrs;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool2dKind;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.classfile.ClassFile;
import java.lang.classfile.constantpool.DynamicConstantPoolEntry;
import java.lang.classfile.constantpool.MemberRefEntry;
import java.lang.classfile.constantpool.MethodHandleEntry;
import java.lang.classfile.instruction.InvokeInstruction;
import java.lang.reflect.AccessFlag;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/** Locks the cross-family structural inventory used by the generated/direct evidence closure. */
class CpuGeneratedDirectEvidenceClosureTest {
    private static final Set<String> NON_POINTWISE_FORMS = Set.of(
            "affine:copy", "affine:internal-view", "movement:pad", "movement:tile",
            "movement:concat", "movement:stack", "movement:unfold-axis",
            "movement:unfold2d", "movement:slice-update", "indexing:gather",
            "indexing:gather-elements", "indexing:gather-nd", "indexing:one-hot",
            "scatter:scatter-elements:none", "scatter:scatter-elements:add",
            "scatter:scatter-elements:min", "scatter:scatter-elements:max",
            "scatter:scatter-elements:mul", "scatter:scatter-add:add",
            "scatter:scatter-nd:none", "scatter:scatter-nd:add", "scatter:scatter-nd:min",
            "scatter:scatter-nd:max", "scatter:scatter-nd:mul", "fold:axis", "fold:2d",
            "ordering:sort", "ordering:argsort", "ordering:top-k", "random:initial-state",
            "random:dropout-f64", "random:dropout-f32", "scan:cum-sum:inclusive-forward",
            "scan:cum-sum:inclusive-reverse", "scan:cum-sum:exclusive-forward",
            "scan:cum-sum:exclusive-reverse", "scan:cum-prod:inclusive-forward",
            "scan:cum-prod:inclusive-reverse", "scan:cum-prod:exclusive-forward",
            "scan:cum-prod:exclusive-reverse", "aggregate:min:full", "aggregate:min:axis",
            "aggregate:min:multi-axis", "aggregate:max:full", "aggregate:max:axis",
            "aggregate:max:multi-axis", "aggregate:all:full", "aggregate:all:axis",
            "aggregate:all:multi-axis", "aggregate:any:full", "aggregate:any:axis",
            "aggregate:any:multi-axis", "aggregate:sum:full", "aggregate:sum:axis",
            "aggregate:sum:multi-axis", "aggregate:mean:full", "aggregate:mean:axis",
            "aggregate:mean:multi-axis", "aggregate:prod:full", "aggregate:prod:axis",
            "aggregate:prod:multi-axis", "advanced:log-sum-exp", "advanced:variance",
            "advanced:standard-deviation", "advanced:l1-norm", "advanced:l2-norm");

    @Test void ledgerCoversEveryPointwiseOpcodeAndCurrentNonPointwiseForm() {
        assertAll(
                () -> assertEquals(48, CpuPointwiseOpcode.values().length),
                () -> assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class),
                        EnumSet.copyOf(List.of(CpuPointwiseOpcode.values()))),
                () -> assertEquals(66, NON_POINTWISE_FORMS.size()),
                () -> assertTrue(NON_POINTWISE_FORMS.stream().noneMatch(String::isBlank)));
    }

    @Test void representativesHaveOneTypedStaticEntryAndClosedMemberReferences() {
        var representatives = new ArrayList<Representative>();
        representatives.add(new Representative(pointwise(false, false), 0));
        representatives.add(new Representative(pointwise(true, false), 0));
        representatives.add(new Representative(pointwise(false, true), 1));
        representatives.add(new Representative(generated(CpuNonAffineMovementLoweringTest.context(
                new Operation(PadKind.PAD,
                        new PadAttrs(List.of(0L), List.of(0L), ScalarValue.int32(0))),
                List.of(0), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                        DataType.INT32, Shape.of(8))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(8)))), 1));
        representatives.add(new Representative(generated(CpuIndexingLoweringTest.context(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, Shape.of(8)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(4))),
                CpuIndexingLoweringTest.descriptor(DataType.FLOAT64, Shape.of(4)))), 2));
        representatives.add(new Representative(generated(CpuScatterLoweringTest.context(
                new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MIN)), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(4))),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)))), 2));
        representatives.add(new Representative(generated(CpuFoldLoweringTest.context(
                new Operation(WindowTransformKind.FOLD_AXIS, new FoldAxisAttrs(0, 8, 1)),
                DataType.FLOAT32, Shape.of(7, 2), Shape.of(8))), 0));
        representatives.add(new Representative(generated(CpuOrderingLoweringTest.context(
                new Operation(OrderingKind.ARGSORT, new SortAttrs(1, true)), DataType.INT64,
                Shape.of(4, 8), Shape.of(4, 8), false)), 0));
        representatives.add(new Representative(generated(CpuScanLoweringTest.context(
                CumulativeScanKind.CUM_PROD, DataType.INT64, Shape.of(4, 8), 1, true, true)), 1));
        representatives.add(new Representative(generated(CpuAggregateLoweringTest.context(
                AggregateReductionKind.MIN, DataType.BFLOAT16, Shape.of(4, 8),
                new AxisReductionAttrs(1, false), Shape.of(4))), 1, true));
        representatives.add(new Representative(generated(CpuAggregateLoweringTest.context(
                AggregateReductionKind.MEAN, DataType.FLOAT32, Shape.of(4, 8),
                new AxisReductionAttrs(1, false), Shape.of(4))), 1, true));
        representatives.add(new Representative(advanced(AggregateReductionKind.LOG_SUM_EXP), 0));
        representatives.add(new Representative(advanced(AggregateReductionKind.VARIANCE), 0));
        representatives.add(new Representative(advanced(
                AggregateReductionKind.STANDARD_DEVIATION), 0));
        representatives.add(new Representative(advanced(AggregateReductionKind.L1_NORM), 0));
        representatives.add(new Representative(advanced(AggregateReductionKind.L2_NORM), 0));
        representatives.forEach(representative -> {
            assertClosedClass(representative.bytes());
            assertSegmentLayoutsHoisted(representative);
        });
    }

    @Test void generalPointwiseAndCrossTypeCastFixturesUseEveryProductionBoundary() throws Exception {
        // These are explicit Model occurrences. They neither infer support from opcode labels nor
        // project the complete typed witness basis across untested carriers, layouts, or strategies.
        List<PointwiseFixture> fixtures = generalPointwiseFixtures();
        assertEquals(169, fixtures.size());
        for (PointwiseFixture fixture : fixtures) {
            var context = fixture.context();
            assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    fixture.operation(), fixture.inputDescriptors(), List.of(fixture.outputDescriptor()))),
                    fixture.id());
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            var route = plan.units().getFirst().portablePlan();
            assertEquals(fixture.opcode(), route.kernelIr().instructions().getFirst().opcode(), fixture.id());
            assertEquals(fixture.boundaryTypes(), route.specialization().boundaryDataTypes(), fixture.id());
            assertEquals(List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                    plan.units().getFirst().generatedCarrierPattern().stream().distinct().toList(), fixture.id());
            assertTrue(plan.units().getFirst().accessBindings().stream().allMatch(binding -> binding.plan().regime()
                    == CpuAccessPlan.Regime.DENSE_LINEAR), fixture.id());
            assertEquals(0, plan.materializations().size(), fixture.id());
            assertEquals(plan.executionStrategy(), route.specialization().executionStrategy(), fixture.id());

            var generator = new CpuClassFileKernelGenerator();
            byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
            byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
            assertArrayEquals(first, second, fixture.id() + " deterministic class-file hash");
            CpuGeneratedCoverageEvidenceRegistry.generated("pointwise:" + fixture.id(), context, plan);
            assertEquals(classFileHash(first), classFileHash(second), fixture.id());
            assertEquals(route.specialization().entryType().descriptorString(), ClassFile.of().parse(first)
                    .methods().getFirst().methodType().stringValue(), fixture.id());
            assertFalse(route.specialization().structuralKey().isBlank(), fixture.id());
            assertClosedClass(first);
        }
    }

    @Test void pointwiseCandidateUniverseClosesRequestedCarrierLayoutAndStrategyAxes() throws Exception {
        // This is a finite occurrence matrix, not a provider-derived Cartesian claim.  The
        // Model operation families determine the 169 non-scalar occurrences below; the scalar
        // occurrences are the separate 256-row finite immediate/clamp basis audited by the
        // checkpoint test.  Every row has a stable id and an owner-defined disposition before
        // CpuCapabilityProvider, preparation, or generation is called.
        var candidates = pointwiseCandidates();
        assertEquals(845, candidates.size());
        assertEquals(48, CpuPointwiseOpcode.values().length);
        assertEquals(169, candidates.stream().map(candidate -> candidate.form()).distinct().count());
        assertEquals(Set.of("HEAP", "SEGMENT", "MIXED"), candidates.stream()
                .map(candidate -> candidate.carrierAxis()).collect(java.util.stream.Collectors.toSet()));
        assertEquals(Set.of("CONTIGUOUS", "GENERAL"), candidates.stream()
                .map(candidate -> candidate.layoutAxis()).collect(java.util.stream.Collectors.toSet()));
        assertEquals(Set.of("DIRECT", "MATERIALIZATION_CANDIDATE"), candidates.stream()
                .map(candidate -> candidate.materializationAxis())
                .collect(java.util.stream.Collectors.toSet()));
        assertEquals(Set.of("SCALAR", "VECTOR", "PARALLEL_SCALAR", "PARALLEL_VECTOR"), candidates.stream()
                .map(candidate -> candidate.requestedStrategy())
                .collect(java.util.stream.Collectors.toSet()));
        assertEquals(Map.of("HEAP", 507L, "SEGMENT", 169L, "MIXED", 169L), counts(candidates,
                PointwiseCandidate::carrierAxis), "every requested carrier mode has one finite owner");
        assertEquals(Map.of("CONTIGUOUS", 507L, "GENERAL", 338L), counts(candidates,
                PointwiseCandidate::layoutAxis), "every requested layout class has one finite owner");
        assertEquals(Map.of("DIRECT", 676L, "MATERIALIZATION_CANDIDATE", 169L), counts(candidates,
                PointwiseCandidate::materializationAxis),
                "materialization candidates remain distinct from direct selection");
        assertEquals(Map.of("SCALAR", 338L, "VECTOR", 169L, "PARALLEL_SCALAR", 169L,
                "PARALLEL_VECTOR", 169L), counts(candidates, PointwiseCandidate::requestedStrategy),
                "every requested execution variant has one finite owner");

        var selectedStrategies = new java.util.TreeMap<String, Long>();
        for (PointwiseCandidate candidate : candidates) {
            assertEquals(PointwiseDisposition.SUPPORTED, candidate.disposition(), candidate.id());
            PointwiseFixture fixture = candidate.fixture();
            var context = fixture.context(candidate.inputLayout(), candidate.inputs(), candidate.execution(),
                    candidate.materializationPolicy());
            // The positive expectation is defined from the Model occurrence family and its
            // descriptor roles.  This real provider call is the evidence, never its definition.
            assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                    fixture.operation(), descriptors(context, context.nodes().getFirst().inputs()),
                    descriptors(context, context.nodes().getFirst().outputs()))), candidate.id());
            var plan = new CpuPartitionPreparer().analyze(context).plan();
            assertEquals(1, plan.units().size(), candidate.id());
            var route = plan.units().getFirst().portablePlan();
            assertEquals(fixture.opcode(), route.kernelIr().instructions().getFirst().opcode(), candidate.id());
            assertEquals(fixture.boundaryTypes(), route.specialization().boundaryDataTypes(), candidate.id());
            assertEquals(candidate.inputs(), route.specialization().carrierPattern(), candidate.id());
            assertEquals(0, plan.materializations().size(), candidate.id());
            // Keep the request and the actual selection separate. The recorded per-row fact
            // captures the current type/family-specific representation selection.
            assertEquals(candidate.materializationSelected(), plan.representationDecisions().stream().anyMatch(decision -> decision instanceof
                    io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision.Variant
                    variant && !variant.identity().materializations().isEmpty()), candidate.id());
            assertEquals(kernelStrategy(plan.executionStrategy()), route.specialization().executionStrategy(),
                    candidate.id() + " parallel orchestration reuses the selected kernel body");
            selectedStrategies.merge(candidate.requestedStrategy() + "->" + plan.executionStrategy(), 1L, Long::sum);
            byte[] first = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
            assertArrayEquals(first, second, candidate.id());
            assertEquals(route.specialization().entryType().descriptorString(), ClassFile.of().parse(first)
                    .methods().getFirst().methodType().stringValue(), candidate.id());
            assertFalse(route.specialization().structuralKey().isBlank(), candidate.id());
            assertClosedClass(first);
            // This is the canonical owner for the finite 845-row execution matrix.  Keeping
            // the candidate id (rather than its opcode/form) makes carrier, layout, request,
            // and materialization distinctions independently joinable by the checkpoint.
            CpuGeneratedCoverageEvidenceRegistry.generated("pointwise-matrix:" + candidate.id(),
                    context, plan);
        }
        assertEquals(Map.of(
                "SCALAR->scalar", 338L,
                "VECTOR->scalar", 95L,
                "VECTOR->vector", 74L,
                "PARALLEL_SCALAR->parallel-scalar", 169L,
                "PARALLEL_VECTOR->parallel-scalar", 95L,
                "PARALLEL_VECTOR->parallel-vector", 74L), selectedStrategies,
                "requested strategy and actual selected strategy are independently closed");
    }

    private static <T> Map<String, Long> counts(List<T> values,
            java.util.function.Function<T, String> axis) {
        return values.stream().collect(java.util.stream.Collectors.groupingBy(axis, java.util.TreeMap::new,
                java.util.stream.Collectors.counting()));
    }

    @Test void pointwiseModelValidButCpuInapplicableTypeRoleUniverseIsExplicitlyRejected() {
        // These rows are derived from the Model expression contracts: numeric promotion permits
        // mixed same-category operands, WHERE promotes floating branches, and each family fixes
        // its BOOL/numeric result role.  They deliberately do not copy CPU provider predicates.
        var candidates = rejectedPointwiseCandidates();
        assertEquals(152, candidates.size());
        assertEquals(candidates.size(), candidates.stream().map(PointwiseRejectedCandidate::id).distinct().count());
        assertEquals(Set.of("MIXED_NUMERIC_INPUTS", "MIXED_FLOATING_BRANCHES", "INVALID_RESULT_ROLE",
                "INVALID_BOOLEAN_NUMERIC_ROLE", "SHAPE_INCOMPATIBLE"), candidates.stream()
                .map(PointwiseRejectedCandidate::reason).collect(java.util.stream.Collectors.toSet()));
        var provider = new CpuCapabilityProvider();
        for (PointwiseRejectedCandidate candidate : candidates) {
            assertFalse(provider.supports(new OperationCapabilityQuery(candidate.operation(),
                    candidate.inputs(), List.of(candidate.output()))), candidate.id() + ": " + candidate.reason());
        }
    }

    @Test void pointwiseOpcodeAndScalarForeignKeysReconcileExactlyOnce() throws Exception {
        // The operation-family ledger owns one row for every executable pointwise opcode.  The
        // scalar-immediate basis owns the 256 parameterized scalar forms, so scalar kinds link
        // there rather than fabricating duplicate general pointwise classes.
        var opcodes = generalPointwiseFixtures().stream().map(PointwiseFixture::opcode)
                .collect(java.util.stream.Collectors.toCollection(() -> EnumSet.noneOf(CpuPointwiseOpcode.class)));
        assertEquals(EnumSet.complementOf(EnumSet.of(CpuPointwiseOpcode.SCALAR_ADD,
                CpuPointwiseOpcode.SCALAR_SUB, CpuPointwiseOpcode.SCALAR_MUL,
                CpuPointwiseOpcode.SCALAR_DIV, CpuPointwiseOpcode.SCALAR_POW,
                CpuPointwiseOpcode.SCALAR_MIN, CpuPointwiseOpcode.SCALAR_MAX,
                CpuPointwiseOpcode.SCALAR_CLAMP)), opcodes);
        try (var stream = CpuGeneratedDirectEvidenceClosureTest.class.getResourceAsStream(
                "operation-family-form-ledger-v3.tsv")) {
            assertTrue(stream != null);
            var rows = new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).lines()
                    .filter(line -> !line.startsWith("# ")).skip(1).map(line -> line.split("\\t", -1)).toList();
            var ledger = rows.stream().filter(row -> row[1].equals("pointwise"))
                    .map(row -> row[0]).filter(CpuGeneratedDirectEvidenceClosureTest::isPointwiseOpcode)
                    .toList();
            assertEquals(CpuPointwiseOpcode.values().length, ledger.size());
            assertEquals(CpuPointwiseOpcode.values().length, ledger.stream().distinct().count());
            assertEquals(EnumSet.allOf(CpuPointwiseOpcode.class), ledger.stream()
                    .map(CpuPointwiseOpcode::valueOf).collect(java.util.stream.Collectors.toCollection(
                            () -> EnumSet.noneOf(CpuPointwiseOpcode.class))));
        }
        Map<String, String> inventoryForeignKeys;
        try (var stream = CpuGeneratedDirectEvidenceClosureTest.class.getResourceAsStream(
                "generated-coverage-inventory.tsv")) {
            assertTrue(stream != null);
            var rows = new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).lines().skip(1)
                    .map(line -> line.split("\\t", -1)).filter(row -> row[0].startsWith("pointwise-matrix:")
                            || row[0].startsWith("scalar-immediate:")).toList();
            inventoryForeignKeys = rows.stream().collect(java.util.stream.Collectors.toUnmodifiableMap(
                    row -> row[2], row -> row[0], (first, ignored) -> first));
            assertEquals(CpuPointwiseOpcode.values().length, inventoryForeignKeys.size());
            for (CpuPointwiseOpcode opcode : CpuPointwiseOpcode.values()) assertTrue(
                    inventoryForeignKeys.containsKey(opcode.name()), opcode.name());
        }
        try (var stream = CpuGeneratedDirectEvidenceClosureTest.class.getResourceAsStream(
                "scalar-immediate-clamp-matrix-forms.tsv")) {
            assertTrue(stream != null);
            var rows = new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8).lines()
                    .filter(line -> !line.startsWith("# ")).skip(1).map(line -> line.split("\\t", -1)).toList();
            assertEquals(263, rows.size());
            assertEquals(263, rows.stream().map(row -> row[0]).distinct().count());
            var linked = rows.stream().map(row -> row[1]).collect(java.util.stream.Collectors.toSet());
            assertEquals(EnumSet.allOf(ScalarElementwiseKind.class).stream().map(Enum::name)
                    .collect(java.util.stream.Collectors.toSet()), linked);
            for (ScalarElementwiseKind kind : ScalarElementwiseKind.values()) {
                String opcode = "SCALAR_" + kind.name();
                assertTrue(inventoryForeignKeys.containsKey(opcode), kind + " inventory foreign key");
                assertTrue(rows.stream().anyMatch(row -> row[1].equals(kind.name())), kind + " scalar row foreign key");
            }
        }
    }

    private static boolean isPointwiseOpcode(String operation) {
        try {
            CpuPointwiseOpcode.valueOf(operation);
            return true;
        } catch (IllegalArgumentException ignored) {
            return false;
        }
    }

    @Test void nonPointwiseRepresentativesUseEveryProductionBoundary() throws Exception {
        // This finite closure is deliberately fixture-backed. It opens only the 30 real Model
        // occurrences below; no family label or inventory row is used to infer another form.
        List<NonPointwiseFixture> fixtures = nonPointwiseFixtures();
        assertEquals(30, fixtures.size());
        for (NonPointwiseFixture fixture : fixtures) assertGeneratedBoundary(fixture);
    }

    @Test void remainingSpecializedGeneratedFamiliesUseEveryProductionBoundary() throws Exception {
        // Each entry is one existing family fixture, not a projection across family variants.
        // The loss entries establish support and generated structure only: their retained 0008I
        // performance disposition remains NON_PASSING and selection remains fail closed.
        List<NonPointwiseFixture> fixtures = specializedGeneratedFixtures();
        assertEquals(28, fixtures.size());
        for (NonPointwiseFixture fixture : fixtures) assertGeneratedBoundary(fixture);
    }

    @Test void conv1dCompositionIsNotCurrentlyBodyIdenticalToEquivalentDirectConv2d()
            throws Exception {
        var context = conv1dCompositionContext();
        var conv2d = context.nodes().stream().filter(node -> node.operation().kind().name()
                .equals("CONV2D")).findFirst().orElseThrow();
        assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(conv2d.operation(),
                descriptors(context, conv2d.inputs()), descriptors(context, conv2d.outputs()))));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(1, plan.units().size());
        var route = plan.units().getFirst().portablePlan();
        var specialization = route.specialization();
        assertTrue(plan.units().getFirst().conv2dGeometry().isPresent(),
                "Conv1d is composition metadata reusing the Conv2d generated form");
        assertTrue(plan.units().getFirst().conv3dGeometry().isEmpty());
        assertEquals(0, plan.materializations().size());
        assertTrue(plan.workspaceDeclaration().isEmpty());
        byte[] first = new CpuClassFileKernelGenerator().generateClassBytes(specialization, route.kernelIr());
        byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(specialization, route.kernelIr());
        assertArrayEquals(first, second);
        assertEquals(specialization.entryType().descriptorString(), ClassFile.of().parse(first)
                .methods().getFirst().methodType().stringValue());
        assertFalse(specialization.structuralKey().isBlank());
        assertClosedSpecializedClass(first, specialization);

        // The composition is intentionally only virtual rank editing.  Its generated route must
        // be precisely the route for the equivalent concrete Conv2d occurrence, not merely a
        // class with the same schema.
        var direct = CpuConv2dLoweringTest.context(List.of(DataType.FLOAT32, DataType.FLOAT32,
                DataType.FLOAT32), Shape.of(1, 2, 1, 4), Shape.of(2, 2, 1, 3),
                Shape.of(1, 2, 1, 4), new Conv2dAttrs(1, 1, 0, 1, 1, 1, 1), null);
        // Carrier representation is bytecode-relevant, so compare the same cold carrier facts.
        direct = new PrepareContext<>(direct.partition(), direct.nodes(), direct.values(),
                direct.memoryRequirements(), direct.constants(), context.backendInputs());
        var directRoute = new CpuPartitionPreparer().analyze(direct).plan().units().getFirst()
                .portablePlan();
        byte[] directBytes = new CpuClassFileKernelGenerator().generateClassBytes(
                directRoute.specialization(), directRoute.kernelIr());
        // This is a machine-checked contradiction to the desired closure claim, not a
        // representative projection: today the composition retains a distinct lowering key and
        // class identity even with equivalent concrete Conv2d geometry and carrier facts.
        // Do not label Conv1d as body-identical until production removes this distinction.
        assertFalse(directRoute.kernelIr().structuralKey().equals(route.kernelIr().structuralKey()));
        assertFalse(java.util.Arrays.equals(directBytes, first));
    }

    @Test void initialRandomStateIsNotMetadataOnlyInTheCurrentProductionRoute() {
        var context = CpuRandomLoweringTest.initialContext(0x1234L, -7L);
        var node = context.nodes().getFirst();
        assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(node.operation(),
                descriptors(context, node.inputs()), descriptors(context, node.outputs()))));
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(0, plan.elementCount());
        assertTrue(plan.randomGeometry().isPresent());
        assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                plan.executionStrategy());
        // This is intentionally counter-evidence to the proposed coverage disposition: the
        // current random emitter has an INITIAL_STATE branch and the generator returns a class.
        // A future metadata-only claim must first remove that production route; this checkpoint
        // must not hide it by omitting the generator boundary.
        var route = plan.units().getFirst().portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(),
                route.kernelIr());
        assertFalse(bytes.length == 0, "INITIAL_STATE currently has a generated class route");
        assertClosedSpecializedClass(bytes, route.specialization());
        var evidence = CpuGeneratedCoverageEvidenceRegistry.generated("nonpointwise:initial-state",
                context, plan);
        assertEquals("GENERATED", evidence.outcome());
        assertEquals(64, evidence.normalizedBodyKey().length());
        assertFalse(evidence.classHash().isBlank());
    }

    private static void assertGeneratedBoundary(NonPointwiseFixture fixture) throws Exception {
        var context = fixture.context();
        var node = context.nodes().getFirst();
        var inputDescriptors = descriptors(context, node.inputs());
        var outputDescriptors = descriptors(context, node.outputs());
        assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(node.operation(),
                inputDescriptors, outputDescriptors)), fixture.id());
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(1, plan.units().size(), fixture.id());
        var unit = plan.units().getFirst();
        var route = unit.portablePlan();
        var specialization = route.specialization();
        assertEquals(boundaryDescriptors(context, plan), specialization.boundaryDataTypes(), fixture.id());
        assertEquals(plan.generatedCarrierPattern(), specialization.carrierPattern(), fixture.id());
        assertEquals(plan.executionStrategy(), specialization.executionStrategy(), fixture.id());
        assertEquals(plan.materializations().isEmpty(), specialization.materializedSourcePosition() == -1,
                fixture.id());
        assertEquals(plan.workspaceDeclaration().isPresent(), specialization.scratchParameter(), fixture.id());
        assertEquals(plan.boundaryValues().size(), plan.accessBindings().size(), fixture.id());
        assertEquals(plan.boundaryValues().size(), specialization.carrierPattern().size(), fixture.id());
        assertTrue(plan.accessBindings().stream().allMatch(binding -> binding.elementCount() >= 0),
                fixture.id());
        assertEquals(route.portableKernelIr().structuralKey(), specialization.loweringFingerprint().hex(),
                fixture.id());
        assertFalse(specialization.structuralKey().isBlank(), fixture.id());

        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(specialization, route.kernelIr());
        byte[] second = generator.generateClassBytes(specialization, route.kernelIr());
        assertArrayEquals(first, second, fixture.id() + " deterministic class-file bytes");
        assertEquals(classFileHash(first), classFileHash(second), fixture.id());
        assertEquals(specialization.entryType().descriptorString(), ClassFile.of().parse(first)
                .methods().getFirst().methodType().stringValue(), fixture.id());
        assertClosedSpecializedClass(first, specialization);
    }

    static List<TensorDescriptor> descriptors(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<ValueId> ids) {
        return ids.stream().map(id -> context.values().stream().filter(value -> value.id().equals(id))
                .findFirst().orElseThrow().descriptor()).toList();
    }

    private static List<DataType> boundaryDescriptors(PrepareContext<CpuPartitionAnalysisInputs> context,
            CpuPartitionPreparationPlan plan) {
        return descriptors(context, plan.boundaryValues()).stream().map(TensorDescriptor::dataType).toList();
    }

    /**
     * Evidence-local attention occurrence.  It stays here because only the generated-coverage
     * audit needs to open this duplicate-role boundary; lowering tests retain package visibility.
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> attentionContext(boolean duplicateQueryKey) {
        Shape q = Shape.of(2, 2);
        TensorDescriptor query = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor key = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor value = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        TensorDescriptor output = CpuScatterLoweringTest.desc(DataType.FLOAT32, q);
        return CpuScatterLoweringTest.context(new Operation(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                duplicateQueryKey ? List.of(0, 0, 1) : List.of(0, 1, 2),
                duplicateQueryKey ? List.of(query, value) : List.of(query, key, value), output);
    }

    /**
     * Evidence-local exact Conv1d-as-Conv2d composition.  The next assertion compares it with
     * the direct Conv2d occurrence, so this construction cannot become a second lowering rule.
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> conv1dCompositionContext() {
        Shape input = Shape.of(1, 2, 4), weight = Shape.of(2, 2, 3), bias = Shape.of(2);
        Shape expandedInput = Shape.of(1, 2, 1, 4), expandedWeight = Shape.of(2, 2, 1, 3);
        Shape convOutput = Shape.of(1, 2, 1, 4), output = Shape.of(1, 2, 4);
        List<Shape> shapes = List.of(input, weight, bias, expandedInput, expandedWeight,
                convOutput, output);
        List<ValueId> ids = java.util.stream.LongStream.range(0, shapes.size())
                .mapToObj(ValueId::new).toList();
        List<CompiledNode> nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(2)), List.of(ids.get(0)), List.of(ids.get(3))),
                new CompiledNode(new NodeId(1), new Operation(AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(2)), List.of(ids.get(1)), List.of(ids.get(4))),
                new CompiledNode(new NodeId(2), new Operation(Conv2dKind.CONV2D,
                        new Conv2dAttrs(1, 1, 0, 1, 1, 1, 1)),
                        List.of(ids.get(3), ids.get(4), ids.get(2)), List.of(ids.get(5))),
                new CompiledNode(new NodeId(3), new Operation(AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(2)), List.of(ids.get(5)), List.of(ids.get(6))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < ids.size(); index++) {
            TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32, shapes.get(index),
                    Optional.of(LayoutDescriptor.contiguous(shapes.get(index))), false);
            values.add(new GraphValue(ids.get(index), descriptor));
            boolean produced = index >= 3;
            boolean published = index == 6;
            memory.add(new LogicalMemoryRequirement(ids.get(index), descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    published ? List.of() : List.of(partition), published));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                new CpuPartitionAnalysisInputs(false, java.util.Collections.nCopies(4,
                        CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY)));
    }

    static List<NonPointwiseFixture> nonPointwiseFixtures() {
        return List.of(
                nonPointwise("pad", CpuNonAffineMovementLoweringTest.context(new Operation(PadKind.PAD,
                        new PadAttrs(List.of(1L), List.of(2L), ScalarValue.int32(-7))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("tile", CpuNonAffineMovementLoweringTest.context(new Operation(TileKind.TILE,
                        new TileAttrs(List.of(3L))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(6)))),
                nonPointwise("concat", CpuNonAffineMovementLoweringTest.context(new Operation(
                        TensorCompositionKind.CONCAT, new CompositionAxisAttrs(0)), List.of(0, 1, 0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(1))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("stack", CpuNonAffineMovementLoweringTest.context(new Operation(
                        TensorCompositionKind.STACK, new CompositionAxisAttrs(1)), List.of(0, 1),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2)))),
                nonPointwise("unfold-axis", CpuNonAffineMovementLoweringTest.context(new Operation(
                        WindowTransformKind.UNFOLD_AXIS, new UnfoldAxisAttrs(1, 2, 1)), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 2, 2)))),
                nonPointwise("unfold2d", CpuNonAffineMovementLoweringTest.context(new Operation(
                        WindowTransformKind.UNFOLD2D, new Unfold2dAttrs(CpuFoldLoweringTest.window(false),
                                ScalarValue.float64(0))), List.of(0),
                        List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.FLOAT64, Shape.of(1, 1, 3, 3))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.FLOAT64, Shape.of(1, 4, 4)))),
                nonPointwise("slice-update", CpuNonAffineMovementLoweringTest.context(new Operation(
                        SliceKind.SLICE_UPDATE, new SliceAttrs(List.of(4L), List.of(2L), List.of(0), List.of(-2L))),
                        List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)),
                                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, Shape.of(5)))),
                nonPointwise("gather", CpuIndexingLoweringTest.context(new Operation(AxisGatherKind.GATHER,
                        new IndexAxisAttrs(1)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                        CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, Shape.of(2, 2)))),
                nonPointwise("gather-elements", CpuIndexingLoweringTest.context(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                        new IndexAxisAttrs(1)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2))),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2)))),
                nonPointwise("gather-nd", CpuIndexingLoweringTest.context(new Operation(GatherNdKind.GATHER_ND,
                        new GatherNdAttrs(0)), List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 1))),
                        CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)))),
                nonPointwise("one-hot", CpuIndexingLoweringTest.context(new Operation(OneHotKind.ONE_HOT,
                        new OneHotAttrs(3)), List.of(0), List.of(CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                        CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3)))),
                nonPointwise("scatter-elements-min", scatter(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0, ScatterReduction.MIN))),
                nonPointwise("scatter-add", CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ADD,
                        new IndexAxisAttrs(1)), List.of(0, 1, 2), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(5, 2)), CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 5, 2, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)))),
                nonPointwise("scatter-nd", CpuScatterLoweringTest.context(new Operation(ScatterNdKind.SCATTER_ND,
                        new ScatterNdAttrs(1, ScatterReduction.MAX)), List.of(0, 1, 2), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 5, 1)), CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 5, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)))),
                nonPointwise("fold-axis", CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                        new FoldAxisAttrs(0, 5, 1)), DataType.INT64, Shape.of(3, 3), Shape.of(5))),
                nonPointwise("fold2d", CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD2D,
                        new io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs(Shape.of(1, 1, 3, 3), CpuFoldLoweringTest.window(false))),
                        DataType.FLOAT32, Shape.of(1, 4, 4), Shape.of(1, 1, 3, 3))),
                nonPointwise("sort", CpuOrderingLoweringTest.context(new Operation(OrderingKind.SORT,
                        new SortAttrs(1, false)), DataType.FLOAT64, Shape.of(2, 4), Shape.of(2, 4), false)),
                nonPointwise("argsort", CpuOrderingLoweringTest.context(new Operation(OrderingKind.ARGSORT,
                        new SortAttrs(1, true)), DataType.INT64, Shape.of(4, 8), Shape.of(4, 8), false)),
                nonPointwise("top-k", CpuOrderingLoweringTest.context(new Operation(TopKKind.TOP_K,
                        new TopKAttrs(1, 2, true, false)), DataType.FLOAT32, Shape.of(3, 5), Shape.of(3, 2), true)),
                nonPointwise("dropout-f64", CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(2, 3), .25d)),
                nonPointwise("dropout-f32", CpuRandomLoweringTest.dropoutContext(DataType.FLOAT32, Shape.of(2, 3), .25d)),
                nonPointwise("cum-sum-inclusive-forward", CpuScanLoweringTest.context(CumulativeScanKind.CUM_SUM,
                        DataType.FLOAT32, Shape.of(2, 3), 1, false, false)),
                nonPointwise("cum-sum-exclusive-reverse", CpuScanLoweringTest.context(CumulativeScanKind.CUM_SUM,
                        DataType.INT64, Shape.of(2, 3), 1, true, true)),
                nonPointwise("cum-prod-inclusive-forward", CpuScanLoweringTest.context(CumulativeScanKind.CUM_PROD,
                        DataType.BFLOAT16, Shape.of(2, 3), 1, false, false)),
                nonPointwise("cum-prod-exclusive-reverse", CpuScanLoweringTest.context(CumulativeScanKind.CUM_PROD,
                        DataType.INT64, Shape.of(4, 8), 1, true, true)),
                nonPointwise("aggregate-min", CpuAggregateLoweringTest.context(AggregateReductionKind.MIN,
                        DataType.BFLOAT16, Shape.of(4, 8), new AxisReductionAttrs(1, false), Shape.of(4))),
                nonPointwise("aggregate-mean", CpuAggregateLoweringTest.context(AggregateReductionKind.MEAN,
                        DataType.FLOAT32, Shape.of(4, 8), new AxisReductionAttrs(1, false), Shape.of(4))),
                nonPointwise("log-sum-exp", advancedContext(AggregateReductionKind.LOG_SUM_EXP)),
                nonPointwise("variance", advancedContext(AggregateReductionKind.VARIANCE)),
                nonPointwise("l2-norm", advancedContext(AggregateReductionKind.L2_NORM)));
    }

    /* Ordinary non-pointwise closure moved to CpuOrdinaryNonPointwiseGeneratedMatrixTest. */
    static List<NonPointwiseFixture> ordinaryFixtureSeeds() {
        var fixtures = new ArrayList<NonPointwiseFixture>();
        // Keep only the families owned by this ordinary increment; advanced reductions have
        // their own closure rows and must not be accidentally counted here.
        fixtures.addAll(nonPointwiseFixtures().stream().filter(fixture -> !Set.of(
                "scatter-elements-min", "scatter-add", "scatter-nd",
                "cum-sum-inclusive-forward", "cum-sum-exclusive-reverse",
                "cum-prod-inclusive-forward", "cum-prod-exclusive-reverse",
                "aggregate-min", "aggregate-mean", "log-sum-exp", "variance", "l2-norm")
                .contains(fixture.id())).toList());
        for (ScatterReduction reduction : ScatterReduction.values()) {
            fixtures.add(nonPointwise("scatter-elements-" + reduction.name().toLowerCase(),
                    scatter(AxisScatterKind.SCATTER_ELEMENTS,
                            new ScatterElementsAttrs(0, reduction))));
            fixtures.add(nonPointwise("scatter-nd-" + reduction.name().toLowerCase(),
                    CpuScatterLoweringTest.context(new Operation(ScatterNdKind.SCATTER_ND,
                            new ScatterNdAttrs(1, reduction)), List.of(0, 1, 2),
                            List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)),
                                    CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 5, 1)),
                                    CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 5, 4))),
                            CpuScatterLoweringTest.desc(DataType.FLOAT64, Shape.of(2, 3, 4)))));
        }
        fixtures.add(nonPointwise("scatter-add", CpuScatterLoweringTest.context(new Operation(
                AxisScatterKind.SCATTER_ADD, new IndexAxisAttrs(1)), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(5, 2)),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 5, 2, 4))),
                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)))));
        List<DataType> scanTypes = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT64, DataType.INT32);
        int scanType = 0;
        for (CumulativeScanKind kind : CumulativeScanKind.values()) for (boolean exclusive : List.of(false, true))
            for (boolean reverse : List.of(false, true)) {
                DataType type = scanTypes.get(scanType++ % scanTypes.size());
                fixtures.add(nonPointwise("scan-" + kind.name().toLowerCase() + '-' + exclusive + '-' + reverse,
                        CpuScanLoweringTest.context(kind, type, Shape.of(2, 3), 1, exclusive, reverse)));
            }
        for (AggregateReductionKind kind : List.of(AggregateReductionKind.MIN,
                AggregateReductionKind.MAX, AggregateReductionKind.ALL, AggregateReductionKind.ANY,
                AggregateReductionKind.SUM, AggregateReductionKind.MEAN, AggregateReductionKind.PROD)) {
            DataType type = kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY
                    ? DataType.BOOL : kind == AggregateReductionKind.MEAN ? DataType.FLOAT32 : DataType.INT64;
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-full",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8),
                            NoOperationAttrs.INSTANCE, Shape.scalar())));
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-axis",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(4, 8),
                            new AxisReductionAttrs(1, false), Shape.of(4))));
            fixtures.add(nonPointwise("aggregate-" + kind.name().toLowerCase() + "-multi",
                    CpuAggregateLoweringTest.context(kind, type, Shape.of(2, 4, 8),
                            new MultiAxisReductionAttrs(List.of(0, 2), false), Shape.of(4))));
        }
        fixtures.add(nonPointwise("sum-to-shape", CpuAggregateLoweringTest.context(
                AggregateReductionKind.SUM, DataType.FLOAT64, Shape.of(2, 3, 4),
                new SumToShapeAttrs(Shape.of(3, 1)), Shape.of(3, 1))));
        fixtures.add(nonPointwise("initial-state", CpuRandomLoweringTest.initialContext(0x1234L, -7L)));
        assertEquals(60, fixtures.size());
        return List.copyOf(fixtures);
    }

    static List<NonPointwiseFixture> specializedGeneratedFixtures() {
        return List.of(
                // Arg extrema is a generated reduction family with a numeric input and an
                // INT64 result role; it must not be silently covered by ordinary aggregate
                // witnesses, whose output role is the input type.
                nonPointwise("arg-min", CpuArgExtremaLoweringTest.context(
                        AggregateReductionKind.ARG_MIN, DataType.FLOAT32, Shape.of(2, 3, 4),
                        1, false, ArgExtremaTiePolicy.FIRST_INDEX)),
                nonPointwise("arg-max", CpuArgExtremaLoweringTest.context(
                        AggregateReductionKind.ARG_MAX, DataType.INT64, Shape.of(2, 3, 4),
                        1, true, ArgExtremaTiePolicy.LAST_INDEX)),
                nonPointwise("masked-sum", CpuMaskedReductionLoweringTest.context(
                        AggregateReductionKind.SUM, DataType.FLOAT32, Shape.of(2, 3, 4), Shape.of(3, 1), 1)),
                nonPointwise("masked-mean", CpuMaskedReductionLoweringTest.context(
                        AggregateReductionKind.MEAN, DataType.FLOAT64, Shape.of(2, 3, 4), Shape.of(2, 1, 4), 1)),
                nonPointwise("advanced-log-sum-exp", advancedContext(AggregateReductionKind.LOG_SUM_EXP)),
                nonPointwise("advanced-variance", advancedContext(AggregateReductionKind.VARIANCE)),
                nonPointwise("advanced-standard-deviation", advancedContext(AggregateReductionKind.STANDARD_DEVIATION)),
                nonPointwise("advanced-l1-norm", advancedContext(AggregateReductionKind.L1_NORM)),
                nonPointwise("advanced-l2-norm", advancedContext(AggregateReductionKind.L2_NORM)),
                nonPointwise("softmax", CpuSoftmaxLoweringTest.context(SoftmaxKind.SOFTMAX,
                        DataType.FLOAT32, Shape.of(2, 3, 4), 1)),
                nonPointwise("log-softmax", CpuSoftmaxLoweringTest.context(SoftmaxKind.LOG_SOFTMAX,
                        DataType.FLOAT64, Shape.of(2, 3, 4), 1)),
                nonPointwise("layer", CpuTrailingNormalizationLoweringTest.context(true, false,
                        DataType.FLOAT32, Shape.of(2, 3), Shape.of(3), List.of(0))),
                nonPointwise("layer-affine", CpuTrailingNormalizationLoweringTest.context(true, true,
                        DataType.FLOAT64, Shape.of(2, 3), Shape.of(3), List.of(0, 1, 2))),
                nonPointwise("rms", CpuTrailingNormalizationLoweringTest.context(false, false,
                        DataType.BFLOAT16, Shape.of(2, 3), Shape.of(3), List.of(0))),
                nonPointwise("rms-scaled", CpuTrailingNormalizationLoweringTest.context(false, true,
                        DataType.FLOAT32, Shape.of(2, 3), Shape.of(3), List.of(0, 1))),
                nonPointwise("batch-norm-inference", CpuBatchNormInferenceLoweringTest.context(
                        List.of(DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
                                DataType.FLOAT32), Shape.of(2, 3, 4), 1, List.of(0, 1, 2, 3, 4))),
                nonPointwise("batch-norm-training", CpuBatchNormTrainingLoweringTest.context(
                        Shape.of(2, 3, 4), 1)),
                nonPointwise("conv2d", CpuConv2dLoweringTest.context(List.of(DataType.FLOAT32,
                        DataType.FLOAT32, DataType.FLOAT32), Shape.of(1, 2, 4, 4), Shape.of(2, 2, 3, 3),
                        Shape.of(1, 2, 2, 2), Conv2dAttrs.defaults(), null)),
                nonPointwise("conv3d", CpuConv3dLoweringTest.context(List.of(DataType.FLOAT32,
                        DataType.FLOAT32), Shape.of(1, 2, 4, 4, 4), Shape.of(2, 2, 3, 3, 3),
                        Shape.of(1, 2, 2, 2, 2), Conv3dAttrs.defaults(), null)),
                nonPointwise("max-pool2d", CpuPool2dLoweringTest.context(Pool2dKind.MAX_POOL2D,
                        new MaxPool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false), DataType.FLOAT32,
                        Shape.of(1, 2, 3, 3), Shape.of(1, 2, 2, 2))),
                nonPointwise("average-pool2d", CpuPool2dLoweringTest.context(Pool2dKind.AVERAGE_POOL2D,
                        new AveragePool2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false), DataType.FLOAT64,
                        Shape.of(1, 2, 3, 3), Shape.of(1, 2, 2, 2))),
                nonPointwise("max-pool3d", CpuPool3dLoweringTest.context(Pool3dKind.MAX_POOL3D,
                        new MaxPool3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false), DataType.FLOAT32,
                        Shape.of(1, 2, 3, 3, 3), Shape.of(1, 2, 2, 2, 2))),
                nonPointwise("average-pool3d", CpuPool3dLoweringTest.context(Pool3dKind.AVERAGE_POOL3D,
                        new AveragePool3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false), DataType.FLOAT64,
                        Shape.of(1, 2, 3, 3, 3), Shape.of(1, 2, 2, 2, 2))),
                nonPointwise("matmul", CpuMatmulLoweringTest.context(DataType.FLOAT32, Shape.of(2, 3),
                        Shape.of(3, 4), Shape.of(2, 4))),
                nonPointwise("attention", attentionContext(true)),
                lossFixture("mse", new Operation(LossKind.MEAN_SQUARED_ERROR,
                        new MeanSquaredErrorAttrs(LossReduction.MEAN)), List.of(0, 1),
                        List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3)),
                                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())),
                lossFixture("dense-cross-entropy", new Operation(
                        LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN)), List.of(0, 1),
                        List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())),
                lossFixture("index-cross-entropy", new Operation(
                        LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                        new IndexCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.MEAN, Optional.empty())),
                        List.of(0, 1), List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.of(2, 3, 4)),
                                CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(2, 4))),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32, Shape.scalar())));
    }

    private static NonPointwiseFixture lossFixture(String id, Operation operation, List<Integer> roles,
            List<TensorDescriptor> inputs, TensorDescriptor output) {
        return nonPointwise(id, CpuScatterLoweringTest.context(operation, roles, inputs, output));
    }

    static PrepareContext<CpuPartitionAnalysisInputs> scatter(AxisScatterKind kind,
            ScatterElementsAttrs attrs) {
        return CpuScatterLoweringTest.context(new Operation(kind, attrs), List.of(0, 1, 2),
                List.of(CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)),
                        CpuScatterLoweringTest.desc(DataType.INT32, Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(4))),
                CpuScatterLoweringTest.desc(DataType.INT64, Shape.of(8)));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> advancedContext(AggregateReductionKind kind) {
        var attrs = kind == AggregateReductionKind.VARIANCE || kind == AggregateReductionKind.STANDARD_DEVIATION
                ? new StatisticalReductionAttrs(List.of(1), false, 1)
                : new MultiAxisReductionAttrs(List.of(1), false);
        return CpuAggregateLoweringTest.context(kind, DataType.FLOAT64, Shape.of(4, 8), attrs, Shape.of(4));
    }

    static NonPointwiseFixture nonPointwise(String id,
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new NonPointwiseFixture(id, context);
    }

    record NonPointwiseFixture(String id, PrepareContext<CpuPartitionAnalysisInputs> context) { }

    static List<PrepareContext<CpuPartitionAnalysisInputs>> coverageFixtureContexts() {
        var contexts = new ArrayList<PrepareContext<CpuPartitionAnalysisInputs>>();
        for (PointwiseFixture fixture : generalPointwiseFixtures()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : nonPointwiseFixtures()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : ordinaryFixtureSeeds()) contexts.add(fixture.context());
        for (NonPointwiseFixture fixture : specializedGeneratedFixtures()) contexts.add(fixture.context());
        contexts.add(CpuRandomLoweringTest.initialContext(0x1234L, -7L));
        return List.copyOf(contexts);
    }

    static List<PointwiseFixture> generalPointwiseFixtures() {
        Shape shape = Shape.of(8);
        var forms = new ArrayList<PointwiseFixture>();
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) for (DataType type : numericTypes())
            if (type.isFloating() || (kind != BinaryArithmeticKind.DIV && kind != BinaryArithmeticKind.POW))
                forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(type, type), type, CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) for (DataType type : floatingTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type), type,
                    kind == UnaryElementwiseKind.GELU ? CpuPointwiseOpcode.GELU_EXACT
                            : CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (FloatingClassificationKind kind : FloatingClassificationKind.values()) for (DataType type : floatingTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type), DataType.BOOL,
                    CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) for (DataType type : numericTypes())
            forms.add(fixture(kind + "_" + type, new Operation(kind, NoOperationAttrs.INSTANCE), List.of(type, type), DataType.BOOL,
                    CpuPointwiseOpcode.valueOf(kind.name()), shape));
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) forms.add(fixture(kind.name(),
                new Operation(kind, NoOperationAttrs.INSTANCE), java.util.Collections.nCopies(
                        kind == BooleanLogicalKind.NOT ? 1 : 2, DataType.BOOL), DataType.BOOL,
                kind == BooleanLogicalKind.NOT ? CpuPointwiseOpcode.LOGICAL_NOT
                        : CpuPointwiseOpcode.valueOf("LOGICAL_" + kind.name()), shape));
        for (DataType type : floatingTypes()) forms.add(fixture("WHERE_" + type,
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE), List.of(DataType.BOOL, type, type), type,
                CpuPointwiseOpcode.WHERE, shape));
        for (DataType source : DataType.values()) for (DataType target : DataType.values()) forms.add(fixture(
                "CAST_" + source + "_" + target, new Operation(CastKind.CAST, new CastAttrs(target)),
                List.of(source), target, CpuPointwiseOpcode.CAST, shape));
        return List.copyOf(forms);
    }

    private static List<DataType> floatingTypes() { return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16); }
    private static List<DataType> numericTypes() { return List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.INT64, DataType.INT32); }

    static List<PointwiseRejectedCandidate> rejectedPointwiseCandidates() {
        Shape shape = Shape.of(8); Shape incompatible = Shape.of(7);
        var result = new ArrayList<PointwiseRejectedCandidate>();
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            for (DataType[] pair : mixedNumericPairs(kind == BinaryArithmeticKind.DIV || kind == BinaryArithmeticKind.POW))
                result.add(rejected(kind + "/mixed-" + pair[0] + '-' + pair[1], new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(descriptor(pair[0], shape), descriptor(pair[1], shape)), descriptor(promote(pair[0], pair[1]), shape), "MIXED_NUMERIC_INPUTS"));
            result.add(rejected(kind + "/bool-role", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.BOOL, shape), descriptor(DataType.BOOL, shape)), descriptor(DataType.BOOL, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
            result.add(rejected(kind + "/shape", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, incompatible)), descriptor(DataType.FLOAT32, shape), "SHAPE_INCOMPATIBLE"));
        }
        for (BinaryComparisonKind kind : BinaryComparisonKind.values()) {
            for (DataType[] pair : mixedNumericPairs(false)) result.add(rejected(kind + "/mixed-" + pair[0] + '-' + pair[1],
                    new Operation(kind, NoOperationAttrs.INSTANCE), List.of(descriptor(pair[0], shape), descriptor(pair[1], shape)),
                    descriptor(DataType.BOOL, shape), "MIXED_NUMERIC_INPUTS"));
            result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape)), descriptor(DataType.FLOAT32, shape), "INVALID_RESULT_ROLE"));
        }
        for (DataType[] pair : mixedFloatingPairs()) result.add(rejected("WHERE/mixed-" + pair[0] + '-' + pair[1],
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE), List.of(descriptor(DataType.BOOL, shape),
                descriptor(pair[0], shape), descriptor(pair[1], shape)), descriptor(promote(pair[0], pair[1]), shape), "MIXED_FLOATING_BRANCHES"));
        result.add(rejected("WHERE/condition-role", new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape), descriptor(DataType.FLOAT32, shape)),
                descriptor(DataType.FLOAT32, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape)), descriptor(DataType.BOOL, shape), "INVALID_RESULT_ROLE"));
        for (FloatingClassificationKind kind : FloatingClassificationKind.values()) result.add(rejected(kind + "/result", new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(descriptor(DataType.FLOAT32, shape)), descriptor(DataType.FLOAT32, shape), "INVALID_RESULT_ROLE"));
        for (BooleanLogicalKind kind : BooleanLogicalKind.values()) result.add(rejected(kind + "/numeric-role", new Operation(kind, NoOperationAttrs.INSTANCE),
                java.util.Collections.nCopies(kind == BooleanLogicalKind.NOT ? 1 : 2, descriptor(DataType.INT32, shape)), descriptor(DataType.BOOL, shape), "INVALID_BOOLEAN_NUMERIC_ROLE"));
        return List.copyOf(result);
    }

    private static List<DataType[]> mixedNumericPairs(boolean floatingOnly) { return pairs(floatingOnly ? floatingTypes() : numericTypes()); }
    private static List<DataType[]> mixedFloatingPairs() { return pairs(floatingTypes()); }
    private static List<DataType[]> pairs(List<DataType> types) { var pairs = new ArrayList<DataType[]>(); for (DataType left : types) for (DataType right : types) if (left != right && left.category() == right.category()) pairs.add(new DataType[] {left, right}); return pairs; }
    private static DataType promote(DataType left, DataType right) { return io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteNumeric(left, right); }
    private static TensorDescriptor descriptor(DataType type, Shape shape) { return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false); }
    private static PointwiseRejectedCandidate rejected(String id, Operation operation, List<TensorDescriptor> inputs, TensorDescriptor output, String reason) { return new PointwiseRejectedCandidate(id, operation, inputs, output, reason); }
    record PointwiseRejectedCandidate(String id, Operation operation, List<TensorDescriptor> inputs,
            TensorDescriptor output, String reason) {
        CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture coverageFixture() {
            return new CpuGeneratedCoverageEvidenceRegistry.ProviderRejectedFixture(id, operation.kind().name(),
                    operation, inputs, List.of(output), reason);
        }
    }

    private static PointwiseFixture fixture(String id, Operation operation, List<DataType> inputs,
            DataType output, CpuPointwiseOpcode opcode, Shape shape) {
        return new PointwiseFixture(id, operation, inputs, output, opcode, shape);
    }

    private static String classFileHash(byte[] bytes) throws java.security.NoSuchAlgorithmException {
        return java.util.HexFormat.of().formatHex(java.security.MessageDigest.getInstance("SHA-256")
                .digest(bytes));
    }

    record PointwiseFixture(String id, Operation operation, List<DataType> inputTypes,
            DataType outputType, CpuPointwiseOpcode opcode, Shape shape) {
        List<DataType> boundaryTypes() {
            var types = new ArrayList<>(inputTypes); types.add(outputType); return List.copyOf(types);
        }
        List<TensorDescriptor> inputDescriptors() { return inputTypes.stream()
                .map(type -> descriptor(type, shape)).toList(); }
        TensorDescriptor outputDescriptor() { return descriptor(outputType, shape); }
        PrepareContext<CpuPartitionAnalysisInputs> context() {
            return context(LayoutDescriptor.contiguous(shape), List.of(),
                    CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT,
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
        }
        PrepareContext<CpuPartitionAnalysisInputs> context(LayoutDescriptor inputLayout,
                List<CpuKernelSpecialization.CarrierAccess> carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
                CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) {
            var inputs = java.util.stream.IntStream.range(0, inputTypes.size()).mapToObj(ValueId::new).toList();
            ValueId output = new ValueId(inputTypes.size());
            var node = new CompiledNode(new NodeId(0), operation, inputs, List.of(output));
            var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
            var descriptors = inputTypes.stream().map(type -> new TensorDescriptor(type, shape,
                    Optional.of(inputLayout), false)).collect(java.util.stream.Collectors.toCollection(ArrayList::new));
            descriptors.add(outputDescriptor());
            var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
            for (int index = 0; index < descriptors.size(); index++) {
                ValueId id = new ValueId(index); TensorDescriptor descriptor = descriptors.get(index);
                boolean result = index == inputTypes.size(); values.add(new GraphValue(id, descriptor));
                memory.add(new LogicalMemoryRequirement(id, descriptor, result ? Optional.of(partition)
                        : Optional.empty(), result ? List.of() : List.of(partition), result));
            }
            return new PrepareContext<>(partition, List.of(node), values, memory, java.util.Map.of(),
                    new CpuPartitionAnalysisInputs(false, carriers, execution, materializationPolicy));
        }
        private static TensorDescriptor descriptor(DataType type, Shape shape) {
            return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
        }
    }

    static List<PointwiseCandidate> pointwiseCandidates() {
        var result = new ArrayList<PointwiseCandidate>();
        var general = LayoutDescriptor.of(Shape.of(8), new long[] {2}, 0, true);
        var materialization = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 1, 20, 1, 3, 1_000_000, 1, 1);
        for (PointwiseFixture fixture : generalPointwiseFixtures()) {
            result.add(candidate(fixture, "heap-contiguous-scalar", "HEAP", "CONTIGUOUS", "DIRECT", "SCALAR",
                    carrierPattern(fixture, false, false), LayoutDescriptor.contiguous(fixture.shape()), scalar(1),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "segment-contiguous-vector", "SEGMENT", "CONTIGUOUS", "DIRECT", "VECTOR",
                    carrierPattern(fixture, true, true), LayoutDescriptor.contiguous(fixture.shape()), vector(1),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "mixed-contiguous-parallel-vector", "MIXED", "CONTIGUOUS", "DIRECT", "PARALLEL_VECTOR",
                    carrierPattern(fixture, false, true), LayoutDescriptor.contiguous(fixture.shape()), vector(4),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "heap-general-parallel-scalar", "HEAP", "GENERAL", "DIRECT", "PARALLEL_SCALAR",
                    carrierPattern(fixture, false, false), general, scalar(4),
                    CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            result.add(candidate(fixture, "heap-general-materialization-candidate", "HEAP", "GENERAL", "MATERIALIZATION_CANDIDATE", "SCALAR",
                    carrierPattern(fixture, false, false), general, scalar(1), materialization));
        }
        return List.copyOf(result);
    }

    /**
     * Exhaustive carrier-role witnesses derived from the same Model occurrences as the 845-row
     * selected matrix.  They are execution witnesses, not additional inventory owners: each
     * ordered input/output position may use its typed heap carrier or a MemorySegment.  Keeping
     * them separate preserves the selected-matrix denominator while making unsupported role
     * combinations fail at real provider/preparer/generator boundaries.
     */
    static List<PointwiseCandidate> pointwiseCarrierRoleWitnesses() {
        var witnesses = new ArrayList<PointwiseCandidate>();
        for (PointwiseFixture fixture : generalPointwiseFixtures()) {
            int positions = fixture.inputTypes().size() + 1;
            for (int mask = 0; mask < (1 << positions); mask++) {
                var carriers = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
                for (int position = 0; position < positions; position++) {
                    boolean segment = (mask & (1 << position)) != 0;
                    carriers.add(segment ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                            : heapCarrier(position == fixture.inputTypes().size() ? fixture.outputType()
                                    : fixture.inputTypes().get(position)));
                }
                String roles = Integer.toBinaryString(mask | (1 << positions)).substring(1);
                witnesses.add(candidate(fixture, "carrier-roles-" + roles, "ROLE_" + roles,
                        "CONTIGUOUS", "DIRECT", "SCALAR", List.copyOf(carriers),
                        LayoutDescriptor.contiguous(fixture.shape()), scalar(1),
                        CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED));
            }
        }
        return List.copyOf(witnesses);
    }

    private static PointwiseCandidate candidate(PointwiseFixture fixture, String suffix, String carrier,
            String layout, String materialization, String strategy,
            List<CpuKernelSpecialization.CarrierAccess> carriers, LayoutDescriptor inputLayout,
            CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        return new PointwiseCandidate(fixture.id() + '/' + suffix, fixture.id(), fixture,
                PointwiseDisposition.SUPPORTED, carrier, layout, materialization, strategy,
                carriers, inputLayout, execution, policy, materialization.equals("MATERIALIZATION_CANDIDATE")
                        && (fixture.operation().kind() == WhereSelectionKind.WHERE || fixture.inputTypes().stream()
                                .noneMatch(type -> type == DataType.BFLOAT16)));
    }

    private static List<CpuKernelSpecialization.CarrierAccess> carrierPattern(PointwiseFixture fixture,
            boolean segments, boolean outputSegments) {
        var pattern = new ArrayList<CpuKernelSpecialization.CarrierAccess>();
        for (DataType type : fixture.inputTypes()) pattern.add(segments ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                : heapCarrier(type));
        pattern.add(outputSegments ? CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT : heapCarrier(fixture.outputType()));
        return List.copyOf(pattern);
    }

    static CpuKernelSpecialization.CarrierAccess heapCarrier(DataType type) {
        return switch (type) {
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case BFLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case BOOL -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
        };
    }

    static CpuPartitionAnalysisInputs.PortableExecutionConfig scalar(int parallelism) {
        return new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                parallelism, parallelism, 1);
    }

    static CpuPartitionAnalysisInputs.PortableExecutionConfig vector(int parallelism) {
        return new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE,
                parallelism, parallelism, 1);
    }

    private static CpuPartitionPreparationPlan.ExecutionStrategy kernelStrategy(
            CpuPartitionPreparationPlan.ExecutionStrategy selected) {
        return selected.orchestration()
                == CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.PARALLEL
                ? new CpuPartitionPreparationPlan.ExecutionStrategy(selected.compute(),
                        CpuPartitionPreparationPlan.ExecutionStrategy.Orchestration.SINGLE_THREAD)
                : selected;
    }

    private enum PointwiseDisposition { SUPPORTED }

    record PointwiseCandidate(String id, String form, PointwiseFixture fixture,
            PointwiseDisposition disposition, String carrierAxis, String layoutAxis,
            String materializationAxis, String requestedStrategy,
            List<CpuKernelSpecialization.CarrierAccess> inputs, LayoutDescriptor inputLayout,
            CpuPartitionAnalysisInputs.PortableExecutionConfig execution,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy,
            boolean materializationSelected) { }

    private static byte[] generated(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst()
                .portablePlan();
        return new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr());
    }

    private static byte[] advanced(AggregateReductionKind kind) {
        var attrs = kind == AggregateReductionKind.VARIANCE
                || kind == AggregateReductionKind.STANDARD_DEVIATION
                ? new StatisticalReductionAttrs(List.of(1), false, 1)
                : new MultiAxisReductionAttrs(List.of(1), false);
        var base = CpuAggregateLoweringTest.context(kind, DataType.FLOAT64,
                Shape.of(4, 8), attrs, Shape.of(4));
        return generated(new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY))));
    }

    private static byte[] pointwise(boolean vector, boolean segment) {
        var denseRead = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var denseWrite = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
        var ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, denseRead),
                new CpuKernelIr.Value(1, DataType.FLOAT32, CpuKernelIr.Value.Kind.OUTPUT, denseWrite)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.TANH, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
        var specialization = new CpuKernelSpecialization(
                io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint
                        .fromHex(ir.structuralKey()),
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                vector ? CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR
                        : CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32, DataType.FLOAT32),
                segment ? List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                        : List.of(CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY,
                                CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY),
                vector ? jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize() : 0,
                -1);
        return new CpuClassFileKernelGenerator().generateClassBytes(specialization, ir);
    }

    private static void assertClosedClass(byte[] bytes) {
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertEquals(1, model.methods().size()),
                () -> assertEquals("invoke", model.methods().getFirst().methodName().stringValue()),
                () -> assertTrue(model.methods().getFirst().flags().has(AccessFlag.STATIC)),
                () -> assertTrue(model.methods().getFirst().methodTypeSymbol().descriptorString()
                        .endsWith("[JJJ)V")),
                () -> assertFalse(model.methods().getFirst().methodTypeSymbol().descriptorString()
                        .contains("Ljava/lang/Object;")),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(
                        model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(members.stream().noneMatch(member -> member.type().stringValue()
                        .contains("Ljava/lang/Object;") || member.owner().asInternalName()
                        .startsWith("java/lang/reflect/") || member.owner().asInternalName()
                        .startsWith("java/lang/invoke/") || member.owner().asInternalName()
                        .startsWith("java/util/"))), () -> assertTrue(members.stream()
                        .filter(member -> member.owner().asInternalName()
                                .startsWith("io/github/pho001/synaptik/"))
                        .allMatch(member -> member.owner().asInternalName().equals(
                                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath")),
                        () -> "unexpected generated helper ownership: " + members.stream()
                                .map(member -> member.owner().asInternalName()).distinct().toList()));
    }

    static void assertClosedSpecializedClass(byte[] bytes, CpuKernelSpecialization specialization) {
        var model = ClassFile.of().parse(bytes);
        var members = java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                .filter(MemberRefEntry.class::isInstance).map(MemberRefEntry.class::cast).toList();
        assertAll(
                () -> assertTrue(model.fields().isEmpty()),
                () -> assertTrue(model.methods().stream().anyMatch(method -> method.methodName()
                        .stringValue().equals("invoke") && method.flags().has(AccessFlag.STATIC)
                        && method.methodTypeSymbol().descriptorString().equals(
                                specialization.entryType().descriptorString()))),
                () -> assertTrue(model.methods().stream().allMatch(method -> method.flags()
                        .has(AccessFlag.STATIC)), "every generated helper method must be static"),
                () -> assertEquals(0, model.constantPool().bootstrapMethodCount()),
                () -> assertTrue(java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                        .noneMatch(MethodHandleEntry.class::isInstance)),
                () -> assertTrue(java.util.stream.StreamSupport.stream(model.constantPool().spliterator(), false)
                        .noneMatch(DynamicConstantPoolEntry.class::isInstance)),
                () -> assertTrue(members.stream().noneMatch(member -> member.type().stringValue()
                        .contains("Ljava/lang/Object;") || member.owner().asInternalName()
                        .startsWith("java/lang/reflect/") || member.owner().asInternalName()
                        .startsWith("java/lang/invoke/") || member.owner().asInternalName()
                        .startsWith("java/util/"))), () -> assertTrue(members.stream()
                        .filter(member -> member.owner().asInternalName()
                                .startsWith("io/github/pho001/synaptik/"))
                        .allMatch(member -> member.owner().asInternalName().equals(
                                "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/CpuVectorMath")
                                || member.owner().asInternalName().startsWith(
                                        "io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/Generated_")),
                        () -> "unexpected generated helper ownership: " + members.stream()
                                .map(member -> member.owner().asInternalName()).distinct().toList()));
    }

    private static void assertSegmentLayoutsHoisted(Representative representative) {
        var invokes = ClassFile.of().parse(representative.bytes()).methods().getFirst().code()
                .orElseThrow().elementStream().filter(InvokeInstruction.class::isInstance)
                .map(InvokeInstruction.class::cast).toList();
        long nativeOrder = invokes.stream().filter(instruction ->
                instruction.owner().asInternalName().equals("java/nio/ByteOrder")
                        && instruction.name().stringValue().equals("nativeOrder")).count();
        long withOrder = invokes.stream().filter(instruction ->
                instruction.owner().asInternalName().equals("java/lang/foreign/ValueLayout")
                        && instruction.name().stringValue().equals("withOrder")).count();
        int firstSegmentAccess = java.util.stream.IntStream.range(0, invokes.size())
                .filter(index -> invokes.get(index).owner().asInternalName()
                        .equals("java/lang/foreign/MemorySegment")
                        && (invokes.get(index).name().stringValue().equals("get")
                            || invokes.get(index).name().stringValue().equals("set")))
                .findFirst().orElse(invokes.size());
        int lastLayoutConstruction = java.util.stream.IntStream.range(0, invokes.size())
                .filter(index -> invokes.get(index).name().stringValue().equals("nativeOrder")
                        || invokes.get(index).name().stringValue().equals("withOrder"))
                .reduce((left, right) -> right).orElse(-1);
        long expectedLayoutConstruction = representative.directNativeLayouts() ? 0
                : representative.orderedLayoutCount();
        assertAll(
                () -> assertEquals(expectedLayoutConstruction, nativeOrder),
                () -> assertEquals(expectedLayoutConstruction, withOrder),
                () -> assertTrue(lastLayoutConstruction < firstSegmentAccess));
    }

    private record Representative(byte[] bytes, int orderedLayoutCount, boolean directNativeLayouts) {
        private Representative(byte[] bytes, int orderedLayoutCount) {
            this(bytes, orderedLayoutCount, false);
        }
    }
}
