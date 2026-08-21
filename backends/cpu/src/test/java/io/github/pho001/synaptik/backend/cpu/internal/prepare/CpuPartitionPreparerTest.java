package io.github.pho001.synaptik.backend.cpu.internal.prepare;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuFoldLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAggregateLoweringTest;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.ShapeBroadcast;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.LongVector;
import jdk.incubator.vector.ByteVector;

public class CpuPartitionPreparerTest {
    @Test void floatingNumericalAggregateDeclaresExactRunOwnedStateBeforeAssignment() {
        var base = CpuAggregateLoweringTest.context(AggregateReductionKind.SUM, DataType.FLOAT64,
                Shape.of(2, 3), new AxisReductionAttrs(1, false), Shape.of(2));
        var config = new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.DOUBLE_ARRAY, CarrierAccess.DOUBLE_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var plan = analysis.plan(); var geometry = plan.aggregateGeometry().orElseThrow();
        assertAll(
                () -> assertEquals(3, analysis.requirements().size()),
                () -> assertEquals(CpuPartitionPreparationPlan.WorkspaceUse.AGGREGATE_EXACT_STATE,
                        plan.workspaceUse()),
                () -> assertEquals(geometry.workspaceBytes(2),
                        plan.workspaceDeclaration().orElseThrow().byteSize()),
                () -> assertEquals(8, plan.workspaceDeclaration().orElseThrow().byteAlignment()),
                () -> assertTrue(plan.units().getFirst().portablePlan().specialization()
                        .scratchParameter()));
    }
    @Test void randomPlansDeclareExactBoundariesScalarStrategyAndZeroWorkspace() {
        var initial = new CpuPartitionPreparer().analyze(
                CpuRandomLoweringTest.initialContext(1, 2)).plan();
        var dropout = new CpuPartitionPreparer().analyze(
                CpuRandomLoweringTest.dropoutContext(DataType.FLOAT64, Shape.of(16), .2)).plan();
        assertAll(() -> assertEquals(1, initial.bufferDeclarations().size()),
                () -> assertEquals(5, dropout.bufferDeclarations().size()),
                () -> assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        initial.executionStrategy()),
                () -> assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                        dropout.executionStrategy()),
                () -> assertTrue(initial.workspaceDeclaration().isEmpty()),
                () -> assertTrue(dropout.workspaceDeclaration().isEmpty()),
                () -> assertTrue(dropout.randomGeometry().isPresent()),
                () -> assertEquals(38, io.github.pho001.synaptik.backend.cpu.internal.cache
                        .CpuGeneratorSchema.CURRENT_VERSION));
    }
    @Test void foldDeclaresExactlyTwoBuffersOneArtifactAndNoWorkspaceOrMaterialization() {
        var base = CpuFoldLoweringTest.context(new Operation(WindowTransformKind.FOLD_AXIS,
                new FoldAxisAttrs(0, 5, 1)), DataType.FLOAT32, Shape.of(3, 3), Shape.of(5));
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 4, 4, 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY, CarrierAccess.FLOAT_ARRAY), config));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var plan = analysis.plan();
        assertAll(() -> assertEquals(2, analysis.requirements().size()),
                () -> assertEquals(2, plan.bufferDeclarations().size()),
                () -> assertEquals(1, plan.units().size()),
                () -> assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR,
                        plan.executionStrategy()),
                () -> assertEquals(4, plan.selectedRangeCount()),
                () -> assertTrue(plan.foldGeometry().isPresent()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertTrue(plan.materialization().isEmpty()),
                () -> assertFalse(plan.units().getFirst().portablePlan().specialization()
                        .scratchParameter()));
    }

    @Test void scatterDeclaresExactUniqueBuffersStrategyAndOnlyEligibleProductScratch() {
        var base=CpuScatterLoweringTest.context(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.MUL)),List.of(0,1,2),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.of(4)),
                        CpuScatterLoweringTest.desc(DataType.INT64,Shape.of(8)),
                        CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.of(8))),
                CpuScatterLoweringTest.desc(DataType.FLOAT32,Shape.of(4)));
        var config=new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                ComputePreference.VECTOR_IF_ELIGIBLE,4,4,1);
        var context=new PrepareContext<>(base.partition(),base.nodes(),base.values(),
                base.memoryRequirements(),base.constants(),new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.FLOAT_ARRAY,CarrierAccess.LONG_ARRAY,
                                CarrierAccess.FLOAT_ARRAY,CarrierAccess.FLOAT_ARRAY),config));
        var analysis=new CpuPartitionPreparer().analyze(context);
        var plan=analysis.plan();
        assertAll(() -> assertEquals(4,plan.bufferDeclarations().size()),
                () -> assertEquals(1,plan.workspaceDeclaration().stream().count()),
                () -> assertEquals(4,plan.selectedRangeCount()),
                () -> assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR,
                        plan.executionStrategy()),
                () -> assertEquals(CpuPartitionPreparationPlan.WorkspaceUse.SCATTER_PRODUCT,
                        plan.workspaceUse()),
                () -> assertEquals(plan.scatterGeometry().orElseThrow().workspaceBytes(4),
                        plan.workspaceDeclaration().orElseThrow().byteSize()),
                () -> assertEquals(5,analysis.requirements().size()));
    }
    @Test void indexingDeclaresUniqueInputsThenOutputWithOneUnitAndNoWorkspace() {
        var distinct = new CpuPartitionPreparer().analyze(CpuIndexingLoweringTest.context(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)), List.of(0, 1),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2)),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2))));
        var deduplicated = new CpuPartitionPreparer().analyze(CpuIndexingLoweringTest.context(
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)), List.of(0, 0),
                List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2))));
        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                        distinct.requirements().stream().map(requirement ->
                                ((io.github.pho001.synaptik.prepare.analysis
                                        .PreparationResourceRequirement.Buffer) requirement)
                                        .valueId()).toList()),
                () -> assertEquals(1, distinct.plan().units().size()),
                () -> assertTrue(distinct.plan().workspaceDeclaration().isEmpty()),
                () -> assertTrue(distinct.plan().materialization().isEmpty()),
                () -> assertEquals(-1, distinct.plan().units().getFirst().portablePlan()
                        .specialization().materializedSourcePosition()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)),
                        deduplicated.requirements().stream().map(requirement ->
                                ((io.github.pho001.synaptik.prepare.analysis
                                        .PreparationResourceRequirement.Buffer) requirement)
                                        .valueId()).toList()),
                () -> assertEquals(2, deduplicated.plan().accessBindings().size()));
    }

    @Test void selectsAllFourStrategiesAndFallsBackFromIneligibleVectorGeometry() {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        var pattern = CpuPartitionAnalysisInputs.DEFAULT.carrierPattern();
        var vector = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1);
        var parallelScalar = new PortableExecutionConfig(ComputePreference.SCALAR, 4, 2, 1);
        var parallelVector = new PortableExecutionConfig(
                ComputePreference.VECTOR_IF_ELIGIBLE, 4, 2, 1);
        var descriptor = descriptor(Shape.of(lanes * 2),
                LayoutDescriptor.contiguous(Shape.of(lanes * 2)));
        Shape generalShape = Shape.of(2, lanes);
        var general = descriptor(generalShape,
                LayoutDescriptor.of(generalShape, new long[] {1, 2}, 0, true));
        var denseGeneral = descriptor(generalShape, LayoutDescriptor.contiguous(generalShape));
        assertAll(
                () -> assertEquals("scalar", analyze(Shape.of(lanes * 2)).plan()
                        .executionStrategy().toString()),
                () -> assertEquals("vector", analyze(descriptor, descriptor, descriptor, descriptor,
                        new CpuPartitionAnalysisInputs(false, pattern, vector)).plan()
                        .executionStrategy().toString()),
                () -> assertEquals("parallel-scalar", analyze(descriptor, descriptor, descriptor,
                        descriptor, new CpuPartitionAnalysisInputs(false, pattern, parallelScalar))
                        .plan().executionStrategy().toString()),
                () -> assertEquals("parallel-vector", analyze(descriptor, descriptor, descriptor,
                        descriptor, new CpuPartitionAnalysisInputs(false, pattern, parallelVector))
                        .plan().executionStrategy().toString()),
                () -> assertEquals("parallel-scalar", analyze(general, denseGeneral, denseGeneral,
                        denseGeneral, new CpuPartitionAnalysisInputs(false, pattern, parallelVector))
                        .plan().executionStrategy().toString()));
    }

    @Test void boundsRangeCountAndRecordsExactPreferredSpecies() {
        int lanes = DoubleVector.SPECIES_PREFERRED.length();
        var config = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 8, 3,
                lanes);
        var analysis = analyze(Shape.of(lanes * 5), config);
        assertAll(
                () -> assertEquals(3, analysis.plan().selectedRangeCount()),
                () -> assertEquals(lanes, analysis.plan().minimumElementsPerWorker()),
                () -> assertEquals(DoubleVector.SPECIES_PREFERRED.vectorBitSize(),
                        analysis.plan().vectorSpeciesBitSize()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new PortableExecutionConfig(ComputePreference.SCALAR, 0, 1, 1)));
    }

    @Test void selectsPreferredFloatSpeciesForHomogeneousFloatIr() {
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        Shape shape = Shape.of(lanes + 1);
        var floatDescriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var inputs = new CpuPartitionAnalysisInputs(false,
                CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1));
        var homogeneous = analyze(floatDescriptor, floatDescriptor, floatDescriptor,
                floatDescriptor, inputs).plan();
        assertAll(
                () -> assertEquals("vector", homogeneous.executionStrategy().toString()),
                () -> assertEquals(FloatVector.SPECIES_PREFERRED.vectorBitSize(),
                        homogeneous.vectorSpeciesBitSize()));
    }

    @Test void floatChainWithOneIneligibleOpcodeFallsBackWithoutSplitting() {
        int lanes = FloatVector.SPECIES_PREFERRED.length();
        Shape shape = Shape.of(lanes * 2);
        var descriptor = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var vector = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1);
        var parallel = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1);
        var scalarPlan = new CpuPartitionPreparer().analyze(context(descriptor, descriptor,
                descriptor, descriptor, new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), vector),
                UnaryElementwiseKind.FLOOR)).plan();
        var parallelPlan = new CpuPartitionPreparer().analyze(context(descriptor, descriptor,
                descriptor, descriptor, new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), parallel),
                UnaryElementwiseKind.FLOOR)).plan();
        assertAll(
                () -> assertEquals("scalar", scalarPlan.executionStrategy().toString()),
                () -> assertEquals("parallel-scalar", parallelPlan.executionStrategy().toString()),
                () -> assertEquals(1, scalarPlan.units().size()),
                () -> assertEquals(0, scalarPlan.vectorSpeciesBitSize()));
    }

    @Test void selectsTypedIntegralAndCanonicalBoolValueVectorsInBothOrchestrationModes() {
        var rows = List.of(
                new VectorRow(DataType.INT32, BinaryArithmeticKind.ADD),
                new VectorRow(DataType.INT64, BinaryArithmeticKind.MAX),
                new VectorRow(DataType.BOOL, BooleanLogicalKind.AND));
        for (VectorRow row : rows) {
            int count = vectorLanes(row.type()) * 2;
            Shape shape = Shape.of(count);
            var descriptor = descriptor(row.type(), shape);
            var scalar = analyze(oneNodeContext(new Operation(row.kind(), NoOperationAttrs.INSTANCE),
                    List.of(descriptor, descriptor), descriptor,
                    new PortableExecutionConfig(ComputePreference.SCALAR, 1, 1, 1)));
            var parallelScalar = analyze(oneNodeContext(
                    new Operation(row.kind(), NoOperationAttrs.INSTANCE),
                    List.of(descriptor, descriptor), descriptor,
                    new PortableExecutionConfig(ComputePreference.SCALAR, 2, 2, 1)));
            var vector = analyze(oneNodeContext(new Operation(row.kind(), NoOperationAttrs.INSTANCE),
                    List.of(descriptor, descriptor), descriptor,
                    new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1)));
            var parallel = analyze(oneNodeContext(new Operation(row.kind(), NoOperationAttrs.INSTANCE),
                    List.of(descriptor, descriptor), descriptor,
                    new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1)));
            assertAll(row.type().name(),
                    () -> assertEquals("scalar", scalar.plan().executionStrategy().toString()),
                    () -> assertEquals("parallel-scalar",
                            parallelScalar.plan().executionStrategy().toString()),
                    () -> assertEquals("vector", vector.plan().executionStrategy().toString()),
                    () -> assertEquals("parallel-vector",
                            parallel.plan().executionStrategy().toString()),
                    () -> assertEquals(vectorSpeciesBits(row.type()),
                            vector.plan().vectorSpeciesBitSize()));
        }
    }

    @Test void selectsOnlyVirtualOrScalarBroadcastFloatingMaskTopologies() {
        int count = FloatVector.SPECIES_PREFERRED.length() * 2;
        var vector = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 1, 1, 1);
        var parallel = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE, 2, 2, 1);
        assertAll(
                () -> assertEquals("vector", analyze(maskWhereContext(count, vector)).plan()
                        .executionStrategy().toString()),
                () -> assertEquals("parallel-vector", analyze(maskWhereContext(count, parallel))
                        .plan().executionStrategy().toString()),
                () -> assertEquals("vector", analyze(externalWhereContext(count, true, vector))
                        .plan().executionStrategy().toString()),
                () -> assertEquals("scalar", analyze(externalWhereContext(count, false, vector))
                        .plan().executionStrategy().toString()),
                () -> assertEquals("scalar", analyze(materializedComparisonContext(count, vector))
                        .plan().executionStrategy().toString()));
    }
    @Test void formsOneFusedUnitAndDeclaresOnlyFourBoundaries() {
        var analysis = analyze(Shape.of(2, 3));
        assertAll(
                () -> assertEquals(1, analysis.plan().units().size()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2),
                                new ValueId(5)),
                        analysis.plan().boundaryValues()),
                () -> assertEquals(4, analysis.requirements().size()),
                () -> assertFalse(analysis.plan().boundaryValues().contains(new ValueId(3))),
                () -> assertFalse(analysis.plan().boundaryValues().contains(new ValueId(4))),
                () -> assertEquals("scalar", analysis.plan().executionStrategy().toString()));
    }

    @Test void failsClosedForPublishedIntermediate() {
        var context = context(Shape.of(2));
        var memory = new ArrayList<>(context.memoryRequirements());
        var old = memory.get(3);
        memory.set(3, new LogicalMemoryRequirement(old.valueId(), old.descriptor(),
                old.producerPartition(), old.consumerPartitions(), true));
        var changed = new PrepareContext<>(context.partition(), context.nodes(), context.values(),
                memory, Map.of(), context.backendInputs());
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(changed));
    }

    @Test void analysisInputsSnapshotPatternAndValidateCountBeforeLoweringConsumption() {
        var mutable = new ArrayList<>(List.of(CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT, CarrierAccess.DOUBLE_ARRAY,
                CarrierAccess.MEMORY_SEGMENT));
        var inputs = new CpuPartitionAnalysisInputs(true, mutable);
        mutable.set(0, CarrierAccess.MEMORY_SEGMENT);
        assertAll(
                () -> assertFalse(CpuPartitionAnalysisInputs.DEFAULT.loweringManifestEnabled()),
                () -> assertTrue(CpuPartitionAnalysisInputs.DEFAULT.carrierPattern().isEmpty()),
                () -> assertEquals(CarrierAccess.DOUBLE_ARRAY, inputs.carrierPattern().getFirst()),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuPartitionAnalysisInputs(false, null)),
                () -> assertThrows(NullPointerException.class,
                        () -> new CpuPartitionAnalysisInputs(false,
                                java.util.Arrays.asList(CarrierAccess.DOUBLE_ARRAY, null))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionPreparer().analyze(new PrepareContext<>(
                                context(Shape.of(2)).partition(), context(Shape.of(2)).nodes(),
                                context(Shape.of(2)).values(), context(Shape.of(2)).memoryRequirements(),
                                Map.of(), new CpuPartitionAnalysisInputs(false,
                                        List.of(CarrierAccess.MEMORY_SEGMENT))))));
    }

    @Test void materializationPolicyAppliesCostBenefitMemoryAndEligibilityGates() {
        Shape shape = Shape.of(2, 3);
        var general = descriptor(shape,
                LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true));
        var dense = descriptor(shape, LayoutDescriptor.contiguous(shape));
        var eligible = new CpuPartitionAnalysisInputs.MaterializationPolicy(
                true, 3, 2, 20, 1, 3, 48, 1, 1);
        var selected = analyze(general, general, general, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        PortableExecutionConfig.DEFAULT, eligible));
        var copy = selected.plan().materialization().orElseThrow();
        assertAll(
                () -> assertEquals(0, copy.sourceBoundaryIndex()),
                () -> assertEquals(1, copy.useCount()),
                () -> assertEquals(3, copy.expectedRunCount()),
                () -> assertEquals(360, copy.directCost()),
                () -> assertEquals(63, copy.copiedTotalCost()),
                () -> assertEquals(8_250, copy.benefitBasisPoints()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2),
                        new ValueId(5)), selected.plan().boundaryValues()),
                () -> assertEquals(5, selected.requirements().size()));

        for (var rejected : List.of(
                new CpuPartitionAnalysisInputs.MaterializationPolicy(
                        false, 0, 0, 100, 0, 1, 48, 0, 0),
                new CpuPartitionAnalysisInputs.MaterializationPolicy(
                        true, 0, 0, 1, 0, 1, 47, 0, 0),
                new CpuPartitionAnalysisInputs.MaterializationPolicy(
                        true, 0, 0, 1, 0, 1, 48, 7, 0),
                new CpuPartitionAnalysisInputs.MaterializationPolicy(
                        true, 1, 0, 1, 0, 1, 48, 0, 9_000))) {
            var direct = analyze(general, dense, dense, dense,
                    new CpuPartitionAnalysisInputs(false,
                            CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                            PortableExecutionConfig.DEFAULT, rejected));
            assertAll(
                    () -> assertTrue(direct.plan().materialization().isEmpty()),
                    () -> assertEquals(4, direct.requirements().size()));
        }
        var denseOnly = analyze(dense, dense, dense, dense,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(),
                        PortableExecutionConfig.DEFAULT, eligible));
        assertTrue(denseOnly.plan().materialization().isEmpty());
    }

    @Test void movementKeepsUniqueBoundaryOrderDisablesWorkspaceAndUsesScalarCompute() {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(TensorCompositionKind.CONCAT, new CompositionAxisAttrs(0)),
                List.of(0, 1, 0),
                List.of(descriptor(DataType.INT32, Shape.of(4)),
                        descriptor(DataType.INT32, Shape.of(2))),
                descriptor(DataType.INT32, Shape.of(10)));
        var preference = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE,
                4, 2, 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), preference));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var plan = analysis.plan();

        var zeroBase = CpuNonAffineMovementLoweringTest.context(
                new Operation(TileKind.TILE, new TileAttrs(List.of(2L))), List.of(0),
                List.of(descriptor(DataType.INT32, Shape.of(0))),
                descriptor(DataType.INT32, Shape.of(0)));
        var zero = new CpuPartitionPreparer().analyze(new PrepareContext<>(zeroBase.partition(),
                zeroBase.nodes(), zeroBase.values(), zeroBase.memoryRequirements(),
                zeroBase.constants(), new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), preference))).plan();
        var windowBase = CpuNonAffineMovementLoweringTest.context(
                new Operation(WindowTransformKind.UNFOLD2D,
                        new Window2dAttrs(1, 1, 1, 1, 0, 0, 1, 1, false)), List.of(0),
                List.of(descriptor(DataType.FLOAT32, Shape.of(2, 0, 3, 3))),
                descriptor(DataType.FLOAT32, Shape.of(2, 0, 9)));
        var windowAnalysis = new CpuPartitionPreparer().analyze(new PrepareContext<>(
                windowBase.partition(), windowBase.nodes(), windowBase.values(),
                windowBase.memoryRequirements(), windowBase.constants(),
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), preference)));
        var windowPlan = windowAnalysis.plan();
        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                        plan.boundaryValues()),
                () -> assertEquals(3, analysis.requirements().size()),
                () -> assertTrue(plan.movementGeometry().isPresent()),
                () -> assertTrue(plan.materialization().isEmpty()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertEquals("parallel-scalar", plan.executionStrategy().toString()),
                () -> assertEquals(0, plan.vectorSpeciesBitSize()),
                () -> assertEquals("scalar", zero.executionStrategy().toString()),
                () -> assertEquals(0, zero.elementCount()),
                () -> assertEquals(2, windowAnalysis.requirements().size()),
                () -> assertTrue(windowPlan.workspaceDeclaration().isEmpty()),
                () -> assertTrue(windowPlan.materialization().isEmpty()),
                () -> assertEquals(0, windowPlan.elementCount()),
                () -> assertEquals("scalar", windowPlan.executionStrategy().toString()),
                () -> assertEquals(0, windowPlan.vectorSpeciesBitSize()));
    }

    @Test void sliceUpdateDeclaresOneParallelScalarUnitAndNoHiddenResources() {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(7L), List.of(4L), List.of(0), List.of(-2L))),
                List.of(0, 1), List.of(descriptor(DataType.INT32, Shape.of(8)),
                        descriptor(DataType.INT32, Shape.of(4))),
                descriptor(DataType.INT32, Shape.of(8)));
        var preference = new PortableExecutionConfig(ComputePreference.VECTOR_IF_ELIGIBLE,
                2, 2, 1);
        var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                        List.of(CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT,
                                CarrierAccess.INT_ARRAY), preference));
        var analysis = new CpuPartitionPreparer().analyze(context);
        var plan = analysis.plan();

        var zeroBase = CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(), List.of(), List.of(), List.of())),
                List.of(0, 0), List.of(descriptor(DataType.INT32, Shape.of(0))),
                descriptor(DataType.INT32, Shape.of(0)));
        var zero = new CpuPartitionPreparer().analyze(zeroBase);
        assertAll(
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                        plan.boundaryValues()),
                () -> assertEquals(3, analysis.requirements().size()),
                () -> assertEquals(1, plan.units().size()),
                () -> assertTrue(plan.movementGeometry().isPresent()),
                () -> assertTrue(plan.materialization().isEmpty()),
                () -> assertTrue(plan.workspaceDeclaration().isEmpty()),
                () -> assertEquals("parallel-scalar", plan.executionStrategy().toString()),
                () -> assertEquals(0, plan.vectorSpeciesBitSize()),
                () -> assertEquals(2, zero.requirements().size()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)),
                        zero.plan().boundaryValues()),
                () -> assertEquals(0, zero.plan().elementCount()),
                () -> assertEquals("scalar", zero.plan().executionStrategy().toString()));
    }

    public static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(Shape shape) {
        return new CpuPartitionPreparer().analyze(context(shape));
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            Shape shape, PortableExecutionConfig config) {
        var descriptor = descriptor(shape, LayoutDescriptor.contiguous(shape));
        return analyze(descriptor, descriptor, descriptor, descriptor,
                new CpuPartitionAnalysisInputs(false,
                        CpuPartitionAnalysisInputs.DEFAULT.carrierPattern(), config));
    }

    private static TensorDescriptor descriptor(Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(DataType.FLOAT64, shape, Optional.of(layout), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            PrepareContext<CpuPartitionAnalysisInputs> context) {
        return new CpuPartitionPreparer().analyze(context);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> oneNodeContext(Operation operation,
            List<TensorDescriptor> inputs, TensorDescriptor output,
            PortableExecutionConfig config) {
        var inputIds = java.util.stream.IntStream.range(0, inputs.size())
                .mapToObj(ValueId::new).toList();
        ValueId outputId = new ValueId(inputs.size());
        var node = new CompiledNode(new NodeId(0), operation, inputIds, List.of(outputId));
        return arbitraryContext(List.of(node), concat(inputs, output),
                new CpuPartitionAnalysisInputs(false, List.of(), config));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> maskWhereContext(int count,
            PortableExecutionConfig config) {
        Shape shape = Shape.of(count);
        var value = descriptor(DataType.FLOAT32, shape);
        var mask = descriptor(DataType.BOOL, shape);
        var compare = new CompiledNode(new NodeId(0),
                new Operation(BinaryComparisonKind.GREATER_THAN, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(4)));
        var where = new CompiledNode(new NodeId(1),
                new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(4), new ValueId(2), new ValueId(3)), List.of(new ValueId(5)));
        return arbitraryContext(List.of(compare, where),
                List.of(value, value, value, value, mask, value),
                new CpuPartitionAnalysisInputs(false, List.of(), config));
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> externalWhereContext(int count,
            boolean scalarCondition, PortableExecutionConfig config) {
        Shape shape = Shape.of(count);
        var value = descriptor(DataType.FLOAT32, shape);
        var condition = descriptor(DataType.BOOL, scalarCondition ? Shape.scalar() : shape);
        return oneNodeContext(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                List.of(condition, value, value), value, config);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> materializedComparisonContext(
            int count, PortableExecutionConfig config) {
        Shape shape = Shape.of(count);
        var value = descriptor(DataType.FLOAT32, shape);
        return oneNodeContext(new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                List.of(value, value), descriptor(DataType.BOOL, shape), config);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> arbitraryContext(
            List<CompiledNode> nodes, List<TensorDescriptor> descriptors,
            CpuPartitionAnalysisInputs inputs) {
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < descriptors.size(); i++) {
            ValueId id = new ValueId(i);
            TensorDescriptor descriptor = descriptors.get(i);
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            boolean graphOutput = nodes.getLast().outputs().contains(id);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), graphOutput));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), inputs);
    }

    private static List<TensorDescriptor> concat(List<TensorDescriptor> inputs,
            TensorDescriptor output) {
        var result = new ArrayList<>(inputs);
        result.add(output);
        return result;
    }

    private static int vectorLanes(DataType type) {
        return switch (type) {
            case INT32 -> IntVector.SPECIES_PREFERRED.length();
            case INT64 -> LongVector.SPECIES_PREFERRED.length();
            case BOOL -> ByteVector.SPECIES_PREFERRED.length();
            default -> throw new IllegalArgumentException("unsupported test vector type");
        };
    }

    private static int vectorSpeciesBits(DataType type) {
        return switch (type) {
            case INT32 -> IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> LongVector.SPECIES_PREFERRED.vectorBitSize();
            case BOOL -> ByteVector.SPECIES_PREFERRED.vectorBitSize();
            default -> throw new IllegalArgumentException("unsupported test vector type");
        };
    }

    private record VectorRow(DataType type, OperationKind kind) { }

    public static BackendPartitionAnalysis<CpuPartitionPreparationPlan> analyze(
            TensorDescriptor a, TensorDescriptor b, TensorDescriptor c, TensorDescriptor output,
            CpuPartitionAnalysisInputs inputs) {
        return new CpuPartitionPreparer().analyze(context(a, b, c, output, inputs));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(Shape shape) {
        var descriptor = new TensorDescriptor(DataType.FLOAT64, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return context(descriptor, descriptor, descriptor, descriptor,
                CpuPartitionAnalysisInputs.DEFAULT);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor cDescriptor, TensorDescriptor outputDescriptor,
            CpuPartitionAnalysisInputs inputs) {
        return context(aDescriptor, bDescriptor, cDescriptor, outputDescriptor, inputs,
                UnaryElementwiseKind.GELU);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(
            TensorDescriptor aDescriptor, TensorDescriptor bDescriptor,
            TensorDescriptor cDescriptor, TensorDescriptor outputDescriptor,
            CpuPartitionAnalysisInputs inputs, UnaryElementwiseKind unaryKind) {
        ValueId a = new ValueId(0), b = new ValueId(1), c = new ValueId(2);
        ValueId sum = new ValueId(3), activated = new ValueId(4), output = new ValueId(5);
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(a, b), List.of(sum)),
                new CompiledNode(new NodeId(1),
                        new Operation(unaryKind, NoOperationAttrs.INSTANCE),
                        List.of(sum), List.of(activated)),
                new CompiledNode(new NodeId(2),
                        new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                        List.of(activated, c), List.of(output)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        Shape addShape = ShapeBroadcast.broadcast(aDescriptor.shape(), bDescriptor.shape());
        var virtualDescriptor = new TensorDescriptor(aDescriptor.dataType(), addShape,
                Optional.of(LayoutDescriptor.contiguous(addShape)), false);
        var descriptors = List.of(aDescriptor, bDescriptor, cDescriptor, virtualDescriptor,
                virtualDescriptor, outputDescriptor);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int i = 0; i < 6; i++) {
            ValueId id = new ValueId(i);
            var descriptor = descriptors.get(i);
            values.add(new GraphValue(id, descriptor));
            boolean produced = i >= 3;
            boolean consumed = i != 5;
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), i == 5));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                inputs);
    }
}
