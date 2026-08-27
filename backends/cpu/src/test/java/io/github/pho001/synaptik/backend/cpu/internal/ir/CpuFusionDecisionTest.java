package io.github.pho001.synaptik.backend.cpu.internal.ir;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparerTest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuFusionDecisionTest {
    @Test void exposesExactlyTheClosedDecisionAndReasonInventories() {
        assertAll(
                () -> assertEquals(List.of("LegalCandidate", "LegalityRejection",
                                "ProfitabilityRejection", "Selection"),
                        java.util.Arrays.stream(CpuFusionDecision.class.getPermittedSubclasses())
                                .map(Class::getSimpleName).sorted().toList()),
                () -> assertArrayEquals(new CpuFusionDecision.LegalityReason[] {
                        CpuFusionDecision.LegalityReason.SEMANTIC_BARRIER,
                        CpuFusionDecision.LegalityReason.PUBLICATION_BARRIER,
                        CpuFusionDecision.LegalityReason.FAN_OUT_BARRIER,
                        CpuFusionDecision.LegalityReason.STATE_OR_RANDOM_BARRIER,
                        CpuFusionDecision.LegalityReason.NUMERICAL_ORDER_BARRIER,
                        CpuFusionDecision.LegalityReason.ALIAS_OR_ACCESS_UNPROVED,
                        CpuFusionDecision.LegalityReason.DEPENDENCY_CYCLE,
                        CpuFusionDecision.LegalityReason.UNSUPPORTED_LOWERING,
                        CpuFusionDecision.LegalityReason.ROUTE_INELIGIBLE,
                        CpuFusionDecision.LegalityReason.HARD_BUDGET_EXCEEDED},
                        CpuFusionDecision.LegalityReason.values()),
                () -> assertArrayEquals(new CpuFusionDecision.ProfitabilityReason[] {
                        CpuFusionDecision.ProfitabilityReason.INSUFFICIENT_MARGIN,
                        CpuFusionDecision.ProfitabilityReason.CODE_SIZE_PRESSURE,
                        CpuFusionDecision.ProfitabilityReason.LIVE_VALUE_PRESSURE,
                        CpuFusionDecision.ProfitabilityReason.MATERIALIZATION_COST,
                        CpuFusionDecision.ProfitabilityReason.SAFE_SPLIT_TIE,
                        CpuFusionDecision.ProfitabilityReason.UNCERTAIN_INPUT},
                        CpuFusionDecision.ProfitabilityReason.values()),
                () -> assertArrayEquals(new CpuFusionDecision.SelectionReason[] {
                        CpuFusionDecision.SelectionReason.PROFITABLE_FUSION,
                        CpuFusionDecision.SelectionReason.CANONICAL_SPLIT,
                        CpuFusionDecision.SelectionReason.TIE_FALLBACK,
                        CpuFusionDecision.SelectionReason.UNCERTAINTY_FALLBACK,
                        CpuFusionDecision.SelectionReason.ENUMERATION_BUDGET_FALLBACK},
                        CpuFusionDecision.SelectionReason.values()));
    }

    @Test void oneUnitPreparationRetainsEqualSplitBaselineAndSelectedTypedIdentities() {
        var plan = oneUnitPlan();
        var selection = plan.fusionDecisions().stream()
                .filter(CpuFusionDecision.Selection.class::isInstance)
                .map(CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow();
        var legal = plan.fusionDecisions().stream()
                .filter(CpuFusionDecision.LegalCandidate.class::isInstance)
                .map(CpuFusionDecision.LegalCandidate.class::cast).toList();
        assertAll(
                () -> assertEquals(2, plan.fusionDecisions().size()),
                () -> assertEquals(1, legal.size()),
                () -> assertTrue(legal.getFirst().canonicalSplit()),
                () -> assertTrue(legal.getFirst().compatibilityBaseline()),
                () -> assertEquals(selection.selected(), selection.canonicalSplit()),
                () -> assertEquals(selection.selected(), selection.compatibilityBaseline()),
                () -> assertEquals(CpuFusionDecision.SelectionReason.CANONICAL_SPLIT,
                        selection.reason()),
                () -> assertTrue(selection.selectedScore().isPresent()));
    }

    @Test void typedIdentityAndCollectionsAreDefensiveAndRejectInvalidGeometry() {
        var plan = oneUnitPlan();
        var identity = plan.fusionDecisions().stream()
                .filter(CpuFusionDecision.Selection.class::isInstance)
                .map(CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow().selected();
        var mutable = new ArrayList<>(identity.units());
        var copied = new CpuFusionDecision.CandidateIdentity(mutable);
        mutable.clear();
        assertAll(
                () -> assertEquals(1, copied.units().size()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> copied.units().clear()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuFusionDecision.BoundaryFact(-1, 0,
                                CpuFusionDecision.BoundaryRole.EXTERNAL_READ,
                                CpuAccessPlan.Regime.DENSE_LINEAR, 8, 8)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuFusionDecision.WorkspaceFact(
                                CpuFusionDecision.WorkspaceRole.AGGREGATE_EXACT_STATE,
                                1, 0)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuFusionDecision.Score(1, 2, 3, 4, 9)));
    }

    @Test void closedFactsReachTheExactCandidateBoundaryWorkspaceAndTotalCeilings() {
        var sourcePlan = oneUnitPlan();
        var seed = sourcePlan.fusionDecisions().stream()
                .filter(CpuFusionDecision.LegalCandidate.class::isInstance)
                .map(CpuFusionDecision.LegalCandidate.class::cast).findFirst().orElseThrow();
        var identities = new ArrayList<CpuFusionDecision.CandidateIdentity>();
        for (int index = 0; index < 64; index++) identities.add(
                atResourceCeilings(seed.identity(), index));
        var facts = new ArrayList<CpuFusionDecision>();
        for (int index = 0; index < 64; index++) facts.add(new CpuFusionDecision.LegalCandidate(
                identities.get(index), new CpuFusionDecision.CandidateFacts(8,
                        64, 8, 512, 16, 4, 8, 0), Optional.of(
                            new CpuFusionDecision.Score(512, 0, 0, 0, 512)),
                index, index == 0, index == 0));
        var pairSource = identities.getFirst();
        for (int index = 0; index < 256; index++) facts.add(
                new CpuFusionDecision.LegalityRejection(pairSource,
                        new CpuFusionDecision.AttemptedPair(0, 1,
                                index % 2 == 0 ? CpuFusionDecision.PairKind.VERTICAL
                                        : CpuFusionDecision.PairKind.HORIZONTAL),
                        CpuFusionDecision.HardFact.BUDGET,
                        CpuFusionDecision.LegalityReason.HARD_BUDGET_EXCEEDED));
        for (int index = 1; index < 64; index++) facts.add(
                new CpuFusionDecision.ProfitabilityRejection(identities.get(index),
                        CpuFusionDecision.ProfitabilityReason.INSUFFICIENT_MARGIN,
                        100, 100, 32));
        facts.add(new CpuFusionDecision.Selection(identities.getFirst(), identities.getFirst(),
                identities.getFirst(), 0, Optional.of(
                        new CpuFusionDecision.Score(512, 0, 0, 0, 512)), Optional.of(
                        new CpuFusionDecision.Score(512, 0, 0, 0, 512)), Optional.of(0L),
                CpuFusionDecision.SelectionReason.CANONICAL_SPLIT));
        assertEquals(384, facts.size());
    }

    @Test void planRejectsMissingDuplicateAndSelectedMismatchedDecisionFacts() {
        var plan = CpuPartitionPreparerTest.analyze(Shape.of(8)).plan();
        var decisions = new ArrayList<>(plan.fusionDecisions());
        var selection = (CpuFusionDecision.Selection) decisions.getLast();
        var missing = new ArrayList<>(decisions);
        missing.removeLast();
        var duplicate = new ArrayList<>(decisions);
        duplicate.add(selection);
        var mismatched = new ArrayList<>(decisions);
        mismatched.set(mismatched.size() - 1, new CpuFusionDecision.Selection(
                selection.canonicalSplit(), selection.canonicalSplit(),
                selection.compatibilityBaseline(), selection.stableRank(),
                selection.canonicalSplitScore(), selection.canonicalSplitScore(),
                Optional.of(0L), CpuFusionDecision.SelectionReason.PROFITABLE_FUSION));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> copyDecisionPlan(plan, missing)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> copyDecisionPlan(plan, duplicate)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> copyDecisionPlan(plan, mismatched)));
    }

    @Test void planRecomputesPublicationRoleFromRetainedLogicalRequirementProjection() {
        var publicationPlan = oneUnitPlan();
        var partitionWritePlan = oneUnitPlanWithoutPublication();
        assertAll(
                () -> assertFalse(publicationPlan.publicationBoundaryPositions().isEmpty()),
                () -> assertTrue(partitionWritePlan.publicationBoundaryPositions().isEmpty()),
                () -> assertThrows(IllegalArgumentException.class, () -> copyDecisionPlan(
                        publicationPlan, forgedBoundaryRole(publicationPlan,
                                CpuFusionDecision.BoundaryRole.PARTITION_WRITE))),
                () -> assertThrows(IllegalArgumentException.class, () -> copyDecisionPlan(
                        partitionWritePlan, forgedBoundaryRole(partitionWritePlan,
                                CpuFusionDecision.BoundaryRole.PUBLICATION))));
    }

    @Test void planRejectsEveryDerivableForgedRetainedRecognitionOverlap() {
        var plan = oneUnitPlan();
        var fact = (CpuSpecializedSubgraph.ExplicitSemanticKernel)
                plan.specializedSubgraphs().getFirst();
        var baseline = fact.structuralIdentity().baselineUnits().getFirst();
        var execution = baseline.execution();
        var specialization = execution.specialization();
        var changedSpecialization = new CpuKernelSpecialization(
                specialization.loweringFingerprint(), specialization.numericalMode(),
                specialization.executionStrategy(), specialization.boundaryDataTypes(),
                specialization.carrierPattern(), specialization.vectorSpeciesBitSize(),
                specialization.materializedSourcePosition(),
                specialization.scalarPowerRealizations(), !specialization.scratchParameter());
        var wrongSpecialization = withExecution(baseline,
                new CpuSpecializedSubgraph.BaselineExecutionFact(execution.route(),
                        changedSpecialization, execution.compute(), execution.orchestration(),
                        execution.extents(), execution.elementCount(),
                        execution.selectedRangeCount(), execution.minimumElementsPerWorker(),
                        execution.vectorSpeciesBitSize(), execution.affineAddressPairs(),
                        execution.materialization(), execution.runtimeTopology(),
                        execution.packedGeometry(), execution.fusionReason()));
        String wrongKey = "02".repeat(32);
        var changedFingerprint = new CpuKernelSpecialization(
                CpuLoweringFingerprint.fromHex(wrongKey), specialization.numericalMode(),
                specialization.executionStrategy(), specialization.boundaryDataTypes(),
                specialization.carrierPattern(), specialization.vectorSpeciesBitSize(),
                specialization.materializedSourcePosition(),
                specialization.scalarPowerRealizations(), specialization.scratchParameter());
        var wrongKeyExecution = new CpuSpecializedSubgraph.BaselineExecutionFact(
                execution.route(), changedFingerprint, execution.compute(),
                execution.orchestration(), execution.extents(), execution.elementCount(),
                execution.selectedRangeCount(), execution.minimumElementsPerWorker(),
                execution.vectorSpeciesBitSize(), execution.affineAddressPairs(),
                execution.materialization(), execution.runtimeTopology(),
                execution.packedGeometry(), execution.fusionReason());
        var wrongStructuralKey = new CpuSpecializedSubgraph.BaselineUnitFact(wrongKey,
                wrongKeyExecution, baseline.dependencies(), baseline.boundaries(),
                baseline.outputCount(), baseline.workspace());
        var wrongDependencies = new CpuSpecializedSubgraph.BaselineUnitFact(
                baseline.structuralKey(), execution, List.of(0), baseline.boundaries(),
                baseline.outputCount(), baseline.workspace());
        var boundaries = new ArrayList<>(baseline.boundaries());
        var boundary = boundaries.getFirst();
        boundaries.set(0, new CpuSpecializedSubgraph.BoundaryResourceFact(boundary.dataType(),
                boundary.role(), boundary.accessPlan(), boundary.extents(),
                boundary.baseElementOffset(), boundary.effectiveStrides(), boundary.elementCount(),
                boundary.start(), boundary.end(),
                Math.addExact(boundary.referencedElementSpan(), 1),
                boundary.startCoordinates(), boundary.startAddress(),
                boundary.accessedElementStart(), boundary.accessedElementEnd(),
                boundary.carrier(), boundary.generatedCarrier()));
        var wrongBoundary = new CpuSpecializedSubgraph.BaselineUnitFact(
                baseline.structuralKey(), execution, baseline.dependencies(), boundaries,
                baseline.outputCount(), baseline.workspace());
        var wrongWorkspace = new CpuSpecializedSubgraph.BaselineUnitFact(
                baseline.structuralKey(), execution, baseline.dependencies(),
                baseline.boundaries(), baseline.outputCount(),
                new CpuSpecializedSubgraph.WorkspaceResourceFact(
                        CpuSpecializedSubgraph.WorkspaceRole.AGGREGATE_EXACT_STATE, 8, 8));
        assertAll(
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, wrongStructuralKey,
                                fact.memberNodeOrdinals())), plan.fusionDecisions())),
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, wrongSpecialization,
                                fact.memberNodeOrdinals())), plan.fusionDecisions())),
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, baseline, List.of(1))),
                        plan.fusionDecisions())),
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, wrongDependencies,
                                fact.memberNodeOrdinals())), plan.fusionDecisions())),
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, wrongBoundary,
                                fact.memberNodeOrdinals())), plan.fusionDecisions())),
                () -> assertThrows(IllegalArgumentException.class, () -> copyPlan(plan,
                        List.of(forgedRecognition(fact, wrongWorkspace,
                                fact.memberNodeOrdinals())), plan.fusionDecisions())));
    }

    private static CpuSpecializedSubgraph.BaselineUnitFact withExecution(
            CpuSpecializedSubgraph.BaselineUnitFact source,
            CpuSpecializedSubgraph.BaselineExecutionFact execution) {
        return new CpuSpecializedSubgraph.BaselineUnitFact(source.structuralKey(), execution,
                source.dependencies(), source.boundaries(), source.outputCount(),
                source.workspace());
    }

    private static CpuSpecializedSubgraph.ExplicitSemanticKernel forgedRecognition(
            CpuSpecializedSubgraph.ExplicitSemanticKernel fact,
            CpuSpecializedSubgraph.BaselineUnitFact baseline, List<Integer> members) {
        var source = fact.structuralIdentity();
        var identity = new CpuSpecializedSubgraph.StructuralIdentity(source.family(),
                source.form(), source.inputDataTypes(), source.resultDataTypes(),
                source.accessFacts(), source.attributes(), source.epilogue(), List.of(baseline));
        return new CpuSpecializedSubgraph.ExplicitSemanticKernel(fact.form(), members,
                fact.baselineUnitIndices(), fact.inputDataTypes(), fact.resultDataTypes(),
                fact.accessFacts(), identity);
    }

    private static List<CpuFusionDecision> forgedBoundaryRole(
            CpuPartitionPreparationPlan plan, CpuFusionDecision.BoundaryRole replacement) {
        var source = (CpuFusionDecision.LegalCandidate) plan.fusionDecisions().getFirst();
        var oldUnit = source.identity().units().getFirst();
        var boundaries = new ArrayList<>(oldUnit.boundaries());
        int output = java.util.stream.IntStream.range(0, boundaries.size())
                .filter(index -> boundaries.get(index).role()
                        == (replacement == CpuFusionDecision.BoundaryRole.PUBLICATION
                            ? CpuFusionDecision.BoundaryRole.PARTITION_WRITE
                            : CpuFusionDecision.BoundaryRole.PUBLICATION))
                .findFirst().orElseThrow();
        var oldBoundary = boundaries.get(output);
        boundaries.set(output, new CpuFusionDecision.BoundaryFact(
                oldBoundary.relativeBoundaryPosition(), oldBoundary.unitBoundaryPosition(),
                replacement, oldBoundary.regime(),
                oldBoundary.referencedBytes(), oldBoundary.byteAlignment()));
        var forgedIdentity = new CpuFusionDecision.CandidateIdentity(List.of(
                new CpuFusionDecision.UnitIdentity(oldUnit.memberNodePositions(),
                        oldUnit.dependencyUnitPositions(), oldUnit.portableIrStructuralKey(),
                        oldUnit.specialization(), oldUnit.strategy(), boundaries,
                        oldUnit.workspace(), oldUnit.topology())));
        var forgedLegal = new CpuFusionDecision.LegalCandidate(forgedIdentity, source.facts(),
                source.score(), source.stableRank(), source.canonicalSplit(),
                source.compatibilityBaseline());
        var oldSelection = (CpuFusionDecision.Selection) plan.fusionDecisions().getLast();
        var forgedSelection = new CpuFusionDecision.Selection(forgedIdentity, forgedIdentity,
                forgedIdentity, oldSelection.stableRank(), oldSelection.selectedScore(),
                oldSelection.canonicalSplitScore(), oldSelection.achievedMargin(),
                oldSelection.reason());
        return List.of(forgedLegal, forgedSelection);
    }

    private static CpuPartitionPreparationPlan copyDecisionPlan(
            CpuPartitionPreparationPlan plan, List<CpuFusionDecision> decisions) {
        return copyPlan(plan, plan.specializedSubgraphs(), decisions);
    }

    private static CpuPartitionPreparationPlan copyPlan(CpuPartitionPreparationPlan plan,
            List<CpuSpecializedSubgraph> recognition, List<CpuFusionDecision> decisions) {
        return new CpuPartitionPreparationPlan(plan.units(), plan.route(),
                plan.executionStrategy(), plan.bufferDeclarations(), plan.boundaryValues(),
                plan.accessBindings(), plan.carrierPattern(), plan.generatedCarrierPattern(),
                plan.extents(), plan.elementCount(), plan.affineAddressPairs(),
                plan.selectedRangeCount(), plan.minimumElementsPerWorker(),
                plan.vectorSpeciesBitSize(), plan.loweringManifest(), plan.materialization(),
                plan.workspaceDeclaration(), plan.workspaceUse(), plan.specializationBudget(),
                plan.movementGeometry(), plan.indexingGeometry(), plan.scatterGeometry(),
                plan.foldGeometry(), plan.orderingGeometry(), plan.randomGeometry(),
                plan.scanGeometry(), plan.aggregateGeometry(), plan.argExtremaGeometry(),
                plan.maskedReductionGeometry(), plan.advancedReductionGeometry(),
                plan.softmaxGeometry(), plan.trailingNormalizationGeometry(),
                plan.batchNormInferenceGeometry(), plan.batchNormTrainingGeometry(),
                plan.conv2dGeometry(), recognition, decisions,
                plan.publicationBoundaryPositions(), plan.materializations(),
                plan.representationUnits(), plan.representationDecisions());
    }

    private static CpuFusionDecision.CandidateIdentity atResourceCeilings(
            CpuFusionDecision.CandidateIdentity source, int suffix) {
        var old = source.units().getFirst();
        var units = new ArrayList<CpuFusionDecision.UnitIdentity>();
        for (int unitIndex = 0; unitIndex < 8; unitIndex++) {
            var octets = new ArrayList<>(old.portableIrStructuralKey().octets());
            octets.add(suffix);
            octets.add(unitIndex);
            var boundaries = new ArrayList<CpuFusionDecision.BoundaryFact>();
            for (int local = 0; local < 8; local++) boundaries.add(
                    new CpuFusionDecision.BoundaryFact(unitIndex * 8 + local, local,
                            CpuFusionDecision.BoundaryRole.CROSS_UNIT,
                            CpuAccessPlan.Regime.DENSE_LINEAR, 8, 8));
            units.add(new CpuFusionDecision.UnitIdentity(List.of(unitIndex),
                    unitIndex == 0 ? List.of() : List.of(unitIndex - 1),
                    new CpuFusionDecision.StructuralKey(octets), old.specialization(),
                    old.strategy(), boundaries, Optional.of(new CpuFusionDecision.WorkspaceFact(
                        CpuFusionDecision.WorkspaceRole.AGGREGATE_EXACT_STATE, 8, 8)),
                    CpuFusionDecision.UnitTopology.SPLIT_POINTWISE));
        }
        return new CpuFusionDecision.CandidateIdentity(units);
    }

    private static io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan
            oneUnitPlan() {
        return new CpuPartitionPreparer().analyze(CpuSoftmaxLoweringTest.context(
                SoftmaxKind.SOFTMAX, DataType.FLOAT64, Shape.of(8), 0)).plan();
    }

    private static CpuPartitionPreparationPlan oneUnitPlanWithoutPublication() {
        var source = CpuSoftmaxLoweringTest.context(
                SoftmaxKind.SOFTMAX, DataType.FLOAT64, Shape.of(8), 0);
        var memory = source.memoryRequirements().stream().map(requirement ->
                new io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement(
                        requirement.valueId(), requirement.descriptor(),
                        requirement.producerPartition(), requirement.consumerPartitions(), false))
                .toList();
        var context = new io.github.pho001.synaptik.prepare.analysis.PrepareContext<>(
                source.partition(), source.nodes(), source.values(), memory, source.constants(),
                source.backendInputs());
        return new CpuPartitionPreparer().analyze(context).plan();
    }
}
