package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuFusionDecision;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.graph.ValueId;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuFusionProfitabilitySelectorTest {
    @Test void verticalCaseRanksBothCompleteTopologiesAndSelectsProfitableFusion() {
        var plan = new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(2)).plan();
        var legal = legal(plan.fusionDecisions());
        var selection = selection(plan.fusionDecisions());
        assertAll(
                () -> assertEquals(2, legal.size()),
                () -> assertEquals(1, plan.units().size()),
                () -> assertEquals(List.of(0, 1),
                        plan.units().getFirst().memberNodeOrdinals()),
                () -> assertEquals(CpuFusionDecision.SelectionReason.PROFITABLE_FUSION,
                        selection.reason()),
                () -> assertTrue(selection.achievedMargin().orElseThrow() >= 32),
                () -> assertTrue(legal.stream().allMatch(value -> value.score().isPresent())));
    }

    @Test void publicationAndDiamondFanOutProduceTypedHardLegalityFacts() {
        var publication = new CpuPartitionPreparer().analyze(
                CpuPartitionDagDecomposerTest.publishedChain(2)).plan();
        var diamond = new CpuPartitionPreparer().analyze(
                CpuPartitionDagDecomposerTest.diamond()).plan();
        var roles = legal(diamond.fusionDecisions()).stream()
                .flatMap(candidate -> candidate.identity().units().stream())
                .flatMap(unit -> unit.boundaries().stream())
                .map(CpuFusionDecision.BoundaryFact::role)
                .collect(java.util.stream.Collectors.toSet());
        assertAll(
                () -> assertEquals(CpuFusionDecision.SelectionReason.CANONICAL_SPLIT,
                        selection(publication.fusionDecisions()).reason()),
                () -> assertTrue(publication.fusionDecisions().stream()
                        .filter(CpuFusionDecision.LegalityRejection.class::isInstance)
                        .map(CpuFusionDecision.LegalityRejection.class::cast)
                        .anyMatch(value -> value.reason()
                                == CpuFusionDecision.LegalityReason.PUBLICATION_BARRIER)),
                () -> assertTrue(diamond.fusionDecisions().stream()
                        .filter(CpuFusionDecision.LegalityRejection.class::isInstance)
                        .map(CpuFusionDecision.LegalityRejection.class::cast)
                        .anyMatch(value -> value.reason()
                                == CpuFusionDecision.LegalityReason.FAN_OUT_BARRIER)),
                () -> assertEquals(2, legal(diamond.fusionDecisions()).size()),
                () -> assertEquals(2, diamond.units().size()),
                () -> assertTrue(roles.contains(CpuFusionDecision.BoundaryRole.EXTERNAL_READ)),
                () -> assertTrue(roles.contains(CpuFusionDecision.BoundaryRole.CROSS_UNIT)),
                () -> assertTrue(roles.contains(CpuFusionDecision.BoundaryRole.PUBLICATION)),
                () -> assertTrue(roles.contains(CpuFusionDecision.BoundaryRole.PARTITION_WRITE)));
    }

    @Test void boundaryFactsMapEveryRoleToExactValueAndCandidateUnit() {
        var publication = new CpuPartitionPreparer().analyze(
                CpuPartitionDagDecomposerTest.publishedChain(2)).plan();
        var publicationCandidate = legal(publication.fusionDecisions()).getFirst().identity();
        var diamond = new CpuPartitionPreparer().analyze(
                CpuPartitionDagDecomposerTest.diamond()).plan();
        var diamondCandidate = legal(diamond.fusionDecisions()).getFirst().identity();

        assertAll(
                () -> assertBoundary(publicationCandidate, publication.boundaryValues(),
                        0, 0, new ValueId(0),
                        CpuFusionDecision.BoundaryRole.EXTERNAL_READ),
                () -> assertBoundary(publicationCandidate, publication.boundaryValues(),
                        0, 1, new ValueId(1),
                        CpuFusionDecision.BoundaryRole.CROSS_UNIT),
                () -> assertBoundary(publicationCandidate, publication.boundaryValues(),
                        1, 0, new ValueId(1),
                        CpuFusionDecision.BoundaryRole.CROSS_UNIT),
                () -> assertBoundary(publicationCandidate, publication.boundaryValues(),
                        1, 1, new ValueId(2),
                        CpuFusionDecision.BoundaryRole.PUBLICATION),
                () -> assertBoundary(diamondCandidate, diamond.boundaryValues(),
                        1, 1, new ValueId(2),
                        CpuFusionDecision.BoundaryRole.PUBLICATION),
                () -> assertBoundary(diamondCandidate, diamond.boundaryValues(),
                        2, 1, new ValueId(3),
                        CpuFusionDecision.BoundaryRole.PARTITION_WRITE));
    }

    @Test void eightNodeEnumerationExhaustsAttemptsRetainsBaselineAndSafelySplits() {
        var plan = new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(8)).plan();
        var legal = legal(plan.fusionDecisions());
        long profitability = plan.fusionDecisions().stream()
                .filter(CpuFusionDecision.ProfitabilityRejection.class::isInstance).count();
        var selection = selection(plan.fusionDecisions());
        assertAll(
                () -> assertEquals(45, legal.size()),
                () -> assertEquals(44, profitability),
                () -> assertEquals(8, plan.units().size()),
                () -> assertEquals(1, selection.compatibilityBaseline().units().size()),
                () -> assertEquals(CpuFusionDecision.SelectionReason.ENUMERATION_BUDGET_FALLBACK,
                        selection.reason()),
                () -> assertTrue(plan.fusionDecisions().size() <= 384));
    }

    @Test void typedTopologyProbePinsCandidateAndAttemptCeilingsAndPendingExcess() {
        var plan = new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(8)).plan();
        var source = selection(plan.fusionDecisions()).canonicalSplit();
        var attempt = new CpuFusionProfitabilitySelector.TopologyAttempt(source,
                new CpuFusionDecision.AttemptedPair(0, 1,
                        CpuFusionDecision.PairKind.VERTICAL));
        assertAll(
                () -> assertTrue(CpuFusionProfitabilitySelector.completeWithinBudgets(
                        new CpuFusionProfitabilitySelector.TopologyProbe(
                                Collections.nCopies(64, source),
                                Collections.nCopies(256, attempt), false, false))),
                () -> assertFalse(CpuFusionProfitabilitySelector.completeWithinBudgets(
                        new CpuFusionProfitabilitySelector.TopologyProbe(
                                Collections.nCopies(65, source),
                                Collections.nCopies(256, attempt), false, false))),
                () -> assertFalse(CpuFusionProfitabilitySelector.completeWithinBudgets(
                        new CpuFusionProfitabilitySelector.TopologyProbe(
                                Collections.nCopies(64, source),
                                Collections.nCopies(257, attempt), false, false))),
                () -> assertFalse(CpuFusionProfitabilitySelector.completeWithinBudgets(
                        new CpuFusionProfitabilitySelector.TopologyProbe(
                                Collections.nCopies(64, source),
                                Collections.nCopies(256, attempt), true, true))));
    }

    @Test void nonWinningTieCannotSuppressTheStrictlyProfitableBestCandidate() {
        var base = legal(new CpuPartitionPreparer().analyze(
                CpuPointwisePartitionLoweringTest.chain(2)).plan().fusionDecisions())
                .getFirst().identity().units().getFirst();
        var split = policyCandidate(identity(base, List.of(List.of(0), List.of(1), List.of(2))),
                2, 48, 240, true);
        var tied = policyCandidate(identity(base, List.of(List.of(0, 1), List.of(2))),
                1, 112, 240, false);
        var winner = policyCandidate(identity(base, List.of(List.of(0, 1, 2))),
                0, 80, 144, false);
        assertAll(
                () -> assertEquals(2, CpuFusionProfitabilitySelector.selectedCandidateIndex(
                        List.of(split, tied, winner), true)),
                () -> assertEquals(0, CpuFusionProfitabilitySelector.selectedCandidateIndex(
                        List.of(split, tied), true)));
    }

    private static CpuFusionDecision.LegalCandidate policyCandidate(
            CpuFusionDecision.CandidateIdentity identity, int rank, long structuralCost,
            long totalScore, boolean split) {
        int units = identity.units().size();
        var facts = new CpuFusionDecision.CandidateFacts(units, 0, 0, 0, 8, 4, units, 0);
        return new CpuFusionDecision.LegalCandidate(identity, facts, Optional.of(
                new CpuFusionDecision.Score(64L * units, 0, structuralCost, 0, totalScore)),
                rank, split, false);
    }

    private static CpuFusionDecision.CandidateIdentity identity(
            CpuFusionDecision.UnitIdentity base, List<List<Integer>> memberships) {
        var units = new ArrayList<CpuFusionDecision.UnitIdentity>();
        for (int index = 0; index < memberships.size(); index++) {
            units.add(new CpuFusionDecision.UnitIdentity(memberships.get(index),
                    index == 0 ? List.of() : List.of(index - 1),
                    base.portableIrStructuralKey(), base.specialization(), base.strategy(),
                    List.of(), Optional.empty(), memberships.get(index).size() == 1
                            ? CpuFusionDecision.UnitTopology.SPLIT_POINTWISE
                            : CpuFusionDecision.UnitTopology.FUSED_POINTWISE));
        }
        return new CpuFusionDecision.CandidateIdentity(units);
    }

    private static List<CpuFusionDecision.LegalCandidate> legal(
            List<CpuFusionDecision> decisions) {
        return decisions.stream().filter(CpuFusionDecision.LegalCandidate.class::isInstance)
                .map(CpuFusionDecision.LegalCandidate.class::cast).toList();
    }

    private static CpuFusionDecision.Selection selection(List<CpuFusionDecision> decisions) {
        return decisions.stream().filter(CpuFusionDecision.Selection.class::isInstance)
                .map(CpuFusionDecision.Selection.class::cast).findFirst().orElseThrow();
    }

    private static void assertBoundary(CpuFusionDecision.CandidateIdentity candidate,
            List<ValueId> boundaryValues, int unitPosition, int boundaryPosition, ValueId value,
            CpuFusionDecision.BoundaryRole role) {
        var unit = candidate.units().get(unitPosition);
        var fact = unit.boundaries().get(boundaryPosition);
        assertAll(
                () -> assertEquals(List.of(unitPosition), unit.memberNodePositions()),
                () -> assertEquals(value,
                        boundaryValues.get(fact.relativeBoundaryPosition())),
                () -> assertEquals(boundaryPosition, fact.unitBoundaryPosition()),
                () -> assertEquals(role, fact.role()));
    }
}
