package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class BackendPartitionTuningHandoffTest {
    @Test
    void retainsExactReferencesForAbsentAndPresentDecisions() {
        PlannedPartition partition = partition();
        CandidateBatch batch = new CandidateBatch("complete-batch");
        Decision decision = new Decision("selected-candidate");
        Optional<Decision> absent = Optional.empty();
        Optional<Decision> present = Optional.of(decision);

        var withoutDecision = new BackendPartitionTuningHandoff<>(partition, batch, absent);
        var withDecision = new BackendPartitionTuningHandoff<>(partition, batch, present);

        assertAll(
                () -> assertSame(partition, withoutDecision.partition()),
                () -> assertSame(batch, withoutDecision.candidateBatch()),
                () -> assertSame(absent, withoutDecision.selectedDecision()),
                () -> assertSame(partition, withDecision.partition()),
                () -> assertSame(batch, withDecision.candidateBatch()),
                () -> assertSame(present, withDecision.selectedDecision()),
                () -> assertSame(decision, withDecision.selectedDecision().orElseThrow()));
    }

    @Test
    void rejectsEveryNullIncludingOptionalNullByItsConstructionRule() {
        PlannedPartition partition = partition();
        CandidateBatch batch = new CandidateBatch("complete-batch");

        assertAll(
                () -> assertThrows(NullPointerException.class,
                        () -> new BackendPartitionTuningHandoff<>(
                                null, batch, Optional.<Decision>empty())),
                () -> assertThrows(NullPointerException.class,
                        () -> new BackendPartitionTuningHandoff<>(
                                partition, null, Optional.<Decision>empty())),
                () -> assertThrows(NullPointerException.class,
                        () -> new BackendPartitionTuningHandoff<>(partition, batch, null)),
                () -> assertThrows(NullPointerException.class,
                        () -> Optional.of((Decision) null)));
    }

    @Test
    void recordValueMethodsUseExactlyTheThreeComponents() {
        PlannedPartition partition = partition();
        CandidateBatch batch = new CandidateBatch("complete-batch");
        Decision decision = new Decision("selected-candidate");
        var first = new BackendPartitionTuningHandoff<>(
                partition, batch, Optional.of(decision));
        var equal = new BackendPartitionTuningHandoff<>(
                partition, new CandidateBatch("complete-batch"),
                Optional.of(new Decision("selected-candidate")));
        var absent = new BackendPartitionTuningHandoff<
                CandidateBatch, Decision>(partition, batch, Optional.empty());

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, absent),
                () -> assertEquals(
                        "BackendPartitionTuningHandoff[partition=" + partition
                                + ", candidateBatch=" + batch
                                + ", selectedDecision=Optional[" + decision + "]]",
                        first.toString()));
    }

    private static PlannedPartition partition() {
        return new PlannedPartition(new BackendId("cpu"), List.of(new NodeId(7)));
    }

    private record CandidateBatch(String name) implements BackendTuningCandidateBatch { }

    private record Decision(String candidate) implements BackendTuningDecision { }
}
