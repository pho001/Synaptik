package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class PreparationResourceRequirementTest {
    @Test
    void bufferAcceptsZeroBytesAndValidatesComponentsInOrder() {
        ValueId valueId = new ValueId(7);
        PreparationResourceRequirement.Buffer zero =
                new PreparationResourceRequirement.Buffer(valueId, 0, 1);
        PreparationResourceRequirement.Buffer maximumAlignment =
                new PreparationResourceRequirement.Buffer(new ValueId(8), 10, 1L << 62);

        NullPointerException nullId = assertThrows(
                NullPointerException.class,
                () -> new PreparationResourceRequirement.Buffer(null, -1, 0));
        IllegalArgumentException negativeSize = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Buffer(valueId, -1, 0));
        IllegalArgumentException zeroAlignment = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Buffer(valueId, 0, 0));
        IllegalArgumentException nonPowerOfTwo = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Buffer(valueId, 0, 3));

        assertAll(
                () -> assertSame(valueId, zero.valueId()),
                () -> assertEquals(0, zero.byteSize()),
                () -> assertEquals(1, zero.byteAlignment()),
                () -> assertEquals(1L << 62, maximumAlignment.byteAlignment()),
                () -> assertEquals("valueId", nullId.getMessage()),
                () -> assertEquals("byteSize must be non-negative", negativeSize.getMessage()),
                () -> assertEquals(
                        "byteAlignment must be a positive power of two",
                        zeroAlignment.getMessage()),
                () -> assertEquals(
                        "byteAlignment must be a positive power of two",
                        nonPowerOfTwo.getMessage()));
    }

    @Test
    void workspaceAcceptsZeroValuesAndValidatesFieldsInOrder() {
        PreparationResourceRequirement.Workspace zero =
                new PreparationResourceRequirement.Workspace(0, 0, 1);

        IllegalArgumentException negativeId = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Workspace(-1, -1, 0));
        IllegalArgumentException negativeSize = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Workspace(0, -1, 0));
        IllegalArgumentException invalidAlignment = assertThrows(
                IllegalArgumentException.class,
                () -> new PreparationResourceRequirement.Workspace(0, 0, Long.MIN_VALUE));

        assertAll(
                () -> assertEquals(0, zero.requirementId()),
                () -> assertEquals(0, zero.byteSize()),
                () -> assertEquals(1, zero.byteAlignment()),
                () -> assertEquals(
                        "requirementId must be non-negative", negativeId.getMessage()),
                () -> assertEquals("byteSize must be non-negative", negativeSize.getMessage()),
                () -> assertEquals(
                        "byteAlignment must be a positive power of two",
                        invalidAlignment.getMessage()));
    }

    @Test
    void analysisValidatesSnapshotsAndKeepsBufferAndWorkspaceIdentityDomainsSeparate() {
        PlannedPartition partition = partition();
        FakePlan plan = new FakePlan("vector");
        PreparationResourceRequirement.Buffer firstBuffer =
                new PreparationResourceRequirement.Buffer(new ValueId(0), 24, 8);
        PreparationResourceRequirement.Workspace firstWorkspace =
                new PreparationResourceRequirement.Workspace(0, 128, 64);
        List<PreparationResourceRequirement> supplied =
                new ArrayList<>(List.of(firstBuffer, firstWorkspace));

        BackendPartitionAnalysis<FakePlan> analysis =
                new BackendPartitionAnalysis<>(partition, plan, supplied);
        supplied.clear();

        assertAll(
                () -> assertSame(partition, analysis.partition()),
                () -> assertSame(plan, analysis.plan()),
                () -> assertEquals(List.of(firstBuffer, firstWorkspace), analysis.requirements()),
                () -> assertSame(firstBuffer, analysis.requirements().getFirst()),
                () -> assertNotSame(supplied, analysis.requirements()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> analysis.requirements().clear()));
    }

    @Test
    void analysisRejectsNullsAndDuplicatesInEncounterOrder() {
        PlannedPartition partition = partition();
        FakePlan plan = new FakePlan("scalar");
        PreparationResourceRequirement.Buffer buffer =
                new PreparationResourceRequirement.Buffer(new ValueId(4), 8, 8);
        PreparationResourceRequirement.Workspace workspace =
                new PreparationResourceRequirement.Workspace(4, 8, 8);
        List<PreparationResourceRequirement> withNull =
                new ArrayList<>(Arrays.asList(buffer, null, null));

        NullPointerException nullPartition = assertThrows(
                NullPointerException.class,
                () -> new BackendPartitionAnalysis<>(null, null, null));
        NullPointerException nullPlan = assertThrows(
                NullPointerException.class,
                () -> new BackendPartitionAnalysis<>(partition, null, null));
        NullPointerException nullRequirements = assertThrows(
                NullPointerException.class,
                () -> new BackendPartitionAnalysis<>(partition, plan, null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class,
                () -> new BackendPartitionAnalysis<>(partition, plan, withNull));
        IllegalArgumentException duplicateBuffer = assertThrows(
                IllegalArgumentException.class,
                () -> new BackendPartitionAnalysis<>(
                        partition,
                        plan,
                        List.of(
                                buffer,
                                workspace,
                                new PreparationResourceRequirement.Buffer(
                                        new ValueId(4), 16, 16))));
        IllegalArgumentException duplicateWorkspace = assertThrows(
                IllegalArgumentException.class,
                () -> new BackendPartitionAnalysis<>(
                        partition,
                        plan,
                        List.of(
                                workspace,
                                buffer,
                                new PreparationResourceRequirement.Workspace(4, 16, 16))));

        assertAll(
                () -> assertEquals("partition", nullPartition.getMessage()),
                () -> assertEquals("plan", nullPlan.getMessage()),
                () -> assertEquals("requirements", nullRequirements.getMessage()),
                () -> assertEquals("requirements[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "requirements[2] duplicates buffer ValueId[value=4]",
                        duplicateBuffer.getMessage()),
                () -> assertEquals(
                        "requirements[2] duplicates workspace requirementId 4",
                        duplicateWorkspace.getMessage()));
    }

    private static PlannedPartition partition() {
        return new PlannedPartition(
                new BackendId("cpu"), List.of(new NodeId(1)));
    }

    private record FakePlan(String route) implements BackendPreparationPlan {}
}
