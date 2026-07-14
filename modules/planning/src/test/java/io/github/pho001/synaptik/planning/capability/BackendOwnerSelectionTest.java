package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class BackendOwnerSelectionTest {
    private static final TensorDescriptor FLOAT_VECTOR =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final OperationCapabilityQuery QUERY =
            new OperationCapabilityQuery(
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(FLOAT_VECTOR),
                    List.of(FLOAT_VECTOR));

    @Test
    void hasTheExactPackagePrivateStatelessSelectorShape() throws ReflectiveOperationException {
        Class<BackendOwnerSelection> type = BackendOwnerSelection.class;
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method[] methods = type.getDeclaredMethods();
        Method select =
                type.getDeclaredMethod(
                        "select", BackendEligibility.class, PartitionScoringConfig.class, List.class);
        ParameterizedType snapshotListType =
                (ParameterizedType) select.getGenericParameterTypes()[2];

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.planning.capability",
                                type.getPackageName()),
                () -> assertFalse(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertFalse(type.isRecord()),
                () -> assertFalse(type.isEnum()),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(1, methods.length),
                () -> assertEquals(select, methods[0]),
                () -> assertTrue(Modifier.isStatic(select.getModifiers())),
                () -> assertFalse(Modifier.isPublic(select.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(select.getModifiers())),
                () -> assertFalse(Modifier.isProtected(select.getModifiers())),
                () -> assertEquals(BackendId.class, select.getReturnType()),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {
                                    BackendEligibility.class,
                                    PartitionScoringConfig.class,
                                    List.class
                                },
                                select.getParameterTypes()),
                () -> assertEquals(List.class, snapshotListType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {BackendAvailabilitySnapshot.class},
                                snapshotListType.getActualTypeArguments()));
    }

    @Test
    void validatesTopLevelInputsInExactOrderBeforeReadingSnapshotElements() {
        List<BackendAvailabilitySnapshot> unreadableSnapshots = unreadableSnapshotList();

        NullPointerException eligibilityFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> BackendOwnerSelection.select(null, null, unreadableSnapshots));
        NullPointerException configFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> BackendOwnerSelection.select(eligibility(), null, unreadableSnapshots));
        NullPointerException snapshotsFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendOwnerSelection.select(
                                        eligibility(), PartitionScoringConfig.neutral(), null));

        assertAll(
                () -> assertEquals("eligibility", eligibilityFailure.getMessage()),
                () -> assertEquals("scoringConfig", configFailure.getMessage()),
                () -> assertEquals("availabilitySnapshots", snapshotsFailure.getMessage()),
                () -> assertEquals(1, unreadableSnapshots.size()));
    }

    @Test
    void emptyEligibilityFailsBeforeReadingAnySnapshotElement() {
        IllegalStateException failure =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                BackendOwnerSelection.select(
                                        new BackendEligibility(QUERY, List.of()),
                                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                                        unreadableSnapshotList()));

        assertEquals(
                "no hard-eligible backend is available for ownership selection",
                failure.getMessage());
    }

    @Test
    void scansTheCompleteSnapshotListBeforeMissingAssociationOrSelection() {
        BackendId cpu = new BackendId("cpu");
        List<BackendAvailabilitySnapshot> snapshots =
                Arrays.asList(snapshot(new BackendId("extra"), DeviceClass.CPU), null);

        NullPointerException failure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendOwnerSelection.select(
                                        eligibility(cpu),
                                        PartitionScoringConfig.neutral(),
                                        snapshots));

        assertEquals("availabilitySnapshots[1]", failure.getMessage());
    }

    @Test
    void rejectsTheFirstDuplicateEqualSnapshotIdentityBeforeMissingAssociations() {
        BackendId cpu = new BackendId("cpu");
        BackendId extra = new BackendId("extra");

        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendOwnerSelection.select(
                                        eligibility(cpu),
                                        PartitionScoringConfig.neutral(),
                                        List.of(
                                                snapshot(extra, DeviceClass.CPU),
                                                snapshot(
                                                        new BackendId("extra"),
                                                        DeviceClass.ACCELERATOR))));

        assertEquals("duplicate availability snapshot backendId: extra", failure.getMessage());
    }

    @Test
    void rejectsTheFirstMissingEligibleAssociationInProviderOrder() {
        BackendId cpu = new BackendId("cpu");
        BackendId metal = new BackendId("metal");

        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendOwnerSelection.select(
                                        eligibility(cpu, metal),
                                        PartitionScoringConfig.neutral(),
                                        List.of(snapshot(new BackendId("extra"), DeviceClass.CPU))));

        assertEquals("missing availability snapshot for backendId: cpu", failure.getMessage());
    }

    @Test
    void neutralSelectionReturnsTheFirstEligibilityReferenceAfterAssociationValidation() {
        BackendId first = new BackendId("metal");
        BackendId second = new BackendId("cpu");

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(first, second),
                        PartitionScoringConfig.neutral(),
                        List.of(
                                snapshot(new BackendId("cpu"), DeviceClass.CPU),
                                snapshot(new BackendId("metal"), DeviceClass.ACCELERATOR)));

        assertSame(first, result);
    }

    @Test
    void preferredClassReturnsTheFirstProviderOrderMatchNotTheFirstSnapshot() {
        BackendId cpu = new BackendId("cpu");
        BackendId metal = new BackendId("metal");
        BackendId cuda = new BackendId("cuda");

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(cpu, metal, cuda),
                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                        List.of(
                                snapshot(new BackendId("cuda"), DeviceClass.ACCELERATOR),
                                snapshot(new BackendId("cpu"), DeviceClass.CPU),
                                snapshot(new BackendId("metal"), DeviceClass.ACCELERATOR)));

        assertSame(metal, result);
    }

    @Test
    void preferredClassMissFallsBackWithoutRemovingHardEligibleBackends() {
        BackendId cpu = new BackendId("cpu");
        BackendId secondCpu = new BackendId("second-cpu");

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(cpu, secondCpu),
                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                        List.of(
                                snapshot(new BackendId("second-cpu"), DeviceClass.CPU),
                                snapshot(new BackendId("cpu"), DeviceClass.CPU)));

        assertSame(cpu, result);
    }

    @Test
    void emptyMatchingSnapshotIsANonmatchAndMayStillWinFallback() {
        BackendId cpu = new BackendId("cpu");
        BackendId metal = new BackendId("metal");

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(cpu, metal),
                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                        List.of(emptySnapshot(new BackendId("cpu")),
                                snapshot(new BackendId("metal"), DeviceClass.CPU)));

        assertSame(cpu, result);
    }

    @Test
    void hybridBackendMatchesEitherClassWithoutSelectingADevice() {
        BackendId hybrid = new BackendId("hybrid");
        BackendId accelerator = new BackendId("accelerator");
        BackendAvailabilitySnapshot hybridSnapshot =
                new BackendAvailabilitySnapshot(
                        new BackendId("hybrid"),
                        Map.of(
                                new BackendDeviceId(new BackendId("hybrid"), "host"),
                                DeviceClass.CPU,
                                new BackendDeviceId(new BackendId("hybrid"), "gpu"),
                                DeviceClass.ACCELERATOR));

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(hybrid, accelerator),
                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                        List.of(
                                snapshot(
                                        new BackendId("accelerator"),
                                        DeviceClass.ACCELERATOR),
                                hybridSnapshot));

        assertSame(hybrid, result);
    }

    @Test
    void extraPreferredSnapshotsNeverBecomeCandidates() {
        BackendId cpu = new BackendId("cpu");

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility(cpu),
                        PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR),
                        List.of(
                                snapshot(new BackendId("extra"), DeviceClass.ACCELERATOR),
                                snapshot(new BackendId("cpu"), DeviceClass.CPU)));

        assertSame(cpu, result);
    }

    @Test
    void doesNotMutateInputsOrRetainSnapshotIdentityAsTheResult() {
        BackendId eligibilityCpu = new BackendId(new String("cpu"));
        BackendId snapshotCpu = new BackendId(new String("cpu"));
        BackendEligibility eligibility = eligibility(eligibilityCpu);
        BackendAvailabilitySnapshot snapshot = snapshot(snapshotCpu, DeviceClass.CPU);
        List<BackendAvailabilitySnapshot> snapshots = new ArrayList<>(List.of(snapshot));
        List<BackendAvailabilitySnapshot> before = List.copyOf(snapshots);

        BackendId result =
                BackendOwnerSelection.select(
                        eligibility,
                        PartitionScoringConfig.preferring(DeviceClass.CPU),
                        snapshots);

        assertAll(
                () -> assertEquals(before, snapshots),
                () -> assertSame(eligibilityCpu, result),
                () -> assertFalse(result == snapshotCpu),
                () -> assertSame(snapshot, snapshots.getFirst()));
    }

    private static BackendEligibility eligibility(BackendId... backendIds) {
        return new BackendEligibility(QUERY, List.of(backendIds));
    }

    private static BackendAvailabilitySnapshot snapshot(
            BackendId backendId, DeviceClass deviceClass) {
        return new BackendAvailabilitySnapshot(
                backendId,
                Map.of(new BackendDeviceId(backendId, "device"), deviceClass));
    }

    private static BackendAvailabilitySnapshot emptySnapshot(BackendId backendId) {
        return new BackendAvailabilitySnapshot(backendId, Map.of());
    }

    private static List<BackendAvailabilitySnapshot> unreadableSnapshotList() {
        return new AbstractList<>() {
            @Override
            public BackendAvailabilitySnapshot get(int index) {
                throw new AssertionError("snapshot element must not be read");
            }

            @Override
            public int size() {
                return 1;
            }
        };
    }
}
