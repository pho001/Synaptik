package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendDeviceIdRequirement;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.contract.DeviceClassRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;
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
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendEligibilityTest {
    private static final TensorDescriptor FLOAT_VECTOR =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final OperationCapabilityQuery QUERY =
            new OperationCapabilityQuery(
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(FLOAT_VECTOR),
                    List.of(FLOAT_VECTOR));

    @Test
    void hasTheExactPackagePrivateGenericRecordAndFactoryShape()
            throws ReflectiveOperationException {
        Class<BackendEligibility> type = BackendEligibility.class;
        RecordComponent[] components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method evaluate =
                type.getDeclaredMethod(
                        "evaluate",
                        OperationCapabilityQuery.class,
                        BackendIntent.class,
                        List.class,
                        List.class);
        ParameterizedType componentListType = (ParameterizedType) components[1].getGenericType();
        ParameterizedType providerListType =
                (ParameterizedType) evaluate.getGenericParameterTypes()[2];
        ParameterizedType snapshotListType =
                (ParameterizedType) evaluate.getGenericParameterTypes()[3];

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.planning.capability",
                                type.getPackageName()),
                () -> assertFalse(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"query", "eligibleBackendIds"},
                                Arrays.stream(components)
                                        .map(RecordComponent::getName)
                                        .toArray(String[]::new)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {OperationCapabilityQuery.class, List.class},
                                Arrays.stream(components)
                                        .map(RecordComponent::getType)
                                        .toArray(Class<?>[]::new)),
                () -> assertEquals(List.class, componentListType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {BackendId.class},
                                componentListType.getActualTypeArguments()),
                () -> assertEquals(2, type.getDeclaredFields().length),
                () ->
                        assertArrayEquals(
                                new String[] {"query", "eligibleBackendIds"},
                                Arrays.stream(type.getDeclaredFields())
                                        .map(field -> field.getName())
                                        .toArray(String[]::new)),
                () -> assertEquals(1, constructors.length),
                () -> assertFalse(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {OperationCapabilityQuery.class, List.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(6, type.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "query",
                                        "eligibleBackendIds",
                                        "evaluate",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                Arrays.stream(type.getDeclaredMethods())
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())),
                () -> assertTrue(Modifier.isPublic(type.getDeclaredMethod("query").getModifiers())),
                () ->
                        assertTrue(
                                Modifier.isPublic(
                                        type.getDeclaredMethod("eligibleBackendIds")
                                                .getModifiers())),
                () -> assertTrue(Modifier.isStatic(evaluate.getModifiers())),
                () -> assertFalse(Modifier.isPublic(evaluate.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(evaluate.getModifiers())),
                () -> assertEquals(BackendEligibility.class, evaluate.getReturnType()),
                () -> assertEquals(List.class, providerListType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {BackendCapabilityProvider.class},
                                providerListType.getActualTypeArguments()),
                () -> assertEquals(List.class, snapshotListType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {BackendAvailabilitySnapshot.class},
                                snapshotListType.getActualTypeArguments()),
                () ->
                        assertEquals(
                                Set.of(
                                        "query",
                                        "eligibleBackendIds",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                Arrays.stream(type.getDeclaredMethods())
                                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())));
    }

    @Test
    void constructorValidatesInExactOrderWithExactMessages() {
        NullPointerException queryFailure =
                assertThrows(
                        NullPointerException.class, () -> new BackendEligibility(null, null));
        NullPointerException listFailure =
                assertThrows(
                        NullPointerException.class, () -> new BackendEligibility(QUERY, null));
        NullPointerException elementFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                new BackendEligibility(
                                        QUERY,
                                        Arrays.asList(new BackendId("cpu"), null, null)));
        IllegalArgumentException duplicateFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new BackendEligibility(
                                        QUERY,
                                        List.of(new BackendId("cpu"), new BackendId("cpu"))));

        assertAll(
                () -> assertEquals("query", queryFailure.getMessage()),
                () -> assertEquals("eligibleBackendIds", listFailure.getMessage()),
                () -> assertEquals("eligibleBackendIds[1]", elementFailure.getMessage()),
                () ->
                        assertEquals(
                                "duplicate eligible backendId: cpu",
                                duplicateFailure.getMessage()));
    }

    @Test
    void constructorSnapshotsMembershipRetainsReferencesAndUsesOrdinaryRecordSemantics() {
        BackendId cpu = new BackendId("cpu");
        BackendId metal = new BackendId("metal");
        List<BackendId> source = new ArrayList<>(List.of(cpu, metal));

        BackendEligibility eligibility = new BackendEligibility(QUERY, source);
        BackendEligibility equalValue =
                new BackendEligibility(QUERY, List.of(new BackendId("cpu"), new BackendId("metal")));
        source.clear();

        assertAll(
                () -> assertSame(QUERY, eligibility.query()),
                () -> assertNotSame(source, eligibility.eligibleBackendIds()),
                () -> assertSame(cpu, eligibility.eligibleBackendIds().get(0)),
                () -> assertSame(metal, eligibility.eligibleBackendIds().get(1)),
                () -> assertEquals(equalValue, eligibility),
                () -> assertEquals(equalValue.hashCode(), eligibility.hashCode()),
                () -> assertEquals(equalValue.toString(), eligibility.toString()),
                () -> assertNotEquals(new BackendEligibility(QUERY, List.of(metal, cpu)), eligibility),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> eligibility.eligibleBackendIds().clear()));
    }

    @Test
    void factoryValidatesTopLevelReferencesBeforeReadingAnyListElement() {
        List<BackendCapabilityProvider> unreadableProviders = unreadableList();
        List<BackendAvailabilitySnapshot> unreadableSnapshots = unreadableList();

        NullPointerException queryFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> BackendEligibility.evaluate(null, null, null, null));
        NullPointerException intentFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> BackendEligibility.evaluate(QUERY, null, null, null));
        NullPointerException providersFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY, BackendIntent.unconstrained(), null, null));
        NullPointerException snapshotsFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        unreadableProviders,
                                        null));

        assertAll(
                () -> assertEquals("query", queryFailure.getMessage()),
                () -> assertEquals("intent", intentFailure.getMessage()),
                () -> assertEquals("providers", providersFailure.getMessage()),
                () -> assertEquals("availabilitySnapshots", snapshotsFailure.getMessage()),
                () -> assertEquals(1, unreadableSnapshots.size()));
    }

    @Test
    void scansAllProvidersBeforeSnapshotsAndCallsEachBackendIdExactlyOnce() {
        List<String> events = new ArrayList<>();
        RecordingProvider cpu = provider(new BackendId("cpu"), true, events);
        RecordingProvider metal = provider(new BackendId("metal"), true, events);
        List<BackendAvailabilitySnapshot> snapshots =
                new AbstractList<>() {
                    @Override
                    public BackendAvailabilitySnapshot get(int index) {
                        events.add("snapshot:" + index);
                        return index == 0
                                ? snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR)
                                : snapshot(new BackendId("cpu"), "host", DeviceClass.CPU);
                    }

                    @Override
                    public int size() {
                        return 2;
                    }
                };

        BackendEligibility result =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.unconstrained(),
                        List.of(cpu, metal),
                        snapshots);

        assertAll(
                () -> assertEquals(1, cpu.backendIdCalls),
                () -> assertEquals(1, metal.backendIdCalls),
                () -> assertEquals(1, cpu.supportsCalls),
                () -> assertEquals(1, metal.supportsCalls),
                () ->
                        assertEquals(
                                List.of(
                                        "backendId:cpu",
                                        "backendId:metal",
                                        "snapshot:0",
                                        "snapshot:1",
                                        "supports:cpu",
                                        "supports:metal"),
                                events),
                () -> assertSame(cpu.backendId, result.eligibleBackendIds().get(0)),
                () -> assertSame(metal.backendId, result.eligibleBackendIds().get(1)));
    }

    @Test
    void reportsProviderNullIdentityAndDuplicateFailuresBeforeSnapshotReadsOrSupportCalls() {
        List<String> events = new ArrayList<>();
        RecordingProvider first = provider(new BackendId("cpu"), true, events);
        RecordingProvider nullIdentity = provider(null, true, events);
        RecordingProvider duplicate = provider(new BackendId("cpu"), true, events);
        List<BackendAvailabilitySnapshot> unreadableSnapshots = unreadableList();

        NullPointerException nullProviderFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        Arrays.asList(first, null, duplicate),
                                        unreadableSnapshots));
        NullPointerException nullIdentityFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(first, nullIdentity),
                                        unreadableSnapshots));
        IllegalArgumentException duplicateFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(first, duplicate),
                                        unreadableSnapshots));

        assertAll(
                () -> assertEquals("providers[1]", nullProviderFailure.getMessage()),
                () ->
                        assertEquals(
                                "providers[1].backendId()", nullIdentityFailure.getMessage()),
                () ->
                        assertEquals(
                                "duplicate provider backendId: cpu",
                                duplicateFailure.getMessage()),
                () -> assertEquals(0, first.supportsCalls),
                () -> assertEquals(0, nullIdentity.supportsCalls),
                () -> assertEquals(0, duplicate.supportsCalls));
    }

    @Test
    void reportsSnapshotNullAndDuplicateFailuresAfterTheCompleteProviderScan() {
        List<String> events = new ArrayList<>();
        RecordingProvider cpu = provider(new BackendId("cpu"), true, events);
        RecordingProvider metal = provider(new BackendId("metal"), true, events);

        NullPointerException nullFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(cpu, metal),
                                        Arrays.asList(snapshot(cpu.backendId, "host", DeviceClass.CPU), null)));
        IllegalArgumentException duplicateFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(cpu, metal),
                                        List.of(
                                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU),
                                                snapshot(new BackendId("cpu"), "host-2", DeviceClass.CPU))));

        assertAll(
                () -> assertEquals("availabilitySnapshots[1]", nullFailure.getMessage()),
                () ->
                        assertEquals(
                                "duplicate availability snapshot backendId: cpu",
                                duplicateFailure.getMessage()),
                () -> assertEquals(2, cpu.backendIdCalls),
                () -> assertEquals(2, metal.backendIdCalls),
                () -> assertEquals(0, cpu.supportsCalls),
                () -> assertEquals(0, metal.supportsCalls));
    }

    @Test
    void validatesMissingSnapshotsThenMissingProvidersInSpecifiedEncounterOrder() {
        List<String> events = new ArrayList<>();
        RecordingProvider cpu = provider(new BackendId("cpu"), true, events);
        RecordingProvider metal = provider(new BackendId("metal"), true, events);
        RecordingProvider cuda = provider(new BackendId("cuda"), true, events);

        IllegalArgumentException missingSnapshot =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(cpu, metal, cuda),
                                        List.of(snapshot(new BackendId("other"), "x", DeviceClass.CPU))));
        IllegalArgumentException missingProvider =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(cpu),
                                        List.of(
                                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU),
                                                snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR),
                                                snapshot(new BackendId("cuda"), "gpu", DeviceClass.ACCELERATOR))));

        assertAll(
                () ->
                        assertEquals(
                                "missing availability snapshot for backendId: cpu",
                                missingSnapshot.getMessage()),
                () ->
                        assertEquals(
                                "missing capability provider for backendId: metal",
                                missingProvider.getMessage()),
                () -> assertEquals(0, cpu.supportsCalls),
                () -> assertEquals(0, metal.supportsCalls),
                () -> assertEquals(0, cuda.supportsCalls));
    }

    @Test
    void unconstrainedEvaluationFiltersAvailabilityAndSupportInProviderOrder() {
        List<String> events = new ArrayList<>();
        RecordingProvider cpu = provider(new BackendId("cpu"), true, events);
        RecordingProvider metal = provider(new BackendId("metal"), false, events);
        RecordingProvider cuda = provider(new BackendId("cuda"), true, events);

        BackendEligibility result =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.unconstrained(),
                        List.of(cpu, metal, cuda),
                        List.of(
                                snapshot(new BackendId("cuda")),
                                snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR),
                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU)));

        assertAll(
                () -> assertEquals(List.of(cpu.backendId), result.eligibleBackendIds()),
                () -> assertSame(cpu.backendId, result.eligibleBackendIds().getFirst()),
                () -> assertEquals(1, cpu.supportsCalls),
                () -> assertEquals(1, metal.supportsCalls),
                () -> assertEquals(0, cuda.supportsCalls),
                () -> assertSame(QUERY, cpu.queries.getFirst()),
                () -> assertSame(QUERY, metal.queries.getFirst()),
                () ->
                        assertEquals(
                                List.of(
                                        "backendId:cpu",
                                        "backendId:metal",
                                        "backendId:cuda",
                                        "supports:cpu",
                                        "supports:metal"),
                                events));
    }

    @Test
    void exactBackendRequirementMatchesByValueWithoutFallback() {
        RecordingProvider cpu = provider(new BackendId("cpu"), true, new ArrayList<>());
        RecordingProvider metal = provider(new BackendId("metal"), true, new ArrayList<>());

        BackendEligibility match =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(new BackendIdRequirement(new BackendId("metal"))),
                        List.of(cpu, metal),
                        List.of(
                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU),
                                snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR)));
        BackendEligibility noMatch =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(new BackendIdRequirement(new BackendId("cuda"))),
                        List.of(cpu, metal),
                        List.of(
                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU),
                                snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR)));

        assertAll(
                () -> assertEquals(List.of(metal.backendId), match.eligibleBackendIds()),
                () -> assertSame(metal.backendId, match.eligibleBackendIds().getFirst()),
                () -> assertTrue(noMatch.eligibleBackendIds().isEmpty()),
                () -> assertEquals(0, cpu.supportsCalls),
                () -> assertEquals(1, metal.supportsCalls));
    }

    @Test
    void exactDeviceRequirementUsesEqualSnapshotKeyOnlyAsAvailabilityProof() {
        BackendId providerMetalId = new BackendId("metal");
        BackendDeviceId requiredDevice = new BackendDeviceId(new BackendId("metal"), "gpu-1");
        RecordingProvider cpu = provider(new BackendId("cpu"), true, new ArrayList<>());
        RecordingProvider metal = provider(providerMetalId, true, new ArrayList<>());
        BackendAvailabilitySnapshot cpuSnapshot =
                snapshot(new BackendId("cpu"), "gpu-1", DeviceClass.CPU);
        BackendAvailabilitySnapshot metalSnapshot =
                snapshot(new BackendId("metal"), "gpu-1", DeviceClass.ACCELERATOR);

        BackendEligibility match =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(new BackendDeviceIdRequirement(requiredDevice)),
                        List.of(cpu, metal),
                        List.of(metalSnapshot, cpuSnapshot));
        BackendEligibility absentDevice =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(
                                new BackendDeviceIdRequirement(
                                        new BackendDeviceId(new BackendId("metal"), "gpu-2"))),
                        List.of(cpu, metal),
                        List.of(metalSnapshot, cpuSnapshot));

        assertAll(
                () -> assertEquals(List.of(providerMetalId), match.eligibleBackendIds()),
                () -> assertSame(providerMetalId, match.eligibleBackendIds().getFirst()),
                () -> assertTrue(absentDevice.eligibleBackendIds().isEmpty()),
                () -> assertEquals(0, cpu.supportsCalls),
                () -> assertEquals(1, metal.supportsCalls),
                () ->
                        assertFalse(
                                match.getClass()
                                        .getRecordComponents()[1]
                                        .getGenericType()
                                        .getTypeName()
                                        .contains("BackendDeviceId")));
    }

    @Test
    void deviceClassRequirementMatchesAnyAvailableDeviceWithoutSelectingIt() {
        RecordingProvider cpu = provider(new BackendId("cpu"), true, new ArrayList<>());
        RecordingProvider hybrid = provider(new BackendId("hybrid"), true, new ArrayList<>());
        BackendAvailabilitySnapshot cpuSnapshot =
                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU);
        BackendAvailabilitySnapshot hybridSnapshot =
                new BackendAvailabilitySnapshot(
                        new BackendId("hybrid"),
                        Map.of(
                                new BackendDeviceId(new BackendId("hybrid"), "host"),
                                DeviceClass.CPU,
                                new BackendDeviceId(new BackendId("hybrid"), "gpu"),
                                DeviceClass.ACCELERATOR));

        BackendEligibility result =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(
                                new DeviceClassRequirement(DeviceClass.ACCELERATOR)),
                        List.of(cpu, hybrid),
                        List.of(cpuSnapshot, hybridSnapshot));

        assertAll(
                () -> assertEquals(List.of(hybrid.backendId), result.eligibleBackendIds()),
                () -> assertEquals(0, cpu.supportsCalls),
                () -> assertEquals(1, hybrid.supportsCalls),
                () -> assertEquals(BackendId.class, result.eligibleBackendIds().getFirst().getClass()));
    }

    @Test
    void availabilityAndRequirementFilteringPrecedeCapabilityCalls() {
        List<String> events = new ArrayList<>();
        AssertionError unavailableFailure = new AssertionError("unavailable provider was called");
        AssertionError mismatchedFailure = new AssertionError("mismatched provider was called");
        RecordingProvider unavailable = provider(new BackendId("cpu"), unavailableFailure, events);
        RecordingProvider mismatched = provider(new BackendId("metal"), mismatchedFailure, events);
        RecordingProvider matching = provider(new BackendId("cuda"), true, events);

        BackendEligibility result =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.requiring(new BackendIdRequirement(new BackendId("cuda"))),
                        List.of(unavailable, mismatched, matching),
                        List.of(
                                snapshot(new BackendId("cpu")),
                                snapshot(new BackendId("metal"), "gpu", DeviceClass.ACCELERATOR),
                                snapshot(new BackendId("cuda"), "gpu", DeviceClass.ACCELERATOR)));

        assertAll(
                () -> assertEquals(List.of(matching.backendId), result.eligibleBackendIds()),
                () -> assertEquals(0, unavailable.supportsCalls),
                () -> assertEquals(0, mismatched.supportsCalls),
                () -> assertEquals(1, matching.supportsCalls));
    }

    @Test
    void providerFailurePropagatesUnchangedAndStopsLaterCalls() {
        List<String> events = new ArrayList<>();
        RecordingProvider first = provider(new BackendId("cpu"), true, events);
        IllegalStateException providerFailure = new IllegalStateException("provider failed");
        RecordingProvider failing = provider(new BackendId("metal"), providerFailure, events);
        RecordingProvider later = provider(new BackendId("cuda"), true, events);

        IllegalStateException actual =
                assertThrows(
                        IllegalStateException.class,
                        () ->
                                BackendEligibility.evaluate(
                                        QUERY,
                                        BackendIntent.unconstrained(),
                                        List.of(first, failing, later),
                                        List.of(
                                                snapshot(new BackendId("cpu"), "host", DeviceClass.CPU),
                                                snapshot(
                                                        new BackendId("metal"),
                                                        "gpu",
                                                        DeviceClass.ACCELERATOR),
                                                snapshot(
                                                        new BackendId("cuda"),
                                                        "gpu",
                                                        DeviceClass.ACCELERATOR))));

        assertAll(
                () -> assertSame(providerFailure, actual),
                () -> assertEquals(1, first.supportsCalls),
                () -> assertEquals(1, failing.supportsCalls),
                () -> assertEquals(0, later.supportsCalls),
                () ->
                        assertEquals(
                                List.of(
                                        "backendId:cpu",
                                        "backendId:metal",
                                        "backendId:cuda",
                                        "supports:cpu",
                                        "supports:metal"),
                                events));
    }

    @Test
    void emptyCompositionAndEveryValidNoMatchReturnImmutableEmptyResults() {
        BackendEligibility emptyComposition =
                BackendEligibility.evaluate(
                        QUERY, BackendIntent.unconstrained(), List.of(), List.of());
        RecordingProvider unsupported =
                provider(new BackendId("cpu"), false, new ArrayList<>());
        BackendEligibility unsupportedResult =
                BackendEligibility.evaluate(
                        QUERY,
                        BackendIntent.unconstrained(),
                        List.of(unsupported),
                        List.of(snapshot(new BackendId("cpu"), "host", DeviceClass.CPU)));

        assertAll(
                () -> assertSame(QUERY, emptyComposition.query()),
                () -> assertTrue(emptyComposition.eligibleBackendIds().isEmpty()),
                () -> assertTrue(unsupportedResult.eligibleBackendIds().isEmpty()),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () ->
                                        emptyComposition
                                                .eligibleBackendIds()
                                                .add(new BackendId("cpu"))),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () ->
                                        unsupportedResult
                                                .eligibleBackendIds()
                                                .add(new BackendId("cpu"))));
    }

    private static RecordingProvider provider(
            BackendId backendId, Object supportOutcome, List<String> events) {
        return new RecordingProvider(backendId, supportOutcome, events);
    }

    private static BackendAvailabilitySnapshot snapshot(BackendId backendId) {
        return new BackendAvailabilitySnapshot(backendId, Map.of());
    }

    private static BackendAvailabilitySnapshot snapshot(
            BackendId backendId, String deviceValue, DeviceClass deviceClass) {
        return new BackendAvailabilitySnapshot(
                backendId,
                Map.of(new BackendDeviceId(new BackendId(backendId.value()), deviceValue), deviceClass));
    }

    private static <T> List<T> unreadableList() {
        return new AbstractList<>() {
            @Override
            public T get(int index) {
                throw new AssertionError("list element must not be read");
            }

            @Override
            public int size() {
                return 1;
            }
        };
    }

    private static final class RecordingProvider implements BackendCapabilityProvider {
        private final BackendId backendId;
        private final Object supportOutcome;
        private final List<String> events;
        private final List<OperationCapabilityQuery> queries = new ArrayList<>();
        private int backendIdCalls;
        private int supportsCalls;

        private RecordingProvider(
                BackendId backendId, Object supportOutcome, List<String> events) {
            this.backendId = backendId;
            this.supportOutcome = supportOutcome;
            this.events = events;
        }

        @Override
        public BackendId backendId() {
            backendIdCalls++;
            events.add("backendId:" + (backendId == null ? "null" : backendId.value()));
            return backendId;
        }

        @Override
        public boolean supports(OperationCapabilityQuery query) {
            supportsCalls++;
            queries.add(query);
            events.add("supports:" + backendId.value());
            if (supportOutcome instanceof RuntimeException failure) {
                throw failure;
            }
            if (supportOutcome instanceof Error failure) {
                throw failure;
            }
            return (boolean) supportOutcome;
        }
    }
}
