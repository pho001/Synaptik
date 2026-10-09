package io.github.pho001.synaptik.trace.payload;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.id.TraceBackendId;
import io.github.pho001.synaptik.trace.id.TraceDeviceId;
import io.github.pho001.synaptik.trace.id.TraceInvocationId;
import io.github.pho001.synaptik.trace.id.TracePreparedUnitId;
import java.io.Serializable;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendOutcomePayloadTest {
    private static final TraceBackendId BACKEND_ID = new TraceBackendId(11L);
    private static final TraceDeviceId DEVICE_ID = new TraceDeviceId(12L);
    private static final TracePreparedUnitId PREPARED_UNIT_ID = new TracePreparedUnitId(13L);
    private static final TraceInvocationId INVOCATION_ID = new TraceInvocationId(14L);
    private static final TraceNativeStatus NATIVE_SUCCESS =
            new TraceNativeStatus(TraceNativeStatusKind.SUCCESS, 0);
    private static final TraceNativeStatus NATIVE_FAILURE =
            new TraceNativeStatus(TraceNativeStatusKind.EXECUTION_FAILED, -77);

    @Test
    void enumsHaveExactlyTheRequiredConstantsAndNoAddedApi() {
        assertExactEnum(
                TraceOutcomeStatus.class,
                new TraceOutcomeStatus[] {
                    TraceOutcomeStatus.SUCCEEDED, TraceOutcomeStatus.FAILED
                });
        assertExactEnum(
                TraceRouteKind.class,
                new TraceRouteKind[] {
                    TraceRouteKind.CUSTOM_KERNEL, TraceRouteKind.GRAPH_EXECUTABLE
                });
        assertExactEnum(
                TraceCacheStatus.class,
                new TraceCacheStatus[] {TraceCacheStatus.NOT_QUERIED});
        assertExactEnum(
                TraceNativeStatusKind.class,
                new TraceNativeStatusKind[] {
                    TraceNativeStatusKind.SUCCESS,
                    TraceNativeStatusKind.INVALID_ARGUMENT,
                    TraceNativeStatusKind.DEVICE_UNAVAILABLE,
                    TraceNativeStatusKind.COMMAND_QUEUE_UNAVAILABLE,
                    TraceNativeStatusKind.ALLOCATION_FAILED,
                    TraceNativeStatusKind.RANGE_OUT_OF_BOUNDS,
                    TraceNativeStatusKind.COPY_FAILED,
                    TraceNativeStatusKind.INTERNAL_ERROR,
                    TraceNativeStatusKind.UNSUPPORTED_SHAPE,
                    TraceNativeStatusKind.COMPILATION_FAILED,
                    TraceNativeStatusKind.INCOMPATIBLE_RESOURCE,
                    TraceNativeStatusKind.EXECUTION_FAILED,
                    TraceNativeStatusKind.UNKNOWN
                });
    }

    @Test
    void removedNumericalSelectorIsNotAnAvailableTraceType() {
        assertThrows(ClassNotFoundException.class, () -> Class.forName(
                "io.github.pho001.synaptik.trace.payload.TraceNumericalProfile"));
    }

    @Test
    void nativeStatusHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        assertRecordShape(
                TraceNativeStatus.class,
                new String[] {"kind", "code"},
                new Class<?>[] {TraceNativeStatusKind.class, int.class},
                Set.of(),
                Set.of("equals", "hashCode", "toString", "kind", "code"));
    }

    @Test
    void preparationOutcomeHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        assertRecordShape(
                BackendPreparationOutcome.class,
                new String[] {
                    "backendId",
                    "deviceId",
                    "preparedUnitId",
                    "status",
                    "route",
                    "cacheStatus",
                    "nativeStatus"
                },
                new Class<?>[] {
                    TraceBackendId.class,
                    TraceDeviceId.class,
                    TracePreparedUnitId.class,
                    TraceOutcomeStatus.class,
                    TraceRouteKind.class,
                    TraceCacheStatus.class,
                    Optional.class
                },
                Set.of(TracePayload.class),
                Set.of(
                        "equals",
                        "hashCode",
                        "toString",
                        "backendId",
                        "deviceId",
                        "preparedUnitId",
                        "status",
                        "route",
                        "cacheStatus",
                        "nativeStatus"));
        assertOptionalNativeStatusComponent(
                BackendPreparationOutcome.class.getRecordComponents()[6]);
    }

    @Test
    void invocationOutcomeHasTheExactPublicRecordShape() throws ReflectiveOperationException {
        assertRecordShape(
                BackendInvocationOutcome.class,
                new String[] {
                    "backendId",
                    "deviceId",
                    "preparedUnitId",
                    "invocationId",
                    "status",
                    "route",
                    "nativeStatus"
                },
                new Class<?>[] {
                    TraceBackendId.class,
                    TraceDeviceId.class,
                    TracePreparedUnitId.class,
                    TraceInvocationId.class,
                    TraceOutcomeStatus.class,
                    TraceRouteKind.class,
                    Optional.class
                },
                Set.of(TracePayload.class),
                Set.of(
                        "equals",
                        "hashCode",
                        "toString",
                        "backendId",
                        "deviceId",
                        "preparedUnitId",
                        "invocationId",
                        "status",
                        "route",
                        "nativeStatus"));
        assertOptionalNativeStatusComponent(
                BackendInvocationOutcome.class.getRecordComponents()[6]);
    }

    @Test
    void publicRecordsExposeOnlyTraceOwnedLeafComponents() {
        for (Class<?> type : new Class<?>[] {
            TraceNativeStatus.class,
            BackendPreparationOutcome.class,
            BackendInvocationOutcome.class
        }) {
            for (RecordComponent component : type.getRecordComponents()) {
                Class<?> componentType = component.getType();
                assertTrue(
                        componentType.isPrimitive()
                                || componentType == Optional.class
                                || componentType.getPackageName()
                                        .startsWith("io.github.pho001.synaptik.trace"),
                        () -> type.getSimpleName() + "." + component.getName()
                                + " exposes " + componentType.getTypeName());
                assertFalse(componentType.isArray());
                assertNotEquals(String.class, componentType);
                assertFalse(Throwable.class.isAssignableFrom(componentType));
                assertFalse(java.util.Map.class.isAssignableFrom(componentType));
                assertFalse(java.util.Collection.class.isAssignableFrom(componentType));
            }
        }
    }

    @Test
    void nativeStatusEnforcesSuccessIfAndOnlyIfCodeIsZero() {
        TraceNativeStatus success = new TraceNativeStatus(TraceNativeStatusKind.SUCCESS, 0);
        assertAll(
                () -> assertSame(TraceNativeStatusKind.SUCCESS, success.kind()),
                () -> assertEquals(0, success.code()),
                () -> assertExactIllegalArgument(
                        "SUCCESS requires code 0",
                        () -> new TraceNativeStatus(TraceNativeStatusKind.SUCCESS, -1)),
                () -> assertExactIllegalArgument(
                        "SUCCESS requires code 0",
                        () -> new TraceNativeStatus(TraceNativeStatusKind.SUCCESS, 1)));

        for (TraceNativeStatusKind kind : TraceNativeStatusKind.values()) {
            if (kind != TraceNativeStatusKind.SUCCESS) {
                assertExactIllegalArgument(
                        "code 0 requires SUCCESS", () -> new TraceNativeStatus(kind, 0));
            }
        }
    }

    @Test
    void nativeStatusRetainsEverySignedNonzeroCodeExactly() {
        for (int code : new int[] {Integer.MIN_VALUE, -1, 1, Integer.MAX_VALUE}) {
            TraceNativeStatus status = new TraceNativeStatus(TraceNativeStatusKind.UNKNOWN, code);
            assertAll(
                    () -> assertSame(TraceNativeStatusKind.UNKNOWN, status.kind()),
                    () -> assertEquals(code, status.code()));
        }
    }

    @Test
    void nativeStatusChecksNullKindBeforeItsCodeInvariant() {
        NullPointerException failure = assertThrows(
                NullPointerException.class, () -> new TraceNativeStatus(null, 0));
        assertEquals("kind", failure.getMessage());
    }

    @Test
    void preparationOutcomeChecksNullsInExactComponentOrder() {
        assertNullMessage("backendId", () -> new BackendPreparationOutcome(
                null, null, null, null, null, null, null));
        assertNullMessage("deviceId", () -> new BackendPreparationOutcome(
                BACKEND_ID, null, null, null, null, null, null));
        assertNullMessage("preparedUnitId", () -> new BackendPreparationOutcome(
                BACKEND_ID, DEVICE_ID, null, null, null, null, null));
        assertNullMessage("status", () -> new BackendPreparationOutcome(
                BACKEND_ID, DEVICE_ID, PREPARED_UNIT_ID, null, null, null, null));
        assertNullMessage("route", () -> new BackendPreparationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                TraceOutcomeStatus.SUCCEEDED,
                null,
                null,
                null));
        assertNullMessage("cacheStatus", () -> new BackendPreparationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                TraceOutcomeStatus.SUCCEEDED,
                TraceRouteKind.CUSTOM_KERNEL,
                null,
                null));
        assertNullMessage("nativeStatus", () -> new BackendPreparationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                TraceOutcomeStatus.SUCCEEDED,
                TraceRouteKind.CUSTOM_KERNEL,
                TraceCacheStatus.NOT_QUERIED,
                null));
    }

    @Test
    void invocationOutcomeChecksNullsInExactComponentOrder() {
        assertNullMessage("backendId", () -> new BackendInvocationOutcome(
                null, null, null, null, null, null, null));
        assertNullMessage("deviceId", () -> new BackendInvocationOutcome(
                BACKEND_ID, null, null, null, null, null, null));
        assertNullMessage("preparedUnitId", () -> new BackendInvocationOutcome(
                BACKEND_ID, DEVICE_ID, null, null, null, null, null));
        assertNullMessage("invocationId", () -> new BackendInvocationOutcome(
                BACKEND_ID, DEVICE_ID, PREPARED_UNIT_ID, null, null, null, null));
        assertNullMessage("status", () -> new BackendInvocationOutcome(
                BACKEND_ID, DEVICE_ID, PREPARED_UNIT_ID, INVOCATION_ID, null, null, null));
        assertNullMessage("route", () -> new BackendInvocationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                INVOCATION_ID,
                TraceOutcomeStatus.SUCCEEDED,
                null,
                null));
        assertNullMessage("nativeStatus", () -> new BackendInvocationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                INVOCATION_ID,
                TraceOutcomeStatus.SUCCEEDED,
                TraceRouteKind.GRAPH_EXECUTABLE,
                null));
    }

    @Test
    void preparationOutcomeEnforcesTheStatusNativeStatusInvariant() {
        assertExactIllegalArgument(
                "SUCCEEDED requires SUCCESS nativeStatus",
                () -> preparation(TraceOutcomeStatus.SUCCEEDED, Optional.empty()));
        assertExactIllegalArgument(
                "SUCCEEDED requires SUCCESS nativeStatus",
                () -> preparation(TraceOutcomeStatus.SUCCEEDED, Optional.of(NATIVE_FAILURE)));
        assertExactIllegalArgument(
                "FAILED forbids SUCCESS nativeStatus",
                () -> preparation(TraceOutcomeStatus.FAILED, Optional.of(NATIVE_SUCCESS)));

        preparation(TraceOutcomeStatus.SUCCEEDED, Optional.of(NATIVE_SUCCESS));
        preparation(TraceOutcomeStatus.FAILED, Optional.empty());
        preparation(TraceOutcomeStatus.FAILED, Optional.of(NATIVE_FAILURE));
    }

    @Test
    void invocationOutcomeEnforcesTheStatusNativeStatusInvariant() {
        assertExactIllegalArgument(
                "SUCCEEDED requires SUCCESS nativeStatus",
                () -> invocation(TraceOutcomeStatus.SUCCEEDED, Optional.empty()));
        assertExactIllegalArgument(
                "SUCCEEDED requires SUCCESS nativeStatus",
                () -> invocation(TraceOutcomeStatus.SUCCEEDED, Optional.of(NATIVE_FAILURE)));
        assertExactIllegalArgument(
                "FAILED forbids SUCCESS nativeStatus",
                () -> invocation(TraceOutcomeStatus.FAILED, Optional.of(NATIVE_SUCCESS)));

        invocation(TraceOutcomeStatus.SUCCEEDED, Optional.of(NATIVE_SUCCESS));
        invocation(TraceOutcomeStatus.FAILED, Optional.empty());
        invocation(TraceOutcomeStatus.FAILED, Optional.of(NATIVE_FAILURE));
    }

    @Test
    void outcomesRetainExactReferencesAndOrdinaryRecordValueSemantics() {
        Optional<TraceNativeStatus> nativeStatus = Optional.of(NATIVE_SUCCESS);
        BackendPreparationOutcome preparation = preparation(
                TraceOutcomeStatus.SUCCEEDED, nativeStatus);
        BackendInvocationOutcome invocation = invocation(
                TraceOutcomeStatus.SUCCEEDED, nativeStatus);

        assertAll(
                () -> assertSame(BACKEND_ID, preparation.backendId()),
                () -> assertSame(DEVICE_ID, preparation.deviceId()),
                () -> assertSame(PREPARED_UNIT_ID, preparation.preparedUnitId()),
                () -> assertSame(nativeStatus, preparation.nativeStatus()),
                () -> assertSame(INVOCATION_ID, invocation.invocationId()),
                () -> assertSame(nativeStatus, invocation.nativeStatus()),
                () -> assertEquals(
                        preparation,
                        new BackendPreparationOutcome(
                                new TraceBackendId(11L),
                                new TraceDeviceId(12L),
                                new TracePreparedUnitId(13L),
                                TraceOutcomeStatus.SUCCEEDED,
                                TraceRouteKind.CUSTOM_KERNEL,
                                TraceCacheStatus.NOT_QUERIED,
                                Optional.of(new TraceNativeStatus(
                                        TraceNativeStatusKind.SUCCESS, 0)))),
                () -> assertNotEquals(
                        invocation,
                        invocation(TraceOutcomeStatus.FAILED, Optional.of(NATIVE_FAILURE))));
    }

    private static BackendPreparationOutcome preparation(
            TraceOutcomeStatus status, Optional<TraceNativeStatus> nativeStatus) {
        return new BackendPreparationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                status,
                TraceRouteKind.CUSTOM_KERNEL,
                TraceCacheStatus.NOT_QUERIED,
                nativeStatus);
    }

    private static BackendInvocationOutcome invocation(
            TraceOutcomeStatus status, Optional<TraceNativeStatus> nativeStatus) {
        return new BackendInvocationOutcome(
                BACKEND_ID,
                DEVICE_ID,
                PREPARED_UNIT_ID,
                INVOCATION_ID,
                status,
                TraceRouteKind.CUSTOM_KERNEL,
                nativeStatus);
    }

    private static void assertRecordShape(
            Class<?> type,
            String[] componentNames,
            Class<?>[] componentTypes,
            Set<Class<?>> expectedInterfaces,
            Set<String> expectedPublicMethods) {
        RecordComponent[] components = type.getRecordComponents();
        var fields = type.getDeclaredFields();
        var constructors = type.getDeclaredConstructors();

        assertAll(
                type.getSimpleName(),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertArrayEquals(
                        componentNames,
                        Arrays.stream(components)
                                .map(RecordComponent::getName)
                                .toArray(String[]::new)),
                () -> assertArrayEquals(
                        componentTypes,
                        Arrays.stream(components)
                                .map(RecordComponent::getType)
                                .toArray(Class<?>[]::new)),
                () -> assertEquals(componentNames.length, fields.length),
                () -> assertArrayEquals(
                        componentNames,
                        Arrays.stream(fields).map(field -> field.getName()).toArray(String[]::new)),
                () -> assertTrue(Arrays.stream(fields)
                        .allMatch(field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertArrayEquals(componentTypes, constructors[0].getParameterTypes()),
                () -> assertEquals(expectedInterfaces, Set.of(type.getInterfaces())),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(
                        expectedPublicMethods,
                        Arrays.stream(type.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .collect(Collectors.toSet())));
    }

    private static void assertOptionalNativeStatusComponent(RecordComponent component) {
        ParameterizedType genericType = (ParameterizedType) component.getGenericType();
        assertAll(
                () -> assertEquals(Optional.class, genericType.getRawType()),
                () -> assertArrayEquals(
                        new java.lang.reflect.Type[] {TraceNativeStatus.class},
                        genericType.getActualTypeArguments()));
    }

    private static <E extends Enum<E>> void assertExactEnum(Class<E> type, E[] expected) {
        Set<String> expectedNames = Arrays.stream(expected)
                .map(Enum::name)
                .collect(Collectors.toSet());
        Set<String> publicFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> Modifier.isPublic(field.getModifiers()))
                .map(field -> field.getName())
                .collect(Collectors.toSet());
        Set<String> publicMethods = Arrays.stream(type.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(Collectors.toSet());

        assertAll(
                () -> assertArrayEquals(expected, type.getEnumConstants()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertEquals(expectedNames, publicFields),
                () -> assertEquals(Set.of("valueOf", "values"), publicMethods),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length));
    }

    private static void assertNullMessage(String message, ThrowingConstructor construction) {
        NullPointerException failure =
                assertThrows(NullPointerException.class, construction::run);
        assertEquals(message, failure.getMessage());
    }

    private static void assertExactIllegalArgument(
            String message, ThrowingConstructor construction) {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, construction::run);
        assertEquals(message, failure.getMessage());
    }

    @FunctionalInterface
    private interface ThrowingConstructor {
        void run();
    }
}
