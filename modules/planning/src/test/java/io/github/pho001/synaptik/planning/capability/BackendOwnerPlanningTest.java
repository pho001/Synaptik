package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class BackendOwnerPlanningTest {
    private static final TensorDescriptor DESCRIPTOR =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final OperationCapabilityQuery QUERY = new OperationCapabilityQuery(
            new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
            List.of(DESCRIPTOR),
            List.of(DESCRIPTOR));

    @Test
    void exposesOnlyTheExactPublicStatelessOwnerSelectionOperation() throws Exception {
        var method = BackendOwnerPlanning.class.getDeclaredMethod(
                "selectOwner",
                OperationCapabilityQuery.class,
                BackendIntent.class,
                List.class,
                List.class,
                PartitionScoringConfig.class);

        assertTrue(Modifier.isPublic(BackendOwnerPlanning.class.getModifiers()));
        assertTrue(Modifier.isFinal(BackendOwnerPlanning.class.getModifiers()));
        assertEquals(0, BackendOwnerPlanning.class.getDeclaredFields().length);
        assertEquals(1, BackendOwnerPlanning.class.getDeclaredMethods().length);
        assertTrue(Modifier.isPublic(method.getModifiers()));
        assertTrue(Modifier.isStatic(method.getModifiers()));
        assertEquals(BackendId.class, method.getReturnType());
        assertTrue(Modifier.isPrivate(
                BackendOwnerPlanning.class.getDeclaredConstructors()[0].getModifiers()));
    }

    @Test
    void validatesTopLevelReferencesInDeclarationOrder() {
        List<BackendCapabilityProvider> unreadableProviders = new ArrayList<>();
        unreadableProviders.add(null);
        List<BackendAvailabilitySnapshot> unreadableSnapshots = new ArrayList<>();
        unreadableSnapshots.add(null);

        assertEquals("query", assertThrows(NullPointerException.class, () ->
                BackendOwnerPlanning.selectOwner(
                        null, null, null, null, null)).getMessage());
        assertEquals("intent", assertThrows(NullPointerException.class, () ->
                BackendOwnerPlanning.selectOwner(
                        QUERY, null, null, null, null)).getMessage());
        assertEquals("providers", assertThrows(NullPointerException.class, () ->
                BackendOwnerPlanning.selectOwner(
                        QUERY, BackendIntent.unconstrained(), null, null, null)).getMessage());
        assertEquals("availabilitySnapshots", assertThrows(NullPointerException.class, () ->
                BackendOwnerPlanning.selectOwner(
                        QUERY,
                        BackendIntent.unconstrained(),
                        unreadableProviders,
                        null,
                        null)).getMessage());
        assertEquals("scoringConfig", assertThrows(NullPointerException.class, () ->
                BackendOwnerPlanning.selectOwner(
                        QUERY,
                        BackendIntent.unconstrained(),
                        unreadableProviders,
                        unreadableSnapshots,
                        null)).getMessage());
    }

    @Test
    void composesEligibilityAndPreferredClassSelectionRetainingExactOwnerReference() {
        BackendId cpu = new BackendId("cpu");
        BackendId accelerator = new BackendId("accelerator");
        List<OperationCapabilityQuery> observed = new ArrayList<>();
        BackendCapabilityProvider cpuProvider = provider(cpu, observed);
        BackendCapabilityProvider acceleratorProvider = provider(accelerator, observed);

        BackendId selected = BackendOwnerPlanning.selectOwner(
                QUERY,
                BackendIntent.unconstrained(),
                List.of(cpuProvider, acceleratorProvider),
                List.of(
                        snapshot(cpu, DeviceClass.CPU),
                        snapshot(accelerator, DeviceClass.ACCELERATOR)),
                PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR));

        assertSame(accelerator, selected);
        assertEquals(Arrays.asList(QUERY, QUERY), observed);
    }

    @Test
    void preservesTheTerminalNoEligibleFailure() {
        BackendId cpu = new BackendId("cpu");
        BackendCapabilityProvider unsupported = new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return cpu;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                return false;
            }
        };

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> BackendOwnerPlanning.selectOwner(
                        QUERY,
                        BackendIntent.unconstrained(),
                        List.of(unsupported),
                        List.of(snapshot(cpu, DeviceClass.CPU)),
                        PartitionScoringConfig.neutral()));

        assertEquals(
                "no hard-eligible backend is available for ownership selection",
                failure.getMessage());
        assertEquals(
                "no hard-eligible backend is available for ownership selection",
                failure.getCause().getMessage());
    }

    @Test
    void propagatesProviderRuntimeFailuresUnchangedEvenWhenTheirMessageMatchesSelection() {
        BackendId cpu = new BackendId("cpu");
        IllegalStateException providerFailure = new IllegalStateException(
                "no hard-eligible backend is available for ownership selection");
        BackendCapabilityProvider failing = new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return cpu;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                throw providerFailure;
            }
        };

        IllegalStateException observed = assertThrows(
                IllegalStateException.class,
                () -> BackendOwnerPlanning.selectOwner(
                        QUERY,
                        BackendIntent.unconstrained(),
                        List.of(failing),
                        List.of(snapshot(cpu, DeviceClass.CPU)),
                        PartitionScoringConfig.neutral()));

        assertSame(providerFailure, observed);
    }

    private static BackendCapabilityProvider provider(
            BackendId backendId,
            List<OperationCapabilityQuery> observed) {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                observed.add(query);
                return true;
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot(
            BackendId backendId,
            DeviceClass deviceClass) {
        BackendDeviceId deviceId = new BackendDeviceId(backendId, "0");
        return new BackendAvailabilitySnapshot(backendId, Map.of(deviceId, deviceClass));
    }
}
