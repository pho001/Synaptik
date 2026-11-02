package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
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
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Objects;
import org.junit.jupiter.api.Test;

final class BackendOwnerPlanningTest {
    private static final TensorDescriptor DESCRIPTOR =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final OperationCapabilityQuery QUERY = new OperationCapabilityQuery(new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE), List.of(DESCRIPTOR), List.of(DESCRIPTOR));

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

    @Test
    void forwardsCompleteScopedScanAndDemandUnchangedToWholeOccurrenceProviders() {
        for (List<Integer> demand : List.of(List.of(0, 1, 3, 4), List.<Integer>of())) {
            OperationCapabilityQuery query = scanFixture(4, UnaryElementwiseKind.TANH, demand);
            CompiledScanBody body = query.scanBody().orElseThrow();
            CompiledGraphModel graph = body.graph();
            BackendId first = new BackendId("first"), preferred = new BackendId("preferred");
            List<OperationCapabilityQuery> observed = new ArrayList<>();
            BackendId owner = BackendOwnerPlanning.selectOwner(query, BackendIntent.unconstrained(),
                    List.of(scanProvider(first, query, true, observed), scanProvider(preferred, query, true, observed)),
                    List.of(snapshot(first, DeviceClass.CPU), snapshot(preferred, DeviceClass.ACCELERATOR)),
                    PartitionScoringConfig.preferring(DeviceClass.ACCELERATOR));
            assertSame(preferred, owner);
            assertEquals(2, observed.size());
            for (OperationCapabilityQuery received : observed) {
                assertSame(query, received);
                assertSame(body, received.scanBody().orElseThrow());
                assertSame(graph, received.scanBody().orElseThrow().graph());
                assertSame(query.operation(), received.operation());
                for (int index = 0; index < query.inputs().size(); index++) {
                    assertSame(query.inputs().get(index), received.inputs().get(index));
                }
                for (int index = 0; index < query.outputs().size(); index++) {
                    assertSame(query.outputs().get(index), received.outputs().get(index));
                }
                assertEquals(demand, received.scanBody().orElseThrow().demandedOutputIndices());
            }
            assertEquals(List.of(new ValueId(0), new ValueId(1), new ValueId(2)), graph.inputs());
            assertEquals(body.nextCarry(), body.emissions());
            assertEquals(body.nextCarry().getFirst(), body.residuals().getLast());
            assertFalse(graph.values().getFirst().descriptor().requiresGrad());
            assertFalse(query.inputs().getFirst().requiresGrad());
            assertTrue(query.inputs().get(1).requiresGrad());
            assertTrue(query.outputs().stream().allMatch(TensorDescriptor::requiresGrad));

            // An independently scoped outer node may reuse numeric IDs, but keeps distinct ports.
            CompiledNode outer = new CompiledNode(new NodeId(0), query.operation(),
                    List.of(new ValueId(0), new ValueId(1), new ValueId(2)),
                    List.of(new ValueId(3), new ValueId(4), new ValueId(5), new ValueId(6), new ValueId(7)),
                    query.scanBody());
            assertEquals(graph.inputs().getFirst(), outer.inputs().getFirst());
            assertNotSame(graph.inputs().getFirst(), outer.inputs().getFirst());
            assertEquals(5, outer.outputs().stream().distinct().count());
        }
    }

    @Test
    void fixtureProviderQualifiesWholeQuestionAndFailsClosedWithoutFallback() {
        OperationKind unknown = new OperationKind() {
            @Override
            public String name() { return "UNKNOWN_SCAN_CHILD"; }
            @Override
            public List<OperationSignature> signatures() {
                return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
            }
        };
        for (long time : List.of(4L, 0L)) {
            OperationCapabilityQuery valid = scanFixture(time, UnaryElementwiseKind.TANH, List.of());
            CompiledScanBody body = valid.scanBody().orElseThrow();
            BackendId id = new BackendId("fixture");
            BackendCapabilityProvider supported = scanProvider(id, valid, true, new ArrayList<>());
            BackendCapabilityProvider noBinder = scanProvider(id, valid, false, new ArrayList<>());
            assertTrue(supported.supports(valid));
            assertTrue(supported.supports(valid));
            for (CompiledNode child : body.graph().nodes()) {
                OperationCapabilityQuery childQuery = childQuery(body.graph(), child);
                assertTrue(supported.supports(childQuery));
                assertTrue(noBinder.supports(childQuery));
            }
            List<TensorDescriptor> badInputs = new ArrayList<>(valid.inputs());
            badInputs.set(1, DESCRIPTOR); // loses time axis and gradient eligibility
            List<TensorDescriptor> badOutputs = new ArrayList<>(valid.outputs());
            badOutputs.set(0, DESCRIPTOR); // loses conservative outer gradient flag
            CompiledScanBody wrongRoles = new CompiledScanBody(body.graph(), body.residuals(),
                    body.emissions(), body.nextCarry(), List.of());
            CompiledGraphModel graph = body.graph();
            CompiledGraphModel missingFormal = new CompiledGraphModel(
                    graph.values().stream().filter(value -> value.id().value() != 2).toList(),
                    graph.nodes(), graph.inputs().subList(0, 2), graph.outputs(), graph.nodePhases());
            CompiledScanBody wrongFormals = new CompiledScanBody(missingFormal, body.nextCarry(),
                    body.emissions(), body.residuals(), List.of());
            CompiledScanBody wrongRoleOrder = new CompiledScanBody(graph, body.nextCarry(),
                    body.emissions(), body.residuals().reversed(), List.of());
            List<GraphValue> wrongValues = new ArrayList<>(graph.values());
            wrongValues.set(3, new GraphValue(new ValueId(3), DESCRIPTOR));
            CompiledScanBody wrongLocalDescriptor = new CompiledScanBody(
                    new CompiledGraphModel(wrongValues, graph.nodes(), graph.inputs(), graph.outputs(), graph.nodePhases()),
                    body.nextCarry(), body.emissions(), body.residuals(), List.of());
            List<OperationCapabilityQuery> rejected = List.of(
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs(), Optional.empty(), Optional.empty()),
                    new OperationCapabilityQuery(new Operation(ScanKind.SCAN,
                            new ScanAttrs(time, 1, 1, 0, 1, 3, false)), valid.inputs(), valid.outputs(), valid.scanBody(), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), badInputs, valid.outputs(), valid.scanBody(), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), badOutputs, valid.scanBody(), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs(), Optional.of(wrongRoles), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs(), Optional.of(wrongFormals), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs(), Optional.of(wrongRoleOrder), Optional.empty()),
                    new OperationCapabilityQuery(valid.operation(), valid.inputs(), valid.outputs(), Optional.of(wrongLocalDescriptor), Optional.empty()),
                    scanFixture(time, UnaryElementwiseKind.RELU, List.of()),
                    scanFixture(time, unknown, List.of()));
            for (OperationCapabilityQuery query : rejected) {
                assertFalse(supported.supports(query));
                assertFalse(supported.supports(query));
                assertNoScanOwner(query, supported);
            }
            assertFalse(noBinder.supports(valid));
            assertNoScanOwner(valid, noBinder);
            assertEquals("query", assertThrows(NullPointerException.class,
                    () -> supported.supports(null)).getMessage());
        }
    }

    private static void assertNoScanOwner(OperationCapabilityQuery query, BackendCapabilityProvider provider) {
        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> BackendOwnerPlanning.selectOwner(query, BackendIntent.unconstrained(), List.of(provider),
                        List.of(snapshot(provider.backendId(), DeviceClass.CPU)), PartitionScoringConfig.neutral()));
        assertEquals("no hard-eligible backend is available for ownership selection", failure.getMessage());
        assertEquals(failure.getMessage(), failure.getCause().getMessage());
    }

    private static OperationCapabilityQuery scanFixture(long time, OperationKind lastKind, List<Integer> demand) {
        TensorDescriptor carry = DESCRIPTOR;
        TensorDescriptor grad = new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), true);
        TensorDescriptor history = new TensorDescriptor(DataType.FLOAT32, Shape.of(time, 2), Optional.empty(), true);
        ValueId h = new ValueId(0), x = new ValueId(1), unused = new ValueId(2);
        ValueId sum = new ValueId(3), next = new ValueId(4);
        CompiledNode add = new CompiledNode(new NodeId(0),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE), List.of(h, x), List.of(sum));
        CompiledNode last = new CompiledNode(new NodeId(1),
                new Operation(lastKind, NoOperationAttrs.INSTANCE), List.of(sum), List.of(next));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(h, carry), new GraphValue(x, grad), new GraphValue(unused, carry),
                        new GraphValue(sum, grad), new GraphValue(next, grad)),
                List.of(add, last), List.of(h, x, unused), List.of(h, sum, next),
                Map.of(add.id(), GraphPhase.FORWARD, last.id(), GraphPhase.FORWARD));
        CompiledScanBody body = new CompiledScanBody(graph, List.of(next), List.of(next), List.of(h, sum, next), demand);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN, new ScanAttrs(time, 1, 1, 1, 1, 3, false)), List.of(carry, history, carry), List.of(grad, history, history, history, history), Optional.of(body), Optional.empty());
    }

    private static OperationCapabilityQuery childQuery(CompiledGraphModel graph, CompiledNode child) {
        // Fixture IDs are contiguous positions; this is not a graph descriptor-inference engine.
        return new OperationCapabilityQuery(child.operation(),
                child.inputs().stream().map(id -> graph.values().get((int) id.value()).descriptor()).toList(),
                child.outputs().stream().map(id -> graph.values().get((int) id.value()).descriptor()).toList());
    }

    private static BackendCapabilityProvider scanProvider(BackendId id, OperationCapabilityQuery fixture,
            boolean binderSupported, List<OperationCapabilityQuery> observed) {
        CompiledScanBody expected = fixture.scanBody().orElseThrow();
        List<OperationCapabilityQuery> supportedChildren = expected.graph().nodes().stream()
                .map(child -> childQuery(expected.graph(), child)).toList();
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() { return id; }
            @Override
            public boolean supports(OperationCapabilityQuery query) {
                Objects.requireNonNull(query, "query");
                observed.add(query);
                // Deliberately bounded exact-fixture qualification, not reusable scan inference.
                if (query.operation().kind() != ScanKind.SCAN) {
                    return supportedChildren.contains(query);
                }
                return binderSupported && fixture.equals(query);
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
