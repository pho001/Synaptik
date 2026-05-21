package graph.execution;

import backend.accelerator.buffer.AcceleratorBufferAccessMode;
import backend.accelerator.buffer.AcceleratorBufferLayout;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.memory.CpuMaterializationReason;
import backend.memory.CpuMaterializationResult;
import backend.memory.DeviceBufferBinding;
import backend.memory.DeviceToCpuMaterializer;
import backend.memory.DeviceToNativeMaterializer;
import backend.memory.ExecutionResource;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.DeviceTransferPolicy;
import config.runtime.RuntimeConfig;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.trace.HostDeviceTransferKind;
import graph.CompiledGraph;
import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExecutionStateResidencyTest {
    @Test
    void deviceCurrentNodeCannotBeReadAsCpuWithoutMaterializer() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();

        fixture.state().markDeviceCurrent(
                outputNodeId,
                StorageResidency.DEVICE_OWNED,
                "GPU_METAL",
                "test device output"
        );

        assertTrue(fixture.state().requiresCpuMaterialization(outputNodeId));
        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRAPH_OUTPUT)
        );
        assertTrue(error.getMessage().contains("reason=graph_output"));
        assertTrue(error.getMessage().contains("backend=GPU_METAL"));
        assertTrue(error.getMessage().contains("prevents publishing stale CPU tensor storage"));
        var trace = fixture.state().cpuMaterializationTraces().getFirst();
        assertEquals(outputNodeId, trace.nodeId());
        assertEquals(CpuMaterializationReason.GRAPH_OUTPUT, trace.reason());
        assertEquals("GPU_METAL", trace.materializedFrom());
        assertEquals(StorageResidency.DEVICE_OWNED, trace.sourceResidency());
        assertEquals(8L, trace.bytes());
        assertFalse(trace.completed());
        assertTrue(trace.detail().contains("no device buffer binding"));
    }

    @Test
    void completedMaterializationRestoresCpuReadableState() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();

        fixture.state().markDeviceCurrent(
                outputNodeId,
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "GPU_METAL",
                "shared output"
        );
        fixture.state().markMaterializedToCpu(outputNodeId, CpuMaterializationReason.GRADIENT_PUBLICATION, 123L);

        fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRADIENT_PUBLICATION);
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_ARRAY, residency.residency());
        assertTrue(residency.cpuCurrent());
        assertFalse(residency.deviceCurrent());
        assertEquals("gradient_publication", residency.lastTransitionReason());
        var trace = fixture.state().cpuMaterializationTraces().getFirst();
        assertEquals(CpuMaterializationReason.GRADIENT_PUBLICATION, trace.reason());
        assertEquals("GPU_METAL", trace.materializedFrom());
        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER, trace.sourceResidency());
        assertEquals(8L, trace.bytes());
        assertEquals(123L, trace.durationNs());
        assertTrue(trace.completed());
    }

    @Test
    void registeredMaterializerRestoresCpuReadableStateForDeviceOwnedBinding() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);
        RecordingMaterializer materializer = new RecordingMaterializer(321L, "fake Metal readback");

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.DEVICE_OWNED,
                "device-owned output"
        );
        fixture.state().registerDeviceToCpuMaterializer("GPU_METAL", materializer);

        fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.PUBLIC_DATA_ACCESS);

        assertEquals(1, materializer.calls);
        assertSame(binding, materializer.binding);
        assertSame(fixture.state().runtimeTensorForNodeId(outputNodeId), materializer.target);
        assertEquals(CpuMaterializationReason.PUBLIC_DATA_ACCESS, materializer.reason);
        assertNull(fixture.state().deviceBufferBindingForNodeId(outputNodeId));
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_ARRAY, residency.residency());
        assertTrue(residency.cpuCurrent());
        assertFalse(residency.deviceCurrent());
        var trace = fixture.state().cpuMaterializationTraces().getFirst();
        assertEquals(outputNodeId, trace.nodeId());
        assertEquals(CpuMaterializationReason.PUBLIC_DATA_ACCESS, trace.reason());
        assertEquals("GPU_METAL", trace.materializedFrom());
        assertEquals(StorageResidency.DEVICE_OWNED, trace.sourceResidency());
        assertEquals(321L, trace.durationNs());
        assertTrue(trace.completed());
        assertEquals("fake Metal readback", trace.detail());
        var transfer = fixture.state().hostDeviceTransferTraces().getFirst();
        assertEquals(outputNodeId, transfer.nodeId());
        assertEquals(HostDeviceTransferKind.DEVICE_TO_CPU_ARRAY_COPY, transfer.transferKind());
        assertEquals(StorageResidency.DEVICE_OWNED, transfer.sourceResidency());
        assertEquals(StorageResidency.CPU_ARRAY, transfer.targetResidency());
        assertEquals(8L, transfer.javaArrayBytes());
        assertTrue(transfer.success());
    }

    @Test
    void executionContextRejectsCpuConsumerForDeviceCurrentInput() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                fixture.metadata(),
                fixture.state()
        );

        context.markDeviceCurrent(
                outputNodeId,
                StorageResidency.DEVICE_OWNED,
                "GPU_METAL",
                "upstream metal output"
        );

        IllegalStateException error = assertThrows(
                IllegalStateException.class,
                () -> context.requireCpuReadable(outputNodeId, CpuMaterializationReason.CPU_CONSUMER)
        );
        assertTrue(error.getMessage().contains("reason=cpu_consumer"));
    }

    @Test
    void sharedDeviceBufferBindingKeepsCpuReadableAndDeviceCurrent() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "shared buffer ready"
        );

        fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRAPH_OUTPUT);
        assertEquals(binding, fixture.state().deviceBufferBindingForNodeId(outputNodeId));
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER, residency.residency());
        assertTrue(residency.cpuCurrent());
        assertTrue(residency.deviceCurrent());
        assertEquals("GPU_METAL", residency.deviceBackend());
    }

    @Test
    void reservedDeviceBufferBindingDoesNotMarkValueCurrent() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);

        fixture.state().reserveDeviceBufferBinding(outputNodeId, binding);

        assertNull(fixture.state().deviceBufferBindingForNodeId(outputNodeId));
        assertSame(binding, fixture.state().writableDeviceBufferBindingForNodeId(outputNodeId));
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_ARRAY, residency.residency());
        assertFalse(residency.cpuCurrent());
        assertFalse(residency.deviceCurrent());
    }

    @Test
    void cpuWriteClearsDeviceBufferBinding() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                "shared buffer ready"
        );
        fixture.state().markCpuCurrent(outputNodeId, "cpu overwrite");

        assertNull(fixture.state().deviceBufferBindingForNodeId(outputNodeId));
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_ARRAY, residency.residency());
        assertTrue(residency.cpuCurrent());
        assertFalse(residency.deviceCurrent());
    }

    @Test
    void nativeCurrentMaterializesToCpuArrayForPublicRead() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 2, "native-output");
        NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
        f32.setFloat32At(0, 11f);
        f32.setFloat32At(1, -7f);

        fixture.state().attachNativeStorage(outputNodeId, storage, "native output");

        assertTrue(fixture.state().requiresCpuMaterialization(outputNodeId));
        assertEquals(StorageResidency.CPU_NATIVE, fixture.state().residencyForNodeId(outputNodeId).residency());

        fixture.state().requireCpuReadable(outputNodeId, CpuMaterializationReason.GRAPH_OUTPUT);

        assertArrayEquals(new float[]{11f, -7f}, fixture.state().runtimeTensorForNodeId(outputNodeId).getFloat32Data(), 0f);
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_ARRAY, residency.residency());
        assertTrue(residency.cpuCurrent());
        assertFalse(residency.nativeCurrent());
        var trace = fixture.state().cpuMaterializationTraces().getFirst();
        assertEquals(CpuMaterializationReason.GRAPH_OUTPUT, trace.reason());
        assertEquals(StorageResidency.CPU_NATIVE, trace.sourceResidency());
        assertTrue(trace.completed());
        assertEquals("native_to_array", trace.detail());
    }

    @Test
    void requireNativeReadableMaterializesCpuArrayToNative() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        Tensor runtime = fixture.state().runtimeTensorForNodeId(outputNodeId);
        runtime.setData(new float[]{5f, 6f});
        fixture.state().markCpuCurrent(outputNodeId, "cpu test data");

        NativeTensorStorage storage = fixture.state().requireNativeReadable(
                outputNodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );

        assertTrue(storage instanceof NativeFloat32Storage);
        NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
        assertEquals(5f, f32.getFloat32At(0), 0f);
        assertEquals(6f, f32.getFloat32At(1), 0f);
        var residency = fixture.state().residencyForNodeId(outputNodeId);
        assertEquals(StorageResidency.CPU_NATIVE, residency.residency());
        assertFalse(residency.cpuCurrent());
        assertTrue(residency.nativeCurrent());
        var trace = fixture.state().cpuMaterializationTraces().getFirst();
        assertEquals(StorageResidency.CPU_ARRAY, trace.sourceResidency());
        assertEquals("array_to_native", trace.detail());
    }

    @Test
    void deviceCurrentRequireNativeReadableRecordsArrayBridgeTransfer() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);
        RecordingMaterializer materializer = new RecordingMaterializer(321L, "fake Metal readback");

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.DEVICE_OWNED,
                "device-owned output"
        );
        fixture.state().registerDeviceToCpuMaterializer("GPU_METAL", materializer);

        NativeTensorStorage storage = fixture.state().requireNativeReadable(
                outputNodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );

        assertTrue(storage instanceof NativeFloat32Storage);
        var transfers = fixture.state().hostDeviceTransferTraces();
        assertEquals(2, transfers.size());
        assertEquals(HostDeviceTransferKind.DEVICE_TO_CPU_ARRAY_COPY, transfers.get(0).transferKind());
        assertEquals(HostDeviceTransferKind.DEVICE_TO_ARRAY_TO_NATIVE_BRIDGE, transfers.get(1).transferKind());
        assertEquals("native-device-direct-transfer-unavailable", transfers.get(1).fallbackReason());
        assertEquals(StorageResidency.CPU_NATIVE, transfers.get(1).targetResidency());
    }

    @Test
    void deviceCurrentRequireNativeReadableUsesDirectNativeMaterializerWhenAvailable() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);
        RecordingNativeMaterializer materializer = new RecordingNativeMaterializer(654L, "fake Metal native readback");

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.DEVICE_OWNED,
                "device-owned output"
        );
        fixture.state().registerDeviceToNativeMaterializer("GPU_METAL", materializer);

        NativeTensorStorage storage = fixture.state().requireNativeReadable(
                outputNodeId,
                CpuMaterializationReason.CPU_CONSUMER
        );

        assertTrue(storage instanceof NativeFloat32Storage);
        NativeFloat32Storage f32 = (NativeFloat32Storage) storage;
        assertEquals(21f, f32.getFloat32At(0), 0f);
        assertEquals(-8f, f32.getFloat32At(1), 0f);
        assertEquals(1, materializer.calls);
        assertTrue(fixture.state().cpuMaterializationTraces().isEmpty());
        var transfer = fixture.state().hostDeviceTransferTraces().getFirst();
        assertEquals(HostDeviceTransferKind.DEVICE_TO_NATIVE_SEGMENT_COPY, transfer.transferKind());
        assertEquals(StorageResidency.DEVICE_OWNED, transfer.sourceResidency());
        assertEquals(StorageResidency.CPU_NATIVE, transfer.targetResidency());
        assertEquals(0L, transfer.javaArrayBytes());
        assertEquals(8L, transfer.nativeBytes());
        assertEquals(8L, transfer.deviceBytes());
        assertTrue(transfer.directTransferSupported());
    }

    @Test
    void requireDirectRejectsDeviceToArrayToNativeBridgeWhenDirectMaterializerMissing() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults().withDeviceTransferPolicy(DeviceTransferPolicy.REQUIRE_DIRECT),
                ExecutionMode.FORWARD,
                fixture.metadata(),
                fixture.state()
        );

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.DEVICE_OWNED,
                "device-owned output"
        );

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> context.requireNativeReadable(outputNodeId, CpuMaterializationReason.CPU_CONSUMER)
        );

        assertTrue(failure.getMessage().contains("DeviceTransferPolicy.REQUIRE_DIRECT"));
        assertTrue(failure.getMessage().contains("direct device-to-native transfer is unavailable"));
        assertTrue(fixture.state().hostDeviceTransferTraces().isEmpty());
    }

    @Test
    void closeResourcesClosesAttachedNativeStorageAndClearsBinding() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        NativeTensorStorage storage = new NativeCpuStorageFactory().allocate(DataType.FLOAT32, 2, "native-close");

        fixture.state().attachNativeStorage(outputNodeId, storage, "native output");
        fixture.state().closeResources();

        assertTrue(storage.closed());
        assertNull(fixture.state().nativeStorageForNodeId(outputNodeId));
    }

    @Test
    void unavailableDeviceBufferBindingIsRejected() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, false);

        IllegalArgumentException error = assertThrows(
                IllegalArgumentException.class,
                () -> fixture.state().attachDeviceBufferBinding(
                        outputNodeId,
                        binding,
                        StorageResidency.HOST_SHARED_DEVICE_BUFFER,
                        "invalid binding"
                )
        );

        assertTrue(error.getMessage().contains("not available"));
        assertNull(fixture.state().deviceBufferBindingForNodeId(outputNodeId));
    }

    @Test
    void executionResourcesCloseInReverseOrderAndClearBindings() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().program().forwardOutputNode().id();
        DeviceBufferBinding binding = fakeBinding(outputNodeId, 8, true);
        List<String> closed = new ArrayList<>();

        fixture.state().attachDeviceBufferBinding(
                outputNodeId,
                binding,
                StorageResidency.DEVICE_OWNED,
                "device output"
        );
        fixture.state().registerResource(new RecordingResource("first", closed));
        fixture.state().registerResource(new RecordingResource("second", closed));

        fixture.state().closeResources();

        assertEquals(List.of("second", "first"), closed);
        assertNull(fixture.state().deviceBufferBindingForNodeId(outputNodeId));
        assertNull(fixture.state().writableDeviceBufferBindingForNodeId(outputNodeId));
    }

    private static Fixture fixture() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference());
        PreparedExecution prepared = compiled.prepare(RuntimeConfig.inferenceDefaults());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedExecutionStep step : prepared.executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                compiled.program().descriptorIndex(),
                metadata,
                compiled.program().forwardOutputNode().id(),
                compiled.publication()
        );
        return new Fixture(compiled, state, Map.copyOf(metadata));
    }

    private record Fixture(
            CompiledGraph compiled,
            ExecutionState state,
            Map<Integer, CompiledNodeExecutionMetadata> metadata
    ) {
    }

    private static DeviceBufferBinding fakeBinding(int nodeId, long logicalByteLength, boolean available) {
        return new FakeDeviceBufferBinding(
                nodeId,
                "GPU_METAL",
                AcceleratorBufferLayout.of(
                        DataType.FLOAT32,
                        new int[]{(int) (logicalByteLength / Float.BYTES)},
                        new int[]{1},
                        0,
                        logicalByteLength / Float.BYTES
                ),
                AcceleratorBufferAccessMode.READ_WRITE,
                "fake-native-" + nodeId,
                available
        );
    }

    private record FakeDeviceBufferBinding(
            int nodeId,
            String backendId,
            AcceleratorBufferLayout layout,
            AcceleratorBufferAccessMode accessMode,
            String nativeHandleIdentity,
            boolean available
    ) implements DeviceBufferBinding {
        @Override
        public String describe() {
            return "fake nodeId=" + nodeId + ", backend=" + backendId
                    + ", bytes=" + layout.logicalByteLength()
                    + ", native=" + nativeHandleIdentity;
        }
    }

    private static final class RecordingMaterializer implements DeviceToCpuMaterializer {
        private final long durationNs;
        private final String detail;
        private int calls;
        private DeviceBufferBinding binding;
        private Tensor target;
        private CpuMaterializationReason reason;

        private RecordingMaterializer(long durationNs, String detail) {
            this.durationNs = durationNs;
            this.detail = detail;
        }

        @Override
        public CpuMaterializationResult materialize(
                DeviceBufferBinding binding,
                Tensor target,
                CpuMaterializationReason reason
        ) {
            calls++;
            this.binding = binding;
            this.target = target;
            this.reason = reason;
            return new CpuMaterializationResult(durationNs, detail);
        }
    }

    private static final class RecordingNativeMaterializer implements DeviceToNativeMaterializer {
        private final long durationNs;
        private final String detail;
        private int calls;

        private RecordingNativeMaterializer(long durationNs, String detail) {
            this.durationNs = durationNs;
            this.detail = detail;
        }

        @Override
        public CpuMaterializationResult materialize(
                DeviceBufferBinding binding,
                Tensor target,
                NativeTensorStorage nativeStorage,
                CpuMaterializationReason reason
        ) {
            calls++;
            NativeFloat32Storage f32 = (NativeFloat32Storage) nativeStorage;
            f32.setFloat32At(0, 21f);
            f32.setFloat32At(1, -8f);
            return new CpuMaterializationResult(durationNs, detail);
        }
    }

    private static final class RecordingResource implements ExecutionResource {
        private final String label;
        private final List<String> closed;
        private boolean closeCalled;

        private RecordingResource(String label, List<String> closed) {
            this.label = label;
            this.closed = closed;
        }

        @Override
        public void close() {
            if (closeCalled) {
                throw new AssertionError("resource closed twice: " + label);
            }
            closeCalled = true;
            closed.add(label);
        }
    }
}
