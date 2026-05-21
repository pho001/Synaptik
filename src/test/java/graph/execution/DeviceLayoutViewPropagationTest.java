package graph.execution;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import backend.ComputeBackend;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.accelerator.buffer.AcceleratorLayoutTransformDecision;
import backend.memory.DeviceBufferBinding;
import backend.memory.CpuMaterializationReason;
import backend.memory.StorageResidency;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalDeviceToCpuMaterializer;
import backend.metal.buffer.MetalBufferHandle;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.device.DeviceLayoutMaterializer;
import graph.execution.device.DeviceLayoutViewPropagator;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DeviceLayoutViewPropagationTest {
    @Test
    void permuteViewPropagatesDeviceBindingWithoutCpuMaterialization() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, true);
        fixture.attachMetalSource();

        assertTrue(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));

        assertNotNull(fixture.state().deviceBufferBindingForNodeId(fixture.targetNode().id()));
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.targetNode().id()).residency());
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void metadataOnlyViewOverHostSharedSourceStaysCpuReadable() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, false);
        fixture.attachHostSharedMetalSource();

        assertTrue(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));

        var targetResidency = fixture.state().residencyForNodeId(fixture.targetNode().id());
        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER, targetResidency.residency());
        assertTrue(targetResidency.cpuCurrent());
        assertTrue(targetResidency.deviceCurrent());

        fixture.context().requireCpuReadable(
                fixture.targetNode().id(),
                backend.memory.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT
        );
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void expandViewPropagatesReadOnlyDeviceBinding() {
        Tensor source = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "input", DataType.FLOAT32);
        Tensor view = source.expand(2, 3);
        Fixture fixture = fixture(source, view, ComputeBackend.GPU_METAL, true);
        fixture.attachMetalSource();

        assertTrue(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));

        MetalBufferBinding binding = (MetalBufferBinding) fixture.state().deviceBufferBindingForNodeId(fixture.targetNode().id());
        assertEquals(MetalBufferAccess.READ, binding.access());
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.targetNode().id()).residency());
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void expandViewMaterializesBroadcastStorageForGradientPublication() {
        Tensor source = new Tensor(new float[]{1f, 2f, 3f}, new int[]{1, 3}, null, "input", DataType.FLOAT32);
        Tensor view = source.expand(2, 3);
        Fixture fixture = fixture(source, view, ComputeBackend.GPU_METAL, false);
        FakeMetalNativeAccess nativeAccess = new FakeMetalNativeAccess();
        MetalBufferAllocator allocator = MetalBufferAllocator.available(nativeAccess);
        MetalBufferBinding sourceBinding = allocator.createInputBinding(fixture.sourceNode().id(), source);

        fixture.state().attachDeviceBufferBinding(
                fixture.sourceNode().id(),
                sourceBinding,
                StorageResidency.DEVICE_OWNED,
                "test metal source"
        );
        fixture.context().registerDeviceToCpuMaterializer(
                ComputeBackend.GPU_METAL.name(),
                new MetalDeviceToCpuMaterializer(allocator)
        );

        assertTrue(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));
        fixture.context().requireCpuReadable(
                fixture.targetNode().id(),
                CpuMaterializationReason.GRADIENT_PUBLICATION
        );

        Tensor target = fixture.state().runtimeTensorForNodeId(fixture.targetNode().id());
        assertArrayEquals(new double[]{1d, 2d, 3d, 1d, 2d, 3d}, target.toDoubleArrayCopy(), 0.0d);
        assertEquals(StorageResidency.CPU_ARRAY, fixture.state().residencyForNodeId(fixture.targetNode().id()).residency());
        assertEquals(1, fixture.state().cpuMaterializationTraces().size());
        assertEquals(CpuMaterializationReason.GRADIENT_PUBLICATION,
                fixture.state().cpuMaterializationTraces().getFirst().reason());
        assertTrue(fixture.state().cpuMaterializationTraces().getFirst().completed());
    }

    @Test
    void contiguousDoesNotUseMetadataOnlyPropagation() {
        Fixture fixture = fixture(input().contiguous(), ComputeBackend.GPU_METAL, true);
        fixture.attachMetalSource();

        assertFalse(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));
    }

    @Test
    void contiguousUsesRegisteredDenseDeviceMaterializer() {
        Fixture fixture = fixture(input().contiguous(), ComputeBackend.GPU_METAL, false);
        fixture.attachMetalSource();
        RecordingLayoutMaterializer materializer = new RecordingLayoutMaterializer();
        fixture.context().registerRuntimeService(DeviceLayoutMaterializer.class, materializer);

        assertTrue(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));

        assertEquals(1, materializer.calls);
        DeviceBufferBinding binding = fixture.state().deviceBufferBindingForNodeId(fixture.targetNode().id());
        assertNotNull(binding);
        assertEquals("GPU_METAL", binding.backendId());
        assertEquals(
                backend.accelerator.buffer.AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS,
                binding.layout().layoutClass()
        );
        assertEquals(StorageResidency.DEVICE_OWNED, fixture.state().residencyForNodeId(fixture.targetNode().id()).residency());
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void missingSourceBindingFallsBackToCpuPath() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, false);

        assertFalse(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    @Test
    void phaseNineteenFallbackPreparedInputsRemainVisible() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, true);
        fixture.attachMetalSource();

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> fixture.context().requireCpuReadable(
                fixture.sourceNode().id(),
                backend.memory.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT
        ));

        assertTrue(failure.getMessage().contains("accelerator_prepared_input"));
        assertEquals(1, fixture.state().cpuMaterializationTraces().size());
        assertEquals(
                backend.memory.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT,
                fixture.state().cpuMaterializationTraces().getFirst().reason()
        );
    }

    @Test
    void requiredModeFailsBeforeCpuMaterializationWhenViewPropagationRejected() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, false, requireMetalRuntime());

        IllegalStateException failure = assertThrows(IllegalStateException.class,
                () -> DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));

        assertTrue(failure.getMessage().contains("Accelerator buffer path is required"));
        assertTrue(failure.getMessage().contains("GPU_LAYOUT_SOURCE_BINDING_UNAVAILABLE"));
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
    }

    private static Tensor input() {
        return new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f
        }, new int[]{2, 3}, null, "input", DataType.FLOAT32);
    }

    private static Fixture fixture(Tensor target, ComputeBackend backend, boolean includeSourceBinding) {
        Tensor source = target.getPrevTensors().getFirst();
        return fixture(source, target, backend, includeSourceBinding);
    }

    private static Fixture fixture(
            Tensor target,
            ComputeBackend backend,
            boolean includeSourceBinding,
            RuntimeConfig runtimeConfig
    ) {
        Tensor source = target.getPrevTensors().getFirst();
        return fixture(source, target, backend, includeSourceBinding, runtimeConfig);
    }

    private static Fixture fixture(Tensor source, Tensor target, ComputeBackend backend, boolean includeSourceBinding) {
        return fixture(source, target, backend, includeSourceBinding, RuntimeConfig.inferenceDefaults());
    }

    private static Fixture fixture(
            Tensor source,
            Tensor target,
            ComputeBackend backend,
            boolean includeSourceBinding,
            RuntimeConfig runtimeConfig
    ) {
        TensorInternalAccess.setBackendIntent(target, backend);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(source, target));
        CompiledNode sourceNode = nodes.getFirst();
        CompiledNode targetNode = nodes.get(1);
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.CPU, null);
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                Map.of(targetNode.id(), metadata),
                targetNode.id(),
                testsupport.PublicationPlans.forRoot(target, nodes, targetNode.id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(runtimeConfig, ExecutionMode.FORWARD, Map.of(targetNode.id(), metadata), state);
        Fixture fixture = new Fixture(sourceNode, targetNode, state, context, new PreparedExecutionStep(targetNode, metadata));
        if (includeSourceBinding && backend == ComputeBackend.GPU_CUDA) {
            fixture.attachCudaSource();
        }
        return fixture;
    }

    private static RuntimeConfig requireMetalRuntime() {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        return defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
                )
        ));
    }

    private record Fixture(
            CompiledNode sourceNode,
            CompiledNode targetNode,
            ExecutionState state,
            ExecutionContext context,
            PreparedExecutionStep step
    ) {
        void attachMetalSource() {
            MetalBufferBinding binding = new MetalBufferBinding(
                    sourceNode.id(),
                    backend.accelerator.buffer.AcceleratorBufferLayout.of(
                            sourceNode.dataType(),
                            sourceNode.shape(),
                            sourceNode.strides(),
                            sourceNode.storageOffset(),
                            sourceNode.flatDataSize()
                    ),
                    new MetalBufferHandle(MemorySegment.ofAddress(100), 64, "shared", "test", true),
                    MetalBufferAccess.READ_WRITE
            );
            state.attachDeviceBufferBinding(sourceNode.id(), binding, StorageResidency.DEVICE_OWNED, "test source");
        }

        void attachHostSharedMetalSource() {
            MetalBufferBinding binding = new MetalBufferBinding(
                    sourceNode.id(),
                    backend.accelerator.buffer.AcceleratorBufferLayout.of(
                            sourceNode.dataType(),
                            sourceNode.shape(),
                            sourceNode.strides(),
                            sourceNode.storageOffset(),
                            sourceNode.flatDataSize()
                    ),
                    new MetalBufferHandle(MemorySegment.ofAddress(150), 64, "shared", "test", true),
                    MetalBufferAccess.READ_WRITE
            );
            state.attachDeviceBufferBinding(sourceNode.id(), binding, StorageResidency.HOST_SHARED_DEVICE_BUFFER, "test shared source");
        }

        void attachCudaSource() {
            CudaBufferBinding binding = new CudaBufferBinding(
                    sourceNode.id(),
                    backend.accelerator.buffer.AcceleratorBufferLayout.of(
                            sourceNode.dataType(),
                            sourceNode.shape(),
                            sourceNode.strides(),
                            sourceNode.storageOffset(),
                            sourceNode.flatDataSize()
                    ),
                    new CudaBufferHandle(MemorySegment.ofAddress(200), 64, true),
                    CudaBufferAccess.READ_WRITE
            );
            state.attachDeviceBufferBinding(sourceNode.id(), binding, StorageResidency.DEVICE_OWNED, "test source");
        }
    }

    private static final class RecordingLayoutMaterializer implements DeviceLayoutMaterializer {
        private int calls;

        @Override
        public DeviceBufferBinding materialize(
                AcceleratorLayoutTransformDecision decision,
                DeviceBufferBinding source,
                ExecutionContext context
        ) {
            calls++;
            return new MetalBufferBinding(
                    decision.targetNodeId(),
                    decision.targetLayout(),
                    new MetalBufferHandle(
                            MemorySegment.ofAddress(300 + decision.targetNodeId()),
                            decision.targetLayout().logicalByteLength(),
                            "shared",
                            "test-materialized",
                            false
                    ),
                    MetalBufferAccess.READ_WRITE
            );
        }
    }

    private static final class FakeMetalNativeAccess implements MetalBufferAllocator.NativeAccess {
        private long nextAddress = 1_000L;
        private final Map<Long, byte[]> buffers = new HashMap<>();

        @Override
        public MetalBufferHandle createBuffer(
                long byteLength,
                int storageMode,
                MemorySegment initialData,
                long initialDataBytes
        ) {
            long address = nextAddress++;
            byte[] storage = new byte[Math.toIntExact(byteLength)];
            if (initialData != null && !initialData.equals(MemorySegment.NULL) && initialDataBytes > 0) {
                byte[] initial = new byte[Math.toIntExact(initialDataBytes)];
                MemorySegment.ofArray(initial).copyFrom(initialData.reinterpret(initialDataBytes));
                System.arraycopy(initial, 0, storage, 0, Math.min(initial.length, storage.length));
            }
            buffers.put(address, storage);
            return new MetalBufferHandle(
                    MemorySegment.ofAddress(address),
                    byteLength,
                    storageMode == 1 ? "shared" : "test",
                    "test",
                    true
            );
        }

        @Override
        public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
            byte[] storage = buffers.get(handle.nativeHandle().address());
            byte[] source = Arrays.copyOf(storage, Math.toIntExact(byteLength));
            destination.copyFrom(MemorySegment.ofArray(source));
        }

        @Override
        public void destroyBuffer(MetalBufferHandle handle) {
            buffers.remove(handle.nativeHandle().address());
        }
    }
}
