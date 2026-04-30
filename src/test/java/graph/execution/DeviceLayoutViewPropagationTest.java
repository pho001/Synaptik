package graph.execution;

import backend.ComputeBackend;
import backend.cuda.buffer.CudaBufferAccess;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.memory.StorageResidency;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Map;

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
    void contiguousDoesNotUseMetadataOnlyPropagation() {
        Fixture fixture = fixture(input().contiguous(), ComputeBackend.GPU_METAL, true);
        fixture.attachMetalSource();

        assertFalse(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));
    }

    @Test
    void missingSourceBindingFallsBackToCpuPath() {
        Fixture fixture = fixture(input().permute(1, 0), ComputeBackend.GPU_METAL, false);

        assertFalse(DeviceLayoutViewPropagator.tryPropagate(fixture.step(), fixture.context()));
        assertEquals(0, fixture.state().cpuMaterializationTraces().size());
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
        TensorInternalAccess.setBackend(target, backend);
        List<CompiledNode> nodes = CompiledNode.snapshot(List.of(source, target));
        CompiledNode sourceNode = nodes.getFirst();
        CompiledNode targetNode = nodes.get(1);
        CompiledNodeExecutionMetadata metadata = new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                null,
                null,
                null,
                null,
                null,
                List.of(),
                PartitionExecutionRole.NONE
        );
        ExecutionState state = ExecutionState.create(nodes, Map.of(targetNode.id(), metadata), targetNode.id());
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(runtimeConfig, ExecutionMode.FORWARD, Map.of(targetNode.id(), metadata), state);
        Fixture fixture = new Fixture(sourceNode, targetNode, state, context, new PreparedNodeExecution(targetNode, metadata));
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
            PreparedNodeExecution step
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
}
