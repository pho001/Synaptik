package graph.execution;

import backend.memory.CpuMaterializationReason;
import backend.memory.DeviceBufferBinding;
import backend.memory.StorageResidency;
import backend.runtime.ExecutionContext;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ExecutionStateResidencyTest {
    @Test
    void deviceCurrentNodeCannotBeReadAsCpuWithoutMaterializer() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();

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
        assertTrue(trace.detail().contains("no device-to-CPU materializer"));
    }

    @Test
    void completedMaterializationRestoresCpuReadableState() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();

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
    void executionContextRejectsCpuConsumerForDeviceCurrentInput() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();
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
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();
        DeviceBufferBinding binding = new FakeDeviceBufferBinding(outputNodeId, "GPU_METAL", 8, true);

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
    void cpuWriteClearsDeviceBufferBinding() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();
        DeviceBufferBinding binding = new FakeDeviceBufferBinding(outputNodeId, "GPU_METAL", 8, true);

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
    void unavailableDeviceBufferBindingIsRejected() {
        Fixture fixture = fixture();
        int outputNodeId = fixture.compiled().compileArtifacts().forwardOutputNode().id();
        DeviceBufferBinding binding = new FakeDeviceBufferBinding(outputNodeId, "GPU_METAL", 8, false);

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

    private static Fixture fixture() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution prepared = compiled.prepare(RuntimeConfig.inferenceDefaults());
        List<CompiledNode> nodes = compiled.compileArtifacts().compiledNodes();
        Map<Integer, CompiledNodeExecutionMetadata> metadata = new HashMap<>();
        for (PreparedNodeExecution step : prepared.executionSteps()) {
            metadata.put(step.compiledNode().id(), step.metadata());
        }
        ExecutionState state = ExecutionState.create(
                nodes,
                metadata,
                compiled.compileArtifacts().forwardOutputNode().id()
        );
        return new Fixture(compiled, state, Map.copyOf(metadata));
    }

    private record Fixture(
            CompiledGraph compiled,
            ExecutionState state,
            Map<Integer, CompiledNodeExecutionMetadata> metadata
    ) {
    }

    private record FakeDeviceBufferBinding(
            int nodeId,
            String backendId,
            long logicalByteLength,
            boolean available
    ) implements DeviceBufferBinding {
        @Override
        public String describe() {
            return "fake nodeId=" + nodeId + ", backend=" + backendId + ", bytes=" + logicalByteLength;
        }
    }
}
