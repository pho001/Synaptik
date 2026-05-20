package graph.execution;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.runtime.ExecutionContext;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class CompiledNodeExecutionMetadataTest {
    @Test
    void nullPartitionRoleDefaultsToNone() {
        PreparedAcceleratorExecutable executable = new PreparedAcceleratorExecutable() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_METAL;
            }

            @Override
            public void execute(ExecutionContext context) {
            }
        };

        CompiledNodeExecutionMetadata metadata =
                testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_METAL, executable, null);

        assertEquals(PartitionExecutionRole.NONE, metadata.partitionRole());
        assertSame(executable, testsupport.MetadataArtifacts.acceleratorExecutable(metadata));
    }
}
