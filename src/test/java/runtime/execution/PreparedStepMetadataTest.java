package runtime.execution;

import backend.contract.ComputeBackend;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class PreparedStepMetadataTest {
    @Test
    void acceleratorMetadataPreservesExecutableAndExplicitResidency() {
        PreparedAcceleratorExecutable executable = new PreparedAcceleratorExecutable() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_METAL;
            }

            @Override
            public void execute(ExecutionContext context) {
            }
        };

        PreparedStepMetadata metadata =
                testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_METAL, executable);

        assertEquals(ComputeBackend.GPU_METAL, metadata.backend());
        assertEquals("NONE", metadata.inputResidencyRequirement().mode().name());
        assertEquals("CPU_CURRENT_IF_UNSET", metadata.outputResidencyEffect().mode().name());
        assertSame(executable, testsupport.MetadataArtifacts.acceleratorExecutable(metadata));
    }
}
