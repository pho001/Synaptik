package backend;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ComputeBackendTest {
    @Test
    void gpuMetalHasDescription() {
        assertTrue(ComputeBackend.GPU_METAL.getDescription().contains("Metal"));
    }

    @Test
    void gpuCudaHasDescription() {
        assertTrue(ComputeBackend.GPU_CUDA.getDescription().contains("CUDA"));
    }
}
