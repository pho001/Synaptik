package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.engine.AdvancedEngine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

/** Crosses the real public Compiler, CPU, Prepare, Engine, and Runtime boundaries. */
final class EngineAdvancedLifecycleIntegrationTest {
    @Test
    void compilesPreparesAndRunsOneCpuGraphConcurrently() throws Exception {
        try (AdvancedEngine engine = AdvancedEngine.takeOwnership(CpuBackendIntegration.open())) {
            Shape shape = Shape.of(4);
            var input = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false));
            var compiled = engine.compile(CompileMode.FORWARD_ONLY,
                    List.of(input.contiguous()), Optional.empty(),
                    GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                    PartitionScoringConfig.neutral());
            var prepared = engine.prepare(compiled);
            try (Arena firstArena = Arena.ofShared(); Arena secondArena = Arena.ofShared();
                    var firstInput = engine.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4,
                            firstArena.allocate(16, 4)));
                    var secondInput = engine.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4,
                            secondArena.allocate(16, 4)));
                    var executor = Executors.newFixedThreadPool(2)) {
                var firstFuture = executor.submit(() -> engine.run(prepared, List.of(firstInput)));
                var secondFuture = executor.submit(() -> engine.run(prepared, List.of(secondInput)));
                try (var first = firstFuture.get(); var second = secondFuture.get()) {
                    assertNotSame(first, second);
                    assertEquals(1, first.resultCount());
                    assertEquals(1, second.resultCount());
                    assertFalse(first.isClosed());
                    assertFalse(second.isClosed());
                }
            }
        }
    }

    @Test
    void preservesCpuZeroNodePreparationRejection() {
        try (AdvancedEngine engine = AdvancedEngine.takeOwnership(CpuBackendIntegration.open())) {
            Shape shape = Shape.of(1);
            var leaf = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)), false));
            var compiled = engine.compile(CompileMode.FORWARD_ONLY, List.of(leaf), Optional.empty(),
                    GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(),
                    PartitionScoringConfig.neutral());
            assertThrows(IllegalArgumentException.class, () -> engine.prepare(compiled));
        }
    }
}
