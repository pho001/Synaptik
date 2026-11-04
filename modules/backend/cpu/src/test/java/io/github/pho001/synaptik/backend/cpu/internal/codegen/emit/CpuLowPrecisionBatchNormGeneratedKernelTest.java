package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormInferenceLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Low BatchNorm ordinary/saved result and scalar-metadata boundaries. */
class CpuLowPrecisionBatchNormGeneratedKernelTest {
    @Test void inferenceConsumesFloatMetadataWithoutPrematureHalfOrFp8Narrowing() throws Throwable {
        for (DataType type : lowTypes()) {
            var base = CpuBatchNormInferenceLoweringTest.context(Collections.nCopies(5, type),
                    Shape.of(2, 1), 1, List.of(0, 1, 2, 3, 4));
            var selected = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                            Collections.nCopies(6, carrier(type))));
            var plan = new CpuPartitionPreparer().analyze(selected).plan();
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var handle = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                    generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Object output = values(type, 0, 0);
            handle.invokeWithArguments(values(type, 1, 2), values(type, 1), values(type, 0),
                    values(type, 0), values(type, 0), output,
                    plan.batchNormInferenceGeometry().orElseThrow().pack(new long[6]),
                    0L, plan.elementCount());
            double[] actual = CpuLayerNormGeneratedKernelTest.decoded(type, output);
            float epsilon = (float) CpuLayerNormGeneratedKernelTest.epsilon(type);
            float denominator = (float) Math.sqrt(epsilon);
            for (int i = 0; i < 2; i++) assertEquals(
                    CpuLayerNormGeneratedKernelTest.raw(type, (i + 1f) / denominator),
                    CpuLayerNormGeneratedKernelTest.raw(type, actual[i]), type.name());
        }
    }

    @Test void trainingPublishesAllFiveLowBoundariesFromFloatWorkingStatistics() throws Throwable {
        for (DataType type : lowTypes()) {
            Shape shape = Shape.of(2, 1), vector = Shape.of(1);
            var layouts = new ArrayList<LayoutDescriptor>();
            for (int i = 0; i < 10; i++) layouts.add(LayoutDescriptor.contiguous(i == 0 || i == 5 ? shape : vector));
            var base = CpuBatchNormTrainingLoweringTest.context(Collections.nCopies(5, type), shape,
                    1, List.of(0, 1, 2, 3, 4), layouts);
            var selected = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                    base.memoryRequirements(), Map.of(), new CpuPartitionAnalysisInputs(false,
                            Collections.nCopies(10, carrier(type))));
            var plan = new CpuPartitionPreparer().analyze(selected).plan();
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var handle = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                    generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
            Object output = values(type, 0, 0), nextMean = values(type, 0), nextVariance = values(type, 0),
                    savedMean = values(type, 0), savedInvStd = values(type, 0);
            var geometry = plan.batchNormTrainingGeometry().orElseThrow();
            try (Arena arena = Arena.ofConfined()) {
                handle.invokeWithArguments(values(type, 1, 3), values(type, 1), values(type, 0),
                        values(type, 0), values(type, 1), output, nextMean, nextVariance, savedMean,
                        savedInvStd, arena.allocate(geometry.scratchSliceBytes(), 8),
                        geometry.pack(new long[10], 0), 0L, plan.elementCount());
            }
            assertArrayEquals(new double[] {-1, 1}, CpuLayerNormGeneratedKernelTest.decoded(type, output), type.name());
            assertArrayEquals(new double[] {.5}, CpuLayerNormGeneratedKernelTest.decoded(type, nextMean), type.name());
            assertArrayEquals(new double[] {1.25}, CpuLayerNormGeneratedKernelTest.decoded(type, nextVariance), type.name());
            assertArrayEquals(new double[] {2}, CpuLayerNormGeneratedKernelTest.decoded(type, savedMean), type.name());
            assertArrayEquals(new double[] {1}, CpuLayerNormGeneratedKernelTest.decoded(type, savedInvStd), type.name());
            if (type == DataType.BFLOAT16) {
                try (Arena arena = Arena.ofConfined()) {
                    handle.invokeWithArguments(values(type, 0x1p64, -0x1p64), values(type, 1), values(type, 0),
                            values(type, 0), values(type, 1), output, nextMean, nextVariance, savedMean,
                            savedInvStd, arena.allocate(geometry.scratchSliceBytes(), 8),
                            geometry.pack(new long[10], 0), 0L, plan.elementCount());
                }
                assertArrayEquals(new double[] {0, 0}, CpuLayerNormGeneratedKernelTest.decoded(type, output));
                assertArrayEquals(new double[] {0}, CpuLayerNormGeneratedKernelTest.decoded(type, savedInvStd));
                assertArrayEquals(new double[] {Double.POSITIVE_INFINITY},
                        CpuLayerNormGeneratedKernelTest.decoded(type, nextVariance));
            }
        }
    }

    private static List<DataType> lowTypes() {
        return List.of(DataType.BFLOAT16, DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2);
    }

    private static CarrierAccess carrier(DataType type) {
        return type.byteWidth() == 1 ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY;
    }

    private static Object values(DataType type, double... values) {
        return CpuLayerNormGeneratedKernelTest.represented(type, values);
    }
}
