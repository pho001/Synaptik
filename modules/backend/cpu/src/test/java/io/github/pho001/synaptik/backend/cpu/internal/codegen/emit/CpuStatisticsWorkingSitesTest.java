package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuBatchNormTrainingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuTrailingNormalizationLoweringTest;
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
import org.junit.jupiter.api.Test;

/** Independent class/sign consumers of mandatory square and saved-statistic working sites. */
class CpuStatisticsWorkingSitesTest {
    @Test void layerAndRmsKeepIndividualAndTotalSquareOverflowWithFiniteTails() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            double individual = type == DataType.FLOAT64 ? Double.MAX_VALUE : 0x1p64;
            double total = type == DataType.FLOAT64 ? 0x1p511 : 0x1p63;
            for (double[] input : List.of(new double[] {individual, -individual, 0, 0},
                    new double[] {total, -total, total, -total, 0, 0})) {
                for (boolean layer : List.of(true, false)) {
                    var base = CpuTrailingNormalizationLoweringTest.context(layer, false, type,
                            Shape.of(1, input.length), Shape.of(input.length), List.of(0));
                    var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                            base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                                    Collections.nCopies(2, carrier(type))));
                    var plan = new CpuPartitionPreparer().analyze(context).plan();
                    var route = plan.units().getFirst().portablePlan();
                    var generator = new CpuClassFileKernelGenerator();
                    var entry = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                            generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
                    Object output = values(type, new double[input.length]);
                    var geometry = plan.trailingNormalizationGeometry().orElseThrow();
                    try (Arena arena = Arena.ofConfined()) {
                        if (layer) entry.invokeWithArguments(values(type, input), output,
                                arena.allocate(geometry.scratchSliceBytes(), 8), geometry.pack(new long[2]), 0L, 1L);
                        else entry.invokeWithArguments(values(type, input), output, geometry.pack(new long[2]), 0L, 1L);
                    }
                    double[] actual = CpuLayerNormGeneratedKernelTest.decoded(type, output);
                    for (int i = 0; i < input.length; i++) {
                        assertTrue(actual[i] == 0d, type + "/" + layer + "/" + i);
                        assertEquals(Double.doubleToRawLongBits(Math.copySign(0d, input[i])),
                                Double.doubleToRawLongBits(actual[i]), "finite value divided by +infinity");
                    }
                }
            }
        }
    }

    @Test void trainingPublishesInfiniteRunningVarianceAndPositiveZeroSavedInverseStd() throws Throwable {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16)) {
            double individual = type == DataType.FLOAT64 ? Double.MAX_VALUE : 0x1p64;
            double total = type == DataType.FLOAT64 ? 0x1p511 : 0x1p63;
            for (double[] input : List.of(new double[] {individual, -individual, 0, 0},
                    new double[] {total, -total, total, -total, 0, 0},
                    new double[] {individual, -individual, Double.NaN, 0})) {
                Shape shape = Shape.of(input.length, 1), vector = Shape.of(1);
                var layouts = new ArrayList<LayoutDescriptor>();
                for (int b = 0; b < 10; b++) layouts.add(LayoutDescriptor.contiguous(b == 0 || b == 5 ? shape : vector));
                var base = CpuBatchNormTrainingLoweringTest.context(Collections.nCopies(5, type), shape,
                        1, List.of(0, 1, 2, 3, 4), layouts);
                var context = new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                        base.memoryRequirements(), base.constants(), new CpuPartitionAnalysisInputs(false,
                                Collections.nCopies(10, carrier(type))));
                var plan = new CpuPartitionPreparer().analyze(context).plan();
                var route = plan.units().getFirst().portablePlan();
                var generator = new CpuClassFileKernelGenerator();
                var entry = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                        generator.generateClassBytes(route.specialization(), route.kernelIr())).entryPoint();
                Object output = values(type, new double[input.length]), nextMean = values(type, 9),
                        nextVariance = values(type, 9), savedMean = values(type, 9), savedInv = values(type, 9);
                var geometry = plan.batchNormTrainingGeometry().orElseThrow();
                try (Arena arena = Arena.ofConfined()) {
                    entry.invokeWithArguments(values(type, input), values(type, 1), values(type, 0),
                            values(type, 0), values(type, 1), output, nextMean, nextVariance, savedMean, savedInv,
                            arena.allocate(geometry.scratchSliceBytes(), 8), geometry.pack(new long[10], 0), 0L, 1L);
                }
                if (Double.isNaN(input[2])) {
                    for (Object boundary : List.of(output, nextMean, nextVariance, savedMean, savedInv))
                        for (double value : CpuLayerNormGeneratedKernelTest.decoded(type, boundary)) assertTrue(Double.isNaN(value));
                } else {
                    assertEquals(Double.POSITIVE_INFINITY, decode(type, nextVariance));
                    assertEquals(0L, Double.doubleToRawLongBits(decode(type, savedInv)));
                    assertEquals(0L, Double.doubleToRawLongBits(decode(type, savedMean)));
                    assertEquals(0L, Double.doubleToRawLongBits(decode(type, nextMean)));
                    for (double value : CpuLayerNormGeneratedKernelTest.decoded(type, output)) assertEquals(0d, value);
                }
            }
        }
    }

    private static CarrierAccess carrier(DataType type) {
        return type == DataType.FLOAT64 ? CarrierAccess.DOUBLE_ARRAY
                : type == DataType.FLOAT32 ? CarrierAccess.FLOAT_ARRAY : CarrierAccess.SHORT_ARRAY;
    }
    private static Object values(DataType type, double... values) {
        return CpuLayerNormGeneratedKernelTest.represented(type, values);
    }
    private static double decode(DataType type, Object values) {
        return CpuLayerNormGeneratedKernelTest.decoded(type, values)[0];
    }
}
