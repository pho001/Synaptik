package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionDagDecomposer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.invoke.MethodHandle;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Deterministic numerical controls for generated multi-store and fused/split kernels. */
class CpuPartitionDagGeneratedEvidenceTest {

    @Test void multiStoreMatchesIndependentJava() throws Throwable {
        CpuKernelIr ir = ir();
        CpuKernelSpecialization specialization = specialization(ir);
        var generator = new CpuClassFileKernelGenerator();
        byte[] bytes = generator.generateClassBytes(specialization, ir);
        MethodHandle entry = generator.defineClassBytes(specialization, ir, bytes).entryPoint();
        double[] left = {1, -2, Double.NaN, -0.0};
        double[] right = {-3, 4, Double.POSITIVE_INFINITY, 0.0};
        double[] first = new double[4], second = new double[4];
        entry.invokeExact(left, right, first, second, geometry(4), 0L, 4L);
        for (int i = 0; i < 4; i++) {
            assertEquals(Double.doubleToRawLongBits(-left[i]),
                    Double.doubleToRawLongBits(first[i]));
            assertEquals(Double.doubleToRawLongBits(-right[i]),
                    Double.doubleToRawLongBits(second[i]));
        }
    }

    @Test void fusedAndSplitPreserveTypedOverflowSubrangesAndCanaries() throws Throwable {
        int size = 129, canary = 0x5a5a5a5a;
        int[] input = new int[size], generatedOutput = new int[size + 2];
        int[] splitIntermediate = new int[size + 2], splitOutput = new int[size + 2];
        int[] expected = new int[size + 2];
        Arrays.fill(generatedOutput, canary);
        Arrays.fill(splitIntermediate, canary);
        Arrays.fill(splitOutput, canary);
        Arrays.fill(expected, canary);
        for (int index = 0; index < size; index++) {
            input[index] = switch (index % 3) {
                case 0 -> Integer.MAX_VALUE - index;
                case 1 -> Integer.MIN_VALUE + index;
                default -> index - size / 2;
            };
            expected[index] = input[index] + 1 + 2;
        }
        int[] originalInput = input.clone();
        var generator = new CpuClassFileKernelGenerator();
        var context = fusionContext(size);
        CpuKernelIr fusedIr = new CpuPartitionPreparer().analyze(context).plan()
                .units().getFirst().portablePlan().kernelIr();
        var enumeration = new CpuPartitionDagDecomposer().enumerate(context,
                new CpuPartitionLowering(), List.of());
        MethodHandle fused = generatedHandle(generator, fusedIr);
        MethodHandle first = generatedHandle(generator,
                enumeration.canonicalSplit().get(0).lowering().kernelIr());
        MethodHandle second = generatedHandle(generator,
                enumeration.canonicalSplit().get(1).lowering().kernelIr());
        for (long[] range : new long[][] {{1, size - 1}, {0, 1}, {size - 1, size}}) {
            long[] geometry = geometry(2, size, range[0]);
            fused.invokeExact(input, generatedOutput, geometry, range[0], range[1]);
            first.invokeExact(input, splitIntermediate, geometry, range[0], range[1]);
            second.invokeExact(splitIntermediate, splitOutput, geometry, range[0], range[1]);
            if (range[0] == 1) {
                assertEquals(canary, generatedOutput[0]);
                assertEquals(canary, generatedOutput[size - 1]);
                assertEquals(canary, splitOutput[0]);
                assertEquals(canary, splitOutput[size - 1]);
            }
        }
        assertArrayEquals(expected, generatedOutput);
        assertArrayEquals(expected, splitOutput);
        assertArrayEquals(originalInput, input);
        assertEquals(canary, splitIntermediate[size]);
        assertEquals(canary, splitIntermediate[size + 1]);
    }

    private static MethodHandle generatedHandle(CpuClassFileKernelGenerator generator,
            CpuKernelIr ir) {
        CpuKernelSpecialization specialization = intSpecialization(ir);
        return generator.defineClassBytes(specialization, ir, generator.generateClassBytes(specialization, ir)).entryPoint();
    }

    private static CpuKernelSpecialization intSpecialization(CpuKernelIr ir) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()), CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
        List.of(DataType.INT32, DataType.INT32),
        List.of(CpuKernelSpecialization.CarrierAccess.INT_ARRAY,
                CpuKernelSpecialization.CarrierAccess.INT_ARRAY), 0, -1);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> fusionContext(long elementCount) {
        Shape shape = Shape.of(elementCount);
        var descriptor = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var nodes = List.of(
                new CompiledNode(new NodeId(0), new Operation(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int32(1))), List.of(new ValueId(0)),
                        List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1), new Operation(ScalarElementwiseKind.ADD,
                        new ScalarValueAttrs(ScalarValue.int32(2))), List.of(new ValueId(1)),
                        List.of(new ValueId(2))));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var values = java.util.stream.LongStream.rangeClosed(0, 2)
                .mapToObj(index -> new GraphValue(new ValueId(index), descriptor)).toList();
        var memory = List.of(
                new LogicalMemoryRequirement(new ValueId(0), descriptor, Optional.empty(),
                        List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(1), descriptor, Optional.of(partition),
                        List.of(partition), false),
                new LogicalMemoryRequirement(new ValueId(2), descriptor, Optional.of(partition),
                        List.of(), true));
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), CpuPartitionAnalysisInputs.DEFAULT);
    }

    private static CpuKernelIr ir() {
        CpuAccessPlan read = dense(CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan write = dense(CpuAccessPlan.AccessKind.WRITE);
        return new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, DataType.FLOAT64, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(2, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, write),
                new CpuKernelIr.Value(3, DataType.FLOAT64, CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(0), 2),
                        new CpuKernelIr.Instruction(CpuPointwiseOpcode.NEG, List.of(1), 3)),
                new CpuKernelIr.Loop("start", "end"),
                List.of(new CpuKernelIr.Store(2, 0), new CpuKernelIr.Store(3, 1)));
    }

    private static CpuKernelSpecialization specialization(CpuKernelIr ir) {
        return new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()), CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
        List.of(DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64, DataType.FLOAT64),
        List.of(CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY), 0, -1);
    }

    private static CpuAccessPlan dense(CpuAccessPlan.AccessKind kind) {
        return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 1,
                List.of(CpuAccessPlan.AxisRole.CONTIGUOUS), 1);
    }

    private static long[] geometry(long extent) {
        return geometry(4, extent);
    }

    private static long[] geometry(int count, long extent) {
        return geometry(count, extent, 0);
    }

    /**
     * Packs rank-one dense boundaries at the invocation's actual first logical element.
     * @param count number of ordered input/output boundaries
     * @param extent complete logical element count
     * @param start first logical element of the requested nonnegative subrange
     * @return fresh common invocation pack with matching addresses and cursor origins
     */
    private static long[] geometry(int count, long extent, long start) {
        long[] result = new long[2 + 4 * count];
        result[0] = extent;
        result[1] = start;
        for (int index = 0; index < count; index++) {
            result[2 + index] = start;
            result[2 + count + index] = 1;
            result[2 + 2 * count + index] = start;
            result[2 + 3 * count + index] = extent;
        }
        return result;
    }

}
