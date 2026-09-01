package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAttentionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.classfile.ClassFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class CpuAttentionGeneratedKernelTest {
  @Test
  void executesDirectFiniteFloatAttention() throws Throwable {
    CpuAccessPlan read = plan(CpuAccessPlan.AccessKind.READ);
    CpuAccessPlan write = plan(CpuAccessPlan.AccessKind.WRITE);
    var attention =
        new CpuAttentionIr(
            DataType.FLOAT32,
            DataType.FLOAT32,
            DataType.FLOAT32,
            DataType.FLOAT32,
            false,
            false,
            2,
            List.of(0, 1, 2),
            List.of(
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32),
            List.of(read, read, read, write, write));
    var ir = attention.encodedKernelIr();
    var specialization =
        new CpuKernelSpecialization(
            CpuLoweringFingerprint.fromHex(ir.structuralKey()),
            CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
            CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
            List.of(
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32,
                DataType.FLOAT32),
            List.of(
                CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY,
                CarrierAccess.FLOAT_ARRAY),
            0,
            -1,
            List.of(),
            true,
            57);
    var generator = new CpuClassFileKernelGenerator();
    byte[] bytes = generator.generateClassBytes(specialization, ir);
    assertEquals(1, ClassFile.of().parse(bytes).methods().size());
    var handle = generator.defineClassBytes(specialization, bytes).entryPoint();
    float[] q = {1, 0, 0, 1};
    float[] k = {1, 0, 0, 1};
    float[] v = {2, 4, 6, 8};
    float[] output = new float[4];
    float[] weights = new float[4];
    var geometry =
        new CpuAttentionLowering.Geometry(
            new long[0],
            2,
            2,
            2,
            2,
            2,
            8,
            DataType.FLOAT32,
            DataType.FLOAT32,
            DataType.FLOAT32,
            DataType.FLOAT32,
            1.0,
            List.of(0, 1, 2),
            3,
            2,
            Optional.of(layout(2, 1)),
            Optional.of(layout(2, 1)),
            Optional.of(layout(2, 1)),
            Optional.empty(),
            layout(2, 1),
            Optional.of(layout(2, 1)));
    try (Arena arena = Arena.ofConfined()) {
      MemorySegment scratch = arena.allocate(8, 8);
      handle.invokeExact(
          q, k, v, output, weights, scratch, geometry.pack(new long[5]), 0L, 2L);
    }
    float a = (float) (Math.exp(1) / (Math.exp(1) + 1));
    float[] expectedWeights = {a, 1 - a, 1 - a, a};
    for (int index = 0; index < weights.length; index++)
      assertEquals(
          expectedWeights[index], weights[index], 1e-6f, java.util.Arrays.toString(weights));
    assertArrayEquals(
        new float[] {
          2 * a + 6 * (1 - a), 4 * a + 8 * (1 - a), 2 * (1 - a) + 6 * a, 4 * (1 - a) + 8 * a
        },
        output,
        1e-5f);
  }

  private static CpuAttentionLowering.NormalizedLayout layout(long... strides) {
    return new CpuAttentionLowering.NormalizedLayout(0, strides);
  }

  private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
    return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0);
  }
}
