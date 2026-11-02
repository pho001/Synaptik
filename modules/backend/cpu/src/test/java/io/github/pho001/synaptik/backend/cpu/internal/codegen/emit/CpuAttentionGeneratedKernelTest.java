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
        new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()), CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
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
    var handle = generator.defineClassBytes(specialization, ir, bytes).entryPoint();
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
    var shared = io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row.shared();
    var query = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(0, 2);
    var feature = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 1);
    for (int mode = 0; mode < 3; mode++) {
      var rows = mode == 0 ? List.of(shared, shared, shared, shared, shared)
          : mode == 1 ? List.of(query, shared, shared, query, query)
          : List.of(shared, shared, feature, feature, shared);
      var controlled = controlledEntry(specialization, ir, rows);
      float[] actual = {-99, -99, -99, -99}, actualWeights = {-99, -99, -99, -99};
      // All-valid has exactly the 16-byte header: no ROW byte exists to accidentally read.
      MemorySegment control = mode == 0 ? MemorySegment.ofArray(new byte[16])
          : CpuAffineCopyGeneratedKernelTest.control(new byte[] {1, 0});
      if (mode == 0) control.set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 1);
      try (Arena arena = Arena.ofConfined()) {
        controlled.invokeWithArguments(mode == 1 ? new float[] {1, 0} : q, k,
            mode == 2 ? new float[] {2, Float.NaN, 6} : v, actual, actualWeights,
            arena.allocate(8, 8), geometry.pack(new long[5]), 0L, 2L, control);
      }
      float[] expected = output.clone(), expectedW = weights.clone();
      if (mode == 1) {
        expected[2] = expected[3] = 0;
        expectedW[2] = expectedW[3] = 0;
      } else if (mode == 2) expected[1] = expected[3] = 0;
      assertArrayEquals(expected, actual, 1e-5f);
      assertArrayEquals(expectedW, actualWeights, 1e-6f);
    }
    // Nonleading prefix ROW(1) repeats across the first prefix axis, so modulo is essential.
    var prefixRow = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row(1, 1);
    var prefixEntry = controlledEntry(specialization, ir,
        List.of(prefixRow, prefixRow, prefixRow, prefixRow, prefixRow));
    var prefixGeometry = new CpuAttentionLowering.Geometry(new long[] {2, 2}, 1, 1, 1, 1, 4, 4,
        DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32, 1.0,
        List.of(0, 1, 2), 3, 2, Optional.of(layout(2, 1, 1, 1)),
        Optional.of(layout(2, 1, 1, 1)), Optional.of(layout(2, 1, 1, 1)), Optional.empty(),
        layout(2, 1, 1, 1), Optional.of(layout(2, 1, 1, 1)));
    float[] prefixOutput = {-99, -99, -99, -99}, prefixWeights = {-99, -99, -99, -99};
    try (Arena arena = Arena.ofConfined()) {
      prefixEntry.invokeWithArguments(new float[] {1, Float.NaN, 1},
          new float[] {1, Float.NaN, 1}, new float[] {2, Float.NaN, 6},
          prefixOutput, prefixWeights, arena.allocate(4, 8), prefixGeometry.pack(new long[5]),
          0L, 4L, CpuAffineCopyGeneratedKernelTest.control(new byte[] {1, 0}));
    }
    assertArrayEquals(new float[] {2, 0, 6, 0}, prefixOutput);
    assertArrayEquals(new float[] {1, 0, 1, 0}, prefixWeights);
  }

  private static java.lang.invoke.MethodHandle controlledEntry(CpuKernelSpecialization specialization,
      io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr ir,
      List<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl.Row> rows)
      throws ReflectiveOperationException {
    var owner = java.lang.constant.ClassDesc.of(
        CpuAttentionGeneratedKernelTest.class.getPackageName() + ".ControlledAttention");
    var facts = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanControl(rows, rows);
    byte[] bytes = ClassFile.of().build(owner, builder -> {
      builder.withVersion(ClassFile.JAVA_26_VERSION, 0).withFlags(ClassFile.ACC_FINAL);
      CpuClassFileKernelGenerator.emitMethods(builder, owner, "", specialization, ir, facts);
    });
    var lookup = java.lang.invoke.MethodHandles.lookup().defineHiddenClass(bytes, true);
    return lookup.findStatic(lookup.lookupClass(), "invoke",
        specialization.entryType().appendParameterTypes(MemorySegment.class));
  }

  private static CpuAttentionLowering.NormalizedLayout layout(long... strides) {
    return new CpuAttentionLowering.NormalizedLayout(0, strides);
  }

  private static CpuAccessPlan plan(CpuAccessPlan.AccessKind kind) {
    return new CpuAccessPlan(kind, CpuAccessPlan.Regime.DENSE_LINEAR, 0, List.of(), 0);
  }
}
