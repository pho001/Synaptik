package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
  void positiveInfinityTiesPublishRoundedThirdsAndExcludeFiniteScores() throws Throwable {
    var read = plan(CpuAccessPlan.AccessKind.READ);
    var write = plan(CpuAccessPlan.AccessKind.WRITE);
    var types = java.util.Collections.nCopies(5, DataType.FLOAT32);
    var attention = new CpuAttentionIr(DataType.FLOAT32, DataType.FLOAT32,
        DataType.FLOAT32, DataType.FLOAT32, false, false, 2, List.of(0, 1, 2),
        types, List.of(read, read, read, write, write));
    var ir = attention.encodedKernelIr();
    var specialization = new CpuKernelSpecialization(
        CpuLoweringFingerprint.fromHex(ir.structuralKey()),
        CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types,
        java.util.Collections.nCopies(5, CarrierAccess.FLOAT_ARRAY),
        0, -1, List.of(), true, CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA);
    var generator = new CpuClassFileKernelGenerator();
    var handle = generator.defineClassBytes(specialization, ir,
        generator.generateClassBytes(specialization, ir)).entryPoint();
    var geometry = new CpuAttentionLowering.Geometry(new long[0], 1, 4, 1, 1, 1, 16,
        DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32, DataType.FLOAT32,
        1.0, List.of(0, 1, 2), 3, 2,
        Optional.of(layout(1, 1)), Optional.of(layout(1, 1)), Optional.of(layout(1, 1)),
        Optional.empty(), layout(1, 1), Optional.of(layout(4, 1)));
    float[] output = {0f}, weights = new float[4];
    try (Arena arena = Arena.ofConfined()) {
      handle.invokeWithArguments(new float[] {1f},
          new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY,
              0f, Float.POSITIVE_INFINITY},
          new float[] {3f, 3f, 100f, 3f}, output, weights,
          arena.allocate(16, 8), geometry.pack(new long[5]), 0L, 1L);
    }
    assertEquals(0x3eaaaaab, Float.floatToRawIntBits(weights[0]));
    assertEquals(0x3eaaaaab, Float.floatToRawIntBits(weights[1]));
    assertEquals(0, Float.floatToRawIntBits(weights[2]));
    assertEquals(0x3eaaaaab, Float.floatToRawIntBits(weights[3]));
    assertEquals(3f, output[0]);
  }

  @Test
  void bfloatPublicationKeepsLongAddressesAcrossMaskModesAndOptionalWeights() throws Throwable {
    DataType type = DataType.BFLOAT16;
    for (boolean masked : List.of(false, true)) for (boolean causal : List.of(false, true))
      for (boolean two : List.of(false, true)) for (boolean segments : List.of(false, true))
        for (int maskMode = 0; maskMode < (masked ? 3 : 1); maskMode++) {
          var types = new java.util.ArrayList<DataType>(List.of(type, type, type));
          var accesses = new java.util.ArrayList<CpuAccessPlan>(List.of(
              plan(CpuAccessPlan.AccessKind.READ), plan(CpuAccessPlan.AccessKind.READ),
              plan(CpuAccessPlan.AccessKind.READ)));
          if (masked) {
            types.add(DataType.BOOL);
            accesses.add(plan(CpuAccessPlan.AccessKind.READ));
          }
          int outputBoundary = types.size();
          types.add(type);
          accesses.add(plan(CpuAccessPlan.AccessKind.WRITE));
          if (two) {
            types.add(type);
            accesses.add(plan(CpuAccessPlan.AccessKind.WRITE));
          }
          var roles = masked ? List.of(0, 1, 2, 3) : List.of(0, 1, 2);
          var attention = new CpuAttentionIr(type, type, type, type, masked, causal,
              two ? 2 : 1, roles, types, accesses);
          var ir = attention.encodedKernelIr();
          var carriers = new java.util.ArrayList<CarrierAccess>();
          for (DataType boundary : types)
            carriers.add(segments ? CarrierAccess.MEMORY_SEGMENT
                : boundary == DataType.BOOL ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY);
          var specialization = new CpuKernelSpecialization(
              CpuLoweringFingerprint.fromHex(ir.structuralKey()),
              CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types, carriers,
              0, -1, List.of(), true, CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA);
          var generator = new CpuClassFileKernelGenerator();
          var handle = generator.defineClassBytes(specialization, ir,
              generator.generateClassBytes(specialization, ir)).entryPoint();
          short[] q = new short[4], k = new short[4];
          short[] v = (short[]) CpuLayerNormGeneratedKernelTest.represented(type,
              new double[] {2, 4, 6, 8});
          short[] output = new short[4], weights = new short[4];
          byte[] mask = maskMode == 0 ? new byte[] {1, 1, 1, 1}
              : maskMode == 1 ? new byte[] {1, 0, 0, 1} : new byte[4];
          var geometry = new CpuAttentionLowering.Geometry(new long[] {1}, 2, 2, 2, 2, 2, 8,
              type, type, type, type, 1.0, roles, outputBoundary, two ? 2 : 1,
              Optional.of(layout(4, 2, 1)), Optional.of(layout(4, 2, 1)),
              Optional.of(layout(4, 2, 1)), masked ? Optional.of(layout(4, 2, 1)) : Optional.empty(),
              layout(4, 2, 1), two ? Optional.of(layout(4, 2, 1)) : Optional.empty());
          var arguments = new java.util.ArrayList<Object>();
          for (short[] input : List.of(q, k, v))
            arguments.add(segments ? MemorySegment.ofArray(input) : input);
          if (masked) arguments.add(segments ? MemorySegment.ofArray(mask) : mask);
          arguments.add(segments ? MemorySegment.ofArray(output) : output);
          if (two) arguments.add(segments ? MemorySegment.ofArray(weights) : weights);
          try (Arena arena = Arena.ofConfined()) {
            arguments.add(arena.allocate(8, 8));
            arguments.add(geometry.pack(new long[types.size()]));
            arguments.add(0L);
            arguments.add(2L);
            handle.invokeWithArguments(arguments);
          }
          double[] expectedOutput = masked && maskMode == 2 ? new double[4]
              : masked && maskMode == 1 ? new double[] {2, 4, 6, 8}
              : causal ? new double[] {2, 4, 4, 6} : new double[] {4, 6, 4, 6};
          double[] expectedWeights = masked && maskMode == 2 ? new double[4]
              : masked && maskMode == 1 ? new double[] {1, 0, 0, 1}
              : causal ? new double[] {1, 0, .5, .5} : new double[] {.5, .5, .5, .5};
          String witness = "masked=" + masked + "/causal=" + causal + "/weights=" + two
              + "/segments=" + segments + "/maskMode=" + maskMode;
          assertArrayEquals(expectedOutput, CpuLayerNormGeneratedKernelTest.decoded(type, output),
              witness);
          if (two) assertArrayEquals(expectedWeights,
              CpuLayerNormGeneratedKernelTest.decoded(type, weights), witness);
        }
  }

  @Test
  void everyLowFormatStoresAttentionOutputsAndWeightsAtItsDeclaredBoundary() throws Throwable {
    for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16,
        DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
      var read = plan(CpuAccessPlan.AccessKind.READ);
      var write = plan(CpuAccessPlan.AccessKind.WRITE);
      var types = java.util.Collections.nCopies(5, type);
      var attention = new CpuAttentionIr(type, type, type, type, false, false, 2,
          List.of(0, 1, 2), types, List.of(read, read, read, write, write));
      var ir = attention.encodedKernelIr();
      var carrier = type.byteWidth() == 1 ? CarrierAccess.BYTE_ARRAY : CarrierAccess.SHORT_ARRAY;
      var specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
          CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, types,
          java.util.Collections.nCopies(5, carrier), 0, -1, List.of(), true,
          CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA);
      var generator = new CpuClassFileKernelGenerator();
      var handle = generator.defineClassBytes(specialization, ir,
          generator.generateClassBytes(specialization, ir)).entryPoint();
      Object q = CpuLayerNormGeneratedKernelTest.represented(type, new double[4]);
      Object k = CpuLayerNormGeneratedKernelTest.represented(type, new double[4]);
      Object v = CpuLayerNormGeneratedKernelTest.represented(type, new double[] {2, 4, 6, 8});
      Object output = CpuLayerNormGeneratedKernelTest.represented(type, new double[4]);
      Object weights = CpuLayerNormGeneratedKernelTest.represented(type, new double[4]);
      var geometry = new CpuAttentionLowering.Geometry(new long[0], 2, 2, 2, 2, 2, 8,
          type, type, type, type, (double) 0.1f, List.of(0, 1, 2), 3, 2,
          Optional.of(layout(2, 1)), Optional.of(layout(2, 1)), Optional.of(layout(2, 1)),
          Optional.empty(), layout(2, 1), Optional.of(layout(2, 1)));
      try (Arena arena = Arena.ofConfined()) {
        handle.invokeWithArguments(q, k, v, output, weights, arena.allocate(8, 8),
            geometry.pack(new long[5]), 0L, 2L);
      }
      assertArrayEquals(new double[] {4, 6, 4, 6}, CpuLayerNormGeneratedKernelTest.decoded(type, output));
      assertArrayEquals(new double[] {.5, .5, .5, .5}, CpuLayerNormGeneratedKernelTest.decoded(type, weights));
    }
  }


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
        CpuKernelSpecialization.ATTENTION_LITERAL_EXP_SCHEMA);
    assertThrows(IllegalArgumentException.class, () -> new CpuKernelSpecialization(
        specialization.loweringFingerprint(), specialization.executionStrategy(),
        specialization.boundaryDataTypes(), specialization.carrierPattern(),
        specialization.vectorSpeciesBitSize(), specialization.materializedSourcePosition(),
        specialization.scalarPowerRealizations(), specialization.scratchParameter(), 57));
    var generator = new CpuClassFileKernelGenerator();
    byte[] bytes = generator.generateClassBytes(specialization, ir);
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
