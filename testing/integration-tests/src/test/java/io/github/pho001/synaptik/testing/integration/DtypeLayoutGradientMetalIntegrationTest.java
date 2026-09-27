package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalBackendConfiguration;
import io.github.pho001.synaptik.backend.metal.MetalBackendIntegration;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.EngineMixedOwnerTestAccess;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/** CPU-free public Engine evidence for Task 0066 dtype, layout, and cast-gradient closure. */
final class DtypeLayoutGradientMetalIntegrationTest {
  private static final DataType[] CARRIERS = {
    DataType.FLOAT64,
    DataType.FLOAT32,
    DataType.BFLOAT16,
    DataType.INT64,
    DataType.INT32,
    DataType.BOOL
  };

  @Test
  void allCarriersCrossThePublicCastAndAffineCanonicalizationBoundary() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        List<Tensor> inputs = new ArrayList<>();
        for (DataType carrier : CARRIERS) {
          inputs.add(tensor(arena, carrier, false, -1, 0, 2));
        }
        List<Tensor> outputs = new ArrayList<>();
        List<byte[]> expected = new ArrayList<>();
        for (int index = 0; index < CARRIERS.length; index++) {
          DataType source = CARRIERS[index];
          DataType target = CARRIERS[(index + 2) % CARRIERS.length];
          outputs.add(inputs.get(index).cast(target));
          expected.add(
              canonical(
                  target, source == DataType.BOOL ? new long[] {1, 0, 1} : new long[] {-1, 0, 2}));
          outputs.add(inputs.get(index).reshape(1, 3).permute(1, 0).contiguous());
          expected.add(
              canonical(
                  source, source == DataType.BOOL ? new long[] {1, 0, 1} : new long[] {-1, 0, 2}));
        }
        var compiled = engine.compile(outputs);
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
        try (var session = engine.session(compiled)) {
          assertResults(session.run(inputs), expected);
          assertResults(session.run(inputs), expected);
        }
      }
    }
  }

  @Test
  void scalarAllWidthTransfersAndExactBoolProgramsStayOnTheCustomMetalRoute() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        List<Tensor> inputs = new ArrayList<>();
        for (DataType carrier : CARRIERS) {
          inputs.add(scalarTensor(arena, carrier, 2));
        }
        Tensor falseScalar = scalarTensor(arena, DataType.BOOL, 0);
        Tensor floating = inputs.get(1);
        Tensor falseFloat = scalarTensor(arena, DataType.FLOAT32, -3);
        Tensor truth = inputs.getLast();
        Tensor selectedBool = tensor(arena, DataType.BOOL, false, 0, 1, 0);
        List<Tensor> outputs = new ArrayList<>();
        List<byte[]> expected = new ArrayList<>();
        for (int index = 0; index < CARRIERS.length; index++) {
          DataType target = CARRIERS[(index + 1) % CARRIERS.length];
          outputs.add(inputs.get(index).cast(target));
          expected.add(canonical(target, new long[] {CARRIERS[index] == DataType.BOOL ? 1 : 2}));
        }
        outputs.add(floating.isFinite());
        outputs.add(floating.isNaN());
        outputs.add(floating.isInf());
        outputs.add(truth.logicalAnd(falseScalar));
        outputs.add(truth.logicalOr(falseScalar));
        outputs.add(truth.logicalNot());
        outputs.add(Tensor.where(truth, floating, falseFloat));
        outputs.add(selectedBool.select(0, 1));
        expected.add(new byte[] {1});
        expected.add(new byte[] {0});
        expected.add(new byte[] {0});
        expected.add(new byte[] {0});
        expected.add(new byte[] {1});
        expected.add(new byte[] {0});
        expected.add(canonical(DataType.FLOAT32, new long[] {2}));
        expected.add(new byte[] {1});

        var compiled = engine.compile(outputs);
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
        try (var session = engine.session(compiled)) {
          var runInputs = new ArrayList<>(inputs);
          runInputs.add(falseScalar);
          runInputs.add(falseFloat);
          runInputs.add(selectedBool);
          assertResults(session.run(runInputs), expected);
        }
      }
    }
  }

  @Test
  void affineBroadcastDescendantsAndExternalMaterializedLayoutsReuseAcrossSessions()
      throws Exception {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor rowInput = tensor(arena, DataType.FLOAT32, false, 1, 2, 3);
        Tensor permuted = rowInput.reshape(1, 3).expand(2, 3).permute(1, 0);
        Tensor physicalSlice =
            permuted.slice(new long[] {1}, new long[] {2}, new int[] {1}, new long[] {1});
        Tensor chain = physicalSlice.contiguous();
        Tensor affineInput = affineFloatTensor(arena);
        Tensor affineRead = affineInput.contiguous();
        var compiled = engine.compile(List.of(chain, physicalSlice, affineRead));
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
        List<Tensor> inputs = List.of(affineInput, rowInput);
        List<byte[]> expected =
            List.of(
                canonical(DataType.FLOAT32, new long[] {1, 2, 3}),
                canonical(DataType.FLOAT32, new long[] {1, 2, 3}),
                canonical(DataType.FLOAT32, new long[] {1, 2, 4, 5}));
        try (var reused = engine.session(compiled);
            var independent = engine.session(compiled)) {
          assertResults(reused.run(inputs), expected);
          assertResults(reused.run(inputs), expected);
          assertResults(independent.run(inputs), expected);
          try (var executor = Executors.newFixedThreadPool(2)) {
            var first =
                executor.submit(
                    () -> {
                      assertResults(reused.run(inputs), expected);
                      return null;
                    });
            var second =
                executor.submit(
                    () -> {
                      assertResults(independent.run(inputs), expected);
                      return null;
                    });
            first.get();
            second.get();
          }
        }
      }
    }
  }

  @Test
  void everyFloatingCastPairPublishesItsReverseCastAdjointOnMetal() {
    Path library = configuredMetalLibrary();
    DataType[] floating = {DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16};
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        for (DataType source : floating) {
          for (DataType target : floating) {
            Tensor input = tensor(arena, source, true, -1, 2);
            Tensor seed = tensor(arena, target, false, 1, 3);
            Tensor cast = input.cast(target).reshape(1, 2).contiguous().reshape(2);
            var compiled = engine.compile(List.of(cast), List.of(seed), List.of(input));
            assertEquals(
                List.of("metal"),
                EngineMixedOwnerTestAccess.partitionOwners(compiled),
                source + " -> " + target);
            try (var session = engine.session(compiled)) {
              assertResults(
                  session.run(List.of(input, seed)),
                  List.of(
                      canonical(target, new long[] {-1, 2}), canonical(source, new long[] {1, 3})));
            }
          }
        }
      }
    }
  }

  @Test
  void savedConditionAndIndexRolesRemainLiveForGradientTargetSubsets() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor conditionInput = tensor(arena, DataType.INT32, false, 1, 0);
        Tensor condition = conditionInput.cast(DataType.BOOL);
        Tensor whenTrue = tensor(arena, DataType.FLOAT32, true, 10, 20);
        Tensor whenFalse = tensor(arena, DataType.FLOAT32, false, 30, 40);
        Tensor whereSeed = tensor(arena, DataType.FLOAT32, false, 2, 3);
        Tensor selected = Tensor.where(condition, whenTrue, whenFalse);
        var whereCompiled =
            engine.compile(List.of(selected), List.of(whereSeed), List.of(whenTrue));
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(whereCompiled));
        try (var session = engine.session(whereCompiled)) {
          assertResults(
              session.run(List.of(whereSeed, conditionInput, whenTrue, whenFalse)),
              List.of(
                  canonical(DataType.FLOAT32, new long[] {10, 40}),
                  canonical(DataType.FLOAT32, new long[] {2, 0})));
        }

        Tensor dataInput = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4, 5, 6);
        Tensor data = dataInput.reshape(2, 3);
        Tensor indicesInput = tensor(arena, DataType.INT32, false, 2, 0, 1, 2);
        Tensor indices = indicesInput.cast(DataType.INT64).reshape(2, 2);
        Tensor updatesInput = tensor(arena, DataType.FLOAT32, true, 10, 11, 12, 13);
        Tensor updates = updatesInput.reshape(2, 2);
        Tensor scatterSeedInput = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4, 5, 6);
        Tensor scatterSeed = scatterSeedInput.reshape(2, 3);
        Tensor scattered = data.scatterElements(indices, updates, 1);
        var scatterCompiled =
            engine.compile(List.of(scattered), List.of(scatterSeed), List.of(updates));
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(scatterCompiled));
        try (var session = engine.session(scatterCompiled)) {
          assertResults(
              session.run(List.of(scatterSeedInput, dataInput, indicesInput, updatesInput)),
              List.of(
                  canonical(DataType.FLOAT32, new long[] {11, 2, 10, 4, 12, 13}),
                  canonical(DataType.FLOAT32, new long[] {3, 1, 5, 6})));
        }
      }
    }
  }

  @Test
  void integerAffineIndicesFeedEveryConsumerAndScatterNdBackwardKeepsInt64Live() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor data = tensor(arena, DataType.FLOAT32, false, 10, 20);
        Tensor permuteSource = tensor(arena, DataType.INT32, false, Shape.of(1, 2), 1, 0);
        Tensor permuted = permuteSource.permute(1, 0);
        Tensor sliceSource = tensor(arena, DataType.INT64, false, 9, 0, 1, 9);
        Tensor sliced =
            sliceSource.slice(new long[] {1}, new long[] {3}, new int[] {0}, new long[] {1});
        Tensor expandSource = tensor(arena, DataType.INT32, false, 1);
        Tensor expanded = expandSource.expand(2);
        Tensor updates = tensor(arena, DataType.FLOAT32, false, 30, 40);
        List<Tensor> publications =
            List.of(
                data.gather(sliced, 0),
                data.gatherElements(sliced, 0),
                data.gatherNd(permuted),
                expanded.oneHot(2),
                data.scatterElements(sliced, updates, 0),
                data.scatterNd(permuted, updates));
        var compiled = engine.compile(publications);
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
        try (var session = engine.session(compiled)) {
          assertResults(
              session.run(List.of(data, permuteSource, sliceSource, expandSource, updates)),
              List.of(
                  canonical(DataType.FLOAT32, new long[] {10, 20}),
                  canonical(DataType.FLOAT32, new long[] {10, 20}),
                  canonical(DataType.FLOAT32, new long[] {20, 10}),
                  canonical(DataType.BOOL, new long[] {0, 1, 0, 1}),
                  canonical(DataType.FLOAT32, new long[] {30, 40}),
                  canonical(DataType.FLOAT32, new long[] {40, 30})));
        }

        Tensor scatterData = tensor(arena, DataType.FLOAT32, false, 1, 2, 3);
        Tensor savedIndexSource = tensor(arena, DataType.INT64, false, 9, 0, 2, 9);
        Tensor savedIndices =
            savedIndexSource
                .slice(new long[] {1}, new long[] {3}, new int[] {0}, new long[] {1})
                .expandDims(1);
        Tensor updateTarget = tensor(arena, DataType.FLOAT32, true, 10, 20);
        Tensor scatterSeed = tensor(arena, DataType.FLOAT32, false, 1, 2, 3);
        Tensor scattered = scatterData.scatterNd(savedIndices, updateTarget);
        var gradientCompiled =
            engine.compile(List.of(scattered), List.of(scatterSeed), List.of(updateTarget));
        assertEquals(
            List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(gradientCompiled));
        try (var session = engine.session(gradientCompiled)) {
          assertResults(
              session.run(List.of(scatterData, savedIndexSource, updateTarget, scatterSeed)),
              List.of(
                  canonical(DataType.FLOAT32, new long[] {10, 2, 20}),
                  canonical(DataType.FLOAT32, new long[] {1, 3})));
        }
      }
    }
  }

  @Test
  void generatedLayoutBackwardPathsAndNegativeSliceTargetSubsetsStayExact() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor padTarget = tensor(arena, DataType.FLOAT32, true, 1, 2);
        Tensor padSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40, 50);
        assertGradientCase(
            engine,
            padTarget.pad(
                new long[] {1},
                new long[] {2},
                io.github.pho001.synaptik.model.datatype.ScalarValue.float32(0)),
            padSeed,
            padTarget,
            List.of(padTarget, padSeed),
            new long[] {0, 1, 2, 0, 0},
            new long[] {20, 30});
        Tensor scalarPadTarget = tensor(arena, DataType.FLOAT32, true, Shape.scalar(), 7);
        Tensor scalarPadSeed = tensor(arena, DataType.FLOAT32, false, Shape.scalar(), 11);
        assertGradientCase(
            engine,
            scalarPadTarget.pad(
                new long[] {},
                new long[] {},
                io.github.pho001.synaptik.model.datatype.ScalarValue.float32(0)),
            scalarPadSeed,
            scalarPadTarget,
            List.of(scalarPadTarget, scalarPadSeed),
            new long[] {7},
            new long[] {11});

        Tensor concatLeft = tensor(arena, DataType.FLOAT32, false, 1, 2);
        Tensor concatTarget = tensor(arena, DataType.FLOAT32, true, 3, 4, 5);
        Tensor concatSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40, 50);
        assertGradientCase(
            engine,
            Tensor.concat(0, concatLeft, concatTarget),
            concatSeed,
            concatTarget,
            List.of(concatLeft, concatTarget, concatSeed),
            new long[] {1, 2, 3, 4, 5},
            new long[] {30, 40, 50});

        Tensor stackTarget = tensor(arena, DataType.FLOAT32, true, 1, 2);
        Tensor stackOther = tensor(arena, DataType.FLOAT32, false, 3, 4);
        Tensor stackSeedInput = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40);
        assertGradientCase(
            engine,
            Tensor.stack(0, stackTarget, stackOther),
            stackSeedInput.reshape(2, 2),
            stackTarget,
            List.of(stackTarget, stackOther, stackSeedInput),
            new long[] {1, 2, 3, 4},
            new long[] {10, 20});

        Tensor selectInput = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4, 5, 6);
        Tensor selectTarget = selectInput.reshape(2, 3);
        Tensor selectSeed = tensor(arena, DataType.FLOAT32, false, 10, 20);
        assertGradientCase(
            engine,
            selectTarget.select(1, 1),
            selectSeed,
            selectTarget,
            List.of(selectInput, selectSeed),
            new long[] {2, 5},
            new long[] {0, 10, 0, 0, 20, 0});

        Tensor updateBase = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4, 5);
        Tensor updateTarget = tensor(arena, DataType.FLOAT32, true, 9, 8);
        Tensor updateSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40, 50);
        assertGradientCase(
            engine,
            updateBase.sliceUpdate(updateTarget, new long[] {1}, new int[] {0}, new long[] {2}),
            updateSeed,
            updateTarget,
            List.of(updateBase, updateTarget, updateSeed),
            new long[] {1, 9, 3, 8, 5},
            new long[] {20, 40});

        Tensor windowTarget = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4);
        Tensor windowSeedInput = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40);
        assertGradientCase(
            engine,
            windowTarget.unfold(0, 2, 2),
            windowSeedInput.reshape(2, 2),
            windowTarget,
            List.of(windowTarget, windowSeedInput),
            new long[] {1, 2, 3, 4},
            new long[] {10, 20, 30, 40});

        Tensor sliceTarget = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4, 5);
        Tensor sliceSeed = tensor(arena, DataType.FLOAT32, false, 10, 20);
        assertGradientCase(
            engine,
            sliceTarget.slice(new long[] {1}, new long[] {5}, new int[] {0}, new long[] {2}),
            sliceSeed,
            sliceTarget,
            List.of(sliceTarget, sliceSeed),
            new long[] {2, 4},
            new long[] {0, 10, 0, 20, 0});

        Tensor inverseTarget =
            tensor(arena, DataType.FLOAT32, true, Shape.of(2, 3), 1, 2, 3, 4, 5, 6);
        Tensor inverseSeed =
            tensor(arena, DataType.FLOAT32, false, Shape.of(3, 2), 10, 20, 30, 40, 50, 60);
        Tensor inverseChain = inverseTarget.permute(1, 0).expandDims(1).squeeze(1);
        assertGradientCase(
            engine,
            inverseChain,
            inverseSeed,
            inverseTarget,
            List.of(inverseTarget, inverseSeed),
            new long[] {1, 4, 2, 5, 3, 6},
            new long[] {10, 30, 50, 20, 40, 60});

        Window2dAttrs window2d = new Window2dAttrs(2, 2, 2, 2, 0, 0, 1, 1, false);
        Tensor image2dTarget =
            tensor(arena, DataType.FLOAT32, true, Shape.of(1, 1, 2, 2), 1, 2, 3, 4);
        Tensor window2dSeed =
            tensor(arena, DataType.FLOAT32, false, Shape.of(1, 4, 1), 10, 20, 30, 40);
        assertGradientCase(
            engine,
            image2dTarget.unfold2d(window2d),
            window2dSeed,
            image2dTarget,
            List.of(image2dTarget, window2dSeed),
            new long[] {1, 2, 3, 4},
            new long[] {10, 20, 30, 40});
        Tensor columns2dTarget =
            tensor(arena, DataType.FLOAT32, true, Shape.of(1, 4, 1), 1, 2, 3, 4);
        Tensor image2dSeed =
            tensor(arena, DataType.FLOAT32, false, Shape.of(1, 1, 2, 2), 10, 20, 30, 40);
        assertGradientCase(
            engine,
            columns2dTarget.fold2d(Shape.of(1, 1, 2, 2), window2d),
            image2dSeed,
            columns2dTarget,
            List.of(columns2dTarget, image2dSeed),
            new long[] {1, 2, 3, 4},
            new long[] {10, 20, 30, 40});

        Window3dAttrs window3d = new Window3dAttrs(2, 2, 2, 2, 2, 2, 0, 0, 0, 1, 1, 1, false);
        Tensor image3dTarget =
            tensor(arena, DataType.FLOAT32, true, Shape.of(1, 1, 2, 2, 2), 1, 2, 3, 4, 5, 6, 7, 8);
        Tensor window3dSeed =
            tensor(
                arena, DataType.FLOAT32, false, Shape.of(1, 8, 1), 10, 20, 30, 40, 50, 60, 70, 80);
        assertGradientCase(
            engine,
            image3dTarget.unfold3d(window3d),
            window3dSeed,
            image3dTarget,
            List.of(image3dTarget, window3dSeed),
            new long[] {1, 2, 3, 4, 5, 6, 7, 8},
            new long[] {10, 20, 30, 40, 50, 60, 70, 80});
        Tensor columns3dTarget =
            tensor(arena, DataType.FLOAT32, true, Shape.of(1, 8, 1), 1, 2, 3, 4, 5, 6, 7, 8);
        Tensor image3dSeed =
            tensor(
                arena,
                DataType.FLOAT32,
                false,
                Shape.of(1, 1, 2, 2, 2),
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80);
        assertGradientCase(
            engine,
            columns3dTarget.fold3d(Shape.of(1, 1, 2, 2, 2), window3d),
            image3dSeed,
            columns3dTarget,
            List.of(columns3dTarget, image3dSeed),
            new long[] {1, 2, 3, 4, 5, 6, 7, 8},
            new long[] {10, 20, 30, 40, 50, 60, 70, 80});

        Tensor negativeBase = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4, 5);
        Tensor negativeUpdate = tensor(arena, DataType.FLOAT32, false, 9, 8);
        Tensor negativeSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40, 50);
        Tensor negativeOutput =
            negativeBase.sliceUpdate(
                negativeUpdate, new long[] {3}, new int[] {0}, new long[] {-2});
        assertGradientCase(
            engine,
            negativeOutput,
            negativeSeed,
            negativeBase,
            List.of(negativeBase, negativeUpdate, negativeSeed),
            new long[] {1, 8, 3, 9, 5},
            new long[] {10, 0, 30, 0, 50});

        Tensor rejectedBase = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4, 5);
        Tensor rejectedUpdate = tensor(arena, DataType.FLOAT32, true, 9, 8);
        Tensor rejectedSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30, 40, 50);
        Tensor rejectedOutput =
            rejectedBase.sliceUpdate(
                rejectedUpdate, new long[] {3}, new int[] {0}, new long[] {-2});
        RuntimeException rejection =
            assertThrows(
                RuntimeException.class,
                () ->
                    engine.compile(
                        List.of(rejectedOutput), List.of(rejectedSeed), List.of(rejectedUpdate)));
        assertTrue(
            rejection.getMessage().contains("SLICE")
                && rejection.getMessage().contains("no hard-eligible backend"),
            rejection.getMessage());
      }
    }
  }

  @Test
  void task0069ScatterAddAndGatherDataGradientRunEndToEnd() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.numericalProfile(NumericalProfile.ACCELERATOR);
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor data = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4);
        Tensor indices = tensor(arena, DataType.INT64, false, 0, 0, 2);
        Tensor updates = tensor(arena, DataType.FLOAT32, false, 10, 20, 30);
        Tensor scattered = data.scatterAdd(indices, updates, 0);
        var direct = engine.compile(List.of(scattered));
        assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(direct));
        try (var session = engine.session(direct)) {
          assertResults(
              session.run(List.of(data, indices, updates)),
              List.of(canonical(DataType.FLOAT32, new long[] {31, 2, 33, 4})));
        }

        Tensor gatherTarget = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4);
        Tensor gatherSeed = tensor(arena, DataType.FLOAT32, false, 10, 20, 30);
        assertGradientCase(
            engine,
            gatherTarget.gather(indices, 0),
            gatherSeed,
            gatherTarget,
            List.of(gatherTarget, indices, gatherSeed),
            new long[] {1, 1, 3},
            new long[] {30, 0, 30, 0});
      }
    }
  }

  @Test
  void unsupportedGradientAndGeometryDomainsFailEarlyWithoutCpuFallback() {
    Path library = configuredMetalLibrary();
    try (Arena arena = Arena.ofShared();
        Engine.Builder builder = Engine.builder()) {
      builder.takeOwnership(MetalBackendIntegration.open(new MetalBackendConfiguration(library)));
      try (Engine engine = builder.build()) {
        Tensor arithmeticScatterData = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4);
        Tensor arithmeticScatterIndices = tensor(arena, DataType.INT64, false, 0, 2);
        Tensor arithmeticScatterUpdates = tensor(arena, DataType.FLOAT32, false, 10, 20);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(
                    List.of(
                        arithmeticScatterData.scatterElements(
                            arithmeticScatterIndices,
                            arithmeticScatterUpdates,
                            0,
                            ScatterReduction.ADD))),
            "SCATTER_ELEMENTS");

        Tensor overlapTarget = tensor(arena, DataType.FLOAT32, true, 1, 2, 3, 4);
        Tensor overlapSeedInput = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4, 5, 6);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(
                    List.of(overlapTarget.unfold(0, 3, 1)),
                    List.of(overlapSeedInput.reshape(2, 3)),
                    List.of(overlapTarget)),
            "FOLD_AXIS");

        Tensor tileTarget = tensor(arena, DataType.FLOAT32, true, 1, 2);
        Tensor tileSeed = tensor(arena, DataType.FLOAT32, false, 1, 2, 3, 4);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(List.of(tileTarget.tile(2)), List.of(tileSeed), List.of(tileTarget)),
            "SUM");
        Tensor expandedTarget = tensor(arena, DataType.FLOAT32, true, 7);
        Tensor expandedSeed = tensor(arena, DataType.FLOAT32, false, 1, 2, 3);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(
                    List.of(expandedTarget.expand(3)),
                    List.of(expandedSeed),
                    List.of(expandedTarget)),
            "SUM");

        Tensor branchTarget = tensor(arena, DataType.FLOAT32, true, 7);
        Tensor condition = tensor(arena, DataType.BOOL, false, 1, 0, 1);
        Tensor other = tensor(arena, DataType.FLOAT32, false, 3, 4, 5);
        Tensor branchSeed = tensor(arena, DataType.FLOAT32, false, 1, 2, 3);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(
                    List.of(Tensor.where(condition, branchTarget.reshape(1).expand(3), other)),
                    List.of(branchSeed),
                    List.of(branchTarget)),
            "SUM");

        Tensor multiPathTarget = tensor(arena, DataType.FLOAT32, true, 7);
        Tensor multiPathSeed = tensor(arena, DataType.FLOAT32, false, Shape.of(2, 1), 1, 2);
        assertMetalCompileRejected(
            engine,
            () ->
                engine.compile(
                    List.of(Tensor.stack(0, multiPathTarget, multiPathTarget)),
                    List.of(multiPathSeed),
                    List.of(multiPathTarget)),
            "ADD");

        Tensor dynamic =
            TensorFactory.create(
                new TensorDescriptor(
                    DataType.FLOAT32,
                    Shape.ofDimensions(new DynamicDimension("N")),
                    Optional.empty(),
                    false));
        assertMetalCompileRejected(
            engine, () -> engine.compile(List.of(dynamic.cast(DataType.FLOAT64))), "CAST");

        Tensor empty =
            TensorFactory.create(
                new TensorDescriptor(
                    DataType.FLOAT32,
                    Shape.of(0),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(0))),
                    false));
        assertMetalCompileRejected(
            engine, () -> engine.compile(List.of(empty.cast(DataType.FLOAT64))), "CAST");

        Shape overlapShape = Shape.of(2, 2);
        Tensor overlapping =
            TensorFactory.create(
                new TensorDescriptor(
                    DataType.FLOAT32,
                    overlapShape,
                    Optional.of(LayoutDescriptor.of(overlapShape, new long[] {1, 1}, 0L, true)),
                    false));
        assertMetalCompileRejected(
            engine, () -> engine.compile(List.of(overlapping.contiguous())), "CONTIGUOUS");

        Shape broadcastShape = Shape.of(3);
        MemorySegment broadcastStorage = arena.allocate(Float.BYTES, Float.BYTES);
        broadcastStorage.set(ValueLayout.JAVA_FLOAT, 0L, 7.0f);
        Tensor externalBroadcast =
            TensorFactory.create(
                new TensorDescriptor(
                    DataType.FLOAT32,
                    broadcastShape,
                    Optional.of(LayoutDescriptor.of(broadcastShape, new long[] {0}, 0L, true)),
                    false),
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, broadcastStorage)));
        var externalBroadcastCompiled = engine.compile(List.of(externalBroadcast.contiguous()));
        assertEquals(
            List.of("metal"),
            EngineMixedOwnerTestAccess.partitionOwners(externalBroadcastCompiled));
        RuntimeException externalBroadcastFailure =
            assertThrows(RuntimeException.class, () -> engine.session(externalBroadcastCompiled));
        assertTrue(
            externalBroadcastFailure
                .getMessage()
                .contains("positive non-overlapping storage geometry"),
            externalBroadcastFailure.getMessage());

        IllegalArgumentException signedStride =
            assertThrows(
                IllegalArgumentException.class,
                () -> LayoutDescriptor.of(Shape.of(2), new long[] {-1}, 1L, true));
        assertTrue(
            signedStride.getMessage().toLowerCase().contains("stride"), signedStride.getMessage());
      }
    }
  }

  private static void assertMetalCompileRejected(
      Engine engine, Runnable compilation, String rejectedKind) {
    RuntimeException failure = assertThrows(RuntimeException.class, compilation::run);
    assertTrue(
        failure.getMessage().contains(rejectedKind)
            && failure.getMessage().contains("no hard-eligible backend"),
        failure.getMessage());
  }

  private static void assertGradientCase(
      Engine engine,
      Tensor output,
      Tensor seed,
      Tensor target,
      List<Tensor> inputs,
      long[] expectedForward,
      long[] expectedGradient) {
    var compiled = engine.compile(List.of(output), List.of(seed), List.of(target));
    assertEquals(List.of("metal"), EngineMixedOwnerTestAccess.partitionOwners(compiled));
    try (var session = engine.session(compiled)) {
      assertResults(
          session.run(inputs),
          List.of(
              canonical(DataType.FLOAT32, expectedForward),
              canonical(DataType.FLOAT32, expectedGradient)));
    }
  }

  private static Tensor scalarTensor(Arena arena, DataType type, long value) {
    long width = type.byteWidth();
    MemorySegment storage = arena.allocate(width, width);
    switch (type) {
      case FLOAT64 -> storage.set(ValueLayout.JAVA_DOUBLE, 0L, (double) value);
      case FLOAT32 -> storage.set(ValueLayout.JAVA_FLOAT, 0L, (float) value);
      case BFLOAT16 ->
          storage.set(
              ValueLayout.JAVA_SHORT, 0L, (short) (Float.floatToRawIntBits((float) value) >>> 16));
      case INT64 -> storage.set(ValueLayout.JAVA_LONG, 0L, value);
      case INT32 -> storage.set(ValueLayout.JAVA_INT, 0L, (int) value);
      case BOOL -> storage.set(ValueLayout.JAVA_BYTE, 0L, (byte) (value == 0 ? 0 : 1));
    }
    Shape shape = Shape.scalar();
    return TensorFactory.create(
        new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false),
        Optional.empty(),
        Optional.of(new MemorySegmentStorage(type, 1, storage)));
  }

  private static Tensor affineFloatTensor(Arena arena) {
    MemorySegment storage = arena.allocate(8L * Float.BYTES, Float.BYTES);
    for (int index = 0; index < 8; index++) {
      storage.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) index);
    }
    Shape shape = Shape.of(2, 2);
    return TensorFactory.create(
        new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.of(shape, new long[] {3, 1}, 1L, true)),
            false),
        Optional.empty(),
        Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 8, storage)));
  }

  private static Tensor tensor(Arena arena, DataType type, boolean requiresGrad, long... values) {
    return tensor(arena, type, requiresGrad, Shape.of(values.length), values);
  }

  private static Tensor tensor(
      Arena arena, DataType type, boolean requiresGrad, Shape shape, long... values) {
    assertEquals(shape.knownElementCount().orElseThrow(), values.length);
    long width =
        switch (type) {
          case FLOAT64, INT64 -> Long.BYTES;
          case FLOAT32, INT32 -> Integer.BYTES;
          case BFLOAT16 -> Short.BYTES;
          case BOOL -> Byte.BYTES;
        };
    MemorySegment storage = arena.allocate(width * values.length, width);
    for (int index = 0; index < values.length; index++) {
      long value = values[index];
      switch (type) {
        case FLOAT64 -> storage.setAtIndex(ValueLayout.JAVA_DOUBLE, index, (double) value);
        case FLOAT32 -> storage.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) value);
        case BFLOAT16 ->
            storage.setAtIndex(
                ValueLayout.JAVA_SHORT,
                index,
                (short) (Float.floatToRawIntBits((float) value) >>> 16));
        case INT64 -> storage.setAtIndex(ValueLayout.JAVA_LONG, index, value);
        case INT32 -> storage.setAtIndex(ValueLayout.JAVA_INT, index, (int) value);
        case BOOL -> storage.setAtIndex(ValueLayout.JAVA_BYTE, index, (byte) (value == 0 ? 0 : 1));
      }
    }
    TensorDescriptor descriptor =
        new TensorDescriptor(
            type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    return TensorFactory.create(
        descriptor,
        Optional.empty(),
        Optional.of(new MemorySegmentStorage(type, values.length, storage)));
  }

  private static byte[] canonical(DataType type, long[] values) {
    int width =
        switch (type) {
          case FLOAT64, INT64 -> Long.BYTES;
          case FLOAT32, INT32 -> Integer.BYTES;
          case BFLOAT16 -> Short.BYTES;
          case BOOL -> Byte.BYTES;
        };
    ByteBuffer bytes = ByteBuffer.allocate(width * values.length).order(ByteOrder.BIG_ENDIAN);
    for (long value : values) {
      switch (type) {
        case FLOAT64 -> bytes.putDouble(value);
        case FLOAT32 -> bytes.putFloat((float) value);
        case BFLOAT16 -> bytes.putShort((short) (Float.floatToRawIntBits((float) value) >>> 16));
        case INT64 -> bytes.putLong(value);
        case INT32 -> bytes.putInt((int) value);
        case BOOL -> bytes.put((byte) (value == 0 ? 0 : 1));
      }
    }
    return bytes.array();
  }

  private static void assertResults(RunResult result, List<byte[]> expected) {
    try (result) {
      assertEquals(expected.size(), result.resultCount());
      for (int index = 0; index < expected.size(); index++) {
        ByteBuffer bytes =
            result
                .materialize(result.publications().get(index), expected.get(index).length)
                .bytes();
        byte[] actual = new byte[bytes.remaining()];
        bytes.get(actual);
        assertArrayEquals(expected.get(index), actual, "publication " + index);
      }
    }
  }

  private static Path configuredMetalLibrary() {
    String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
    Assumptions.assumeTrue(
        configured != null && !configured.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
    return Path.of(configured).toAbsolutePath().normalize();
  }
}
