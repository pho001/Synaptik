package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Exact small-integer CPU oracles for batched matrix/vector cotangents and both mixed adjoints. */
final class CompilerBatchedMatvecCpuIntegrationTest {
    @Test
    void coincidentDimensionsWithUnitSeedProduceSixteenAndTwenty() {
        verify(Shape.of(2, 2, 2), true);
    }

    @Test
    void coincidentDimensionsWithNonuniformSeedKeepBatchAndRowRolesDistinct() {
        verify(Shape.of(2, 2, 2), false);
    }

    @Test
    void unequalDimensionsRestoreTheCotangentColumn() {
        verify(Shape.of(2, 3, 4), false);
    }

    @Test
    void multipleBatchAxesReduceIntoTheSharedVector() {
        verify(Shape.of(2, 3, 4, 5), false);
    }

    private static void verify(Shape matrixShape, boolean unitSeed) {
        int columns = Math.toIntExact(
                ((StaticDimension) matrixShape.dimension(matrixShape.rank() - 1)).size());
        int elements = Math.toIntExact(matrixShape.knownElementCount().orElseThrow());
        int rows = elements / columns;
        float[] a = new float[elements];
        float[] b = new float[columns];
        float[] g = new float[rows];
        float[] h = new float[elements];
        float[] v = new float[columns];
        for (int i = 0; i < elements; i++) {
            a[i] = i + 1;
            h[i] = i % 7 - 3;
        }
        for (int k = 0; k < columns; k++) {
            b[k] = k - 2;
            v[k] = k + 1;
        }
        for (int row = 0; row < rows; row++) g[row] = unitSeed ? 1 : row % 5 - 2;

        float[] output = new float[rows];
        float[] da = new float[elements];
        float[] db = new float[columns];
        float[] dDbDa = new float[elements];
        float[] dDaDb = new float[columns];
        // Flatten only the batch/row prefix: each original row contributes to the shared vector.
        for (int row = 0; row < rows; row++) {
            for (int k = 0; k < columns; k++) {
                int index = row * columns + k;
                output[row] += a[index] * b[k];
                da[index] = g[row] * b[k];
                db[k] += a[index] * g[row];
                dDbDa[index] = g[row] * v[k];
                dDaDb[k] += g[row] * h[index];
            }
        }
        if (unitSeed) assertArrayEquals(new float[] {16, 20}, db);

        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor left = tensor(arena, matrixShape, true, a);
            Tensor right = tensor(arena, Shape.of(columns), true, b);
            Tensor product = left.matmul(right);
            Tensor seed = tensor(arena, product.descriptor().shape(), false, g);
            Tensor vectorSeed = tensor(arena, right.descriptor().shape(), false, v);
            Tensor matrixSeed = tensor(arena, matrixShape, false, h);
            for (var optimization : List.of(
                    GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                for (int selectedFirstRole : List.of(0, 1)) {
                    var first = new FunctionalGradientRequest.Stage(
                            List.of(new FunctionalGradientRequest.ForwardTensorReference(product)),
                            List.of(Optional.of(seed)), List.of(left, right), true,
                            FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                    Tensor secondSeed = selectedFirstRole == 0 ? matrixSeed : vectorSeed;
                    Tensor secondTarget = selectedFirstRole == 0 ? right : left;
                    var second = new FunctionalGradientRequest.Stage(
                            List.of(new FunctionalGradientRequest.FirstStageGradientReference(
                                    selectedFirstRole)),
                            List.of(Optional.of(secondSeed)), List.of(secondTarget), false,
                            FunctionalGradientRequest.DisconnectedPolicy.ERROR);
                    var advanced = EngineMixedOwnerTestAccess.advancedEngine(engine).compile(
                            CompileMode.FORWARD_AND_BACKWARD, List.of(product),
                            Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                            optimization, BackendIntent.unconstrained(), PartitionScoringConfig.neutral());
                    // Reuse the existing staged-compile seam and ordinary publication materializer.
                    var compiled = new CompiledGraph(
                            engine, EngineMixedOwnerTestAccess.compileArtifacts(advanced));
                    List<Tensor> supplied = List.of(left, right, seed, secondSeed);
                    List<Tensor> inputs = compiled.inputs().stream().map(binding -> supplied.stream()
                            .filter(tensor -> tensor.id().equals(binding.tensorId()))
                            .findFirst().orElseThrow()).toList();
                    try (var prepared = engine.prepare(compiled);
                            var result = engine.run(prepared, inputs)) {
                        assertEquals(4, result.resultCount());
                        assertValues(result, 0, product.descriptor().shape(), output);
                        assertValues(result, 1, matrixShape, da);
                        assertValues(result, 2, right.descriptor().shape(), db);
                        assertValues(result, 3, secondTarget.descriptor().shape(),
                                selectedFirstRole == 0 ? dDaDb : dDbDa);
                        assertEquals(2, result.publications().get(3).derivativeOrder().orElseThrow());
                    }
                }
            }
        }
    }

    private static void assertValues(RunResult result, int index, Shape shape, float[] expected) {
        var publication = result.publications().get(index);
        assertEquals(shape, publication.descriptor().shape());
        var bytes = result.materialize(publication, (long) expected.length * Float.BYTES).bytes();
        float[] actual = new float[expected.length];
        for (int i = 0; i < actual.length; i++) actual[i] = bytes.getFloat();
        // All nonzero results are exactly representable; MATMUL permits either exact-zero sign.
        assertArrayEquals(expected, actual, 0.0f);
    }

    private static Tensor tensor(Arena arena, Shape shape, boolean requiresGrad, float[] values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad),
                Optional.empty(), Optional.of(
                        new MemorySegmentStorage(DataType.FLOAT32, values.length, segment)));
    }
}
