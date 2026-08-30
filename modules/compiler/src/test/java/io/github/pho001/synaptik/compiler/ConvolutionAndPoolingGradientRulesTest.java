package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.Fold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.Unfold3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayDeque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ConvolutionAndPoolingGradientRulesTest {
    @Test
    void linearRemainsTheVisiblePermuteMatmulAndOptionalAddComposition() {
        Tensor input = tensor(Shape.of(2, 4));
        Tensor weight = tensor(Shape.of(3, 4));
        Tensor bias = tensor(Shape.of(3));
        Tensor output = input.linear(weight, bias);
        var addition = output.provenance().orElseThrow();
        Tensor product = addition.inputs().getFirst();
        Tensor transposedWeight =
                product.provenance().orElseThrow().inputs().get(1);

        assertEquals(
                io.github.pho001.synaptik.model.operation.elementwise.binary
                        .BinaryArithmeticKind.ADD,
                addition.operation().kind());
        assertEquals(MatmulKind.MATMUL,
                product.provenance().orElseThrow().operation().kind());
        assertEquals(AxisTransformKind.PERMUTE,
                transposedWeight.provenance().orElseThrow().operation().kind());
        assertCompiles(output.sum(), input);
        assertCompiles(output.sum(), weight);
        assertCompiles(output.sum(), bias);
    }

    @Test
    void groupedConvolutionBuildsWindowMatmulFoldAndBiasReduction() {
        Tensor input = tensor(Shape.of(2, 4, 5, 5));
        Tensor weight = tensor(Shape.of(6, 2, 3, 3));
        Tensor bias = tensor(Shape.of(6));
        Tensor output = input.conv2d(
                weight, bias, new Conv2dAttrs(1, 1, 1, 1, 1, 1, 2));
        Tensor objective = output.sum();

        Tensor inputGradient = gradient(objective, input);
        Tensor weightGradient = gradient(objective, weight);
        Tensor biasGradient = gradient(objective, bias);
        assertTrue(containsKind(inputGradient, WindowTransformKind.FOLD2D));
        assertTrue(containsKind(weightGradient, WindowTransformKind.UNFOLD2D));
        assertTrue(containsKind(biasGradient, AggregateReductionKind.SUM));
        assertCompiles(objective, input);
        assertCompiles(objective, weight);
        assertCompiles(objective, bias);
    }

    @Test
    void averageAndMaximumPoolingBothFoldToTheExactInputShape() {
        Tensor input = tensor(Shape.of(2, 3, 5, 5));
        AveragePool2dAttrs averageAttrs =
                new AveragePool2dAttrs(3, 3, 2, 2, 1, 1, 1, 1, true);
        MaxPool2dAttrs maxAttrs =
                new MaxPool2dAttrs(3, 3, 2, 2, 1, 1, 1, 1, true);

        Tensor average = input.averagePool2d(averageAttrs);
        Tensor maximum = input.maxPool2d(maxAttrs);
        Tensor averageGradient = gradient(average.sum(), input);
        Tensor maximumGradient = gradient(maximum.sum(), input);
        assertEquals(input.descriptor().shape(), averageGradient.descriptor().shape());
        assertEquals(input.descriptor().shape(), maximumGradient.descriptor().shape());
        assertTrue(containsKind(averageGradient, WindowTransformKind.FOLD2D));
        assertTrue(containsKind(maximumGradient, AggregateReductionKind.ARG_MAX));
        assertTrue(reaches(maximumGradient, maximum));
        assertCompiles(average.sum(), input);
        assertCompiles(maximum.sum(), input);
    }

    @Test
    void structuredWindowFormulasRetainDynamicSpatialDimensions() {
        DynamicDimension batch = new DynamicDimension("N");
        DynamicDimension height = new DynamicDimension("H");
        DynamicDimension width = new DynamicDimension("W");
        Tensor input = tensor(Shape.ofDimensions(
                batch,
                new io.github.pho001.synaptik.model.shape.StaticDimension(4),
                height,
                width));
        Tensor weight = tensor(Shape.of(6, 2, 3, 3));
        Tensor convolution = input.conv2d(
                weight, new Conv2dAttrs(2, 2, 1, 1, 1, 1, 2));
        Tensor average = input.averagePool2d(
                new AveragePool2dAttrs(3, 3, 2, 2, 1, 1, 1, 1, false));

        assertEquals(input.descriptor().shape(),
                gradient(convolution.sum(), input).descriptor().shape());
        assertEquals(input.descriptor().shape(),
                gradient(average.sum(), input).descriptor().shape());
        assertCompiles(convolution.sum(), input);
        assertCompiles(average.sum(), input);
    }

    @Test
    void pool3dGradientsUseFixedDivisorAndExactSameOccurrenceWinnerReconstruction() {
        Tensor input = tensor(Shape.of(1, 2, 4, 4, 4));
        AveragePool3dAttrs averageAttrs = new AveragePool3dAttrs(
                2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, true);
        MaxPool3dAttrs maximumAttrs = new MaxPool3dAttrs(
                2, 3, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, true);
        Tensor average = input.averagePool3d(averageAttrs);
        Tensor maximum = input.maxPool3d(maximumAttrs);

        Tensor averageGradient = gradient(average.sum(), input);
        assertEquals(input.descriptor().shape(), averageGradient.descriptor().shape());
        assertTrue(containsKind(averageGradient, WindowTransformKind.FOLD3D));
        assertTrue(containsKind(averageGradient, AggregateReductionKind.SUM));
        assertTrue(containsKind(
                averageGradient,
                io.github.pho001.synaptik.model.operation.elementwise.binary
                        .BinaryArithmeticKind.DIV));

        Tensor maximumGradient = gradient(maximum.sum(), input);
        assertEquals(input.descriptor().shape(), maximumGradient.descriptor().shape());
        assertTrue(containsKind(maximumGradient, WindowTransformKind.FOLD3D));
        assertTrue(containsKind(maximumGradient, AggregateReductionKind.ARG_MAX));
        assertTrue(containsKind(
                maximumGradient,
                io.github.pho001.synaptik.model.operation.index.OneHotKind.ONE_HOT));
        assertTrue(reaches(maximumGradient, maximum));
        assertEquals(1, countKind(maximumGradient, Pool3dKind.MAX_POOL3D));
        assertEquals(2, countKind(maximumGradient, WindowTransformKind.UNFOLD3D));

        List<Unfold3dAttrs> unfolds = collect(maximumGradient).stream()
                .flatMap(tensor -> tensor.provenance().stream())
                .filter(provenance -> provenance.operation().kind()
                        == WindowTransformKind.UNFOLD3D)
                .map(provenance -> provenance.operation().attrs())
                .filter(Unfold3dAttrs.class::isInstance)
                .map(Unfold3dAttrs.class::cast)
                .toList();
        assertEquals(1, unfolds.size());
        assertEquals(ScalarValue.float32(Float.NEGATIVE_INFINITY),
                unfolds.getFirst().paddingValue());
        assertCompiles(average.sum(), input);
        assertCompiles(maximum.sum(), input);
    }

    @Test
    void pool3dGradientFormulasPreserveEveryFloatingTypeAndSymbolicShape() {
        for (DataType dataType : List.of(
                DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            DynamicDimension depth = new DynamicDimension("D" + dataType);
            Tensor input = tensor(
                    dataType,
                    Shape.ofDimensions(
                            new io.github.pho001.synaptik.model.shape.StaticDimension(1),
                            new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                            depth,
                            new io.github.pho001.synaptik.model.shape.StaticDimension(4),
                            new io.github.pho001.synaptik.model.shape.StaticDimension(4)));
            Tensor result = input.averagePool3d(new AveragePool3dAttrs(
                    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, true));
            Tensor resultGradient = gradient(result.sum(), input);
            assertEquals(dataType, resultGradient.descriptor().dataType());
            assertEquals(input.descriptor().shape(), resultGradient.descriptor().shape());
            Fold3dAttrs fold = collect(resultGradient).stream()
                    .flatMap(tensor -> tensor.provenance().stream())
                    .filter(provenance -> provenance.operation().kind()
                            == WindowTransformKind.FOLD3D)
                    .map(provenance -> provenance.operation().attrs())
                    .map(Fold3dAttrs.class::cast)
                    .findFirst()
                    .orElseThrow();
            assertTrue(fold.outputShape() == input.descriptor().shape());

            Tensor maximum = input.maxPool3d(new MaxPool3dAttrs(
                    2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, true));
            Tensor maximumGradient = gradient(maximum.sum(), input);
            assertEquals(dataType, maximumGradient.descriptor().dataType());
            assertEquals(input.descriptor().shape(), maximumGradient.descriptor().shape());
            assertTrue(reaches(maximumGradient, maximum));
        }
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        var plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                FunctionalGradientTestSupport.stage(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
        return FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();
    }

    private static void assertCompiles(Tensor objective, Tensor target) {
        GraphCompilation compilation = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(FunctionalGradientTestSupport.request(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), compilation.gradientResults().getFirst().target());
    }

    private static boolean containsKind(Tensor root, Enum<?> kind) {
        return traverse(root, tensor -> tensor.provenance()
                .map(provenance -> provenance.operation().kind() == kind)
                .orElse(false));
    }

    private static boolean reaches(Tensor root, Tensor expected) {
        return traverse(root, tensor -> tensor == expected);
    }

    private static long countKind(Tensor root, Enum<?> kind) {
        return collect(root).stream()
                .filter(tensor -> tensor.provenance()
                        .map(provenance -> provenance.operation().kind() == kind)
                        .orElse(false))
                .count();
    }

    private static List<Tensor> collect(Tensor root) {
        List<Tensor> result = new java.util.ArrayList<>();
        traverse(root, tensor -> {
            result.add(tensor);
            return false;
        });
        return List.copyOf(result);
    }

    private static boolean traverse(
            Tensor root, java.util.function.Predicate<Tensor> predicate) {
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (predicate.test(tensor)) {
                return true;
            }
            if (seen.put(tensor, Boolean.TRUE) == null) {
                tensor.provenance().ifPresent(
                        provenance -> provenance.inputs().forEach(pending::addLast));
            }
        }
        return false;
    }

    private static Tensor tensor(Shape shape) {
        return tensor(DataType.FLOAT32, shape);
    }

    private static Tensor tensor(DataType dataType, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), true));
    }
}
