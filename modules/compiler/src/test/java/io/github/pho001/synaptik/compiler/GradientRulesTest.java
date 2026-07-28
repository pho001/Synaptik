package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.BatchNormTrainingResult;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

final class GradientRulesTest {
    @Test
    void erfUsesTheFixedCorrectlyRoundedCoefficientForEveryFloatingType() {
        for (DataType dataType :
                List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor target = tensor(dataType, Shape.of(3));
            Tensor objective = target.erf().sum();

            Tensor gradient = gradient(objective, target);
            ScalarValue coefficient = assertInstanceOf(
                    ScalarValueAttrs.class,
                    gradient.provenance().orElseThrow().operation().attrs())
                    .value();

            switch (dataType) {
                case BFLOAT16 -> assertEquals((short) 0x3F90, coefficient.bfloat16Bits());
                case FLOAT32 -> assertEquals(
                        0x3F906EBB, Float.floatToRawIntBits(coefficient.float32Value()));
                case FLOAT64 -> assertEquals(
                        0x3FF20DD750429B6DL,
                        Double.doubleToRawLongBits(coefficient.float64Value()));
                case INT32, INT64, BOOL -> throw new AssertionError(dataType);
            }
        }
    }

    @Test
    void extremaUseOrderedSelectionNumericTieSharingAndMixedRoleNormalization() {
        Tensor narrow = tensor(DataType.BFLOAT16, Shape.of(2, 1));
        Tensor wide = tensor(DataType.FLOAT64, Shape.of(2, 3));

        for (Tensor output : List.of(narrow.minimum(wide), narrow.maximum(wide))) {
            Tensor narrowGradient = gradient(output.sum(), narrow);
            assertEquals(CastKind.CAST,
                    narrowGradient.provenance().orElseThrow().operation().kind());
            Tensor unbroadcast = narrowGradient.provenance().orElseThrow().inputs().getFirst();
            assertTrue(unbroadcast.provenance().orElseThrow().operation().attrs()
                    instanceof SumToShapeAttrs);

            Tensor routed = unbroadcast.provenance().orElseThrow().inputs().getFirst();
            var outerWhere = routed.provenance().orElseThrow();
            assertEquals(WhereSelectionKind.WHERE, outerWhere.operation().kind());
            assertInstanceOf(
                    BinaryComparisonKind.class,
                    outerWhere.inputs().get(0).provenance().orElseThrow().operation().kind());
            Tensor tieWhere = outerWhere.inputs().get(2);
            assertEquals(
                    WhereSelectionKind.WHERE,
                    tieWhere.provenance().orElseThrow().operation().kind());
            assertTrue(collectScalarValues(routed).contains(ScalarValue.float64(0.5d)));
        }
    }

    @Test
    void scalarExtremaAndClampUseExplicitCachedBoundSplatsAndOrderedComposition() {
        Tensor input = tensor(Shape.of(3));
        ScalarValue lower = ScalarValue.float32(
                Float.intBitsToFloat(0x80000000));
        ScalarValue upper = ScalarValue.float32(2.0f);

        Tensor scalarMinimum = gradient(input.minimum(lower).sum(), input);
        assertEquals(
                WhereSelectionKind.WHERE,
                scalarMinimum.provenance().orElseThrow().operation().kind());
        assertEquals(
                1,
                expansion(input.minimum(lower).sum(), input).ingress().bindings().stream()
                        .filter(binding -> binding.splat().value().equals(lower))
                        .count());

        Tensor forwardClamp = input.clamp(lower, upper);
        assertEquals(
                ScalarElementwiseKind.CLAMP,
                forwardClamp.provenance().orElseThrow().operation().kind());
        Tensor clampGradient = gradient(forwardClamp.sum(), input);
        assertEquals(
                WhereSelectionKind.WHERE,
                clampGradient.provenance().orElseThrow().operation().kind());
        assertTrue(collectScalarValues(clampGradient).containsAll(List.of(
                lower, ScalarValue.float32(0.5f))));

        FirstOrderAutograd.Expansion expansion = expansion(forwardClamp.sum(), input);
        long lowerBindings = expansion.ingress().bindings().stream()
                .filter(binding -> binding.splat().value().equals(lower))
                .count();
        long upperBindings = expansion.ingress().bindings().stream()
                .filter(binding -> binding.splat().value().equals(upper))
                .count();
        assertEquals(1, lowerBindings);
        assertEquals(1, upperBindings);
    }

    @Test
    void powUsesExactTensorAndRepresentedScalarExponentFormulas() {
        Tensor left = tensor(DataType.FLOAT32, Shape.of(2, 1));
        Tensor right = tensor(DataType.FLOAT64, Shape.of(2, 3));
        Tensor output = left.pow(right);

        Tensor leftGradient = gradient(output.sum(), left);
        assertEquals(CastKind.CAST,
                leftGradient.provenance().orElseThrow().operation().kind());
        Tensor rightGradient = gradient(output.sum(), right);
        assertTrue(containsOperation(rightGradient, UnaryElementwiseKind.LOG));
        assertTrue(containsExactTensor(rightGradient, output));

        for (ScalarValue exponent : List.of(
                ScalarValue.bfloat16Bits((short) 0x4000),
                ScalarValue.float32(2.0f),
                ScalarValue.float64(2.0d))) {
            Tensor input = tensor(exponent.dataType(), Shape.of(2));
            Tensor scalarGradient = gradient(input.pow(exponent).sum(), input);
            ScalarValue expected = switch (exponent.dataType()) {
                case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3F80);
                case FLOAT32 -> ScalarValue.float32(1.0f);
                case FLOAT64 -> ScalarValue.float64(1.0d);
                case INT32, INT64, BOOL -> throw new AssertionError();
            };
            assertTrue(collectScalarValues(scalarGradient).contains(exponent));
            assertTrue(collectScalarValues(scalarGradient).contains(expected));
        }
    }

    @Test
    void everyNewUnaryFormulaUsesTheSelectedOrdinaryTensorStructure() {
        Tensor input = tensor(Shape.of(3));
        for (Tensor output : List.of(
                input.abs(),
                input.reciprocal(),
                input.log(),
                input.log1p(),
                input.sqrt(),
                input.rsqrt(),
                input.relu(),
                input.gelu(),
                input.geluTanhApproximation(),
                input.silu())) {
            Tensor result = gradient(output.sum(), input);
            assertEquals(input.descriptor().shape(), result.descriptor().shape());
            assertEquals(input.descriptor().dataType(), result.descriptor().dataType());
        }

        assertEquals(
                WhereSelectionKind.WHERE,
                gradient(input.abs().sum(), input)
                        .provenance().orElseThrow().operation().kind());
        assertEquals(
                BinaryArithmeticKind.DIV,
                gradient(input.log().sum(), input)
                        .provenance().orElseThrow().operation().kind());
        assertEquals(
                WhereSelectionKind.WHERE,
                gradient(input.relu().sum(), input)
                        .provenance().orElseThrow().operation().kind());
        for (Tensor activation : List.of(input.gelu(), input.geluTanhApproximation(), input.silu())) {
            Tensor result = gradient(activation.sum(), input);
            assertEquals(
                    WhereSelectionKind.WHERE,
                    result.provenance().orElseThrow().operation().kind());
            assertEquals(
                    io.github.pho001.synaptik.model.operation.elementwise.classification
                            .FloatingClassificationKind.IS_INF,
                    result.provenance().orElseThrow().inputs().get(0)
                            .provenance().orElseThrow().operation().kind());
        }
    }

    @Test
    void fixedCoefficientTableIsUsedForEveryFloatingTypeWithoutHostDerivation() {
        for (DataType dataType :
                List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor input = tensor(dataType, Shape.of(2));
            Set<ScalarValue> values = new HashSet<>();
            for (Tensor output : List.of(
                    input.sqrt(), input.rsqrt(), input.gelu(), input.geluTanhApproximation())) {
                values.addAll(collectScalarValues(gradient(output.sum(), input)));
            }

            for (long[] bits : List.of(
                    new long[] {0x3F00L, 0x3F000000L, 0x3FE0000000000000L},
                    new long[] {0xBF00L, 0xBF000000L, 0xBFE0000000000000L},
                    new long[] {0x4000L, 0x40000000L, 0x4000000000000000L},
                    new long[] {0x3F35L, 0x3F3504F3L, 0x3FE6A09E667F3BCDL},
                    new long[] {0x3ECCL, 0x3ECC422AL, 0x3FD9884533D43651L},
                    new long[] {0x3F4CL, 0x3F4C422AL, 0x3FE9884533D43651L},
                    new long[] {0x3D37L, 0x3D372713L, 0x3FA6E4E26D4801F7L},
                    new long[] {0x3E09L, 0x3E095D4FL, 0x3FC12BA9D1F60179L})) {
                assertTrue(values.contains(scalarFromBits(dataType, bits)));
            }
        }
    }

    @Test
    void exceptionalAndBoundaryInputsSelectTheSameFixedStructuralPolicies() {
        for (float[] pair : List.of(
                new float[] {2.0f, 2.0f},
                new float[] {0.0f, -0.0f},
                new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY},
                new float[] {Float.NaN, 1.0f})) {
            Tensor left = TensorFactory.scalar(pair[0], Optional.empty(), true);
            Tensor right = TensorFactory.scalar(pair[1], Optional.empty(), true);
            for (Tensor output : List.of(left.minimum(right), left.maximum(right))) {
                assertEquals(
                        WhereSelectionKind.WHERE,
                        gradient(output, left).provenance().orElseThrow().operation().kind());
            }
        }

        for (ScalarValue[] bounds : List.of(
                new ScalarValue[] {
                    ScalarValue.float32(0.0f), ScalarValue.float32(1.0f)
                },
                new ScalarValue[] {
                    ScalarValue.float32(1.0f), ScalarValue.float32(1.0f)
                },
                new ScalarValue[] {
                    ScalarValue.float32(-0.0f), ScalarValue.float32(0.0f)
                },
                new ScalarValue[] {
                    ScalarValue.float32(0.0f), ScalarValue.float32(-0.0f)
                })) {
            Tensor input = TensorFactory.scalar(
                    bounds[0].float32Value(), Optional.empty(), true);
            assertEquals(
                    WhereSelectionKind.WHERE,
                    gradient(input.clamp(bounds[0], bounds[1]), input)
                            .provenance().orElseThrow().operation().kind());
        }

        Tensor zero = TensorFactory.scalar(0.0f, Optional.empty(), true);
        Tensor negative = TensorFactory.scalar(-2.0f, Optional.empty(), true);
        Tensor positiveInfinity =
                TensorFactory.scalar(Float.POSITIVE_INFINITY, Optional.empty(), true);
        Tensor negativeInfinity =
                TensorFactory.scalar(Float.NEGATIVE_INFINITY, Optional.empty(), true);
        Tensor nan = TensorFactory.scalar(Float.NaN, Optional.empty(), true);
        for (Tensor output : List.of(
                zero.abs(),
                zero.relu(),
                negative.floor(),
                negative.ceil(),
                negative.sign(),
                negative.log(),
                zero.reciprocal(),
                negative.pow(ScalarValue.float32(0.5f)),
                positiveInfinity.gelu(),
                negativeInfinity.geluTanhApproximation(),
                nan.silu())) {
            Tensor source = output.provenance().orElseThrow().inputs().getFirst();
            assertGradientCompiles(output, source);
        }
    }

    @Test
    void maskedSumRestoresTheFullShapeAndRoutesNoMaskCotangent() {
        Tensor target = tensor(Shape.of(2, 0, 3));
        Tensor mask = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, Shape.of(1, 0, 3), Optional.empty(), false));
        Tensor objective = target.sum(1, mask).sum();

        Tensor gradient = gradient(objective, target);
        var where = gradient.provenance().orElseThrow();

        assertEquals(WhereSelectionKind.WHERE, where.operation().kind());
        assertSame(mask, where.inputs().get(0));
        assertEquals(target.descriptor().shape(), gradient.descriptor().shape());
        assertEquals(target.descriptor().shape(), where.inputs().get(1).descriptor().shape());
        assertEquals(target.descriptor().shape(), where.inputs().get(2).descriptor().shape());
    }

    @Test
    void sumToShapeInvertsLeadingExactSingletonScalarAndZeroExtentCases() {
        Tensor leading = tensor(Shape.of(2, 3, 1));
        Tensor singleton = tensor(Shape.of(4, 1));
        Tensor scalar = tensor(Shape.of(2, 3));
        Tensor zero = tensor(Shape.of(0, 1));

        assertEquals(
                leading.descriptor().shape(),
                gradient(leading.sumToShape(Shape.of(1)).sum(), leading)
                        .descriptor().shape());
        assertEquals(
                singleton.descriptor().shape(),
                gradient(singleton.sumToShape(Shape.of(1, 1)).sum(), singleton)
                        .descriptor().shape());
        assertEquals(
                scalar.descriptor().shape(),
                gradient(scalar.sumToShape(Shape.scalar()), scalar)
                        .descriptor().shape());
        Tensor zeroGradient = gradient(zero.sumToShape(Shape.of(1)).sum(), zero);
        assertEquals(zero.descriptor().shape(), zeroGradient.descriptor().shape());
        assertEquals(
                ShapeTransformKind.EXPAND,
                zeroGradient.provenance().orElseThrow().operation().kind());
    }

    @Test
    void productExtremaAndAdvancedReductionRulesUseOnlyTheSpecifiedTensorAlgebra() {
        Tensor input = tensor(Shape.of(2, 3));
        for (Tensor reduced : List.of(
                input.prod(),
                input.prod(1),
                input.prod(1, true),
                input.prod(new int[] {1, 0}, false),
                input.prod(new int[] {}, false),
                input.min(),
                input.max(1, true),
                input.logSumExp(new int[] {1}, false),
                input.variance(new int[] {1}, false, 1),
                input.standardDeviation(new int[] {1}, true, 1),
                input.l1Norm(new int[] {1}, false),
                input.l2Norm(new int[] {1}, true))) {
            Tensor objective = reduced.descriptor().shape().rank() == 0
                    ? reduced
                    : reduced.sum();
            assertGradientCompiles(objective, input);
        }

        Tensor productGradient = gradient(input.prod(), input);
        Set<OperationKind> productKinds = collectKinds(productGradient);
        assertTrue(productKinds.contains(CumulativeScanKind.CUM_PROD));
        assertTrue(!productKinds.contains(BinaryArithmeticKind.DIV));

        Tensor extremaGradient = gradient(input.max(1).sum(), input);
        Set<OperationKind> extremaKinds = collectKinds(extremaGradient);
        assertTrue(extremaKinds.contains(BinaryComparisonKind.EQUAL));
        assertTrue(extremaKinds.contains(WhereSelectionKind.WHERE));
        assertTrue(extremaKinds.contains(AggregateReductionKind.SUM));

        Tensor varianceGradient =
                gradient(input.variance(new int[] {1}, false, 1).sum(), input);
        assertTrue(collectKinds(varianceGradient).contains(BinaryArithmeticKind.DIV));
        assertTrue(collectScalarValues(varianceGradient).contains(ScalarValue.float32(2.0f)));
    }

    @Test
    void cumulativeProductUsesZeroSafeFormulaForEveryTraversalMode() {
        Tensor input = tensor(Shape.of(2, 3));
        for (boolean exclusive : List.of(false, true)) {
            for (boolean reverse : List.of(false, true)) {
                Tensor gradient = gradient(
                        input.cumProd(1, exclusive, reverse).sum(), input);
                Set<OperationKind> kinds = collectKinds(gradient);
                assertTrue(kinds.contains(CumulativeScanKind.CUM_PROD));
                assertTrue(kinds.contains(CumulativeScanKind.CUM_SUM));
                assertTrue(kinds.contains(WhereSelectionKind.WHERE));
                assertTrue(kinds.contains(BinaryArithmeticKind.DIV));
            }
        }
        Tensor empty = tensor(Shape.of(2, 0));
        assertGradientCompiles(empty.cumProd(1, true, true).sum(), empty);
    }

    @Test
    void softmaxRulesConsumeTheExactSavedForwardOutput() {
        Tensor input = tensor(Shape.of(2, 3));
        for (Tensor output : List.of(input.softmax(1), input.logSoftmax(1))) {
            Tensor gradient = gradient(output.sum(), input);
            assertTrue(containsExactTensor(gradient, output));
            assertEquals(input.descriptor().shape(), gradient.descriptor().shape());
            assertGradientCompiles(output.sum(), input);
        }
    }

    @Test
    void layerRmsAndBatchNormalizationCoverEverySelectedFloatingRole() {
        Tensor input = tensor(DataType.BFLOAT16, Shape.of(2, 3, 4));
        Tensor scale32 = tensor(DataType.FLOAT32, Shape.of(3, 4));
        Tensor bias64 = tensor(DataType.FLOAT64, Shape.of(3, 4));
        Tensor layer = input.layerNorm(
                Shape.of(3, 4),
                scale32,
                bias64,
                ScalarValue.float64(1.0e-5d));
        for (Tensor target : List.of(input, scale32, bias64)) {
            assertGradientCompiles(layer.sum(), target);
        }
        assertGradientCompiles(
                input.layerNorm(
                                Shape.of(3, 4),
                                ScalarValue.bfloat16Bits((short) 0x3728))
                        .sum(),
                input);

        Tensor rmsScale = tensor(DataType.FLOAT32, Shape.of(3, 4));
        Tensor rms = input.rmsNorm(
                Shape.of(3, 4), rmsScale, ScalarValue.float32(1.0e-5f));
        for (Tensor target : List.of(input, rmsScale)) {
            assertGradientCompiles(rms.sum(), target);
        }
        assertGradientCompiles(
                input.rmsNorm(
                                Shape.of(3, 4),
                                ScalarValue.bfloat16Bits((short) 0x3728))
                        .sum(),
                input);

        Tensor channelInput = tensor(DataType.BFLOAT16, Shape.of(2, 3, 4));
        Tensor channelScale = tensor(DataType.FLOAT32, Shape.of(3));
        Tensor channelBias = tensor(DataType.FLOAT64, Shape.of(3));
        Tensor runningMean = tensor(DataType.FLOAT32, Shape.of(3));
        Tensor runningVariance = tensor(DataType.FLOAT64, Shape.of(3));
        Tensor inference = channelInput.batchNormInference(
                1,
                channelScale,
                channelBias,
                runningMean,
                runningVariance,
                ScalarValue.float64(1.0e-5d));
        for (Tensor target : List.of(
                channelInput, channelScale, channelBias, runningMean, runningVariance)) {
            assertGradientCompiles(inference.sum(), target);
        }

        BatchNormTrainingResult training = channelInput.batchNormTraining(
                1,
                channelScale,
                channelBias,
                runningMean,
                runningVariance,
                ScalarValue.float64(0.1d),
                ScalarValue.float64(1.0e-5d));
        for (Tensor target : List.of(channelInput, channelScale, channelBias)) {
            assertGradientCompiles(training.output().sum(), target);
        }
        assertGradientCompiles(training.nextRunningMean().sum(), channelInput);
        assertGradientCompiles(training.nextRunningMean().sum(), runningMean);
        assertGradientCompiles(training.nextRunningVariance().sum(), channelInput);
        assertGradientCompiles(training.nextRunningVariance().sum(), runningVariance);
        Tensor inputGradient = gradient(training.output().sum(), channelInput);
        Tensor producerOutput = training.output()
                .provenance().orElseThrow().producer().output(4);
        assertTrue(containsExactTensor(inputGradient, producerOutput));
    }

    @Test
    void matmulBuildsEveryRankCaseAndUnbroadcastsEitherBatchOperand() {
        assertMatmulGradient(Shape.of(3), Shape.of(3), 0);
        assertMatmulGradient(Shape.of(3), Shape.of(3, 4), 0);
        assertMatmulGradient(Shape.of(2, 3), Shape.of(3), 1);
        assertMatmulGradient(Shape.of(2, 3), Shape.of(3, 4), 0);
        assertMatmulGradient(Shape.of(1, 2, 3), Shape.of(5, 3, 4), 0);
        assertMatmulGradient(Shape.of(5, 2, 3), Shape.of(1, 3, 4), 1);
        assertMatmulGradient(Shape.of(2, 0), Shape.of(0, 4), 0);
    }

    @Test
    void matmulSupportsTheSelectedPromotedRoleAndRepeatedOperandMultiplicity() {
        Tensor selectedLeft = tensor(DataType.FLOAT32, Shape.of(2, 3));
        Tensor narrowerRight = tensor(DataType.BFLOAT16, Shape.of(3, 4));
        assertEquals(
                selectedLeft.descriptor().shape(),
                gradient(selectedLeft.matmul(narrowerRight).sum(), selectedLeft)
                        .descriptor().shape());

        Tensor narrowerLeft = tensor(DataType.BFLOAT16, Shape.of(2, 3));
        Tensor selectedRight = tensor(DataType.FLOAT32, Shape.of(3, 4));
        assertEquals(
                selectedRight.descriptor().shape(),
                gradient(narrowerLeft.matmul(selectedRight).sum(), selectedRight)
                        .descriptor().shape());

        Tensor repeated = tensor(Shape.of(2, 2));
        Tensor repeatedGradient = gradient(repeated.matmul(repeated).sum(), repeated);
        assertEquals(
                BinaryArithmeticKind.ADD,
                repeatedGradient.provenance().orElseThrow().operation().kind());
    }

    @Test
    void sliceAndSliceUpdatePreserveSignedEmptyAndRepeatedGeometry() {
        Tensor target = tensor(Shape.of(5));
        assertGradientCompiles(
                target.slice(
                                new long[] {1},
                                new long[] {5},
                                new int[] {0},
                                new long[] {2})
                        .sum(),
                target);
        assertGradientCompiles(
                target.slice(
                                new long[] {4},
                                new long[] {-6},
                                new int[] {0},
                                new long[] {-1})
                        .sum(),
                target);
        assertGradientCompiles(
                target.slice(
                                new long[] {0},
                                new long[] {0},
                                new int[] {0},
                                new long[] {-3})
                        .sum(),
                target);
        assertGradientCompiles(
                target.slice(new long[] {}, new long[] {}, new int[] {}, new long[] {}).sum(),
                target);

        Tensor base = tensor(Shape.of(5));
        Tensor update = tensor(Shape.of(2));
        Tensor replaced = base.sliceUpdate(
                update, new long[] {4}, new int[] {0}, new long[] {-2});
        Tensor baseGradient = gradient(replaced.sum(), base);
        Tensor updateGradient = gradient(replaced.sum(), update);
        assertEquals(SliceKind.SLICE_UPDATE,
                baseGradient.provenance().orElseThrow().operation().kind());
        assertEquals(SliceKind.SLICE,
                updateGradient.provenance().orElseThrow().operation().kind());
        SliceAttrs extraction = assertInstanceOf(
                SliceAttrs.class,
                updateGradient.provenance().orElseThrow().operation().attrs());
        assertEquals(List.of(4L), extraction.starts());
        assertEquals(List.of(2L), extraction.lengths());
        assertEquals(List.of(-2L), extraction.steps());

        Tensor repeated = tensor(Shape.of(2));
        Tensor repeatedGradient = gradient(
                repeated.sliceUpdate(
                                repeated,
                                new long[] {},
                                new int[] {},
                                new long[] {})
                        .sum(),
                repeated);
        assertEquals(
                BinaryArithmeticKind.ADD,
                repeatedGradient.provenance().orElseThrow().operation().kind());
    }

    @Test
    void selectPadTileConcatAndStackUseExactPublicInverseCompositions() {
        Tensor selected = tensor(Shape.of(2, 3));
        Tensor selectGradient = gradient(selected.select(1, 2).sum(), selected);
        assertEquals(
                SliceKind.SLICE_UPDATE,
                selectGradient.provenance().orElseThrow().operation().kind());

        Tensor scalar = tensor(Shape.scalar());
        assertGradientCompiles(scalar.tile(), scalar);
        assertGradientCompiles(
                scalar.pad(new long[] {}, new long[] {}, ScalarValue.float32(0.0f)),
                scalar);

        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor dynamicInput = tensor(Shape.ofDimensions(dynamic, new StaticDimension(0)));
        assertGradientCompiles(dynamicInput.pad(
                        new long[] {2, 0},
                        new long[] {3, 1},
                        ScalarValue.float32(0.0f))
                .sum(), dynamicInput);
        assertGradientCompiles(dynamicInput.tile(2, 3).sum(), dynamicInput);

        Tensor first = tensor(Shape.ofDimensions(dynamic, new StaticDimension(2)));
        Tensor second = tensor(Shape.ofDimensions(new DynamicDimension("M"), new StaticDimension(2)));
        Tensor concatGradient = gradient(Tensor.concat(0, first, second).sum(), second);
        CropToShapeAttrs crop = assertInstanceOf(
                CropToShapeAttrs.class,
                concatGradient.provenance().orElseThrow().operation().attrs());
        assertEquals(first.descriptor().shape().dimension(0), crop.prefixShape().dimension(0));

        Tensor stackGradient = gradient(Tensor.stack(1, first, first).sum(), first);
        assertEquals(
                BinaryArithmeticKind.ADD,
                stackGradient.provenance().orElseThrow().operation().kind());
        for (Tensor contribution : stackGradient.provenance().orElseThrow().inputs()) {
            assertEquals(Shape.ofDimensions(dynamic, new StaticDimension(2)),
                    contribution.descriptor().shape());
            assertEquals(
                    SelectKind.SELECT,
                    contribution.provenance().orElseThrow().operation().kind());
        }
    }

    @Test
    void multiplyUsesExactForwardOperandAndUnbroadcastsOnlySelectedRole() {
        Tensor target = tensor(Shape.of(2, 1));
        Tensor right = tensor(Shape.of(2, 3));
        Tensor objective = target.mul(right).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        FirstOrderAutograd.Expansion expansion =
                FirstOrderAutograd.expand(plan, CompileTimeConstantGraph.Ingress.empty());
        Tensor gradient = expansion.targetGradients().getFirst().gradient();
        var sumToShape = gradient.provenance().orElseThrow();
        Tensor multiplied = sumToShape.inputs().getFirst();

        assertTrue(sumToShape.operation().attrs() instanceof SumToShapeAttrs);
        assertSame(right, multiplied.provenance().orElseThrow().inputs().get(1));
        assertEquals(
                BinaryArithmeticKind.MUL,
                multiplied.provenance().orElseThrow().operation().kind());
    }

    @Test
    void selectedReductionAndLayoutRulesRestoreTheExactInputShape() {
        Tensor target = tensor(Shape.of(2, 3));
        Tensor objective = target.permute(1, 0).sum(0).sum();
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());

        Tensor gradient = FirstOrderAutograd.expand(
                        plan, CompileTimeConstantGraph.Ingress.empty())
                .targetGradients().getFirst().gradient();

        assertEquals(target.descriptor().shape(), gradient.descriptor().shape());
    }

    @Test
    void compilesEverySupportedElementwiseVariantForEveryFloatingType() {
        for (DataType dataType :
                List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor target = tensor(dataType, Shape.of(2, 3));
            Tensor other = tensor(dataType, Shape.of(2, 3));
            ScalarValue scalar = one(dataType);
            Tensor condition = TensorFactory.create(new TensorDescriptor(
                    DataType.BOOL, Shape.of(2, 3), Optional.empty(), false));

            assertGradientCompiles(target.add(other).sum(), target);
            assertGradientCompiles(target.sub(other).sum(), target);
            assertGradientCompiles(target.mul(other).sum(), target);
            assertGradientCompiles(target.div(other).sum(), target);
            assertGradientCompiles(target.minimum(other).sum(), target);
            assertGradientCompiles(target.maximum(other).sum(), target);
            assertGradientCompiles(target.pow(other).sum(), target);
            assertGradientCompiles(target.add(scalar).sum(), target);
            assertGradientCompiles(target.sub(scalar).sum(), target);
            assertGradientCompiles(target.mul(scalar).sum(), target);
            assertGradientCompiles(target.div(scalar).sum(), target);
            assertGradientCompiles(target.minimum(scalar).sum(), target);
            assertGradientCompiles(target.maximum(scalar).sum(), target);
            assertGradientCompiles(target.pow(scalar).sum(), target);
            assertGradientCompiles(target.clamp(zero(dataType), scalar).sum(), target);
            assertGradientCompiles(Tensor.where(condition, target, other).sum(), target);
            assertGradientCompiles(target.cast(dataType).sum(), target);
            assertGradientCompiles(target.neg().sum(), target);
            assertGradientCompiles(target.abs().sum(), target);
            assertGradientCompiles(target.reciprocal().sum(), target);
            assertGradientCompiles(target.log().sum(), target);
            assertGradientCompiles(target.log1p().sum(), target);
            assertGradientCompiles(target.exp().sum(), target);
            assertGradientCompiles(target.expm1().sum(), target);
            assertGradientCompiles(target.sigmoid().sum(), target);
            assertGradientCompiles(target.tanh().sum(), target);
            assertGradientCompiles(target.floor().sum(), target);
            assertGradientCompiles(target.ceil().sum(), target);
            assertGradientCompiles(target.sign().sum(), target);
            assertGradientCompiles(target.sqrt().sum(), target);
            assertGradientCompiles(target.rsqrt().sum(), target);
            assertGradientCompiles(target.relu().sum(), target);
            assertGradientCompiles(target.gelu().sum(), target);
            assertGradientCompiles(target.geluTanhApproximation().sum(), target);
            assertGradientCompiles(target.silu().sum(), target);
            assertGradientCompiles(target.sum(), target);
            assertGradientCompiles(target.sum(1).sum(), target);
            assertGradientCompiles(target.sum(1, true).sum(), target);
            assertGradientCompiles(target.sum(new int[] {1, 0}, false).sum(), target);
            assertGradientCompiles(target.sum(new int[] {}, false).sum(), target);
            assertGradientCompiles(target.mean(), target);
            assertGradientCompiles(target.mean(1).sum(), target);
            assertGradientCompiles(target.mean(1, true).sum(), target);
            assertGradientCompiles(target.mean(new int[] {1, 0}, false), target);
            assertGradientCompiles(target.mean(new int[] {}, false).sum(), target);
            assertGradientCompiles(target.cumSum(1, true, false).sum(), target);
            assertGradientCompiles(target.contiguous().sum(), target);
            assertGradientCompiles(target.reshape(3, 2).sum(), target);
            assertGradientCompiles(target.expandDims(1).sum(), target);
            assertGradientCompiles(target.permute(1, 0).sum(), target);

            Tensor expandable = tensor(dataType, Shape.of(2, 1));
            assertGradientCompiles(expandable.expand(2, 3).sum(), expandable);
            Tensor squeezable = tensor(dataType, Shape.of(2, 1, 3));
            assertGradientCompiles(squeezable.squeeze(1).sum(), squeezable);
        }
    }

    @Test
    void mixedFloatingContributionsUnbroadcastBeforeOneReverseCast() {
        for (List<DataType> pair : List.of(
                List.of(DataType.BFLOAT16, DataType.FLOAT32),
                List.of(DataType.BFLOAT16, DataType.FLOAT64),
                List.of(DataType.FLOAT32, DataType.FLOAT64))) {
            Tensor narrow = tensor(pair.get(0), Shape.of(2, 1));
            Tensor wide = tensor(pair.get(1), Shape.of(2, 3));
            for (Tensor output : List.of(
                    narrow.add(wide), narrow.sub(wide), narrow.mul(wide), narrow.div(wide))) {
                Tensor contribution = gradient(output.sum(), narrow);
                assertEquals(CastKind.CAST,
                        contribution.provenance().orElseThrow().operation().kind());
                Tensor unbroadcast = contribution.provenance().orElseThrow().inputs().getFirst();
                assertTrue(unbroadcast.provenance().orElseThrow().operation().attrs()
                        instanceof SumToShapeAttrs);
                assertEquals(narrow.descriptor().shape(), contribution.descriptor().shape());
                assertEquals(narrow.descriptor().dataType(), contribution.descriptor().dataType());
            }
        }
    }

    @Test
    void divisionDirectZeroAndMeanUseOnlyOrdinaryTensorExpressions() {
        Tensor target = tensor(Shape.of(2, 3));
        Tensor other = tensor(Shape.of(2, 3));
        Tensor rightGradient = gradient(target.div(other).sum(), other);
        assertEquals(BinaryArithmeticKind.DIV,
                rightGradient.provenance().orElseThrow().operation().kind());

        for (Tensor output : List.of(target.floor(), target.ceil(), target.sign())) {
            Tensor zero = gradient(output.sum(), target);
            assertEquals(ShapeTransformKind.EXPAND,
                    zero.provenance().orElseThrow().operation().kind());
            assertTrue(zero.provenance().orElseThrow().inputs().getFirst()
                    .provenance().isEmpty());
        }

        for (Shape shape : List.of(
                Shape.of(0, 3),
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(3)),
                Shape.ofDimensions(
                        DimensionExpressions.addConstant(new DynamicDimension("M"), 1),
                        new StaticDimension(3)))) {
            Tensor input = tensor(shape);
            assertGradientCompiles(input.mean(), input);
            assertGradientCompiles(input.mean(0).sum(), input);
            assertGradientCompiles(input.mean(new int[] {1, 0}, false), input);
            assertGradientCompiles(input.mean(new int[] {}, false).sum(), input);
        }

        Tensor mask = TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, Shape.of(1, 3), Optional.empty(), false));
        Tensor masked = gradient(target.mean(0, mask).sum(), target);
        assertEquals(WhereSelectionKind.WHERE,
                masked.provenance().orElseThrow().operation().kind());
        Tensor quotient = masked.provenance().orElseThrow().inputs().get(1);
        assertEquals(BinaryArithmeticKind.DIV,
                quotient.provenance().orElseThrow().operation().kind());
        Tensor count = quotient.provenance().orElseThrow().inputs().get(1);
        assertEquals(AggregateReductionKind.SUM,
                count.provenance().orElseThrow().inputs().getFirst()
                        .provenance().orElseThrow().operation().kind());
    }

    private static void assertGradientCompiles(Tensor objective, Tensor target) {
        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
        assertEquals(target.id(), result.gradientResults().getFirst().target());
    }

    private static Tensor gradient(Tensor objective, Tensor target) {
        return expansion(objective, target).targetGradients().getFirst().gradient();
    }

    private static FirstOrderAutograd.Expansion expansion(Tensor objective, Tensor target) {
        AutogradPreflight.Plan plan = AutogradPreflight.preflight(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                new AutogradPreflight.FirstOrderRequest(objective, List.of(target)),
                CompileTimeConstantGraph.Ingress.empty());
        return FirstOrderAutograd.expand(
                plan, CompileTimeConstantGraph.Ingress.empty());
    }

    private static void assertMatmulGradient(
            Shape leftShape, Shape rightShape, int selectedInput) {
        Tensor left = tensor(leftShape);
        Tensor right = tensor(rightShape);
        Tensor result = left.matmul(right);
        Tensor objective = result.descriptor().shape().rank() == 0 ? result : result.sum();
        Tensor selected = selectedInput == 0 ? left : right;
        assertEquals(
                selected.descriptor().shape(),
                gradient(objective, selected).descriptor().shape());
    }

    private static ScalarValue one(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3F80);
            case FLOAT32 -> ScalarValue.float32(1.0f);
            case FLOAT64 -> ScalarValue.float64(1.0d);
            case INT32, INT64, BOOL -> throw new AssertionError(dataType);
        };
    }

    private static ScalarValue zero(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x0000);
            case FLOAT32 -> ScalarValue.float32(0.0f);
            case FLOAT64 -> ScalarValue.float64(0.0d);
            case INT32, INT64, BOOL -> throw new AssertionError(dataType);
        };
    }

    private static Set<ScalarValue> collectScalarValues(Tensor root) {
        Set<ScalarValue> values = new HashSet<>();
        Set<Tensor> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (!seen.add(tensor)) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> {
                if (provenance.operation().attrs() instanceof ScalarValueAttrs attrs) {
                    values.add(attrs.value());
                }
                pending.addAll(provenance.inputs());
            });
        }
        return values;
    }

    private static Set<OperationKind> collectKinds(Tensor root) {
        Set<OperationKind> kinds = new HashSet<>();
        Set<Tensor> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (!seen.add(tensor)) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> {
                kinds.add(provenance.operation().kind());
                pending.addAll(provenance.inputs());
            });
        }
        return kinds;
    }

    private static boolean containsOperation(Tensor root, Object kind) {
        Set<Tensor> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (!seen.add(tensor)) {
                continue;
            }
            var provenance = tensor.provenance().orElse(null);
            if (provenance == null) {
                continue;
            }
            if (provenance.operation().kind() == kind) {
                return true;
            }
            pending.addAll(provenance.inputs());
        }
        return false;
    }

    private static boolean containsExactTensor(Tensor root, Tensor expected) {
        Set<Tensor> seen = java.util.Collections.newSetFromMap(new java.util.IdentityHashMap<>());
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (tensor == expected) {
                return true;
            }
            if (!seen.add(tensor)) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> pending.addAll(provenance.inputs()));
        }
        return false;
    }

    private static ScalarValue scalarFromBits(DataType dataType, long[] bits) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) bits[0]);
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat((int) bits[1]));
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(bits[2]));
            case INT32, INT64, BOOL -> throw new AssertionError(dataType);
        };
    }

    private static Tensor tensor(Shape shape) {
        return tensor(DataType.FLOAT32, shape);
    }

    private static Tensor tensor(DataType dataType, Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), true));
    }
}
