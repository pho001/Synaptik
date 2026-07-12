package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorLinearExpressionTest {
    private static final AtomicLong IDS = new AtomicLong(190_000);

    @Test
    void helperAndPublicMethodsHaveExactlyTheRequiredSurface()
            throws ReflectiveOperationException {
        int modifiers = TensorLinearExpressions.class.getModifiers();
        var constructors = TensorLinearExpressions.class.getDeclaredConstructors();
        List<Method> methods = Arrays.asList(TensorLinearExpressions.class.getDeclaredMethods());
        Set<List<Class<?>>> helperParameters = methods.stream()
                .map(method -> List.of(method.getParameterTypes()))
                .collect(Collectors.toSet());
        Method withoutBias = Tensor.class.getDeclaredMethod("linear", Tensor.class);
        Method withBias = Tensor.class.getDeclaredMethod(
                "linear", Tensor.class, Tensor.class);
        long publicTensorMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count();
        long publicLinearMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals("linear"))
                .count();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertFalse(TensorLinearExpressions.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(TensorLinearExpressions.class.getInterfaces())),
                () -> assertEquals(0, TensorLinearExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0, TensorLinearExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(0, constructors[0].getParameterCount()),
                () -> assertEquals(2, methods.size()),
                () -> assertTrue(methods.stream().allMatch(
                        method -> method.getName().equals("apply"))),
                () -> assertEquals(
                        Set.of(
                                List.of(Tensor.class, Tensor.class),
                                List.of(Tensor.class, Tensor.class, Tensor.class)),
                        helperParameters),
                () -> assertTrue(methods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(methods.stream().noneMatch(
                        method -> Modifier.isPublic(method.getModifiers())
                                || Modifier.isProtected(method.getModifiers())
                                || Modifier.isPrivate(method.getModifiers()))),
                () -> assertTrue(methods.stream().allMatch(
                        method -> method.getReturnType() == Tensor.class)),
                () -> assertFalse(Modifier.isStatic(withoutBias.getModifiers())),
                () -> assertFalse(Modifier.isStatic(withBias.getModifiers())),
                () -> assertEquals(177, publicTensorMethods),
                () -> assertEquals(2, publicLinearMethods));
    }

    @Test
    void derivesRankOneMatrixAndHigherRankShapesWithExactDimensionReferences() {
        Dimension batch = new DynamicDimension("B");
        Dimension time = DimensionExpressions.addConstant(new DynamicDimension("T"), 1);
        Dimension inputFeatures = new DynamicDimension("K");
        Dimension outputFeatures = DimensionExpressions.multiply(
                new DynamicDimension("N"), 2);
        Tensor weight = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(outputFeatures, inputFeatures),
                Optional.empty(),
                false);

        Tensor vector = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(inputFeatures),
                Optional.empty(),
                false).linear(weight);
        Tensor matrix = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, inputFeatures),
                Optional.empty(),
                false).linear(weight);
        Tensor higherRank = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, time, inputFeatures),
                Optional.empty(),
                false).linear(weight);
        Tensor empty = tensor(
                DataType.FLOAT32, Shape.of(0, 3), Optional.empty(), false)
                .linear(tensor(DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), false));
        Tensor zeroOutput = tensor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false).linear(
                        tensor(DataType.FLOAT32, Shape.of(0, 3), Optional.empty(), false),
                        tensor(DataType.FLOAT32, Shape.of(0), Optional.empty(), false));
        Tensor singletonOutput = tensor(
                DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false).linear(
                        tensor(DataType.FLOAT32, Shape.of(1, 3), Optional.empty(), false),
                        tensor(DataType.FLOAT32, Shape.of(1), Optional.empty(), false));

        assertAll(
                () -> assertEquals(Shape.ofDimensions(outputFeatures),
                        vector.descriptor().shape()),
                () -> assertSame(outputFeatures, vector.descriptor().shape().dimension(0)),
                () -> assertSame(batch, matrix.descriptor().shape().dimension(0)),
                () -> assertSame(outputFeatures, matrix.descriptor().shape().dimension(1)),
                () -> assertSame(batch, higherRank.descriptor().shape().dimension(0)),
                () -> assertSame(time, higherRank.descriptor().shape().dimension(1)),
                () -> assertSame(outputFeatures, higherRank.descriptor().shape().dimension(2)),
                () -> assertEquals(Shape.of(0, 4), empty.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 0), zeroOutput.descriptor().shape()),
                () -> assertEquals(Shape.of(2, 1), singletonOutput.descriptor().shape()));
    }

    @Test
    void acceptsEveryFloatingAndIntegralPromotionAndBiasMayWidenOnlyTheAddResult() {
        DataType[] floating = {DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64};
        for (DataType inputType : floating) {
            for (DataType weightType : floating) {
                DataType expected = widestFloating(inputType, weightType);
                Tensor input = tensor(inputType, Shape.of(2, 3), Optional.empty(), false);
                Tensor weight = tensor(weightType, Shape.of(4, 3), Optional.empty(), false);
                assertSame(expected, input.linear(weight).descriptor().dataType());
                for (DataType biasType : floating) {
                    Tensor biased = input.linear(
                            weight,
                            tensor(biasType, Shape.of(4), Optional.empty(), false));
                    Tensor product = biased.provenance().orElseThrow().inputs().getFirst();
                    assertSame(expected, product.descriptor().dataType());
                    assertSame(widestFloating(expected, biasType),
                            biased.descriptor().dataType());
                }
            }
        }
        DataType[] integral = {DataType.INT32, DataType.INT64};
        for (DataType inputType : integral) {
            for (DataType weightType : integral) {
                DataType expected = inputType == DataType.INT64 || weightType == DataType.INT64
                        ? DataType.INT64 : DataType.INT32;
                Tensor input = tensor(inputType, Shape.of(2, 3), Optional.empty(), false);
                Tensor weight = tensor(weightType, Shape.of(4, 3), Optional.empty(), false);
                assertSame(expected, input.linear(weight).descriptor().dataType());
                for (DataType biasType : integral) {
                    Tensor biased = input.linear(
                            weight,
                            tensor(biasType, Shape.of(4), Optional.empty(), false));
                    Tensor product = biased.provenance().orElseThrow().inputs().getFirst();
                    DataType finalType = expected == DataType.INT64 || biasType == DataType.INT64
                            ? DataType.INT64 : DataType.INT32;
                    assertSame(expected, product.descriptor().dataType());
                    assertSame(finalType, biased.descriptor().dataType());
                }
            }
        }

        Tensor widened = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), true)
                .linear(
                        tensor(DataType.BFLOAT16, Shape.of(4, 3), Optional.empty(), false),
                        tensor(DataType.FLOAT64, Shape.of(4), Optional.empty(), true));
        Tensor product = widened.provenance().orElseThrow().inputs().getFirst();
        Tensor integralWidened = tensor(DataType.INT32, Shape.of(2, 3), Optional.empty(), false)
                .linear(
                        tensor(DataType.INT32, Shape.of(4, 3), Optional.empty(), false),
                        tensor(DataType.INT64, Shape.of(4), Optional.empty(), false));

        assertAll(
                () -> assertSame(DataType.FLOAT32, product.descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64, widened.descriptor().dataType()),
                () -> assertTrue(widened.descriptor().requiresGrad()),
                () -> assertSame(DataType.INT64, integralWidened.descriptor().dataType()),
                () -> assertFalse(integralWidened.descriptor().requiresGrad()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.BOOL, Shape.of(2, 3), Optional.empty(), false)
                                .linear(tensor(
                                        DataType.BOOL, Shape.of(4, 3), Optional.empty(), false))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false)
                                .linear(tensor(
                                        DataType.INT32, Shape.of(4, 3), Optional.empty(), false))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false)
                                .linear(
                                        tensor(DataType.FLOAT32, Shape.of(4, 3),
                                                Optional.empty(), false),
                                        tensor(DataType.INT32, Shape.of(4),
                                                Optional.empty(), false))));
    }

    @Test
    void appliesDeferredContractionAndStrictStructuralBiasDimensionPolicies() {
        Dimension inputFeatures = new DynamicDimension("inputK");
        Dimension weightInputFeatures = DimensionExpressions.unknown(0, Optional.empty());
        Dimension namedOutput = new DynamicDimension("N");
        Dimension equalNamedOutput = new DynamicDimension("N");
        Tensor deferred = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"), inputFeatures),
                Optional.empty(),
                false).linear(tensor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(namedOutput, weightInputFeatures),
                        Optional.empty(),
                        false));
        Tensor namedBias = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(inputFeatures),
                Optional.empty(),
                false).linear(
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(namedOutput, inputFeatures),
                                Optional.empty(), false),
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(equalNamedOutput),
                                Optional.empty(), false));

        Dimension expressionOutput = DimensionExpressions.addConstant(
                new DynamicDimension("M"), 2);
        Dimension equalExpressionOutput = DimensionExpressions.addConstant(
                new DynamicDimension("M"), 2);
        Tensor expressionBias = tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), false)
                .linear(
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(expressionOutput, new StaticDimension(3)),
                                Optional.empty(), false),
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(equalExpressionOutput),
                                Optional.empty(), false));
        Dimension unknownOutput = DimensionExpressions.unknown(0, Optional.empty());
        Tensor identicalUnknownBias = tensor(
                DataType.FLOAT32, Shape.of(3), Optional.empty(), false).linear(
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(unknownOutput, new StaticDimension(3)),
                                Optional.empty(), false),
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(unknownOutput),
                                Optional.empty(), false));

        assertAll(
                () -> assertSame(namedOutput, deferred.descriptor().shape().dimension(1)),
                () -> assertEquals(Shape.ofDimensions(namedOutput),
                        namedBias.descriptor().shape()),
                () -> assertSame(namedOutput, namedBias.descriptor().shape().dimension(0)),
                () -> assertSame(expressionOutput,
                        expressionBias.descriptor().shape().dimension(0)),
                () -> assertSame(unknownOutput,
                        identicalUnknownBias.descriptor().shape().dimension(0)));

        Dimension otherUnknown = DimensionExpressions.unknown(0, Optional.empty());
        assertAll(
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), false)
                                .linear(
                                        tensor(DataType.FLOAT32,
                                                Shape.ofDimensions(
                                                        unknownOutput,
                                                        new StaticDimension(3)),
                                                Optional.empty(), false),
                                        tensor(DataType.FLOAT32,
                                                Shape.ofDimensions(otherUnknown),
                                                Optional.empty(), false))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), false)
                                .linear(
                                        tensor(DataType.FLOAT32, Shape.of(4, 3),
                                                Optional.empty(), false),
                                        tensor(DataType.FLOAT32, Shape.of(1),
                                                Optional.empty(), false))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> tensor(DataType.FLOAT32, Shape.of(3), Optional.empty(), false)
                                .linear(
                                        tensor(DataType.FLOAT32,
                                                Shape.ofDimensions(
                                                        unknownOutput,
                                                        new StaticDimension(3)),
                                                Optional.empty(), false),
                                        tensor(DataType.FLOAT32, Shape.of(4),
                                                Optional.empty(), false))));
    }

    @Test
    void validatesInExactOrderWithExactMessagesAndConsumesNoIdOnEveryLocalFailure()
            throws Exception {
        AtomicLong next = nextTensorIdState();
        Tensor scalarFloat = tensor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
        Tensor vectorFloat = tensor(
                DataType.FLOAT32, Shape.of(3), Optional.empty(), false);
        Tensor matrixFloat = tensor(
                DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), false);
        Tensor rankThreeFloat = tensor(
                DataType.FLOAT32, Shape.of(1, 4, 3), Optional.empty(), false);
        Tensor scalarBool = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        Tensor matrixBool = tensor(DataType.BOOL, Shape.of(4, 3), Optional.empty(), false);
        Tensor scalarBiasBool = tensor(DataType.BOOL, Shape.scalar(), Optional.empty(), false);
        long before = next.get();

        NullPointerException inputNull = assertThrows(NullPointerException.class,
                () -> TensorLinearExpressions.apply(null, null));
        NullPointerException weightNull = assertThrows(NullPointerException.class,
                () -> TensorLinearExpressions.apply(vectorFloat, null));
        NullPointerException biasedInputNull = assertThrows(NullPointerException.class,
                () -> TensorLinearExpressions.apply(null, null, null));
        NullPointerException biasedWeightNull = assertThrows(NullPointerException.class,
                () -> TensorLinearExpressions.apply(vectorFloat, null, null));
        NullPointerException biasNull = assertThrows(NullPointerException.class,
                () -> TensorLinearExpressions.apply(vectorFloat, matrixFloat, null));
        IllegalArgumentException promotionBeforeRank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(scalarBool, matrixBool));
        IllegalArgumentException inputRank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(scalarFloat, rankThreeFloat));
        IllegalArgumentException weightRank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(vectorFloat, rankThreeFloat));
        IllegalArgumentException contraction = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(
                        tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false),
                        matrixFloat));
        IllegalArgumentException contractionBeforeBiasPromotion = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(
                        tensor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false),
                        matrixFloat,
                        scalarBiasBool));
        IllegalArgumentException biasPromotionBeforeRank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(
                        vectorFloat,
                        matrixFloat,
                        scalarBiasBool));
        IllegalArgumentException biasRank = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(
                        vectorFloat,
                        matrixFloat,
                        tensor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false)));
        IllegalArgumentException biasDimension = assertThrows(
                IllegalArgumentException.class,
                () -> TensorLinearExpressions.apply(
                        vectorFloat,
                        matrixFloat,
                        tensor(DataType.FLOAT32, Shape.of(5), Optional.empty(), false)));

        assertAll(
                () -> assertEquals("input", inputNull.getMessage()),
                () -> assertEquals("weight", weightNull.getMessage()),
                () -> assertEquals("input", biasedInputNull.getMessage()),
                () -> assertEquals("weight", biasedWeightNull.getMessage()),
                () -> assertEquals("bias", biasNull.getMessage()),
                () -> assertEquals("left must be a numeric data type, but was BOOL",
                        promotionBeforeRank.getMessage()),
                () -> assertEquals("input rank must be at least 1: 0", inputRank.getMessage()),
                () -> assertEquals("weight rank must be exactly 2: 3", weightRank.getMessage()),
                () -> assertEquals(
                        "linear input feature dimension must match weight in-features dimension: input=StaticDimension[size=2], weight=StaticDimension[size=3]",
                        contraction.getMessage()),
                () -> assertEquals(contraction.getMessage(),
                        contractionBeforeBiasPromotion.getMessage()),
                () -> assertEquals("right must be a numeric data type, but was BOOL",
                        biasPromotionBeforeRank.getMessage()),
                () -> assertEquals("bias rank must be exactly 1: 0", biasRank.getMessage()),
                () -> assertEquals(
                        "linear bias dimension must match weight out-features dimension: bias=StaticDimension[size=5], weight=StaticDimension[size=4]",
                        biasDimension.getMessage()),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void noBiasCreatesExactPermuteThenMatmulChainForResolvedAndUnresolvedWeightLayouts()
            throws Exception {
        Shape inputShape = Shape.of(2, 3);
        Shape weightShape = Shape.of(4, 3);
        Tensor input = tensor(DataType.FLOAT32, inputShape, Optional.empty(), true);
        LayoutDescriptor weightLayout = LayoutDescriptor.of(
                weightShape, new long[] {3, 1}, 5, false);
        Tensor resolvedWeight = tensor(
                DataType.FLOAT64, weightShape, Optional.of(weightLayout), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor result = input.linear(resolvedWeight);
        TensorProvenance matmul = result.provenance().orElseThrow();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProvenance permute = transposedWeight.provenance().orElseThrow();
        LayoutDescriptor transposedLayout = transposedWeight.descriptor().layout().orElseThrow();

        assertAll(
                () -> assertEquals(before + 2, next.get()),
                () -> assertEquals(before, transposedWeight.id().value()),
                () -> assertEquals(before + 1, result.id().value()),
                () -> assertSame(MatmulKind.MATMUL, matmul.operation().kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, matmul.operation().attrs()),
                () -> assertEquals(0, matmul.outputIndex()),
                () -> assertEquals(1, matmul.producer().outputCount()),
                () -> assertSame(result.descriptor(), matmul.outputDescriptor()),
                () -> assertEquals(2, matmul.inputs().size()),
                () -> assertSame(input, matmul.inputs().get(0)),
                () -> assertSame(transposedWeight, matmul.inputs().get(1)),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.operation().kind()),
                () -> assertEquals(List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertEquals(0, permute.outputIndex()),
                () -> assertEquals(1, permute.producer().outputCount()),
                () -> assertEquals(List.of(resolvedWeight), permute.inputs()),
                () -> assertSame(resolvedWeight, permute.inputs().getFirst()),
                () -> assertSame(DataType.FLOAT64, transposedWeight.descriptor().dataType()),
                () -> assertTrue(transposedWeight.descriptor().requiresGrad()),
                () -> assertTrue(transposedWeight.label().isEmpty()),
                () -> assertTrue(transposedWeight.hostStorage().isEmpty()),
                () -> assertSame(weightShape.dimension(1),
                        transposedWeight.descriptor().shape().dimension(0)),
                () -> assertSame(weightShape.dimension(0),
                        transposedWeight.descriptor().shape().dimension(1)),
                () -> assertArrayEquals(new long[] {1, 3}, transposedLayout.strides()),
                () -> assertEquals(5, transposedLayout.storageOffset()),
                () -> assertTrue(transposedLayout.isView()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 4), result.descriptor().shape()),
                () -> assertSame(inputShape.dimension(0),
                        result.descriptor().shape().dimension(0)),
                () -> assertSame(weightShape.dimension(0),
                        result.descriptor().shape().dimension(1)),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()),
                () -> assertSame(inputShape, input.descriptor().shape()),
                () -> assertSame(weightShape, resolvedWeight.descriptor().shape()),
                () -> assertSame(weightLayout,
                        resolvedWeight.descriptor().layout().orElseThrow()),
                () -> assertTrue(input.provenance().isEmpty()),
                () -> assertTrue(resolvedWeight.provenance().isEmpty()));

        Tensor unresolvedWeight = tensor(
                DataType.FLOAT32, weightShape, Optional.empty(), false);
        Tensor unresolved = input.linear(unresolvedWeight);
        Tensor unresolvedTranspose = unresolved.provenance().orElseThrow().inputs().get(1);
        assertTrue(unresolvedTranspose.descriptor().layout().isEmpty());
    }

    @Test
    void biasCreatesExactPermuteMatmulAddChainAndPreservesOrderedDimensionReferences()
            throws Exception {
        Dimension batch = new DynamicDimension("B");
        Dimension inputFeatures = new DynamicDimension("K");
        Dimension outputFeatures = new DynamicDimension("N");
        Tensor input = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(batch, inputFeatures),
                Optional.empty(),
                false);
        Tensor weight = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(outputFeatures, inputFeatures),
                Optional.empty(),
                true);
        Tensor bias = tensor(
                DataType.FLOAT64,
                Shape.ofDimensions(new DynamicDimension("N")),
                Optional.empty(),
                true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        Tensor result = input.linear(weight, bias);
        TensorProvenance add = result.provenance().orElseThrow();
        Tensor product = add.inputs().getFirst();
        TensorProvenance matmul = product.provenance().orElseThrow();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProvenance permute = transposedWeight.provenance().orElseThrow();
        Shape productShape = product.descriptor().shape();
        Shape resultShape = result.descriptor().shape();

        assertAll(
                () -> assertEquals(before + 3, next.get()),
                () -> assertEquals(before, transposedWeight.id().value()),
                () -> assertEquals(before + 1, product.id().value()),
                () -> assertEquals(before + 2, result.id().value()),
                () -> assertSame(BinaryArithmeticKind.ADD, add.operation().kind()),
                () -> assertSame(NoOperationAttrs.INSTANCE, add.operation().attrs()),
                () -> assertEquals(0, add.outputIndex()),
                () -> assertEquals(1, add.producer().outputCount()),
                () -> assertSame(result.descriptor(), add.outputDescriptor()),
                () -> assertEquals(2, add.inputs().size()),
                () -> assertSame(product, add.inputs().get(0)),
                () -> assertSame(bias, add.inputs().get(1)),
                () -> assertSame(MatmulKind.MATMUL, matmul.operation().kind()),
                () -> assertEquals(List.of(input, transposedWeight), matmul.inputs()),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.operation().kind()),
                () -> assertEquals(List.of(weight), permute.inputs()),
                () -> assertEquals(List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertEquals(0, matmul.outputIndex()),
                () -> assertEquals(0, permute.outputIndex()),
                () -> assertSame(DataType.FLOAT32, product.descriptor().dataType()),
                () -> assertTrue(product.descriptor().requiresGrad()),
                () -> assertTrue(product.descriptor().layout().isEmpty()),
                () -> assertTrue(product.label().isEmpty()),
                () -> assertTrue(product.hostStorage().isEmpty()),
                () -> assertSame(DataType.FLOAT64, result.descriptor().dataType()),
                () -> assertEquals(productShape, resultShape),
                () -> assertNotSame(productShape, resultShape),
                () -> assertSame(productShape.dimension(0), resultShape.dimension(0)),
                () -> assertSame(productShape.dimension(1), resultShape.dimension(1)),
                () -> assertSame(batch, resultShape.dimension(0)),
                () -> assertSame(outputFeatures, resultShape.dimension(1)),
                () -> assertTrue(result.descriptor().layout().isEmpty()),
                () -> assertTrue(result.descriptor().requiresGrad()),
                () -> assertTrue(result.label().isEmpty()),
                () -> assertTrue(result.hostStorage().isEmpty()));
    }

    @Test
    void invalidBiasCreatesNoPartialChainAndRepeatedValidCallsAreFresh() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), false);
        Tensor invalidBias = tensor(DataType.FLOAT32, Shape.of(5), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        long beforeFailure = next.get();

        assertThrows(IllegalArgumentException.class,
                () -> input.linear(weight, invalidBias));
        assertEquals(beforeFailure, next.get());
        assertTrue(input.provenance().isEmpty());
        assertTrue(weight.provenance().isEmpty());
        assertTrue(invalidBias.provenance().isEmpty());

        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), Optional.empty(), false);
        Tensor first = input.linear(weight, bias);
        Tensor second = input.linear(weight, bias);
        Tensor firstProduct = first.provenance().orElseThrow().inputs().getFirst();
        Tensor secondProduct = second.provenance().orElseThrow().inputs().getFirst();
        Tensor firstTranspose = firstProduct.provenance().orElseThrow().inputs().get(1);
        Tensor secondTranspose = secondProduct.provenance().orElseThrow().inputs().get(1);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertNotSame(firstProduct, secondProduct),
                () -> assertNotSame(firstProduct.provenance().orElseThrow().producer(),
                        secondProduct.provenance().orElseThrow().producer()),
                () -> assertNotSame(firstTranspose, secondTranspose),
                () -> assertNotSame(firstTranspose.provenance().orElseThrow().producer(),
                        secondTranspose.provenance().orElseThrow().producer()));
    }

    @Test
    void identifierExhaustionDoesNotRollBackSuccessfulIntermediates() throws Exception {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 3), Optional.empty(), false);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), Optional.empty(), false);
        AtomicLong next = nextTensorIdState();
        AtomicBoolean claimed = maximumClaimedState();
        long originalNext = next.get();
        boolean originalClaimed = claimed.get();
        try {
            next.set(Long.MAX_VALUE - 1);
            claimed.set(false);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> input.linear(weight, bias));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(claimed.get()));
        } finally {
            next.set(originalNext);
            claimed.set(originalClaimed);
        }
    }

    private static Tensor tensor(
            DataType dataType,
            Shape shape,
            Optional<LayoutDescriptor> layout,
            boolean requiresGrad) {
        return new Tensor(
                new TensorId(IDS.getAndIncrement()),
                new TensorDescriptor(dataType, shape, layout, requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static DataType widestFloating(DataType left, DataType right) {
        if (left == DataType.FLOAT64 || right == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (left == DataType.FLOAT32 || right == DataType.FLOAT32) {
            return DataType.FLOAT32;
        }
        return DataType.BFLOAT16;
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumClaimedState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
