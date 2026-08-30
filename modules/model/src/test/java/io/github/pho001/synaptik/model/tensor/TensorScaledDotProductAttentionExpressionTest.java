package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class TensorScaledDotProductAttentionExpressionTest {
    @Test
    void helperAndTensorExposeOnlyTheExactRequiredAttentionSurface()
            throws ReflectiveOperationException {
        int modifiers = TensorScaledDotProductAttentionExpressions.class.getModifiers();
        var constructors = TensorScaledDotProductAttentionExpressions.class.getDeclaredConstructors();
        List<Method> applyMethods = Arrays.stream(
                        TensorScaledDotProductAttentionExpressions.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("apply"))
                .toList();
        List<Method> applyWithWeightsMethods = Arrays.stream(
                        TensorScaledDotProductAttentionExpressions.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("applyWithWeights"))
                .toList();
        List<Method> publicMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals("scaledDotProductAttention"))
                .toList();
        List<Method> publicWithWeightsMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .filter(method -> method.getName().equals(
                        "scaledDotProductAttentionWithWeights"))
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isFinal(modifiers)),
                () -> assertFalse(Modifier.isPublic(modifiers)),
                () -> assertFalse(Modifier.isProtected(modifiers)),
                () -> assertFalse(TensorScaledDotProductAttentionExpressions.class.isRecord()),
                () -> assertEquals(Set.of(),
                        Set.of(TensorScaledDotProductAttentionExpressions.class.getInterfaces())),
                () -> assertEquals(0,
                        TensorScaledDotProductAttentionExpressions.class.getDeclaredFields().length),
                () -> assertEquals(0,
                        TensorScaledDotProductAttentionExpressions.class.getDeclaredClasses().length),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(2, applyMethods.size()),
                () -> assertTrue(applyMethods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(applyMethods.stream().noneMatch(
                        method -> Modifier.isPublic(method.getModifiers()))),
                () -> assertEquals(2, applyWithWeightsMethods.size()),
                () -> assertTrue(applyWithWeightsMethods.stream().allMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertTrue(applyWithWeightsMethods.stream().noneMatch(
                        method -> Modifier.isPublic(method.getModifiers()))),
                () -> assertEquals(4, publicMethods.size()),
                () -> assertTrue(publicMethods.stream().allMatch(
                        method -> method.getReturnType() == Tensor.class)),
                () -> assertTrue(publicMethods.stream().noneMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertEquals(4, publicWithWeightsMethods.size()),
                () -> assertTrue(publicWithWeightsMethods.stream().allMatch(method ->
                        method.getReturnType() == ScaledDotProductAttentionResult.class)),
                () -> assertTrue(publicWithWeightsMethods.stream().noneMatch(
                        method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertEquals(213, Arrays.stream(Tensor.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()));

        assertMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttention", Tensor.class, Tensor.class));
        assertMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttention",
                Tensor.class,
                Tensor.class,
                ScaledDotProductAttentionAttrs.class));
        assertMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttention", Tensor.class, Tensor.class, Tensor.class));
        assertMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttention",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                ScaledDotProductAttentionAttrs.class));
        assertWithWeightsMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttentionWithWeights", Tensor.class, Tensor.class));
        assertWithWeightsMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttentionWithWeights",
                Tensor.class,
                Tensor.class,
                ScaledDotProductAttentionAttrs.class));
        assertWithWeightsMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttentionWithWeights",
                Tensor.class,
                Tensor.class,
                Tensor.class));
        assertWithWeightsMethod(Tensor.class.getDeclaredMethod(
                "scaledDotProductAttentionWithWeights",
                Tensor.class,
                Tensor.class,
                Tensor.class,
                ScaledDotProductAttentionAttrs.class));
    }

    @Test
    void createsExactOutputAndWeightsDescriptorsFromOneSharedProducer() {
        Dimension batch = new StaticDimension(2);
        Dimension queryLength = new DynamicDimension("L");
        Dimension embedding = new StaticDimension(8);
        Dimension keyLength = new DynamicDimension("S");
        Dimension valueWidth = new StaticDimension(10);
        Tensor query = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, queryLength, embedding), false);
        Tensor key = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), keyLength, embedding), true);
        Tensor value = tensor(DataType.FLOAT64,
                Shape.ofDimensions(batch, keyLength, valueWidth), false);
        Tensor mask = tensor(DataType.BOOL, Shape.scalar(), false);
        var attrs = new ScaledDotProductAttentionAttrs(
                Optional.of(ScalarValue.float64(0.25d)), true);

        ScaledDotProductAttentionResult result = query.scaledDotProductAttentionWithWeights(
                key, value, mask, attrs);
        Tensor output = result.output();
        Tensor weights = result.weights();
        TensorProvenance outputProvenance = output.provenance().orElseThrow();
        TensorProvenance weightsProvenance = weights.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(
                        Shape.ofDimensions(batch, queryLength, valueWidth),
                        output.descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, queryLength, keyLength),
                        weights.descriptor().shape()),
                () -> assertSame(batch, output.descriptor().shape().dimension(0)),
                () -> assertSame(batch, weights.descriptor().shape().dimension(0)),
                () -> assertSame(queryLength, output.descriptor().shape().dimension(1)),
                () -> assertSame(queryLength, weights.descriptor().shape().dimension(1)),
                () -> assertSame(valueWidth, output.descriptor().shape().dimension(2)),
                () -> assertSame(keyLength, weights.descriptor().shape().dimension(2)),
                () -> assertSame(DataType.FLOAT64, output.descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64, weights.descriptor().dataType()),
                () -> assertTrue(output.descriptor().requiresGrad()),
                () -> assertTrue(weights.descriptor().requiresGrad()),
                () -> assertTrue(output.descriptor().layout().isEmpty()),
                () -> assertTrue(weights.descriptor().layout().isEmpty()),
                () -> assertTrue(output.label().isEmpty()),
                () -> assertTrue(weights.label().isEmpty()),
                () -> assertTrue(output.hostStorage().isEmpty()),
                () -> assertTrue(weights.hostStorage().isEmpty()),
                () -> assertSame(outputProvenance.producer(), weightsProvenance.producer()),
                () -> assertSame(outputProvenance.operation(), weightsProvenance.operation()),
                () -> assertSame(attrs, outputProvenance.operation().attrs()),
                () -> assertEquals(0, outputProvenance.outputIndex()),
                () -> assertEquals(1, weightsProvenance.outputIndex()),
                () -> assertEquals(2, outputProvenance.producer().outputCount()),
                () -> assertSame(output.descriptor(), outputProvenance.outputDescriptor()),
                () -> assertSame(weights.descriptor(), weightsProvenance.outputDescriptor()),
                () -> assertEquals(4, outputProvenance.inputs().size()),
                () -> assertSame(query, outputProvenance.inputs().get(0)),
                () -> assertSame(key, outputProvenance.inputs().get(1)),
                () -> assertSame(value, outputProvenance.inputs().get(2)),
                () -> assertSame(mask, outputProvenance.inputs().get(3)));
    }

    @Test
    void separatesOutputAndWeightsGradientEligibilityAndPreservesDefaults() {
        Tensor query = tensor(DataType.FLOAT32, Shape.of(0, 3), false);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(4, 0), true);

        ScaledDotProductAttentionResult defaultResult =
                query.scaledDotProductAttentionWithWeights(key, value);
        ScaledDotProductAttentionAttrs defaultAttrs =
                (ScaledDotProductAttentionAttrs) defaultResult.output().provenance()
                        .orElseThrow().operation().attrs();

        assertAll(
                () -> assertEquals(Shape.of(0, 0), defaultResult.output().descriptor().shape()),
                () -> assertEquals(Shape.of(0, 4), defaultResult.weights().descriptor().shape()),
                () -> assertTrue(defaultResult.output().descriptor().requiresGrad()),
                () -> assertFalse(defaultResult.weights().descriptor().requiresGrad()),
                () -> assertTrue(defaultAttrs.scale().isEmpty()),
                () -> assertFalse(defaultAttrs.causal()));
    }

    @Test
    void allFourWithWeightsFormsRetainExactDefaultsAttrsAndOrderedInputs() {
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(4, 5), false);
        Tensor mask = tensor(DataType.BOOL, Shape.of(2, 4), false);
        var attrs = new ScaledDotProductAttentionAttrs(Optional.empty(), true);

        List<ScaledDotProductAttentionResult> results = List.of(
                query.scaledDotProductAttentionWithWeights(key, value),
                query.scaledDotProductAttentionWithWeights(key, value, attrs),
                query.scaledDotProductAttentionWithWeights(key, value, mask),
                query.scaledDotProductAttentionWithWeights(key, value, mask, attrs));

        assertAll(
                () -> assertSame(attrs, results.get(1).output().provenance()
                        .orElseThrow().operation().attrs()),
                () -> assertSame(attrs, results.get(3).output().provenance()
                        .orElseThrow().operation().attrs()),
                () -> assertFalse(((ScaledDotProductAttentionAttrs) results.get(0).output()
                        .provenance().orElseThrow().operation().attrs()).causal()),
                () -> assertFalse(((ScaledDotProductAttentionAttrs) results.get(2).output()
                        .provenance().orElseThrow().operation().attrs()).causal()),
                () -> assertEquals(3, results.get(0).output().provenance()
                        .orElseThrow().inputs().size()),
                () -> assertEquals(4, results.get(2).output().provenance()
                        .orElseThrow().inputs().size()),
                () -> assertSame(mask, results.get(2).weights().provenance()
                        .orElseThrow().inputs().get(3)));
    }

    @Test
    void derivesTheConceptualExampleShapeAndExactSelectedDimensionReferences() {
        Dimension batch = new StaticDimension(2);
        Dimension queryLength = new StaticDimension(4);
        Dimension embedding = new StaticDimension(8);
        Dimension keyLength = new StaticDimension(6);
        Dimension valueWidth = new StaticDimension(10);
        Tensor query = tensor(DataType.FLOAT32,
                Shape.ofDimensions(batch, queryLength, embedding), true);
        Tensor key = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(1), keyLength, embedding), false);
        Tensor value = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), keyLength, valueWidth), false);
        Tensor mask = tensor(DataType.BOOL, Shape.of(4, 6), false);
        var attrs = new ScaledDotProductAttentionAttrs(Optional.empty(), true);

        Tensor output = query.scaledDotProductAttention(key, value, mask, attrs);
        List<Dimension> dimensions = output.descriptor().shape().dimensions();

        assertAll(
                () -> assertEquals(Shape.of(2, 4, 10), output.descriptor().shape()),
                () -> assertSame(batch, dimensions.get(0)),
                () -> assertSame(queryLength, dimensions.get(1)),
                () -> assertSame(valueWidth, dimensions.get(2)),
                () -> assertSame(DataType.FLOAT32, output.descriptor().dataType()),
                () -> assertTrue(output.descriptor().requiresGrad()),
                () -> assertTrue(output.descriptor().layout().isEmpty()),
                () -> assertTrue(output.label().isEmpty()),
                () -> assertTrue(output.hostStorage().isEmpty()),
                () -> assertSame(attrs, output.provenance().orElseThrow().operation().attrs()));
    }

    @Test
    void implementsThreeWayBatchBroadcastSelectionAndDeferredPolicies() {
        Dimension queryBatch = new DynamicDimension("Q");
        Dimension staticFour = new StaticDimension(4);
        Tensor selectedStatic = attention(
                Shape.ofDimensions(queryBatch, new StaticDimension(2), new StaticDimension(3)),
                Shape.of(1, 5, 3),
                Shape.ofDimensions(staticFour, new StaticDimension(5), new StaticDimension(7)));
        Dimension shared = new DynamicDimension("B");
        Tensor selectedUnresolved = attention(
                Shape.ofDimensions(shared, new StaticDimension(2), new StaticDimension(3)),
                Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(5),
                        new StaticDimension(3)),
                Shape.of(5, 7));
        Dimension unpaired = new DynamicDimension("H");
        Tensor selectedUnpaired = attention(
                Shape.of(2, 3),
                Shape.ofDimensions(unpaired, new StaticDimension(5), new StaticDimension(3)),
                Shape.of(5, 7));

        assertAll(
                () -> assertSame(staticFour, selectedStatic.descriptor().shape().dimension(0)),
                () -> assertSame(shared, selectedUnresolved.descriptor().shape().dimension(0)),
                () -> assertSame(unpaired, selectedUnpaired.descriptor().shape().dimension(0)));
    }

    @Test
    void rejectsConflictingStaticAndUnderivableUnresolvedBatchDimensions() {
        IllegalArgumentException staticFailure = assertThrows(IllegalArgumentException.class,
                () -> attention(Shape.of(2, 3, 4), Shape.of(3, 5, 4), Shape.of(2, 5, 7)));
        Dimension queryBatch = new DynamicDimension("Q");
        Dimension keyBatch = DimensionExpressions.addConstant(new DynamicDimension("K"), 1);
        IllegalArgumentException dynamicFailure = assertThrows(IllegalArgumentException.class,
                () -> attention(
                        Shape.ofDimensions(queryBatch, new StaticDimension(3), new StaticDimension(4)),
                        Shape.ofDimensions(keyBatch, new StaticDimension(5), new StaticDimension(4)),
                        Shape.of(5, 7)));

        assertAll(
                () -> assertEquals(
                        "cannot broadcast attention batch dimensions at result batch axis 0: "
                                + "query=StaticDimension[size=2], key=StaticDimension[size=3], "
                                + "value=StaticDimension[size=2]",
                        staticFailure.getMessage()),
                () -> assertEquals(
                        "cannot derive exact attention batch dimension at result batch axis 0: "
                                + "query=" + queryBatch + ", key=" + keyBatch
                                + ", value=StaticDimension[size=1]",
                        dynamicFailure.getMessage()));
    }

    @Test
    void validatesMaskTypeRankAndEveryExactBroadcastCategory() {
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 4, 8), false);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(1, 6, 8), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(2, 6, 10), false);
        for (Shape maskShape : List.of(
                Shape.scalar(), Shape.of(1), Shape.of(6), Shape.of(4, 6), Shape.of(1, 4, 6),
                Shape.of(2, 4, 6),
                Shape.ofDimensions(new DynamicDimension("M"), new StaticDimension(6)))) {
            Tensor output = query.scaledDotProductAttention(
                    key, value, tensor(DataType.BOOL, maskShape, false));
            assertEquals(Shape.of(2, 4, 10), output.descriptor().shape());
            assertFalse(output.descriptor().requiresGrad());
        }

        IllegalArgumentException type = assertThrows(IllegalArgumentException.class,
                () -> query.scaledDotProductAttention(
                        key, value, tensor(DataType.FLOAT32, Shape.scalar(), false)));
        IllegalArgumentException rank = assertThrows(IllegalArgumentException.class,
                () -> query.scaledDotProductAttention(
                        key, value, tensor(DataType.BOOL, Shape.of(1, 2, 4, 6), false)));
        IllegalArgumentException axis = assertThrows(IllegalArgumentException.class,
                () -> query.scaledDotProductAttention(
                        key, value, tensor(DataType.BOOL, Shape.of(5, 6), false)));
        Dimension dynamicSequence = new DynamicDimension("S");
        IllegalArgumentException staticAgainstDynamic = assertThrows(IllegalArgumentException.class,
                () -> tensor(DataType.FLOAT32, Shape.of(2, 4, 8), false)
                        .scaledDotProductAttention(
                                tensor(DataType.FLOAT32,
                                        Shape.ofDimensions(dynamicSequence, new StaticDimension(8)),
                                        false),
                                tensor(DataType.FLOAT32,
                                        Shape.ofDimensions(dynamicSequence, new StaticDimension(10)),
                                        false),
                                tensor(DataType.BOOL, Shape.of(4, 6), false)));

        assertAll(
                () -> assertEquals("mask must have BOOL data type, but was FLOAT32",
                        type.getMessage()),
                () -> assertEquals(
                        "mask rank must not exceed attention score rank: mask=4, score=3",
                        rank.getMessage()),
                () -> assertEquals(
                        "mask cannot broadcast exactly to attention score shape at axis 1: "
                                + "mask=StaticDimension[size=5], score=StaticDimension[size=4]",
                        axis.getMessage()),
                () -> assertEquals(
                        "mask cannot broadcast exactly to attention score shape at axis 2: "
                                + "mask=StaticDimension[size=6], score=" + dynamicSequence,
                        staticAgainstDynamic.getMessage()));
    }

    @Test
    void promotesInQueryKeyValueOrderAndRequiresAnExactExplicitScaleType() {
        Tensor query = tensor(DataType.BFLOAT16, Shape.of(2, 3), true);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT64, Shape.of(4, 5), true);
        var exact = new ScaledDotProductAttentionAttrs(
                Optional.of(ScalarValue.float64(0.25d)), false);
        Tensor output = query.scaledDotProductAttention(key, value, exact);
        Tensor defaultOutput = query.scaledDotProductAttention(key, value);
        var wrong = new ScaledDotProductAttentionAttrs(
                Optional.of(ScalarValue.float32(0.25f)), false);
        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> query.scaledDotProductAttention(key, value, wrong));

        assertAll(
                () -> assertSame(DataType.FLOAT64, output.descriptor().dataType()),
                () -> assertTrue(output.descriptor().requiresGrad()),
                () -> assertSame(exact, output.provenance().orElseThrow().operation().attrs()),
                () -> assertTrue(((ScaledDotProductAttentionAttrs) defaultOutput.provenance()
                        .orElseThrow().operation().attrs()).scale().isEmpty()),
                () -> assertEquals(
                        "scale data type must match promoted attention data type: "
                                + "scale=FLOAT32, promoted=FLOAT64",
                        failure.getMessage()));
    }

    @Test
    void acceptsDeferredContractionsAndEmptyAxesButRejectsProvenStaticFailures() {
        Dimension queryEmbedding = new DynamicDimension("EQ");
        Dimension keyEmbedding = new DynamicDimension("EK");
        Dimension keySequence = new DynamicDimension("SK");
        Dimension valueSequence = new DynamicDimension("SV");
        Tensor deferred = attention(
                Shape.ofDimensions(new StaticDimension(0), queryEmbedding),
                Shape.ofDimensions(keySequence, keyEmbedding),
                Shape.ofDimensions(valueSequence, new StaticDimension(0)));

        assertEquals(Shape.of(0, 0), deferred.descriptor().shape());
        assertEquals("attention embedding dimension must be positive: StaticDimension[size=0]",
                assertThrows(IllegalArgumentException.class,
                        () -> attention(Shape.of(2, 0), Shape.of(3, 0), Shape.of(3, 4)))
                        .getMessage());
        assertEquals(
                "attention query/key embedding dimensions must match: "
                        + "query=StaticDimension[size=3], key=StaticDimension[size=4]",
                assertThrows(IllegalArgumentException.class,
                        () -> attention(Shape.of(2, 3), Shape.of(5, 4), Shape.of(5, 6)))
                        .getMessage());
        assertEquals(
                "attention key/value sequence dimensions must match: "
                        + "key=StaticDimension[size=5], value=StaticDimension[size=6]",
                assertThrows(IllegalArgumentException.class,
                        () -> attention(Shape.of(2, 3), Shape.of(5, 3), Shape.of(6, 7)))
                        .getMessage());
        assertEquals(Shape.of(2, 4),
                attention(Shape.of(2, 3), Shape.of(0, 3), Shape.of(0, 4))
                        .descriptor().shape());
    }

    @Test
    void createsOneFreshExactProducerAndOneIdWithoutMutatingInputs()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 3), true);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(4, 5), false);
        Tensor mask = tensor(DataType.BOOL, Shape.of(2, 4), false);
        var attrs = new ScaledDotProductAttentionAttrs(Optional.empty(), true);
        long before = next.get();

        Tensor first = query.scaledDotProductAttention(key, value, mask, attrs);
        Tensor second = query.scaledDotProductAttention(key, value, mask, attrs);
        TensorProvenance provenance = first.provenance().orElseThrow();

        assertAll(
                () -> assertEquals(before, first.id().value()),
                () -> assertEquals(before + 1, second.id().value()),
                () -> assertEquals(before + 2, next.get()),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first.id(), second.id()),
                () -> assertNotSame(provenance.producer(),
                        second.provenance().orElseThrow().producer()),
                () -> assertSame(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        provenance.operation().kind()),
                () -> assertSame(attrs, provenance.operation().attrs()),
                () -> assertEquals(4, provenance.inputs().size()),
                () -> assertSame(query, provenance.inputs().get(0)),
                () -> assertSame(key, provenance.inputs().get(1)),
                () -> assertSame(value, provenance.inputs().get(2)),
                () -> assertSame(mask, provenance.inputs().get(3)),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(first.descriptor(), provenance.outputDescriptor()),
                () -> assertTrue(query.provenance().isEmpty()),
                () -> assertTrue(key.provenance().isEmpty()),
                () -> assertTrue(value.provenance().isEmpty()),
                () -> assertTrue(mask.provenance().isEmpty()));
    }

    @Test
    void createsTwoFreshIdsInSlotOrderAndDistinctOccurrences()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(4, 5), false);
        long before = next.get();

        ScaledDotProductAttentionResult first =
                query.scaledDotProductAttentionWithWeights(key, value);
        ScaledDotProductAttentionResult second =
                query.scaledDotProductAttentionWithWeights(key, value);

        assertAll(
                () -> assertEquals(before, first.output().id().value()),
                () -> assertEquals(before + 1, first.weights().id().value()),
                () -> assertEquals(before + 2, second.output().id().value()),
                () -> assertEquals(before + 3, second.weights().id().value()),
                () -> assertEquals(before + 4, next.get()),
                () -> assertNotSame(first.output(), second.output()),
                () -> assertNotSame(first.weights(), second.weights()),
                () -> assertNotSame(first.output().provenance().orElseThrow().producer(),
                        second.output().provenance().orElseThrow().producer()),
                () -> assertSame(first.output().provenance().orElseThrow().producer(),
                        first.weights().provenance().orElseThrow().producer()));
    }

    @Test
    void retainsConsumedSlotZeroIdWhenSlotOneExhaustsIdentifierSpace()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long originalNext = next.get();
        boolean originalMaximumClaimed = maximumClaimed.get();
        Tensor query = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor key = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor value = tensor(DataType.FLOAT32, Shape.of(4, 5), false);

        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(false);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> query.scaledDotProductAttentionWithWeights(key, value));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(originalNext);
            maximumClaimed.set(originalMaximumClaimed);
        }
    }

    @Test
    void validatesInExactOrderAndConsumesNoIdForEveryLocalFailure()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        Tensor bool = tensor(DataType.BOOL, Shape.scalar(), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor validQuery = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor validKey = tensor(DataType.FLOAT32, Shape.of(4, 3), false);
        Tensor validValue = tensor(DataType.FLOAT32, Shape.of(4, 5), false);
        Tensor validMask = tensor(DataType.BOOL, Shape.scalar(), false);
        var attrs = new ScaledDotProductAttentionAttrs(Optional.empty(), false);
        long before = next.get();

        assertEquals("query", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.apply(null, null, null, attrs))
                .getMessage());
        assertEquals("key", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.apply(
                        validQuery, null, null, attrs)).getMessage());
        assertEquals("value", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.apply(
                        validQuery, validKey, null, attrs)).getMessage());
        assertEquals("mask", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.apply(
                        validQuery, validKey, validValue, null, attrs)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.apply(
                        validQuery, validKey, validValue, validMask, null)).getMessage());
        assertEquals("query", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                        null, null, null, attrs)).getMessage());
        assertEquals("key", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                        validQuery, null, null, attrs)).getMessage());
        assertEquals("value", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                        validQuery, validKey, null, attrs)).getMessage());
        assertEquals("mask", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                        validQuery, validKey, validValue, null, attrs)).getMessage());
        assertEquals("attrs", assertThrows(NullPointerException.class,
                () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                        validQuery, validKey, validValue, validMask, null)).getMessage());
        assertEquals("query must have a floating data type, but was BOOL",
                assertThrows(IllegalArgumentException.class,
                        () -> TensorScaledDotProductAttentionExpressions.apply(
                                bool, bool, bool, attrs)).getMessage());
        assertEquals("query rank must be at least 2: 0",
                assertThrows(IllegalArgumentException.class,
                        () -> TensorScaledDotProductAttentionExpressions.apply(
                                scalar, scalar, scalar, attrs)).getMessage());
        assertEquals("query must have a floating data type, but was BOOL",
                assertThrows(IllegalArgumentException.class,
                        () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                                bool, bool, bool, attrs)).getMessage());
        assertEquals("query rank must be at least 2: 0",
                assertThrows(IllegalArgumentException.class,
                        () -> TensorScaledDotProductAttentionExpressions.applyWithWeights(
                                scalar, scalar, scalar, attrs)).getMessage());
        assertEquals(before, next.get());
    }

    private static void assertMethod(Method method) {
        assertAll(
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(method.isVarArgs()),
                () -> assertSame(Tensor.class, method.getReturnType()));
    }

    private static void assertWithWeightsMethod(Method method) {
        assertAll(
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(method.isVarArgs()),
                () -> assertSame(ScaledDotProductAttentionResult.class, method.getReturnType()));
    }

    private static Tensor attention(Shape query, Shape key, Shape value) {
        return tensor(DataType.FLOAT32, query, false).scaledDotProductAttention(
                tensor(DataType.FLOAT32, key, false),
                tensor(DataType.FLOAT32, value, false));
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return new Tensor(
                new TensorId(1),
                new TensorDescriptor(dataType, shape, Optional.empty(), requiresGrad),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState()
            throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
