package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.Sequential;
import io.github.pho001.synaptik.nn.module.StateKind;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class RnnCellTest {
    @Test
    void exposesExactlyThePlannedFinalDirectModuleSurface() throws ReflectiveOperationException {
        Set<List<Class<?>>> constructors = Arrays.stream(RnnCell.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(RnnCell.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(RnnCell.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(RnnCell.class.getModifiers())),
                () -> assertSame(Module.class, RnnCell.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(RnnCell.class)),
                () -> assertFalse(Sequential.class.isAssignableFrom(RnnCell.class)),
                () -> assertEquals(Set.of(
                        List.of(Tensor.class, Tensor.class),
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        List.of(
                                long.class,
                                long.class,
                                boolean.class,
                                DataType.class,
                                RandomGenerator.class)), constructors),
                () -> assertEquals(
                        Set.of("inputWeight", "hiddenWeight", "bias", "forward"), methods),
                () -> assertSame(
                        Parameter.class,
                        RnnCell.class.getDeclaredMethod("inputWeight").getReturnType()),
                () -> assertSame(
                        Parameter.class,
                        RnnCell.class.getDeclaredMethod("hiddenWeight").getReturnType()),
                () -> assertSame(
                        Optional.class,
                        RnnCell.class.getDeclaredMethod("bias").getReturnType()),
                () -> assertSame(
                        Tensor.class,
                        RnnCell.class.getDeclaredMethod("forward", Tensor.class, Tensor.class)
                                .getReturnType()),
                () -> assertEquals(4, Arrays.stream(RnnCell.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Arrays.stream(RnnCell.class.getDeclaredFields())
                        .noneMatch(field -> Modifier.isPublic(field.getModifiers())
                                || Modifier.isProtected(field.getModifiers()))));
    }

    @Test
    void suppliedStateRetainsExactBindingsNamesOrderMetadataAndDiscovery() {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), true);
        RnnCell noBias = new RnnCell(inputWeight, hiddenWeight);
        RnnCell biased = new RnnCell(inputWeight, hiddenWeight, bias);
        Owner owner = new Owner(biased);

        assertAll(
                () -> assertSame(inputWeight, noBias.inputWeight().value()),
                () -> assertSame(hiddenWeight, noBias.hiddenWeight().value()),
                () -> assertTrue(noBias.bias().isEmpty()),
                () -> assertEquals(
                        List.of("inputWeight", "hiddenWeight"), names(noBias.parameters())),
                () -> assertSame(inputWeight, biased.inputWeight().value()),
                () -> assertSame(hiddenWeight, biased.hiddenWeight().value()),
                () -> assertSame(bias, biased.bias().orElseThrow().value()),
                () -> assertEquals(
                        List.of("inputWeight", "hiddenWeight", "bias"),
                        names(biased.parameters())),
                () -> assertTrue(biased.buffers().isEmpty()),
                () -> assertTrue(biased.children().isEmpty()),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        List.copyOf(owner.parametersRecursively().keySet())),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        owner.stateDictionary().entries().stream().map(entry -> entry.path()).toList()),
                () -> assertTrue(owner.stateDictionary().entries().stream()
                        .allMatch(entry -> entry.kind() == StateKind.PARAMETER)),
                () -> assertSame(
                        biased.inputWeight(), owner.parametersRecursively().get("cell.inputWeight")),
                () -> assertSame(
                        biased.hiddenWeight(),
                        owner.parametersRecursively().get("cell.hiddenWeight")),
                () -> assertSame(
                        biased.bias().orElseThrow(),
                        owner.parametersRecursively().get("cell.bias")));
    }

    @Test
    void suppliedConstructionChecksNullsAndEveryInputWeightRuleBeforeAllocatingIds()
            throws ReflectiveOperationException {
        Tensor integral = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("H"), new StaticDimension(3)),
                true);
        Tensor zeroHidden = tensor(DataType.FLOAT32, Shape.of(0, 0), true);
        Tensor zeroInput = tensor(DataType.FLOAT32, Shape.of(4, 0), true);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals(
                        "inputWeight",
                        assertThrows(
                                NullPointerException.class,
                                () -> new RnnCell(null, null, null)).getMessage()),
                () -> assertEquals(
                        "hiddenWeight",
                        assertThrows(
                                NullPointerException.class,
                                () -> new RnnCell(integral, null, null)).getMessage()),
                () -> assertEquals(
                        "bias",
                        assertThrows(
                                NullPointerException.class,
                                () -> new RnnCell(integral, integral, null)).getMessage()),
                () -> assertContains("inputWeight", "floating", () -> new RnnCell(integral, validHidden)),
                () -> assertContains(
                        "inputWeight", "requiresGrad", () -> new RnnCell(noGradient, validHidden)),
                () -> assertContains("inputWeight", "rank two", () -> new RnnCell(scalar, validHidden)),
                () -> assertContains(
                        "inputWeight", "fully static", () -> new RnnCell(dynamic, validHidden)),
                () -> assertContains(
                        "inputWeight hiddenSize", "positive",
                        () -> new RnnCell(zeroHidden, validHidden)),
                () -> assertContains(
                        "inputWeight inputSize", "positive",
                        () -> new RnnCell(zeroInput, validHidden)),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void suppliedConstructionChecksEveryHiddenWeightAndBiasRuleBeforeDeclaration()
            throws ReflectiveOperationException {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor integral = tensor(DataType.INT32, Shape.of(4, 4), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.of(4, 4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("H"), new StaticDimension(4)),
                true);
        Tensor zeroAxisZero = tensor(DataType.FLOAT32, Shape.of(0, 4), true);
        Tensor zeroAxisOne = tensor(DataType.FLOAT32, Shape.of(4, 0), true);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(4, 4), true);
        Tensor wrongAxisZero = tensor(DataType.FLOAT32, Shape.of(5, 4), true);
        Tensor wrongAxisOne = tensor(DataType.FLOAT32, Shape.of(4, 5), true);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertContains("hiddenWeight", "floating", () -> new RnnCell(inputWeight, integral)),
                () -> assertContains(
                        "hiddenWeight", "requiresGrad", () -> new RnnCell(inputWeight, noGradient)),
                () -> assertContains("hiddenWeight", "rank two", () -> new RnnCell(inputWeight, scalar)),
                () -> assertContains(
                        "hiddenWeight", "fully static", () -> new RnnCell(inputWeight, dynamic)),
                () -> assertContains(
                        "hiddenWeight axis zero", "positive",
                        () -> new RnnCell(inputWeight, zeroAxisZero)),
                () -> assertContains(
                        "hiddenWeight axis one", "positive",
                        () -> new RnnCell(inputWeight, zeroAxisOne)),
                () -> assertContains("hiddenWeight", "data type", () -> new RnnCell(inputWeight, wrongType)),
                () -> assertContains("axis zero", "hidden size", () -> new RnnCell(inputWeight, wrongAxisZero)),
                () -> assertContains("axis one", "hidden size", () -> new RnnCell(inputWeight, wrongAxisOne)),
                () -> assertEquals(before, ids.get()),
                () -> assertBiasFailures(inputWeight, validHidden));
    }

    @Test
    void forwardBuildsExactNoBiasAndBiasedChainsInIdentifierOrder() throws Exception {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), true);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor hidden = tensor(DataType.BFLOAT16, Shape.of(2, 4), false);
        AtomicLong ids = nextTensorIdState();

        long noBiasStart = ids.get();
        Tensor noBias = new RnnCell(inputWeight, hiddenWeight).forward(input, hidden);
        assertCellChain(noBias, input, hidden, inputWeight, hiddenWeight, Optional.empty());
        assertIds(noBias, noBiasStart, false);

        long biasedStart = ids.get();
        Tensor biased = new RnnCell(inputWeight, hiddenWeight, bias).forward(input, hidden);
        assertCellChain(biased, input, hidden, inputWeight, hiddenWeight, Optional.of(bias));
        assertIds(biased, biasedStart, true);
        assertAll(
                () -> assertSame(DataType.FLOAT64, biased.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 4), biased.descriptor().shape()),
                () -> assertTrue(biased.descriptor().requiresGrad()),
                () -> assertTrue(biased.descriptor().layout().isEmpty()),
                () -> assertTrue(biased.label().isEmpty()),
                () -> assertTrue(biased.hostStorage().isEmpty()));
    }

    @Test
    void forwardSupportsRankOneMixedFloatingAndOrdinaryLeadingBroadcasts() {
        RnnCell cell = cell(DataType.FLOAT32, true);
        DynamicDimension batch = new DynamicDimension("B");

        Tensor rankOne = cell.forward(
                tensor(DataType.BFLOAT16, Shape.of(3), false),
                tensor(DataType.FLOAT32, Shape.of(4), false));
        Tensor hiddenVectorBroadcast = cell.forward(
                tensor(DataType.FLOAT64, Shape.of(7, 3), false),
                tensor(DataType.BFLOAT16, Shape.of(4), false));
        Tensor singletonBroadcast = cell.forward(
                tensor(DataType.FLOAT32, Shape.of(2, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(7, 4), false));
        Tensor symbolic = cell.forward(
                tensor(DataType.FLOAT32, Shape.ofDimensions(batch, new StaticDimension(3)), false),
                tensor(DataType.FLOAT32, Shape.ofDimensions(batch, new StaticDimension(4)), false));

        assertAll(
                () -> assertEquals(Shape.of(4), rankOne.descriptor().shape()),
                () -> assertSame(DataType.FLOAT32, rankOne.descriptor().dataType()),
                () -> assertEquals(Shape.of(7, 4), hiddenVectorBroadcast.descriptor().shape()),
                () -> assertSame(DataType.FLOAT64, hiddenVectorBroadcast.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 7, 4), singletonBroadcast.descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, new StaticDimension(4)),
                        symbolic.descriptor().shape()));
    }

    @Test
    void forwardPrevalidatesNullTypeRankFeatureAndBroadcastFailuresBeforeAnyExpressionId()
            throws ReflectiveOperationException {
        RnnCell cell = cell(DataType.FLOAT32, true);
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor integralInput = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor integralHidden = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor wrongInput = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongHidden = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor incompatibleHidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        DynamicDimension left = new DynamicDimension("L");
        DynamicDimension right = new DynamicDimension("R");
        Tensor symbolicInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(left, new StaticDimension(3)), false);
        Tensor symbolicHidden = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(right, new StaticDimension(4)), false);
        Tensor symbolicStaticHidden = tensor(DataType.FLOAT32, Shape.of(5, 4), false);
        Tensor unresolvedInputFeature = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("I")), false);
        Tensor unresolvedHiddenFeature = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("H")), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals(
                        "input",
                        assertThrows(NullPointerException.class, () -> cell.forward(null, null))
                                .getMessage()),
                () -> assertEquals(
                        "hidden",
                        assertThrows(
                                NullPointerException.class,
                                () -> cell.forward(validInput, null)).getMessage()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> cell.forward(integralInput, validHidden)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> cell.forward(validInput, integralHidden)),
                () -> assertContains("input", "rank", () -> cell.forward(scalar, validHidden)),
                () -> assertContains("hidden", "rank", () -> cell.forward(validInput, scalar)),
                () -> assertContains("input", "feature", () -> cell.forward(wrongInput, validHidden)),
                () -> assertContains("hidden", "feature", () -> cell.forward(validInput, wrongHidden)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> cell.forward(validInput, incompatibleHidden)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> cell.forward(symbolicInput, symbolicHidden)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> cell.forward(symbolicInput, symbolicStaticHidden)),
                () -> assertEquals(before, ids.get()));

        Tensor unresolved = cell.forward(unresolvedInputFeature, unresolvedHiddenFeature);
        assertEquals(Shape.of(4), unresolved.descriptor().shape());
    }

    @Test
    void modeDoesNotChangeCompositionAndExplicitHiddenStateIsNeverRetained() {
        RnnCell cell = cell(DataType.FLOAT32, false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor initialHidden = tensor(DataType.FLOAT32, Shape.of(4), false);

        cell.eval();
        Tensor evaluation = cell.forward(input, initialHidden);
        cell.train();
        Tensor training = cell.forward(input, initialHidden);
        Tensor threaded = cell.forward(input, evaluation);

        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, cell.mode()),
                () -> assertEquals(evaluation.descriptor(), training.descriptor()),
                () -> assertNotSame(evaluation, training),
                () -> assertNotSame(training, threaded),
                () -> assertSame(
                        initialHidden,
                        hiddenProjection(training).provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(
                        evaluation,
                        hiddenProjection(threaded).provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(cell.buffersRecursively().isEmpty()),
                () -> assertEquals(
                        List.of("inputWeight", "hiddenWeight"),
                        List.copyOf(cell.parametersRecursively().keySet())),
                () -> assertEquals(2, cell.stateDictionary().entries().size()));
    }

    @Test
    void compatibleReplacementAffectsOnlyLaterSnapshotsAndKeepsStableWrappers() {
        Tensor oldInputWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor oldHiddenWeight = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        Tensor oldBias = tensor(DataType.FLOAT32, Shape.of(4), true);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(4), false);
        RnnCell cell = new RnnCell(oldInputWeight, oldHiddenWeight, oldBias);
        Parameter inputHandle = cell.inputWeight();
        Parameter hiddenHandle = cell.hiddenWeight();
        Parameter biasHandle = cell.bias().orElseThrow();
        Tensor before = cell.forward(input, hidden);

        Tensor newInputWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor newHiddenWeight = tensor(DataType.FLOAT32, Shape.of(4, 4), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(4), true);
        inputHandle.replace(newInputWeight);
        hiddenHandle.replace(newHiddenWeight);
        biasHandle.replace(newBias);
        Tensor after = cell.forward(input, hidden);

        assertCellChain(
                before, input, hidden, oldInputWeight, oldHiddenWeight, Optional.of(oldBias));
        assertCellChain(
                after, input, hidden, newInputWeight, newHiddenWeight, Optional.of(newBias));
        assertAll(
                () -> assertSame(inputHandle, cell.inputWeight()),
                () -> assertSame(hiddenHandle, cell.hiddenWeight()),
                () -> assertSame(biasHandle, cell.bias().orElseThrow()),
                () -> assertSame(newInputWeight, inputHandle.value()),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()),
                () -> assertSame(newBias, biasHandle.value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> hiddenHandle.replace(
                                tensor(DataType.FLOAT32, Shape.of(3, 4), true))),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()));
    }

    private static void assertBiasFailures(Tensor inputWeight, Tensor hiddenWeight)
            throws ReflectiveOperationException {
        Tensor integral = tensor(DataType.INT32, Shape.of(4), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(
                DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("H")), true);
        Tensor wrongType = tensor(DataType.FLOAT64, Shape.of(4), true);
        Tensor wrongShape = tensor(DataType.FLOAT32, Shape.of(5), true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertContains(
                        "bias", "floating", () -> new RnnCell(inputWeight, hiddenWeight, integral)),
                () -> assertContains(
                        "bias", "requiresGrad",
                        () -> new RnnCell(inputWeight, hiddenWeight, noGradient)),
                () -> assertContains(
                        "bias", "rank one", () -> new RnnCell(inputWeight, hiddenWeight, scalar)),
                () -> assertContains(
                        "bias", "fully static",
                        () -> new RnnCell(inputWeight, hiddenWeight, dynamic)),
                () -> assertContains(
                        "bias", "data type",
                        () -> new RnnCell(inputWeight, hiddenWeight, wrongType)),
                () -> assertContains(
                        "bias", "hidden size",
                        () -> new RnnCell(inputWeight, hiddenWeight, wrongShape)),
                () -> assertEquals(before, ids.get()));
    }

    private static void assertCellChain(
            Tensor result,
            Tensor input,
            Tensor hidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias) {
        TensorProvenance tanh = result.provenance().orElseThrow();
        Tensor sum = tanh.inputs().getFirst();
        TensorProvenance projectionAdd = sum.provenance().orElseThrow();
        Tensor inputProjection = projectionAdd.inputs().getFirst();
        Tensor hiddenProjection = projectionAdd.inputs().get(1);

        assertAll(
                () -> assertSame(UnaryElementwiseKind.TANH, tanh.operation().kind()),
                () -> assertSame(BinaryArithmeticKind.ADD, projectionAdd.operation().kind()),
                () -> assertSame(sum, tanh.inputs().getFirst()),
                () -> assertLinearChain(inputProjection, input, inputWeight, bias),
                () -> assertLinearChain(
                        hiddenProjection, hidden, hiddenWeight, Optional.empty()));
    }

    private static void assertLinearChain(
            Tensor projection, Tensor input, Tensor weight, Optional<Tensor> bias) {
        Tensor product;
        if (bias.isPresent()) {
            TensorProvenance biasAdd = projection.provenance().orElseThrow();
            product = biasAdd.inputs().getFirst();
            assertAll(
                    () -> assertSame(BinaryArithmeticKind.ADD, biasAdd.operation().kind()),
                    () -> assertSame(bias.orElseThrow(), biasAdd.inputs().get(1)));
        } else {
            product = projection;
        }
        TensorProvenance matmul = product.provenance().orElseThrow();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProvenance permute = transposedWeight.provenance().orElseThrow();
        assertAll(
                () -> assertSame(MatmulKind.MATMUL, matmul.operation().kind()),
                () -> assertSame(input, matmul.inputs().getFirst()),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.operation().kind()),
                () -> assertEquals(
                        List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertSame(weight, permute.inputs().getFirst()));
    }

    private static void assertIds(Tensor result, long start, boolean biased) {
        Tensor sum = result.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProjection = sum.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProduct = sum.provenance().orElseThrow().inputs().get(1);
        Tensor inputProduct = biased
                ? inputProjection.provenance().orElseThrow().inputs().getFirst()
                : inputProjection;
        Tensor inputTranspose = inputProduct.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenTranspose = hiddenProduct.provenance().orElseThrow().inputs().get(1);
        long biasOffset = biased ? 1L : 0L;
        assertAll(
                () -> assertEquals(start, inputTranspose.id().value()),
                () -> assertEquals(start + 1, inputProduct.id().value()),
                () -> {
                    if (biased) {
                        assertEquals(start + 2, inputProjection.id().value());
                    }
                },
                () -> assertEquals(start + 2 + biasOffset, hiddenTranspose.id().value()),
                () -> assertEquals(start + 3 + biasOffset, hiddenProduct.id().value()),
                () -> assertEquals(start + 4 + biasOffset, sum.id().value()),
                () -> assertEquals(start + 5 + biasOffset, result.id().value()));
    }

    private static Tensor hiddenProjection(Tensor result) {
        Tensor sum = result.provenance().orElseThrow().inputs().getFirst();
        return sum.provenance().orElseThrow().inputs().get(1);
    }

    private static RnnCell cell(DataType dataType, boolean bias) {
        Tensor inputWeight = tensor(dataType, Shape.of(4, 3), true);
        Tensor hiddenWeight = tensor(dataType, Shape.of(4, 4), true);
        return bias
                ? new RnnCell(inputWeight, hiddenWeight, tensor(dataType, Shape.of(4), true))
                : new RnnCell(inputWeight, hiddenWeight);
    }

    private static List<String> names(List<Parameter> parameters) {
        return parameters.stream().map(Parameter::name).toList();
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static void assertContains(
            String first, String second, ThrowingConstruction construction) {
        String message = assertThrows(IllegalArgumentException.class, construction::run).getMessage();
        assertTrue(message.contains(first), message);
        assertTrue(message.contains(second), message);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    @FunctionalInterface
    private interface ThrowingConstruction {
        void run();
    }

    private static final class Owner extends Module {
        private Owner(RnnCell cell) {
            child("cell", cell);
        }
    }
}
