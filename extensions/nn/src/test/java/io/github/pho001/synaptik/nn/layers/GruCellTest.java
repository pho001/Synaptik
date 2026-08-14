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
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.Buffer;
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
class GruCellTest {
    @Test
    void exposesExactlyThePlannedFinalDirectModuleSurfaceAndNoHiddenState() throws Exception {
        Set<List<Class<?>>> constructors = Arrays.stream(GruCell.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(GruCell.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(GruCell.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GruCell.class.getModifiers())),
                () -> assertSame(Module.class, GruCell.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(GruCell.class)),
                () -> assertFalse(Sequential.class.isAssignableFrom(GruCell.class)),
                () -> assertEquals(3, GruCell.class.getDeclaredConstructors().length),
                () -> assertTrue(Arrays.stream(GruCell.class.getDeclaredConstructors())
                        .allMatch(constructor -> Modifier.isPublic(constructor.getModifiers()))),
                () -> assertEquals(Set.of(
                        List.of(Tensor.class, Tensor.class),
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        List.of(long.class, long.class, boolean.class,
                                DataType.class, RandomGenerator.class)), constructors),
                () -> assertEquals(
                        Set.of("inputWeight", "hiddenWeight", "bias", "forward"), methods),
                () -> assertSame(Parameter.class,
                        GruCell.class.getDeclaredMethod("inputWeight").getReturnType()),
                () -> assertSame(Parameter.class,
                        GruCell.class.getDeclaredMethod("hiddenWeight").getReturnType()),
                () -> assertSame(Optional.class,
                        GruCell.class.getDeclaredMethod("bias").getReturnType()),
                () -> assertSame(Tensor.class,
                        GruCell.class.getDeclaredMethod("forward", Tensor.class, Tensor.class)
                                .getReturnType()),
                () -> assertEquals(4L, Arrays.stream(GruCell.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers())).count()),
                () -> assertTrue(Arrays.stream(GruCell.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isProtected(method.getModifiers()))),
                () -> assertEquals(Set.of("inputWeight", "hiddenWeight", "bias"),
                        Arrays.stream(GruCell.class.getDeclaredFields())
                                .map(Field::getName)
                                .collect(Collectors.toSet())),
                () -> assertTrue(Arrays.stream(GruCell.class.getDeclaredFields())
                        .allMatch(field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(GruCell.class.getDeclaredFields())
                        .noneMatch(field -> field.getType() == Tensor.class
                                || field.getType() == Buffer.class
                                || RandomGenerator.class.isAssignableFrom(field.getType()))),
                () -> assertEquals(0, GruCell.class.getDeclaredClasses().length));
    }

    @Test
    void suppliedStateRetainsPackedBindingsNamesOrderAndDiscovery() {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(12), true);
        GruCell noBias = new GruCell(inputWeight, hiddenWeight);
        GruCell biased = new GruCell(inputWeight, hiddenWeight, bias);
        Owner owner = new Owner(biased);

        assertAll(
                () -> assertSame(inputWeight, noBias.inputWeight().value()),
                () -> assertSame(hiddenWeight, noBias.hiddenWeight().value()),
                () -> assertTrue(noBias.bias().isEmpty()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight"), names(noBias)),
                () -> assertSame(inputWeight, biased.inputWeight().value()),
                () -> assertSame(hiddenWeight, biased.hiddenWeight().value()),
                () -> assertSame(bias, biased.bias().orElseThrow().value()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight", "bias"), names(biased)),
                () -> assertTrue(biased.buffers().isEmpty()),
                () -> assertTrue(biased.children().isEmpty()),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        List.copyOf(owner.parametersRecursively().keySet())),
                () -> assertTrue(owner.stateDictionary().entries().stream()
                        .allMatch(entry -> entry.kind() == StateKind.PARAMETER)),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        owner.stateDictionary().entries().stream().map(entry -> entry.path()).toList()),
                () -> assertTrue(owner.buffersRecursively().isEmpty()));
    }

    @Test
    void suppliedConstructionValidatesCompletePackedSchemaBeforeDeclarationOrIds() throws Exception {
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        Tensor integral = tensor(DataType.INT32, Shape.of(12, 3), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.of(12, 3), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P"), new StaticDimension(3)), true);
        Tensor zeroPacked = tensor(DataType.FLOAT32, Shape.of(0, 3), true);
        Tensor unpacked = tensor(DataType.FLOAT32, Shape.of(10, 3), true);
        Tensor zeroInput = tensor(DataType.FLOAT32, Shape.of(12, 0), true);
        assertHiddenAndBiasFailures(validInput, validHidden);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("inputWeight",
                        assertThrows(NullPointerException.class,
                                () -> new GruCell(null, null, null)).getMessage()),
                () -> assertEquals("hiddenWeight",
                        assertThrows(NullPointerException.class,
                                () -> new GruCell(validInput, null, null)).getMessage()),
                () -> assertEquals("bias",
                        assertThrows(NullPointerException.class,
                                () -> new GruCell(validInput, validHidden, null)).getMessage()),
                () -> assertContains("inputWeight", "floating",
                        () -> new GruCell(integral, validHidden)),
                () -> assertContains("inputWeight", "requiresGrad",
                        () -> new GruCell(noGradient, validHidden)),
                () -> assertContains("inputWeight", "rank two",
                        () -> new GruCell(scalar, validHidden)),
                () -> assertContains("inputWeight", "fully static",
                        () -> new GruCell(dynamic, validHidden)),
                () -> assertContains("packed hidden size", "positive",
                        () -> new GruCell(zeroPacked, validHidden)),
                () -> assertContains("packed hidden size", "divisible",
                        () -> new GruCell(unpacked, validHidden)),
                () -> assertContains("inputSize", "positive",
                        () -> new GruCell(zeroInput, validHidden)),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void forwardBuildsExactResetAfterFormulaSlicesBiasSideAndIdentifierOrder() throws Exception {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(12), true);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor hidden = tensor(DataType.BFLOAT16, Shape.of(2, 4), false);
        AtomicLong ids = nextTensorIdState();

        long noBiasStart = ids.get();
        Tensor noBias = new GruCell(inputWeight, hiddenWeight).forward(input, hidden);
        assertGruChain(noBias, input, hidden, inputWeight, hiddenWeight, Optional.empty());
        assertIds(noBias, noBiasStart, false);

        long biasedStart = ids.get();
        Tensor biased = new GruCell(inputWeight, hiddenWeight, bias).forward(input, hidden);
        assertGruChain(biased, input, hidden, inputWeight, hiddenWeight, Optional.of(bias));
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
    void forwardSupportsRankOneMixedFloatingAndLeadingBatchBroadcasting() {
        GruCell cell = cell(DataType.FLOAT32, true);
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
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(batch, new StaticDimension(3)), false),
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(batch, new StaticDimension(4)), false));

        assertAll(
                () -> assertEquals(Shape.of(4), rankOne.descriptor().shape()),
                () -> assertSame(DataType.FLOAT32, rankOne.descriptor().dataType()),
                () -> assertEquals(Shape.of(7, 4), hiddenVectorBroadcast.descriptor().shape()),
                () -> assertSame(DataType.FLOAT64,
                        hiddenVectorBroadcast.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 7, 4), singletonBroadcast.descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, new StaticDimension(4)),
                        symbolic.descriptor().shape()));
    }

    @Test
    void forwardPrevalidatesEveryLocallyKnowableFailureBeforeFirstExpressionId() throws Exception {
        GruCell cell = cell(DataType.FLOAT32, true);
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
        Tensor symbolicInput = tensor(DataType.FLOAT32,
                Shape.ofDimensions(left, new StaticDimension(3)), false);
        Tensor symbolicHidden = tensor(DataType.FLOAT32,
                Shape.ofDimensions(right, new StaticDimension(4)), false);
        Tensor unresolvedInputFeature = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("I")), false);
        Tensor unresolvedHiddenFeature = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("H")), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("input",
                        assertThrows(NullPointerException.class,
                                () -> cell.forward(null, null)).getMessage()),
                () -> assertEquals("hidden",
                        assertThrows(NullPointerException.class,
                                () -> cell.forward(validInput, null)).getMessage()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> cell.forward(integralInput, validHidden)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> cell.forward(validInput, integralHidden)),
                () -> assertContains("input", "rank",
                        () -> cell.forward(scalar, validHidden)),
                () -> assertContains("hidden", "rank",
                        () -> cell.forward(validInput, scalar)),
                () -> assertContains("input", "feature",
                        () -> cell.forward(wrongInput, validHidden)),
                () -> assertContains("hidden", "feature",
                        () -> cell.forward(validInput, wrongHidden)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> cell.forward(validInput, incompatibleHidden)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> cell.forward(symbolicInput, symbolicHidden)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> cell.forward(validInput, unresolvedHiddenFeature)),
                () -> assertEquals(before, ids.get()));

        Tensor unresolved = cell.forward(unresolvedInputFeature, validHidden);
        assertEquals(Shape.of(2, 4), unresolved.descriptor().shape());
    }

    @Test
    void modeAndRepeatedExplicitHiddenThreadingNeverCreateRetainedState() {
        GruCell cell = cell(DataType.FLOAT32, false);
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
                () -> assertSame(initialHidden, finalDifference(training).inputs().getFirst()),
                () -> assertSame(evaluation, finalDifference(threaded).inputs().getFirst()),
                () -> assertTrue(cell.buffersRecursively().isEmpty()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight"), names(cell)),
                () -> assertEquals(2, cell.stateDictionary().entries().size()));
    }

    @Test
    void compatibleReplacementAffectsOnlyLaterSnapshotsAndKeepsStableWrappers() {
        Tensor oldInputWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor oldHiddenWeight = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        Tensor oldBias = tensor(DataType.FLOAT32, Shape.of(12), true);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(4), false);
        GruCell cell = new GruCell(oldInputWeight, oldHiddenWeight, oldBias);
        Parameter inputHandle = cell.inputWeight();
        Parameter hiddenHandle = cell.hiddenWeight();
        Parameter biasHandle = cell.bias().orElseThrow();
        Tensor before = cell.forward(input, hidden);

        Tensor newInputWeight = tensor(DataType.FLOAT32, Shape.of(12, 3), true);
        Tensor newHiddenWeight = tensor(DataType.FLOAT32, Shape.of(12, 4), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(12), true);
        inputHandle.replace(newInputWeight);
        hiddenHandle.replace(newHiddenWeight);
        biasHandle.replace(newBias);
        Tensor after = cell.forward(input, hidden);

        assertGruChain(before, input, hidden,
                oldInputWeight, oldHiddenWeight, Optional.of(oldBias));
        assertGruChain(after, input, hidden,
                newInputWeight, newHiddenWeight, Optional.of(newBias));
        assertAll(
                () -> assertSame(inputHandle, cell.inputWeight()),
                () -> assertSame(hiddenHandle, cell.hiddenWeight()),
                () -> assertSame(biasHandle, cell.bias().orElseThrow()),
                () -> assertSame(newInputWeight, inputHandle.value()),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()),
                () -> assertSame(newBias, biasHandle.value()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> hiddenHandle.replace(
                                tensor(DataType.FLOAT32, Shape.of(12, 5), true))),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()));
    }

    private static void assertHiddenAndBiasFailures(Tensor inputWeight, Tensor validHidden)
            throws Exception {
        Tensor integralHidden = tensor(DataType.INT32, Shape.of(12, 4), false);
        Tensor noGradientHidden = tensor(DataType.FLOAT32, Shape.of(12, 4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamicHidden = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P"), new StaticDimension(4)), true);
        Tensor zeroAxisZero = tensor(DataType.FLOAT32, Shape.of(0, 4), true);
        Tensor zeroAxisOne = tensor(DataType.FLOAT32, Shape.of(12, 0), true);
        Tensor wrongTypeHidden = tensor(DataType.FLOAT64, Shape.of(12, 4), true);
        Tensor wrongPacked = tensor(DataType.FLOAT32, Shape.of(15, 4), true);
        Tensor wrongHiddenSize = tensor(DataType.FLOAT32, Shape.of(12, 5), true);
        Tensor integralBias = tensor(DataType.INT32, Shape.of(12), false);
        Tensor noGradientBias = tensor(DataType.FLOAT32, Shape.of(12), false);
        Tensor dynamicBias = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P")), true);
        Tensor wrongTypeBias = tensor(DataType.FLOAT64, Shape.of(12), true);
        Tensor wrongBias = tensor(DataType.FLOAT32, Shape.of(11), true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertContains("hiddenWeight", "floating",
                        () -> new GruCell(inputWeight, integralHidden)),
                () -> assertContains("hiddenWeight", "requiresGrad",
                        () -> new GruCell(inputWeight, noGradientHidden)),
                () -> assertContains("hiddenWeight", "rank two",
                        () -> new GruCell(inputWeight, scalar)),
                () -> assertContains("hiddenWeight", "fully static",
                        () -> new GruCell(inputWeight, dynamicHidden)),
                () -> assertContains("axis zero", "positive",
                        () -> new GruCell(inputWeight, zeroAxisZero)),
                () -> assertContains("axis one", "positive",
                        () -> new GruCell(inputWeight, zeroAxisOne)),
                () -> assertContains("hiddenWeight", "data type",
                        () -> new GruCell(inputWeight, wrongTypeHidden)),
                () -> assertContains("axis zero", "packed extent",
                        () -> new GruCell(inputWeight, wrongPacked)),
                () -> assertContains("axis one", "hidden size",
                        () -> new GruCell(inputWeight, wrongHiddenSize)),
                () -> assertContains("bias", "floating",
                        () -> new GruCell(inputWeight, validHidden, integralBias)),
                () -> assertContains("bias", "requiresGrad",
                        () -> new GruCell(inputWeight, validHidden, noGradientBias)),
                () -> assertContains("bias", "rank one",
                        () -> new GruCell(inputWeight, validHidden, scalar)),
                () -> assertContains("bias", "fully static",
                        () -> new GruCell(inputWeight, validHidden, dynamicBias)),
                () -> assertContains("bias", "data type",
                        () -> new GruCell(inputWeight, validHidden, wrongTypeBias)),
                () -> assertContains("bias", "packed hidden size",
                        () -> new GruCell(inputWeight, validHidden, wrongBias)),
                () -> assertEquals(before, ids.get()));
    }

    private static void assertGruChain(
            Tensor result,
            Tensor input,
            Tensor hidden,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias) {
        TensorProvenance finalAdd = result.provenance().orElseThrow();
        Tensor candidate = finalAdd.inputs().getFirst();
        Tensor weightedDifference = finalAdd.inputs().get(1);
        TensorProvenance weightedMul = weightedDifference.provenance().orElseThrow();
        Tensor update = weightedMul.inputs().getFirst();
        Tensor difference = weightedMul.inputs().get(1);
        TensorProvenance differenceSub = difference.provenance().orElseThrow();
        TensorProvenance candidateTanh = candidate.provenance().orElseThrow();
        Tensor candidateAdd = candidateTanh.inputs().getFirst();
        TensorProvenance candidateAddition = candidateAdd.provenance().orElseThrow();
        Tensor inputCandidate = candidateAddition.inputs().getFirst();
        Tensor resetProduct = candidateAddition.inputs().get(1);
        TensorProvenance resetMul = resetProduct.provenance().orElseThrow();
        Tensor reset = resetMul.inputs().getFirst();
        Tensor hiddenCandidate = resetMul.inputs().get(1);
        Tensor resetAdd = reset.provenance().orElseThrow().inputs().getFirst();
        Tensor updateAdd = update.provenance().orElseThrow().inputs().getFirst();
        Tensor inputReset = resetAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenReset = resetAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputUpdate = updateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenUpdate = updateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection = inputReset.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection = hiddenReset.provenance().orElseThrow().inputs().getFirst();

        assertAll(
                () -> assertSame(BinaryArithmeticKind.ADD, finalAdd.operation().kind()),
                () -> assertSame(candidate, finalAdd.inputs().getFirst()),
                () -> assertSame(BinaryArithmeticKind.MUL, weightedMul.operation().kind()),
                () -> assertSame(BinaryArithmeticKind.SUB, differenceSub.operation().kind()),
                () -> assertSame(hidden, differenceSub.inputs().getFirst()),
                () -> assertSame(candidate, differenceSub.inputs().get(1)),
                () -> assertSame(UnaryElementwiseKind.SIGMOID,
                        reset.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.SIGMOID,
                        update.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.TANH, candidateTanh.operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL, resetMul.operation().kind()),
                () -> assertSame(inputCandidate, candidateAddition.inputs().getFirst()),
                () -> assertSame(reset, resetMul.inputs().getFirst()),
                () -> assertSame(hiddenCandidate, resetMul.inputs().get(1)),
                () -> assertSlice(inputReset, inputProjection, 0, 4),
                () -> assertSlice(inputUpdate, inputProjection, 4, 4),
                () -> assertSlice(inputCandidate, inputProjection, 8, 4),
                () -> assertSlice(hiddenReset, hiddenProjection, 0, 4),
                () -> assertSlice(hiddenUpdate, hiddenProjection, 4, 4),
                () -> assertSlice(hiddenCandidate, hiddenProjection, 8, 4),
                () -> assertLinearChain(inputProjection, input, inputWeight, bias),
                () -> assertLinearChain(hiddenProjection, hidden, hiddenWeight, Optional.empty()));
    }

    private static void assertSlice(
            Tensor slice, Tensor projection, long start, long length) {
        TensorProvenance provenance = slice.provenance().orElseThrow();
        SliceAttrs attrs = (SliceAttrs) provenance.operation().attrs();
        Shape projectionShape = projection.descriptor().shape();
        Shape sliceShape = slice.descriptor().shape();
        int finalAxis = projectionShape.rank() - 1;
        assertAll(
                () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                () -> assertSame(projection, provenance.inputs().getFirst()),
                () -> assertEquals(List.of(start), attrs.starts()),
                () -> assertEquals(List.of(length), attrs.lengths()),
                () -> assertEquals(List.of(finalAxis), attrs.axes()),
                () -> assertEquals(List.of(1L), attrs.steps()),
                () -> assertEquals(projectionShape.rank(), sliceShape.rank()),
                () -> assertEquals(new StaticDimension(length),
                        sliceShape.dimension(finalAxis)),
                () -> assertTrue(leadingDimensionsMatch(projectionShape, sliceShape)));
    }

    private static boolean leadingDimensionsMatch(Shape first, Shape second) {
        for (int axis = 0; axis < first.rank() - 1; axis++) {
            if (!first.dimension(axis).equals(second.dimension(axis))) {
                return false;
            }
        }
        return true;
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
                () -> assertEquals(List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertSame(weight, permute.inputs().getFirst()));
    }

    private static void assertIds(Tensor result, long start, boolean biased) {
        List<Tensor> ordered = orderedCreatedTensors(result, biased);
        assertEquals(biased ? 21 : 20, ordered.size());
        for (int index = 0; index < ordered.size(); index++) {
            assertEquals(start + index, ordered.get(index).id().value(), "ID at " + index);
        }
    }

    private static List<Tensor> orderedCreatedTensors(Tensor result, boolean biased) {
        Tensor candidate = result.provenance().orElseThrow().inputs().getFirst();
        Tensor weighted = result.provenance().orElseThrow().inputs().get(1);
        Tensor update = weighted.provenance().orElseThrow().inputs().getFirst();
        Tensor difference = weighted.provenance().orElseThrow().inputs().get(1);
        Tensor candidateAdd = candidate.provenance().orElseThrow().inputs().getFirst();
        Tensor resetProduct = candidateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor reset = resetProduct.provenance().orElseThrow().inputs().getFirst();
        Tensor resetAdd = reset.provenance().orElseThrow().inputs().getFirst();
        Tensor updateAdd = update.provenance().orElseThrow().inputs().getFirst();
        Tensor inputReset = resetAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenReset = resetAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputUpdate = updateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenUpdate = updateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenCandidate = resetProduct.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection = inputReset.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection = hiddenReset.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProduct = biased
                ? inputProjection.provenance().orElseThrow().inputs().getFirst()
                : inputProjection;
        Tensor inputTranspose = inputProduct.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenTranspose = hiddenProjection.provenance().orElseThrow().inputs().get(1);
        java.util.ArrayList<Tensor> tensors = new java.util.ArrayList<>();
        tensors.add(inputTranspose);
        tensors.add(inputProduct);
        if (biased) {
            tensors.add(inputProjection);
        }
        tensors.add(hiddenTranspose);
        tensors.add(hiddenProjection);
        tensors.addAll(List.of(inputReset, inputUpdate, inputCandidate,
                hiddenReset, hiddenUpdate, hiddenCandidate,
                resetAdd, reset, updateAdd, update, resetProduct,
                candidateAdd, candidate, difference, weighted, result));
        return tensors;
    }

    private static TensorProvenance finalDifference(Tensor result) {
        Tensor weighted = result.provenance().orElseThrow().inputs().get(1);
        Tensor difference = weighted.provenance().orElseThrow().inputs().get(1);
        return difference.provenance().orElseThrow();
    }

    private static GruCell cell(DataType dataType, boolean bias) {
        Tensor inputWeight = tensor(dataType, Shape.of(12, 3), true);
        Tensor hiddenWeight = tensor(dataType, Shape.of(12, 4), true);
        return bias
                ? new GruCell(inputWeight, hiddenWeight,
                        tensor(dataType, Shape.of(12), true))
                : new GruCell(inputWeight, hiddenWeight);
    }

    private static List<String> names(GruCell cell) {
        return cell.parameters().stream().map(Parameter::name).toList();
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

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    @FunctionalInterface
    private interface ThrowingConstruction {
        void run();
    }

    private static final class Owner extends Module {
        private Owner(GruCell cell) {
            child("cell", cell);
        }
    }
}
