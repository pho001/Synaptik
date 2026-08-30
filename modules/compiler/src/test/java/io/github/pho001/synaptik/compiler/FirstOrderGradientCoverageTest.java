package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.ordering.OrderingKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool2dAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.BatchNormTrainingResult;
import io.github.pho001.synaptik.model.tensor.DropoutResult;
import io.github.pho001.synaptik.model.tensor.GraphRngState;
import io.github.pho001.synaptik.model.tensor.ScaledDotProductAttentionResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TopKResult;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.jar.JarFile;
import org.junit.jupiter.api.Test;

final class FirstOrderGradientCoverageTest {
    private static final String OPERATION_PACKAGE =
            "io.github.pho001.synaptik.model.operation";
    private static final String OPERATION_PATH =
            "io/github/pho001/synaptik/model/operation/";

    @Test
    void supportedAndDeferredInventoriesPartitionCompleteProductionModelInventory()
            throws Exception {
        Set<Class<? extends OperationKind>> families = discoverKindFamilies();
        Set<FirstOrderGradientCoverage.SignatureFingerprint> discovered = new HashSet<>();
        Set<OperationKind> kinds = new HashSet<>();
        for (Class<? extends OperationKind> family : families) {
            Object[] constants = family.getEnumConstants();
            assertNotNull(constants, family.getName());
            for (Object constant : constants) {
                OperationKind kind = (OperationKind) constant;
                assertTrue(kinds.add(kind), kind.toString());
                for (OperationSignature signature : kind.signatures()) {
                    assertTrue(discovered.add(new FirstOrderGradientCoverage.SignatureFingerprint(
                            kind,
                            signature.attributesType(),
                            signature.minimumInputs(),
                            signature.maximumInputs(),
                            signature.minimumOutputs(),
                            signature.maximumOutputs())));
                }
            }
        }

        Set<FirstOrderGradientCoverage.SignatureFingerprint> supported =
                new HashSet<>(FirstOrderGradientCoverage.signatures());
        Set<Class<?>> supportedFamilies = supported.stream()
                .map(signature -> signature.kind().getClass())
                .collect(java.util.stream.Collectors.toSet());
        Set<OperationKind> supportedKinds = supported.stream()
                .map(FirstOrderGradientCoverage.SignatureFingerprint::kind)
                .collect(java.util.stream.Collectors.toSet());
        assertEquals(
                FirstOrderGradientCoverage.signatures().size(),
                supported.size(),
                "checker inventory contains a duplicate exact row");
        assertEquals(38, supportedFamilies.size());
        assertEquals(111, supportedKinds.size());
        assertEquals(133, supported.size());

        Set<FirstOrderGradientCoverage.SignatureFingerprint> deferred =
                deferredSignatures();
        assertEquals(4, deferred.size());
        Set<FirstOrderGradientCoverage.SignatureFingerprint> overlap =
                new HashSet<>(supported);
        overlap.retainAll(deferred);
        assertTrue(overlap.isEmpty(), "deferred signatures became compiler-supported");

        Set<FirstOrderGradientCoverage.SignatureFingerprint> complete =
                new HashSet<>(supported);
        complete.addAll(deferred);
        assertEquals(discovered, complete);
        assertEquals(40, families.size());
        assertEquals(115, kinds.size());
        assertEquals(137, discovered.size());
    }

    @Test
    void failsClosedForEveryDeferredSignatureWithoutAllocation() throws Exception {
        long before = nextTensorId();
        for (FirstOrderGradientCoverage.SignatureFingerprint signature
                : deferredSignatures()) {
            for (int inputCount : boundaryCounts(
                    signature.minimumInputs(), signature.maximumInputs())) {
                for (int outputIndex = 0;
                        outputIndex < signature.minimumOutputs();
                        outputIndex++) {
                    assertDeferredFailClosed(FirstOrderGradientCoverage.classify(
                            signature.kind(),
                            signature.attributesType(),
                            inputCount,
                            signature.minimumOutputs(),
                            outputIndex,
                            -1,
                            null,
                            DataType.FLOAT32));
                    for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
                        assertDeferredFailClosed(FirstOrderGradientCoverage.classify(
                                signature.kind(),
                                signature.attributesType(),
                                inputCount,
                                signature.minimumOutputs(),
                                outputIndex,
                                inputIndex,
                                DataType.FLOAT32,
                                DataType.FLOAT32));
                    }
                }
            }
        }
        assertEquals(before, nextTensorId());
    }

    @Test
    void classifiesEveryLegalCheckpointRoleAndRangedBoundaryExactlyOnce() {
        int differentiable = 0;
        int nonDifferentiable = 0;
        int failClosed = 0;
        for (FirstOrderGradientCoverage.SignatureFingerprint signature
                : FirstOrderGradientCoverage.signatures()) {
            List<Integer> inputCounts = boundaryCounts(
                    signature.minimumInputs(), signature.maximumInputs());
            List<Integer> outputCounts = boundaryCounts(
                    signature.minimumOutputs(), signature.maximumOutputs());
            for (int inputCount : inputCounts) {
                for (int outputCount : outputCounts) {
                    for (int outputIndex = 0; outputIndex < outputCount; outputIndex++) {
                        FirstOrderGradientCoverage.Decision output =
                                FirstOrderGradientCoverage.classify(
                                        signature.kind(),
                                        signature.attributesType(),
                                        inputCount,
                                        outputCount,
                                        outputIndex,
                                        -1,
                                        null,
                                        DataType.FLOAT32);
                        assertWellFormed(output);
                        assertEquals(
                                expectedOutputDisposition(signature.kind(), outputIndex),
                                output.disposition(),
                                signature + " output " + outputIndex);
                        for (int inputIndex = 0; inputIndex < inputCount; inputIndex++) {
                            FirstOrderGradientCoverage.Decision decision =
                                    FirstOrderGradientCoverage.classify(
                                            signature.kind(),
                                            signature.attributesType(),
                                            inputCount,
                                            outputCount,
                                            outputIndex,
                                            inputIndex,
                                            DataType.FLOAT32,
                                            DataType.FLOAT32);
                            assertWellFormed(decision);
                            assertEquals(
                                    expectedInputDisposition(
                                            signature.kind(),
                                            outputCount,
                                            outputIndex,
                                            inputIndex,
                                            DataType.FLOAT32,
                                            DataType.FLOAT32),
                                    decision.disposition(),
                                    signature + " output " + outputIndex
                                            + " input " + inputIndex);
                            switch (decision.disposition()) {
                                case D -> differentiable++;
                                case ND -> nonDifferentiable++;
                                case FC -> failClosed++;
                            }
                        }
                    }
                }
            }
        }
        assertTrue(differentiable > 0);
        assertTrue(nonDifferentiable > 0);
        assertTrue(failClosed > 0);
    }

    @Test
    void failsClosedForUnknownMalformedAndIllegalFactsWithoutAllocation() throws Exception {
        long before = nextTensorId();
        OperationKind unknown = new OperationKind() {
            @Override
            public String name() {
                return "UNKNOWN";
            }

            @Override
            public List<OperationSignature> signatures() {
                return List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 1));
            }
        };
        assertFailClosed(FirstOrderGradientCoverage.classify(
                unknown,
                NoOperationAttrs.class,
                1,
                1,
                0,
                0,
                DataType.FLOAT32,
                DataType.FLOAT32));

        var known = FirstOrderGradientCoverage.signatures().getFirst();
        assertFailClosed(FirstOrderGradientCoverage.classify(
                known.kind(),
                known.attributesType(),
                known.minimumInputs() - 1,
                known.minimumOutputs(),
                0,
                0,
                DataType.FLOAT32,
                DataType.FLOAT32));
        assertFailClosed(FirstOrderGradientCoverage.classify(
                known.kind(),
                known.attributesType(),
                known.minimumInputs(),
                known.minimumOutputs(),
                known.minimumOutputs(),
                -1,
                null,
                DataType.FLOAT32));
        assertFailClosed(FirstOrderGradientCoverage.classify(
                known.kind(),
                known.attributesType(),
                known.minimumInputs(),
                known.minimumOutputs(),
                0,
                known.minimumInputs(),
                DataType.FLOAT32,
                DataType.FLOAT32));
        assertEquals(before, nextTensorId());
    }

    @Test
    void classifiesNonFloatingDataAndFixedPolicyRolesAsNonDifferentiable() {
        for (FirstOrderGradientCoverage.SignatureFingerprint signature
                : FirstOrderGradientCoverage.signatures()) {
            if (signature.minimumInputs() == 0) {
                continue;
            }
            for (int input = 0; input < signature.minimumInputs(); input++) {
                FirstOrderGradientCoverage.Decision decision =
                        FirstOrderGradientCoverage.classify(
                                signature.kind(),
                                signature.attributesType(),
                                signature.minimumInputs(),
                                signature.minimumOutputs(),
                                0,
                                input,
                                DataType.INT64,
                                DataType.FLOAT32);
                assertFalse(
                        decision.disposition() == FirstOrderGradientCoverage.Disposition.D,
                        signature + " input " + input);
                assertNull(decision.owner());
                assertFalse(decision.reason().isBlank());
            }
        }
    }

    @Test
    void closesGeneratedFormulaEdgesAndRunsOnlyIdentityConnectedNestedPasses() {
        Set<String> emitted = new HashSet<>();
        int connected = 0;
        int notApplicable = 0;
        for (ClosureCase closureCase : closureCases()) {
            AutogradPreflight.StagePlan firstPlan = AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(closureCase.objective()),
                    FunctionalGradientTestSupport.stage(
                            closureCase.objective(), List.of(closureCase.target())),
                    CompileTimeConstantGraph.Ingress.empty());
            FirstOrderAutograd.Expansion first = FirstOrderAutograd.expand(
                    firstPlan, CompileTimeConstantGraph.Ingress.empty());
            Tensor firstGradient = first.targetGradients().getFirst().gradient();
            emitted.addAll(generatedEdges(firstGradient, firstPlan.originalProducers()));

            if (!hasDifferentiableIdentityAncestry(
                    firstGradient, closureCase.target())) {
                notApplicable++;
                assertFalse(
                        hasDifferentiableIdentityAncestry(
                                firstGradient, closureCase.target()),
                        "SECOND_PASS_NOT_APPLICABLE: " + closureCase.label()
                                + " first gradient has no D-edge identity path");
                continue;
            }
            connected++;
            Tensor nestedObjective = firstGradient.sum();
            assertTrue(
                    nestedObjective.descriptor().dataType().isFloating()
                            && nestedObjective.descriptor().requiresGrad(),
                    closureCase.label());
            AutogradPreflight.StagePlan secondPlan = AutogradPreflight.preflight(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(nestedObjective),
                    FunctionalGradientTestSupport.stage(
                            nestedObjective, List.of(closureCase.target())),
                    first.ingress());
            FirstOrderAutograd.Expansion second =
                    FirstOrderAutograd.expand(secondPlan, first.ingress());
            assertSameTarget(
                    second.targetGradients().getFirst().target(), closureCase.target());
            emitted.addAll(generatedEdges(
                    second.targetGradients().getFirst().gradient(),
                    secondPlan.originalProducers()));
        }
        assertFalse(emitted.isEmpty());
        assertEquals(10, connected);
        assertEquals(12, notApplicable);
        assertEquals(64, emitted.size());
    }

    private static List<ClosureCase> closureCases() {
        List<ClosureCase> cases = new ArrayList<>();

        Tensor elementwise = tensor(Shape.of(2, 3));
        cases.add(new ClosureCase(
                "elementwise-pow-exp", elementwise.pow(elementwise).exp().sum(), elementwise));
        Tensor directZero = tensor(Shape.of(2, 3));
        cases.add(new ClosureCase("direct-zero-floor", directZero.floor().sum(), directZero));
        Tensor whereBranch = tensor(Shape.of(2, 3));
        Tensor condition = boolTensor(Shape.of(2, 3));
        cases.add(new ClosureCase(
                "where-repeated-branches",
                Tensor.where(condition, whereBranch, whereBranch).sum(),
                whereBranch));

        Tensor reduction = tensor(Shape.of(2, 3));
        cases.add(new ClosureCase("reduction-product", reduction.prod(1).sum(), reduction));
        cases.add(new ClosureCase(
                "reduction-statistics",
                reduction.standardDeviation(new int[] {1}, false, 1).sum(),
                reduction));
        cases.add(new ClosureCase(
                "cumulative-product", reduction.cumProd(1, true, true).sum(), reduction));
        cases.add(new ClosureCase(
                "softmax", reduction.softmax(1).mul(reduction).sum(), reduction));

        Tensor channelInput = tensor(Shape.of(2, 3, 4));
        Tensor channelVector = tensor(Shape.of(3));
        BatchNormTrainingResult training = channelInput.batchNormTraining(
                1,
                channelVector,
                channelVector,
                channelVector,
                channelVector,
                ScalarValue.float32(0.1f),
                ScalarValue.float32(1.0e-5f));
        cases.add(new ClosureCase(
                "batch-normalization-training",
                training.output().sum()
                        .add(training.nextRunningMean().sum())
                        .add(training.nextRunningVariance().sum()),
                channelInput));

        Tensor matrix = tensor(Shape.of(2, 2));
        cases.add(new ClosureCase("matmul-repeated", matrix.matmul(matrix).sum(), matrix));

        Tensor query = tensor(Shape.of(2, 3, 4));
        Tensor key = tensor(Shape.of(2, 5, 4));
        Tensor value = tensor(Shape.of(2, 5, 6));
        ScaledDotProductAttentionResult attention =
                query.scaledDotProductAttentionWithWeights(key, value);
        cases.add(new ClosureCase(
                "attention-both-outputs",
                attention.output().sum().add(attention.weights().sum()),
                query));

        Tensor image = tensor(Shape.of(2, 4, 5, 5));
        Tensor weight = tensor(Shape.of(6, 2, 3, 3));
        Tensor bias = tensor(Shape.of(6));
        cases.add(new ClosureCase(
                "grouped-convolution",
                image.conv2d(weight, bias, new Conv2dAttrs(1, 1, 1, 1, 1, 1, 2)).sum(),
                image));
        cases.add(new ClosureCase(
                "maximum-pooling",
                image.maxPool2d(new MaxPool2dAttrs(3, 3, 1, 1, 1, 1, 1, 1, false))
                        .sum(),
                image));

        Tensor lossPrediction = tensor(Shape.of(2, 3));
        Tensor lossTarget = tensor(Shape.of(2, 3));
        cases.add(new ClosureCase(
                "mean-squared-error-target",
                lossPrediction.meanSquaredError(lossTarget, LossReduction.MEAN),
                lossTarget));
        Tensor logits = tensor(Shape.of(2, 3, 4));
        Tensor indexTarget = indexTensor(Shape.of(2, 4));
        cases.add(new ClosureCase(
                "index-categorical-loss",
                logits.categoricalCrossEntropyWithLogits(
                        indexTarget, 1, LossReduction.MEAN, ScalarValue.int64(-1)),
                logits));

        Tensor sliceBase = tensor(Shape.of(5));
        Tensor sliceUpdate = tensor(Shape.of(2));
        cases.add(new ClosureCase(
                "slice-update-base",
                sliceBase.sliceUpdate(
                                sliceUpdate,
                                new long[] {4},
                                new int[] {0},
                                new long[] {-2})
                        .sum(),
                sliceBase));
        Tensor windowInput = tensor(Shape.of(1, 1, 3, 3));
        cases.add(new ClosureCase(
                "unfold2d",
                windowInput.unfold2d(
                                new io.github.pho001.synaptik.model.operation.layout.Window2dAttrs(
                                        2, 2, 1, 1, 0, 0, 1, 1, false))
                        .sum(),
                windowInput));

        Tensor indexedData = tensor(Shape.of(3));
        Tensor indices = indexTensor(Shape.of(2));
        Tensor updates = tensor(Shape.of(2));
        cases.add(new ClosureCase(
                "gather-elements", indexedData.gatherElements(indices, 0).sum(), indexedData));
        cases.add(new ClosureCase(
                "scatter-elements-max",
                indexedData
                        .scatterElements(indices, updates, 0, ScatterReduction.MAX)
                        .sum(),
                updates));

        Tensor ordered = tensor(Shape.of(3));
        cases.add(new ClosureCase("sort", ordered.sort(0, true).sum(), ordered));
        TopKResult topK = ordered.topK(2, 0, true, false);
        cases.add(new ClosureCase("top-k", topK.values().sum(), ordered));
        DropoutResult dropout = ordered.dropout(0.25d, GraphRngState.initial(7L, 11L));
        cases.add(new ClosureCase("dropout", dropout.output().sum(), ordered));

        Tensor linear = tensor(Shape.of(2, 3));
        cases.add(new ClosureCase("linear-sum", linear.sum(), linear));
        return List.copyOf(cases);
    }

    private static Set<String> generatedEdges(
            Tensor root, Set<TensorProducer> originalProducers) {
        Set<String> emitted = new HashSet<>();
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (seen.put(tensor, Boolean.TRUE) != null) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> {
                TensorProducer producer = provenance.producer();
                if (!originalProducers.contains(producer)) {
                    for (int input = 0; input < producer.inputs().size(); input++) {
                        FirstOrderGradientCoverage.Decision decision =
                                FirstOrderGradientCoverage.classify(
                                        producer, provenance.outputIndex(), input);
                        assertFalse(
                                decision.disposition()
                                        == FirstOrderGradientCoverage.Disposition.FC,
                                producer.operation() + " output "
                                        + provenance.outputIndex() + " input " + input
                                        + ": " + decision.reason());
                        emitted.add(
                                producer.operation().kind().getClass().getName()
                                        + "." + producer.operation().kind().name()
                                        + "|" + producer.operation().attrs().getClass().getName()
                                        + "|" + provenance.outputIndex()
                                        + "|" + input
                                        + "|" + decision.disposition());
                    }
                }
                producer.inputs().forEach(pending::addLast);
            });
        }
        return Set.copyOf(emitted);
    }

    private static boolean hasDifferentiableIdentityAncestry(
            Tensor root, Tensor target) {
        IdentityHashMap<Tensor, Boolean> seen = new IdentityHashMap<>();
        ArrayDeque<Tensor> pending = new ArrayDeque<>();
        pending.add(root);
        while (!pending.isEmpty()) {
            Tensor tensor = pending.removeLast();
            if (tensor == target) {
                return true;
            }
            if (seen.put(tensor, Boolean.TRUE) != null) {
                continue;
            }
            tensor.provenance().ifPresent(provenance -> {
                for (int input = 0; input < provenance.inputs().size(); input++) {
                    FirstOrderGradientCoverage.Decision decision =
                            FirstOrderGradientCoverage.classify(
                                    provenance.producer(),
                                    provenance.outputIndex(),
                                    input);
                    if (decision.disposition()
                            == FirstOrderGradientCoverage.Disposition.D) {
                        pending.addLast(provenance.inputs().get(input));
                    }
                }
            });
        }
        return false;
    }

    private static void assertSameTarget(Tensor actual, Tensor expected) {
        assertTrue(actual == expected, "nested expansion must retain exact target identity");
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), true));
    }

    private static Tensor boolTensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.BOOL, shape, Optional.empty(), false));
    }

    private static Tensor indexTensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.INT64, shape, Optional.empty(), false));
    }

    private static void assertWellFormed(FirstOrderGradientCoverage.Decision decision) {
        if (decision.disposition() == FirstOrderGradientCoverage.Disposition.D) {
            assertNotNull(decision.owner());
            assertTrue(decision.reason().isEmpty());
        } else {
            assertNull(decision.owner());
            assertFalse(decision.reason().isBlank());
        }
    }

    private static FirstOrderGradientCoverage.Disposition expectedOutputDisposition(
            OperationKind kind, int outputIndex) {
        if (kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof OneHotKind
                || kind == OrderingKind.ARGSORT
                || kind instanceof GraphRngKind
                || (kind instanceof TopKKind && outputIndex == 1)
                || (kind instanceof DropoutKind && outputIndex != 0)) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if (kind == BatchNormKind.BATCH_NORM_TRAINING && outputIndex >= 3) {
            return FirstOrderGradientCoverage.Disposition.FC;
        }
        return FirstOrderGradientCoverage.Disposition.D;
    }

    private static FirstOrderGradientCoverage.Disposition expectedInputDisposition(
            OperationKind kind,
            int outputCount,
            int outputIndex,
            int inputIndex,
            DataType inputType,
            DataType outputType) {
        if (kind instanceof BinaryComparisonKind
                || kind instanceof BooleanLogicalKind
                || kind instanceof FloatingClassificationKind
                || kind instanceof OneHotKind
                || kind == OrderingKind.ARGSORT
                || kind instanceof GraphRngKind) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if (kind == WhereSelectionKind.WHERE && inputIndex == 0) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if (kind instanceof AggregateReductionKind reduction) {
            if (reduction == AggregateReductionKind.ALL
                    || reduction == AggregateReductionKind.ANY
                    || reduction == AggregateReductionKind.ARG_MIN
                    || reduction == AggregateReductionKind.ARG_MAX
                    || inputIndex != 0) {
                return FirstOrderGradientCoverage.Disposition.ND;
            }
        }
        if ((kind instanceof AxisGatherKind || kind instanceof GatherNdKind)
                && inputIndex != 0) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if ((kind instanceof AxisScatterKind || kind instanceof ScatterNdKind)
                && inputIndex == 1) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if (kind == BatchNormKind.BATCH_NORM_TRAINING) {
            boolean differentiable = switch (outputIndex) {
                case 0 -> inputIndex <= 2;
                case 1 -> inputIndex == 0 || inputIndex == 3;
                case 2 -> inputIndex == 0 || inputIndex == 4;
                default -> false;
            };
            if (!differentiable) {
                return outputIndex >= 3
                        ? FirstOrderGradientCoverage.Disposition.FC
                        : FirstOrderGradientCoverage.Disposition.ND;
            }
        }
        if (kind == ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION) {
            if (inputIndex == 3 || (outputIndex == 1 && inputIndex >= 2)) {
                return FirstOrderGradientCoverage.Disposition.ND;
            }
            if (outputCount == 1) {
                return FirstOrderGradientCoverage.Disposition.FC;
            }
        }
        if (kind == LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                && inputIndex == 1) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if ((kind instanceof TopKKind && outputIndex == 1)
                || (kind instanceof DropoutKind && (outputIndex != 0 || inputIndex != 0))) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if (kind == CastKind.CAST
                && (!inputType.isFloating() || !outputType.isFloating())) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        if ((kind == OrderingKind.SORT || kind instanceof CumulativeScanKind)
                && !inputType.isFloating()) {
            return FirstOrderGradientCoverage.Disposition.ND;
        }
        return inputType.isFloating()
                ? FirstOrderGradientCoverage.Disposition.D
                : FirstOrderGradientCoverage.Disposition.ND;
    }

    private static void assertFailClosed(FirstOrderGradientCoverage.Decision decision) {
        assertEquals(FirstOrderGradientCoverage.Disposition.FC, decision.disposition());
        assertNull(decision.owner());
        assertFalse(decision.reason().isBlank());
    }

    private static void assertDeferredFailClosed(
            FirstOrderGradientCoverage.Decision decision) {
        assertFailClosed(decision);
        assertEquals(
                "unknown or unclassified operation kind/attributes pairing",
                decision.reason());
    }

    private static Set<FirstOrderGradientCoverage.SignatureFingerprint>
            deferredSignatures() {
        return Set.of(
                recurrentSignature(RecurrentScanKind.RNN_TANH, 5, 6, 2),
                recurrentSignature(RecurrentScanKind.GRU_RESET_AFTER, 5, 6, 2),
                recurrentSignature(RecurrentScanKind.LSTM, 6, 7, 3),
                new FirstOrderGradientCoverage.SignatureFingerprint(
                        Conv3dKind.CONV3D,
                        Conv3dAttrs.class,
                        2,
                        3,
                        1,
                        1));
    }

    private static FirstOrderGradientCoverage.SignatureFingerprint fixedSignature(
            OperationKind kind, Class<? extends OperationAttrs> attributesType) {
        return new FirstOrderGradientCoverage.SignatureFingerprint(
                kind, attributesType, 1, 1, 1, 1);
    }

    private static FirstOrderGradientCoverage.SignatureFingerprint recurrentSignature(
            RecurrentScanKind kind, int minimumInputs, int maximumInputs, int outputCount) {
        return new FirstOrderGradientCoverage.SignatureFingerprint(
                kind,
                RecurrentDirection.class,
                minimumInputs,
                maximumInputs,
                outputCount,
                outputCount);
    }

    private static List<Integer> boundaryCounts(int minimum, int maximum) {
        if (minimum == maximum) {
            return List.of(minimum);
        }
        return List.of(minimum, maximum == Integer.MAX_VALUE ? minimum + 2 : maximum);
    }

    @SuppressWarnings("unchecked")
    private static Set<Class<? extends OperationKind>> discoverKindFamilies() throws Exception {
        URI location = OperationKind.class
                .getProtectionDomain()
                .getCodeSource()
                .getLocation()
                .toURI();
        Path root = Path.of(location);
        List<String> classNames = Files.isDirectory(root)
                ? classesFromDirectory(root)
                : classesFromArchive(root);
        Set<Class<? extends OperationKind>> families = new HashSet<>();
        ClassLoader loader = OperationKind.class.getClassLoader();
        for (String className : classNames) {
            Class<?> candidate = Class.forName(className, false, loader);
            if (candidate.isEnum()
                    && OperationKind.class.isAssignableFrom(candidate)
                    && candidate.getEnclosingClass() == null) {
                families.add((Class<? extends OperationKind>) candidate);
            }
        }
        return Set.copyOf(families);
    }

    private static List<String> classesFromDirectory(Path root) throws IOException {
        Path operationRoot = root.resolve(OPERATION_PATH);
        List<String> names = new ArrayList<>();
        try (var paths = Files.walk(operationRoot)) {
            paths.filter(Files::isRegularFile)
                    .map(operationRoot::relativize)
                    .map(Path::toString)
                    .filter(name -> name.endsWith(".class") && !name.contains("$"))
                    .map(name -> OPERATION_PACKAGE + "."
                            + name.substring(0, name.length() - ".class".length())
                                    .replace('/', '.')
                                    .replace('\\', '.'))
                    .forEach(names::add);
        }
        return List.copyOf(names);
    }

    private static List<String> classesFromArchive(Path archive) throws IOException {
        List<String> names = new ArrayList<>();
        try (JarFile jar = new JarFile(archive.toFile())) {
            jar.stream()
                    .map(entry -> entry.getName())
                    .filter(name -> name.startsWith(OPERATION_PATH)
                            && name.endsWith(".class")
                            && !name.contains("$"))
                    .map(name -> name.substring(0, name.length() - ".class".length())
                            .replace('/', '.'))
                    .forEach(names::add);
        }
        return List.copyOf(names);
    }

    private static long nextTensorId() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
    }

    private record ClosureCase(String label, Tensor objective, Tensor target) {}
}
