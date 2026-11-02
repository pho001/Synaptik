package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import static io.github.pho001.synaptik.compiler.SymbolicScanCompilerTest.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Structural Compiler evidence only: no length-mode execution oracle is used here. */
final class LengthAwareScanBodyInferenceTest {
    @Test void arbitrarySavedRowHeadersQualifyBeforeDerivativeIdentities() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1, 2}) for (long time : new long[] {0, 3}) {
                Tensor h = tensor(type, true, batch, 2), x = tensor(type, true, batch, 2);
                Tensor next = h.expandDims(0).tanh().squeeze(0);
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of());
                var source = Scan.apply(List.of(tensor(type, true, batch, 2)),
                        List.of(tensor(type, true, time, batch, 2)), List.of(),
                        tensor(DataType.INT64, false, batch), 1, List.of(OptionalInt.of(0)), body, false);
                var producer = source.finalCarry().getFirst().provenance().orElseThrow().producer();
                Tensor marker = tensor(DataType.BOOL, false);
                var sourcePlan = ScanGradientRules.preflightSource(producer);
                assertDoesNotThrow(() -> ScanGradientRules.plan(sourcePlan,
                        new ScanGradientRules.Demand(List.of(new ScanGradientRules.Pair(0, 0)))));
                assertEquals(marker.id().value() + 1, tensor(DataType.BOOL, false).id().value());
            }
    }

    @Test void emptyCarryAndZeroRowLengthSourceInferWhileColdAdmissionRemainsIndependent() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1}) for (long time : new long[] {0, 1})
                for (boolean reverse : List.of(false, true)) for (int topology = 0; topology < 3; topology++) {
                    Tensor x = tensor(type, true, batch, 2), old = tensor(type, false, 2);
                    boolean shared = topology == 2;
                    var body = new ScanBody(shared ? List.of(old) : List.of(), List.of(x), List.of(),
                            shared ? List.of(old.add(x.sum(0, false))) : List.of(), List.of(x));
                    var carry = shared ? List.of(tensor(type, false, 2)) : List.<Tensor>of();
                    var sequences = List.of(tensor(type, true, time, batch, 2));
                    var result = topology == 0 ? Scan.apply(carry, sequences, List.of(), body, reverse)
                            : Scan.apply(carry, sequences, List.of(), tensor(DataType.INT64, false, batch),
                                    0, List.of(OptionalInt.of(0)), body, reverse);
                    Tensor root = result.outputs().getFirst();
                    var raw = GraphCapture.capture(List.of(root));
                    var node = raw.nodes().getLast();
                    var inputs = node.inputs().stream().map(id -> raw.values().stream()
                            .filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor()).toList();
                    // The raw inference and cold owners are reached independently, not via compile.
                    assertDoesNotThrow(() -> ScanBodyInference.infer(node, inputs));
                    assertThrows(IllegalArgumentException.class, () -> SourceGraphReconstruction.preflight(raw));
                }
    }

    @Test void broaderForwardCoordinatesDoNotWidenColdAdmissionOrRestrictSharedOrdinaryWork() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1, 3}) for (long time : new long[] {0, 2}) {
                Tensor h = tensor(type, false, batch, 2), x = tensor(type, false, batch, 2);
                Tensor shared = tensor(type, false, 2, 2, 2), weight = tensor(type, false, 2, 2, 2);
                for (Tensor sharedResult : List.of(shared.relu(), shared.matmul(weight))) {
                    Tensor next = h.add(x).add(sharedResult.sum());
                    var body = new ScanBody(List.of(h), List.of(x), List.of(shared, weight), List.of(next), List.of(next));
                    var source = Scan.apply(List.of(tensor(type, false, batch, 2)),
                            List.of(tensor(type, false, time, batch, 2)),
                            List.of(tensor(type, false, 2, 2, 2), tensor(type, false, 2, 2, 2)), body, false);
                    var raw = lengthIR(GraphCapture.capture(source.finalCarry()), 1, List.of(OptionalInt.of(0)),
                            tensor(DataType.INT64, false, batch).descriptor());
                    Tensor marker = tensor(type, false);
                    assertTrue(CapturedGraphInference.inferAndValidate(raw).constraints().isEmpty());
                    assertThrows(IllegalArgumentException.class, () -> SourceGraphReconstruction.preflight(raw));
                    Tensor after = tensor(type, false);
                    assertEquals(marker.id().value() + 1, after.id().value());
                    // Exercise AD through a newly source-valid length occurrence independently.
                    var length = Scan.apply(List.of(tensor(type, false, batch, 2)),
                            List.of(tensor(type, false, time, batch, 2)),
                            List.of(tensor(type, false, 2, 2, 2), tensor(type, false, 2, 2, 2)),
                            tensor(DataType.INT64, false, batch), 1, List.of(OptionalInt.of(0)), body, false);
                    var producer = length.finalCarry().getFirst().provenance().orElseThrow().producer();
                    assertDoesNotThrow(() -> ScanGradientRules.preflightSource(producer));
                }
            }
    }

    @Test void completeReshapeWitnessRetainsOldSingletonColdRoundtripWithoutSpecialCases() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1, 3}) for (long time : new long[] {0, 2}) {
                Tensor h = tensor(type, false, batch, 2), x = tensor(type, false, batch, 2);
                Tensor next = h.reshape(io.github.pho001.synaptik.model.shape.Shape.of(batch, 1, 2)).sum(1).add(x);
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(next), List.of(h));
                var scan = Scan.apply(List.of(tensor(type, false, batch, 2)),
                        List.of(tensor(type, false, time, batch, 2)), List.of(),
                        tensor(DataType.INT64, false, batch), 1, List.of(OptionalInt.of(0)), body, false);
                var captured = GraphCompilationPort.captureSource(scan.finalCarry());
                var rebuilt = GraphCompilationPort.reconstructSource(captured.graph(), captured.leaves());
                assertEquals(captured.graph(), GraphCompilationPort.captureSource(rebuilt).graph());
            }
    }

    @Test void arbitraryIntermediateRowAxisSupportsActualGroupedConvolutionGradientTopology() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64))
            for (long batch : new long[] {0, 1, 3}) {
                Tensor h = tensor(type, false, batch, 2, 3, 4), x = tensor(type, false, batch, 2, 3, 4);
                Tensor old = tensor(type, false, 2, 3, 5), weight = tensor(type, false, 2, 1, 4, 5);
                Tensor contribution = x.permute(1, 0, 2, 3).matmul(weight).sum(1);
                var body = new ScanBody(List.of(h, old), List.of(x), List.of(weight),
                        List.of(h.add(x), old.add(contribution)), List.of(h));
                var source = Scan.apply(List.of(tensor(type, false, batch, 2, 3, 4), tensor(type, false, 2, 3, 5)),
                        List.of(tensor(type, false, 0, batch, 2, 3, 4)), List.of(tensor(type, false, 2, 1, 4, 5)),
                        tensor(DataType.INT64, false, batch), 1, List.of(OptionalInt.of(0)), body, false);
                var raw = GraphCapture.capture(source.finalCarry());
                assertTrue(CapturedGraphInference.inferAndValidate(raw).constraints().isEmpty());
                assertThrows(IllegalArgumentException.class, () -> SourceGraphReconstruction.preflight(raw));
            }
    }

    @Test void outputSpecificRoutesCannotHideSyntacticSharedCarryUsesInRawInputPositions() {
        Tensor h = tensor(DataType.FLOAT64, false, 2, 2), x = tensor(DataType.FLOAT64, false, 2, 2);
        Tensor old = tensor(DataType.FLOAT64, false, 2), data = tensor(DataType.FLOAT64, false, 3, 2);
        Tensor scale = tensor(DataType.FLOAT64, false, 2), bias = tensor(DataType.FLOAT64, false, 2);
        Tensor variance = tensor(DataType.FLOAT64, false, 2);
        var training = data.batchNormTraining(1, scale, bias, old, variance,
                io.github.pho001.synaptik.model.datatype.ScalarValue.float64(0.1),
                io.github.pho001.synaptik.model.datatype.ScalarValue.float64(0.01));
        Tensor savedMean = training.output().provenance().orElseThrow().producer().output(3);
        var body = new ScanBody(List.of(h, old), List.of(x), List.of(data, scale, bias, variance),
                List.of(h, old.add(savedMean)), List.of());
        var source = Scan.apply(List.of(tensor(DataType.FLOAT64, false, 2, 2), tensor(DataType.FLOAT64, false, 2)),
                List.of(tensor(DataType.FLOAT64, false, 0, 2, 2)),
                List.of(tensor(DataType.FLOAT64, false, 3, 2), tensor(DataType.FLOAT64, false, 2),
                        tensor(DataType.FLOAT64, false, 2), tensor(DataType.FLOAT64, false, 2)), body, false);
        var raw = lengthIR(GraphCapture.capture(source.finalCarry()), 1, List.of(OptionalInt.of(0)),
                tensor(DataType.INT64, false, 2).descriptor());
        assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(raw));
    }

    /**
     * Constructs newly valid source without invoking any Compiler consumer.
     * @param type non-F32 interior or unused-slot dtype
     * @param batch static or named row Dimension
     * @param time fixed time count, including zero
     * @param unusedRole -2 for homogeneous identity, -1 for a cast interior, otherwise unused carry/sequence/invariant (0/1/2)
     * @return source with exact typed declarations; live boundaries are F32 except in the homogeneous case
     */
    static ScanResult typedLengthSource(DataType type, io.github.pho001.synaptik.model.shape.Dimension batch,
            long time, int unusedRole) {
        var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch, fixed(2));
        var layout = row.isFullyStatic() ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(row)) : Optional.<io.github.pho001.synaptik.model.layout.LayoutDescriptor>empty();
        DataType boundary = unusedRole == -2 ? type : DataType.FLOAT32;
        Tensor h = TensorFactory.create(new TensorDescriptor(boundary, row, layout, boundary.isDifferentiable()));
        Tensor x = TensorFactory.create(h.descriptor());
        Tensor unused = TensorFactory.create(new TensorDescriptor(type, unusedRole == 2 ? io.github.pho001.synaptik.model.shape.Shape.of(2) : row,
                unusedRole == 2 ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(io.github.pho001.synaptik.model.shape.Shape.of(2))) : layout, false));
        var carries = unusedRole == 0 ? List.of(h, unused) : List.of(h);
        var steps = unusedRole == 1 ? List.of(x, unused) : List.of(x);
        var invariants = unusedRole == 2 ? List.of(unused) : List.<Tensor>of();
        Tensor next = unusedRole == -2 ? h : unusedRole == -1 ? h.cast(type).cast(DataType.FLOAT32).add(x).mul(x) : h.add(x);
        var body = new ScanBody(carries, steps, invariants, unusedRole == 0 ? List.of(next, unused) : List.of(next), List.of(next));
        var sequences = steps.stream().map(t -> {
            var shape = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(fixed(time), batch, fixed(2));
            return TensorFactory.create(new TensorDescriptor(t.descriptor().dataType(), shape, shape.isFullyStatic()
                    ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)) : Optional.empty(), t.descriptor().requiresGrad()));
        }).toList();
        var lengthsShape = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch);
        Tensor lengths = TensorFactory.create(new TensorDescriptor(DataType.INT64, lengthsShape, lengthsShape.isFullyStatic()
                ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(lengthsShape)) : Optional.empty(), false));
        return Scan.apply(carries.stream().map(t -> TensorFactory.create(t.descriptor())).toList(), sequences,
                invariants.stream().map(t -> TensorFactory.create(t.descriptor())).toList(), lengths, carries.size(),
                Collections.nCopies(steps.size(), OptionalInt.of(0)), body, false);
    }

    @Test void perValueTypedSourceInfersHiddenAndUnusedTypesIndependently() {
        for (var batch : List.of(fixed(0), fixed(2), B)) for (long time : new long[] {0, 3})
            for (DataType type : DataType.values())
                for (int role = -2; role < 3; role++) {
                    var source = typedLengthSource(type, batch, time, role);
                    var raw = GraphCapture.capture(source.finalCarry());
                    var node = raw.nodes().getFirst();
                    var inputs = node.inputs().stream().map(id -> raw.values().stream().filter(v -> v.id().equals(id))
                            .findFirst().orElseThrow().descriptor()).toList();
                    assertDoesNotThrow(() -> ScanBodyInference.infer(node, inputs));
                }
    }

    @Test void unusedTypedFormalMustMatchItsActualEvenInEmptyDomainsBeforeAnyTensorAllocation() {
        for (DataType type : DataType.values()) for (long time : new long[] {0, 3}) {
            var source = typedLengthSource(type, fixed(0), time, 2);
            var raw = GraphCapture.capture(source.finalCarry());
            var node = raw.nodes().getFirst();
            var body = node.scanBody().orElseThrow();
            ValueId unused = body.graph().inputs().getLast();
            DataType wrong = type == DataType.BOOL ? DataType.INT64 : DataType.BOOL;
            var values = body.graph().values().stream().map(v -> v.id().equals(unused)
                    ? new GraphValue(v.id(), new TensorDescriptor(wrong, v.descriptor().shape(),
                            v.descriptor().layout(), false)) : v).toList();
            var badBody = new CompiledScanBody(new CompiledGraphModel(values, body.graph().nodes(),
                    body.graph().inputs(), body.graph().outputs(), body.graph().nodePhases()),
                    body.nextCarry(), body.emissions(), body.residuals(), List.of());
            var badNode = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(badBody));
            var bad = new CompiledGraphModel(raw.values(), List.of(badNode), raw.inputs(), raw.outputs(), raw.nodePhases());
            Tensor marker = tensor(DataType.BOOL, false);
            assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(bad));
            assertEquals(marker.id().value() + 1, tensor(DataType.BOOL, false).id().value());
        }
    }

    /** Adds a length header/actual to all-valid raw IR, deliberately bypassing Scan.apply rows. */
    static CompiledGraphModel lengthIR(CompiledGraphModel raw, int rowCarries,
            List<OptionalInt> axes, TensorDescriptor lengths) {
        var old = raw.nodes().getLast();
        var a = (ScanAttrs) old.operation().attrs();
        var id = new ValueId(raw.values().stream().mapToLong(v -> v.id().value()).max().orElseThrow() + 1);
        var values = new ArrayList<>(raw.values()); values.add(new GraphValue(id, lengths));
        var actuals = new ArrayList<>(old.inputs()); actuals.add(id);
        var inputs = new ArrayList<>(raw.inputs()); inputs.add(id);
        var nodes = new ArrayList<>(raw.nodes());
        nodes.set(nodes.size() - 1, new CompiledNode(old.id(), new Operation(ScanKind.SCAN,
                new ScanAttrs(a.timeCount(), a.carryCount(), a.sequenceCount(), a.invariantCount(), a.emissionCount(), a.residualCount(), a.reverse(), true, rowCarries, axes)),
                actuals, old.outputs(), old.scanBody()));
        return new CompiledGraphModel(values, nodes, inputs, raw.outputs(), raw.nodePhases());
    }

    @Test void validatesRawLengthMetadataAndAllUnusedDeclarationsEvenInEmptyDomains() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (long batch : List.of(0L, 1L, 2L)) for (long time : List.of(0L, 3L)) {
                Tensor h = tensor(type, false, batch, 2), x = tensor(type, false, batch, 2);
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of(h));
                var scan = Scan.apply(List.of(tensor(type, false, batch, 2)),
                        List.of(tensor(type, false, time, batch, 2)), List.of(), body, false);
                var raw = GraphCapture.capture(scan.finalCarry());
                var good = lengthIR(raw, 1, List.of(OptionalInt.of(0)), tensor(DataType.INT64, false, batch).descriptor());
                assertTrue(CapturedGraphInference.inferAndValidate(good).constraints().isEmpty());
                var node = good.nodes().getFirst(); var compiledBody = node.scanBody().orElseThrow();
                var localValues = new ArrayList<>(compiledBody.graph().values());
                ValueId unused = compiledBody.graph().inputs().get(1);
                localValues.replaceAll(v -> v.id().equals(unused)
                        ? new GraphValue(v.id(), tensor(type, true, batch, 2).descriptor()) : v);
                var badLocal = new CompiledGraphModel(localValues, compiledBody.graph().nodes(), compiledBody.graph().inputs(),
                        compiledBody.graph().outputs(), compiledBody.graph().nodePhases());
                var badBody = new CompiledScanBody(badLocal, compiledBody.nextCarry(), compiledBody.emissions(), compiledBody.residuals(), List.of());
                var badNode = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(badBody));
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                        new CompiledGraphModel(good.values(), List.of(badNode), good.inputs(), good.outputs(), good.nodePhases())));
                var badTime = new ArrayList<>(good.values());
                badTime.replaceAll(v -> v.id().equals(node.inputs().get(1))
                        ? new GraphValue(v.id(), tensor(type, false, time + 1, batch, 2).descriptor()) : v);
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                        new CompiledGraphModel(badTime, good.nodes(), good.inputs(), good.outputs(), good.nodePhases())));
                for (TensorDescriptor bad : List.of(tensor(DataType.INT32, false, batch).descriptor(),
                        tensor(DataType.FLOAT32, true, batch).descriptor(), tensor(DataType.INT64, false, batch, 1).descriptor(),
                        tensor(DataType.INT64, false, batch + 1).descriptor())) {
                    assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                            lengthIR(raw, 1, List.of(OptionalInt.of(0)), bad)));
                }
                // ROW(1) declaration must be checked even when unused; do not index into empties.
                if (batch != 2) assertThrows(IllegalArgumentException.class,
                        () -> CapturedGraphInference.inferAndValidate(lengthIR(raw, 1,
                                List.of(OptionalInt.of(1)), tensor(DataType.INT64, false, batch).descriptor())));
            }
        }
    }

    @Test void rawRowFeedbackAndInvalidAccumulatorUsesRejectWithoutSourceValidation() {
        for (long batch : List.of(0L, 1L, 2L)) for (long time : List.of(0L, 2L)) {
            Tensor h = tensor(DataType.FLOAT64, false, batch, 2), x = tensor(DataType.FLOAT64, false, batch, 2);
            for (Tensor badRoot : List.of(h.sum(0, true).expand(h.descriptor().shape()),
                    h.add(x.sum(0, true).expand(h.descriptor().shape())))) {
                var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(badRoot), List.of());
                var raw = GraphCapture.capture(Scan.apply(List.of(tensor(DataType.FLOAT64, false, batch, 2)),
                        List.of(tensor(DataType.FLOAT64, false, time, batch, 2)), List.of(), body, false).finalCarry());
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                        lengthIR(raw, 1, List.of(OptionalInt.of(0)), tensor(DataType.INT64, false, batch).descriptor())));
            }
            Tensor old = tensor(DataType.FLOAT64, false, 2);
            for (Tensor root : List.of(old, old.add(old), old.add(h.sum(0, false)).add(old))) {
                var body = new ScanBody(List.of(h, old), List.of(x), List.of(), List.of(h, root), List.of());
                var raw = GraphCapture.capture(Scan.apply(List.of(tensor(DataType.FLOAT64, false, batch, 2),
                        tensor(DataType.FLOAT64, false, 2)), List.of(tensor(DataType.FLOAT64, false, time, batch, 2)),
                        List.of(), body, true).finalCarry());
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                        lengthIR(raw, 1, List.of(OptionalInt.of(0)), tensor(DataType.INT64, false, batch).descriptor())));
            }
            Tensor matrixOld = tensor(DataType.FLOAT64, false, 2, 2);
            // Broadcasting would hide a non-exact shared contribution shape without row checks.
            var broadcastBody = new ScanBody(List.of(h, matrixOld), List.of(x), List.of(),
                    List.of(h, matrixOld.add(h.sum(0, false))), List.of());
            var broadcastRaw = GraphCapture.capture(Scan.apply(List.of(tensor(DataType.FLOAT64, false, batch, 2),
                    tensor(DataType.FLOAT64, false, 2, 2)), List.of(tensor(DataType.FLOAT64, false, time, batch, 2)),
                    List.of(), broadcastBody, false).finalCarry());
            assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(
                    lengthIR(broadcastRaw, 1, List.of(OptionalInt.of(0)), tensor(DataType.INT64, false, batch).descriptor())));
        }
    }

    @Test void admitsForwardSharedSuffixAndMixedSequenceRolesWithoutInferringFromEqualExtents() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            for (long batch : List.of(0L, 1L, 2L)) {
                Tensor h = tensor(type, true, batch, 2), old = tensor(type, false, 2);
                Tensor x = tensor(type, false, 2, batch), shared = tensor(type, false, 2, 2);
                var body = new ScanBody(List.of(h, old), List.of(x, shared), List.of(),
                        List.of(h.add(x.permute(1, 0)), old.add(h.sum(0, false))), List.of(h));
                var result = Scan.apply(List.of(tensor(type, true, batch, 2), tensor(type, false, 2)),
                        List.of(tensor(type, false, 3, 2, batch), tensor(type, false, 3, 2, 2)), List.of(),
                        tensor(DataType.INT64, false, batch), 1,
                        List.of(OptionalInt.of(1), OptionalInt.empty()), body, reverse);
                for (var opt : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                    assertDoesNotThrow(() -> GraphCompiler.compile(CompileMode.FORWARD_ONLY,
                            result.finalCarry(), Optional.empty(), CompileTimeConstantGraph.Ingress.empty(), opt));
                }
            }
        }
    }

    @Test void lengthActualSitesRemainDistinctAcrossCseAndExactBypassRewrites() {
        for (boolean doubleNeg : List.of(false, true)) {
            Tensor base = tensor(DataType.FLOAT64, false, 1, 2);
            Tensor a = doubleNeg ? base.neg().neg() : base.neg();
            Tensor b = doubleNeg ? base.neg().neg() : base.neg();
            Tensor h = tensor(DataType.FLOAT64, false, 1, 2), x = tensor(DataType.FLOAT64, false, 1, 2);
            Tensor w = tensor(DataType.FLOAT64, false, 1, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.add(w)), List.of());
            var scan = Scan.apply(List.of(a), List.of(tensor(DataType.FLOAT64, false, 3, 1, 2)), List.of(b),
                    tensor(DataType.INT64, false, 1), 1, List.of(OptionalInt.of(0)), body, false);
            for (var opt : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                var graph = GraphCompiler.compile(CompileMode.FORWARD_ONLY, scan.finalCarry(), Optional.empty(),
                        CompileTimeConstantGraph.Ingress.empty(), opt).validatedGraph().graph();
                var node = graph.nodes().getLast();
                var tied = new ArrayList<>(node.inputs()); tied.set(2, tied.getFirst());
                var badNode = new CompiledNode(node.id(), node.operation(), tied, node.outputs(), node.scanBody());
                var nodes = new ArrayList<>(graph.nodes()); nodes.set(nodes.size() - 1, badNode);
                var bad = new CompiledGraphModel(graph.values(), nodes, graph.inputs(), graph.outputs(), graph.nodePhases());
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(bad));
            }
        }
    }

    @Test void discreteFirstCarryDoesNotChooseFloatingCarryOrHistoryTypes() {
        for (long batch : new long[] {0, 1, 3}) for (long time : new long[] {0, 2}) {
            Tensor control = tensor(DataType.BOOL, false, batch, 2);
            Tensor h = tensor(DataType.FLOAT64, true, batch, 2), x = fresh(h, true);
            Tensor next = h.add(x).expandDims(0).tanh().squeeze(0);
            var body = new ScanBody(List.of(control, h), List.of(x), List.of(),
                    List.of(control, next), List.of(next));
            var source = Scan.apply(List.of(fresh(control, false), fresh(h, true)),
                    List.of(tensor(DataType.FLOAT64, true, time, batch, 2)), List.of(),
                    tensor(DataType.INT64, false, batch), 2, List.of(OptionalInt.of(0)), body, false);
            var raw = GraphCapture.capture(source.finalCarry());
            assertDoesNotThrow(() -> CapturedGraphInference.inferAndValidate(raw));
            var checked = ScanGradientRules.preflightSource(
                    source.finalCarry().getFirst().provenance().orElseThrow().producer());
            Tensor seed = fresh(h, false);
            var plan = ScanGradientRules.plan(checked,
                    new ScanGradientRules.Demand(List.of(new ScanGradientRules.Pair(1, 1))));
            var result = ScanGradientRules.apply(plan, List.of(seed), new FirstOrderAutograd.DerivativeConstants());
            assertNull(result[0]);
            assertEquals(DataType.FLOAT64, result[1].descriptor().dataType());
            assertEquals(h.descriptor().shape(), result[1].descriptor().shape());
            if (time == 0) assertSame(seed, result[1]);
        }
    }

    @Test void symbolicTerminalFeedbackSharedRowInventionAndCrossAccumulatorLeakageRejectInRawInference() {
        for (var batch : List.of(B, fixed(1))) for (long time : new long[] {0, 3}) {
            Tensor h = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, shape(batch, fixed(2)),
                    batch.isStatic() ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape(batch, fixed(2))))
                            : Optional.empty(), false));
            Tensor x = fresh(h, false), bias = tensor(DataType.FLOAT64, false, 2);
            Tensor initial = fresh(h, false);
            var sq = shape(fixed(time), batch, fixed(2));
            Tensor sequence = TensorFactory.create(new TensorDescriptor(DataType.FLOAT64, sq, sq.isFullyStatic()
                    ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(sq)) : Optional.empty(), false));
            var ls = shape(batch);
            var ld = new TensorDescriptor(DataType.INT64, ls, ls.isFullyStatic()
                    ? Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(ls)) : Optional.empty(), false);
            for (Tensor root : List.of(bias.expand(h.descriptor().shape()), h.sum(0, true).expand(h.descriptor().shape()),
                    h.add(x.sum(0, true).expand(h.descriptor().shape())))) {
                var body = new ScanBody(List.of(h), List.of(x), List.of(bias), List.of(root), List.of());
                var allValid = Scan.apply(List.of(initial), List.of(sequence), List.of(fresh(bias, false)), body, false);
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(lengthIR(
                        GraphCapture.capture(allValid.finalCarry()), 1, List.of(OptionalInt.of(0)), ld)));
            }
            Tensor a = tensor(DataType.FLOAT64, false, 2), b = fresh(a, false);
            for (List<Tensor> roots : List.of(List.of(h, a, b.add(h.sum(0, false))),
                    List.of(h, a.add(b), b.add(h.sum(0, false))),
                    List.of(h, a.add(h.sum(0, false)), b.add(a)))) {
                var body = new ScanBody(List.of(h, a, b), List.of(x), List.of(), roots, List.of());
                var allValid = Scan.apply(List.of(initial, fresh(a, false), fresh(b, false)), List.of(sequence), List.of(), body, true);
                assertThrows(IllegalArgumentException.class, () -> CapturedGraphInference.inferAndValidate(lengthIR(
                        GraphCapture.capture(allValid.finalCarry()), 1, List.of(OptionalInt.of(0)), ld)));
            }
        }
    }
}
