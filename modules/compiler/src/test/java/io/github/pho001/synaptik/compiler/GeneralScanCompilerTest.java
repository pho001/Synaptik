package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.compiler.ScanBodyInferenceTest.tensor;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.planning.capability.*;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

final class GeneralScanCompilerTest {

    @Test void allValidSymbolicReshapeAcceptsWholeTensorEqualityWithoutInventingRowMapping() {
        var batch = new DynamicDimension("ReshapeBatch");
        var width = new StaticDimension(2);
        var stepShape = Shape.ofDimensions(width, batch);
        var emissionShape = Shape.ofDimensions(batch, width);
        var sequenceShape = Shape.ofDimensions(new StaticDimension(3), width, batch);
        var outputShape = Shape.ofDimensions(new StaticDimension(3), batch, width);
        var backend = new BackendId("scoped-symbolic-reshape-test");
        var snapshots = List.of(new BackendAvailabilitySnapshot(backend,
                Map.of(new BackendDeviceId(backend, "0"), DeviceClass.CPU)));
        BackendCapabilityProvider accept = new BackendCapabilityProvider() {
            public BackendId backendId() { return backend; }
            public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (boolean reverse : List.of(false, true)) for (boolean withCarry : List.of(false, true)) {
                var descriptor = new TensorDescriptor(type, stepShape, Optional.empty(), true);
                Tensor step = TensorFactory.create(descriptor);
                Tensor carry = TensorFactory.create(descriptor);
                Tensor initial = TensorFactory.create(descriptor);
                Tensor sequence = TensorFactory.create(new TensorDescriptor(
                        type, sequenceShape, Optional.empty(), true));
                // The carry, when declared, retains its own initial value exactly. Only the
                // emission changes the batch coordinate's position, with no length/ROW promise.
                var body = new ScanBody(withCarry ? List.of(carry) : List.of(), List.of(step),
                        List.of(), withCarry ? List.of(carry) : List.of(),
                        List.of(step.reshape(emissionShape)));
                var scan = Scan.apply(withCarry ? List.of(initial) : List.of(), List.of(sequence),
                        List.of(), body, reverse);
                var roots = new ArrayList<>(scan.finalCarry());
                roots.addAll(scan.outputs());
                assertEquals(outputShape, scan.outputs().getFirst().descriptor().shape());
                var targets = withCarry ? List.of(initial, sequence) : List.of(sequence);
                var seeds = roots.stream().map(root -> TensorFactory.create(new TensorDescriptor(
                        type, root.descriptor().shape(), Optional.empty(), false))).toList();
                var request = ScanGradientRulesTest.request(roots, seeds, targets);
                var expanded = FirstOrderAutograd.expand(AutogradPreflight.preflight(
                        CompileMode.FORWARD_AND_BACKWARD, roots, request,
                        CompileTimeConstantGraph.Ingress.empty()).stageOne(),
                        CompileTimeConstantGraph.Ingress.empty());
                for (int i = 0; i < targets.size(); i++) {
                    var gradient = expanded.targetGradients().get(i).gradient().descriptor();
                    assertEquals(targets.get(i).descriptor().shape(), gradient.shape());
                    assertEquals(type, gradient.dataType());
                    assertTrue(gradient.layout().isEmpty());
                }
                for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                    var forward = GraphCompiler.compile(CompileMode.FORWARD_ONLY, roots, Optional.empty(),
                            CompileTimeConstantGraph.Ingress.empty(), optimization, BackendIntent.unconstrained(),
                            PartitionScoringConfig.neutral(), List.of(accept), snapshots);
                    var combined = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, roots,
                            Optional.of(request), CompileTimeConstantGraph.Ingress.empty(), optimization,
                            BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(accept), snapshots);
                    for (var artifacts : List.of(forward, combined)) for (Tensor root : roots) {
                        var binding = artifacts.publication().forwardBindings().stream()
                                .filter(candidate -> candidate.tensorId().equals(root.id())).findFirst().orElseThrow();
                        var published = artifacts.graph().values().stream()
                                .filter(value -> value.id().equals(binding.valueId())).findFirst().orElseThrow();
                        assertEquals(root.descriptor(), published.descriptor());
                    }
                    for (Tensor target : targets) {
                        var binding = combined.publication().gradientBindings().stream()
                                .filter(candidate -> candidate.target().equals(target.id())).findFirst().orElseThrow();
                        var gradient = combined.graph().values().stream()
                                .filter(value -> value.id().equals(binding.valueId())).findFirst().orElseThrow()
                                .descriptor();
                        assertEquals(target.descriptor().shape(), gradient.shape());
                        assertEquals(type, gradient.dataType());
                        assertTrue(gradient.layout().isEmpty());
                    }
                }
            }
    }

    @Test void symbolicLengthsAdmitPublicForwardAndActualAdWithoutOpeningColdBoundaries() throws Exception {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16,
                DataType.FLOAT16, DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2))
            for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                var source = symbolicLengthSource(type, time, reverse);
                var p = source.finalCarry().getFirst().provenance().orElseThrow().producer();
                Tensor seed = TensorFactory.create(new TensorDescriptor(type,
                        source.finalCarry().getFirst().descriptor().shape(), Optional.empty(), false));
                var request = ScanGradientRulesTest.request(source.finalCarry(), List.of(seed), List.of(p.inputs().get(1)));
                Tensor target = tensor(type, true), objective = target.mul(target);
                for (var opt : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                    GraphCompiler.compile(CompileMode.FORWARD_ONLY, source.finalCarry(), Optional.empty(),
                            CompileTimeConstantGraph.Ingress.empty(), opt);
                    var combined = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                            source.finalCarry(), Optional.of(request), CompileTimeConstantGraph.Ingress.empty(), opt);
                    GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                            List.of(objective, source.finalCarry().getFirst()),
                            Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                            CompileTimeConstantGraph.Ingress.empty(), opt);
                }
                assertNoIds(() -> GraphCompilationPort.captureSource(source.finalCarry()));
                var raw = GraphCapture.capture(source.finalCarry());
                var leaves = raw.inputs().stream().map(id -> raw.values().stream().filter(v -> v.id().equals(id))
                        .findFirst().orElseThrow().descriptor()).map(TensorFactory::create).toList();
                assertNoIds(() -> GraphCompilationPort.reconstructSource(raw, leaves));
            }
    }

    /**
     * Builds a storage-free symbolic-length source used to separate compile and cold admission.
     * @param type homogeneous floating body type, including either OFP8 type
     * @param time non-negative fixed time count
     * @param reverse whether traversal decreases original time
     * @return non-null source occurrence with one named batch and a separate discrete lengths input;
     *     all Tensors remain storage-free and no Compiler admission is attempted
     * @throws NullPointerException if type is null
     * @throws IllegalArgumentException if type is nonfloating or time is negative
     * @throws IllegalStateException if Tensor identities are exhausted while constructing the fixture
     */
    private static ScanResult symbolicLengthSource(DataType type, long time, boolean reverse) {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Customers");
        var width = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
        var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch, width);
        var sequence = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(
                new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch, width);
        var d = new TensorDescriptor(type, row, Optional.empty(), true);
        Tensor h = TensorFactory.create(d), x = TensorFactory.create(d);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
        return Scan.apply(List.of(TensorFactory.create(d)), List.of(TensorFactory.create(new TensorDescriptor(type, sequence, Optional.empty(), true))),
                List.of(), TensorFactory.create(new TensorDescriptor(DataType.INT64,
                        io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch), Optional.empty(), false)),
                1, List.of(OptionalInt.of(0)), body, reverse);
    }

    @Test void symbolicScanAdmitsInferenceAndActualAdButColdBoundariesRemainStaticOnly() throws Exception {
        var batch = new io.github.pho001.synaptik.model.shape.DynamicDimension("Rows");
        var row = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(batch,
                new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        for (boolean unusedOnly : List.of(false, true))
                for (long time : List.of(0L, 3L)) for (boolean reverse : List.of(false, true)) {
            var sequenceShape = io.github.pho001.synaptik.model.shape.Shape.ofDimensions(
                    new io.github.pho001.synaptik.model.shape.StaticDimension(time), batch,
                    new io.github.pho001.synaptik.model.shape.StaticDimension(2));
            var publicShape = unusedOnly ? io.github.pho001.synaptik.model.shape.Shape.of(2, 2) : row;
            TensorDescriptor rd = unusedOnly ? tensor(DataType.FLOAT32, true, 2, 2).descriptor()
                    : new TensorDescriptor(DataType.FLOAT32, row, Optional.empty(), true);
            Tensor h = TensorFactory.create(rd), x = TensorFactory.create(rd);
            var declarations = unusedOnly ? List.of(TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, row, Optional.empty(), true)))
                    : List.<Tensor>of();
            var actualInvariants = unusedOnly ? List.of(TensorFactory.create(declarations.getFirst().descriptor())) : List.<Tensor>of();
            var body = new ScanBody(List.of(h), List.of(x), declarations, List.of(h.add(x).tanh()), List.of(h));
            Tensor initial = TensorFactory.create(rd);
            Tensor sequence = unusedOnly ? tensor(DataType.FLOAT32, true, time, 2, 2)
                    : TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, sequenceShape, Optional.empty(), true));
            var scan = Scan.apply(List.of(initial), List.of(sequence), actualInvariants, body, reverse);
            var roots = scan.finalCarry();
            if (unusedOnly) assertTrue(scan.finalCarry().getFirst().provenance().orElseThrow().producer().outputDescriptors()
                    .stream().allMatch(d -> d.shape().isFullyStatic() && d.layout().isPresent()));
            Tensor seed = unusedOnly ? tensor(DataType.FLOAT32, false, 2, 2)
                    : TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, publicShape, Optional.empty(), false));
            var request = ScanGradientRulesTest.request(roots, List.of(seed), List.of(initial, sequence));
            Tensor target = tensor(DataType.FLOAT32, true), objective = target.mul(target);
            for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                GraphCompiler.compile(CompileMode.FORWARD_ONLY, roots, Optional.empty(),
                        CompileTimeConstantGraph.Ingress.empty(), optimization);
                var combined = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, roots, Optional.of(request),
                        CompileTimeConstantGraph.Ingress.empty(), optimization);
                GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(objective, roots.getFirst()),
                        Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                        CompileTimeConstantGraph.Ingress.empty(), optimization);
            }
            assertNoIds(() -> GraphCompilationPort.captureSource(roots));
            // Structural capture is not inference, execution or cold qualification.
            var raw = GraphCapture.capture(roots);
            var leaves = raw.inputs().stream().map(id -> raw.values().stream().filter(v -> v.id().equals(id))
                    .findFirst().orElseThrow().descriptor()).map(TensorFactory::create).toList();
            assertNoIds(() -> GraphCompilationPort.reconstructSource(raw, leaves));
        }
    }

    @Test void lengthModeInferenceSucceedsButUnrelatedUnsupportedOriginalAdRejectsBeforeIds() throws Exception {
        for (long time : List.of(0L, 3L)) {
            Tensor h = tensor(DataType.FLOAT32, false, 2, 2), x = tensor(DataType.FLOAT32, false, 2, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h), List.of());
            var scan = Scan.apply(List.of(tensor(DataType.FLOAT32, false, 2, 2)),
                    List.of(tensor(DataType.FLOAT32, false, time, 2, 2)), List.of(), tensor(DataType.INT64, false, 2),
                    1, List.of(OptionalInt.empty()), body, false);
            var graph = GraphCapture.capture(scan.finalCarry());
            var n = graph.nodes().getFirst();
            var inputs = n.inputs().stream().map(id -> graph.values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor()).toList();
            GraphCompiler.compile(CompileMode.FORWARD_ONLY, scan.finalCarry(), Optional.empty(),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled());
            Tensor target = tensor(DataType.FLOAT32, true), objective = target.mul(target);
            assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(objective, scan.finalCarry().getFirst()),
                    Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        }
    }
    @Test void demandUsesOuterPortsNotRetainedSiblingLiveness() {
        Tensor h = tensor(DataType.FLOAT32, false, 2), x = tensor(DataType.FLOAT32, false, 2);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
        var result = Scan.apply(List.of(tensor(DataType.FLOAT32, false, 2)), List.of(tensor(DataType.FLOAT32, false, 4, 2)), List.of(), body, false);
        for (var optimization : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            var compilation = GraphCompiler.compile(CompileMode.FORWARD_ONLY, List.of(result.finalCarry().getFirst()),
                    Optional.empty(), CompileTimeConstantGraph.Ingress.empty(), optimization);
            var scan = compilation.validatedGraph().graph().nodes().getFirst();
            assertEquals(List.of(0), scan.scanBody().orElseThrow().demandedOutputIndices());
        }
    }

    @Test void residualAndHigherOrderRejectionsConsumeNoTensorIds() throws Exception {
        Tensor h = tensor(DataType.FLOAT32, true), x = tensor(DataType.FLOAT32, true);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of());
        Tensor initial = tensor(DataType.FLOAT32, true), seq = tensor(DataType.FLOAT32, true, 3);
        var good = Scan.apply(List.of(initial), List.of(seq), List.of(), body, false);
        for (Tensor residualRoot : List.of(good.residuals().getFirst(), good.residuals().getFirst().sum())) {
            Tensor seed = tensor(DataType.FLOAT32, false, ScanGradientRulesTest.Oracle.shape(residualRoot).length == 0 ? new long[0] : new long[] {3});
            var request = ScanGradientRulesTest.request(List.of(residualRoot), List.of(seed), List.of(seq));
            assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(residualRoot), Optional.of(request),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        }
        var first = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.ForwardTensorReference(good.finalCarry().getFirst())),
                List.of(Optional.empty()), List.of(seq), true, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var second = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.of(tensor(DataType.FLOAT32, false, 3))), List.of(seq), false, FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        var higher = new FunctionalGradientRequest(List.of(first, second));
        assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(good.finalCarry().getFirst()), Optional.of(higher),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        var cold = GraphCompilationPort.captureSource(List.of(good.finalCarry().getFirst()));
        var raw = GraphCapture.capture(List.of(good.finalCarry().getFirst()));
        assertEquals(raw, cold.graph());
        assertEquals(raw, GraphCompilationPort.captureSource(
                GraphCompilationPort.reconstructSource(raw, List.of(initial, seq))).graph());
    }

    /**
     * Checks that a deliberate preflight rejection consumes no derivative identities.
     * @param action non-null, already-constructed compile request, invoked synchronously once
     * @return non-null rejected request's exact exception for diagnostic assertions
     * @throws Exception if the action unexpectedly fails outside the checked rejection
     * @throws AssertionError if the request does not reject with IllegalArgumentException or consumes an identity
     */
    private static IllegalArgumentException assertNoIds(Runnable action) throws Exception {
        Tensor marker = tensor(DataType.BOOL, false);
        var failure = assertThrows(IllegalArgumentException.class, action::run);
        assertEquals(marker.id().value() + 1, tensor(DataType.BOOL, false).id().value(),
                "known preflight rejection must precede every derivative allocation");
        return failure;
    }

    @Test void lengthOriginalAdmissionChecksAllUnusedRolesSharedCarriesAndDerivativeGuards() throws Exception {
        Tensor target = tensor(DataType.FLOAT64, true), objective = target.mul(target);
        for (long time : List.of(0L, 3L)) {
            for (OptionalInt axis : List.of(OptionalInt.of(1), OptionalInt.empty())) {
                Tensor h = tensor(DataType.FLOAT64, false, 2, 2), x = tensor(DataType.FLOAT64, false, 2, 2);
                Tensor unused = tensor(DataType.FLOAT64, false, 2, 2);
                var body = new ScanBody(List.of(h), List.of(x, unused), List.of(), List.of(h), List.of());
                var scan = Scan.apply(List.of(tensor(DataType.FLOAT64, false, 2, 2)),
                        List.of(tensor(DataType.FLOAT64, false, time, 2, 2), tensor(DataType.FLOAT64, false, time, 2, 2)),
                        List.of(), tensor(DataType.INT64, false, 2), 1, List.of(OptionalInt.of(0), axis), body, false);
                assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                        List.of(objective, scan.finalCarry().getFirst()),
                        Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                        CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard()));
            }
            Tensor h = tensor(DataType.FLOAT64, true, 2, 2), x = tensor(DataType.FLOAT64, true, 2, 2);
            Tensor old = tensor(DataType.FLOAT64, false, 2), seq = tensor(DataType.FLOAT64, true, time, 2, 2);
            var sharedBody = new ScanBody(List.of(h, old), List.of(x), List.of(),
                    List.of(h, old.add(h.sum(0, false))), List.of());
            var sharedScan = Scan.apply(List.of(tensor(DataType.FLOAT64, true, 2, 2), tensor(DataType.FLOAT64, false, 2)),
                    List.of(seq), List.of(), tensor(DataType.INT64, false, 2), 1,
                    List.of(OptionalInt.of(0)), sharedBody, false);
            assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective, sharedScan.finalCarry().getFirst()),
                    Optional.of(FunctionalGradientTestSupport.request(objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of());
            var good = Scan.apply(List.of(tensor(DataType.FLOAT64, true, 2, 2)), List.of(seq), List.of(),
                    tensor(DataType.INT64, false, 2), 1, List.of(OptionalInt.of(0)), body, false);
            Tensor residual = good.residuals().getFirst().sum();
            assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(residual),
                    Optional.of(FunctionalGradientTestSupport.request(residual, List.of(seq))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
            var first = new FunctionalGradientRequest.Stage(
                    List.of(new FunctionalGradientRequest.ForwardTensorReference(good.finalCarry().getFirst())),
                    List.of(Optional.of(tensor(DataType.FLOAT64, false, 2, 2))), List.of(seq), true,
                    FunctionalGradientRequest.DisconnectedPolicy.ZERO);
            var second = new FunctionalGradientRequest.Stage(List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                    List.of(Optional.of(tensor(DataType.FLOAT64, false, time, 2, 2))), List.of(seq), false,
                    FunctionalGradientRequest.DisconnectedPolicy.ZERO);
            assertNoIds(() -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, good.finalCarry(),
                    Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                    CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.disabled()));
        }
    }

    @Test void lengthFinalQueriesCarryExactRolesLengthsAndRefreshedDemandWithoutGrantingBinderSupport() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            Tensor h = tensor(type, false, 2, 2), x = tensor(type, true, 2, 2), w = tensor(type, true, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(h.mul(w).add(x).tanh()), List.of(h));
            Tensor seq = tensor(type, true, 3, 2, 2), weight = tensor(type, true, 2);
            Tensor lengths = tensor(DataType.INT64, false, 2);
            var source = Scan.apply(List.of(tensor(type, false, 2, 2)), List.of(seq), List.of(weight), lengths,
                    1, List.of(OptionalInt.of(0)), body, reverse);
            var roots = List.of(source.finalCarry().getFirst());
            var req = ScanGradientRulesTest.request(roots, List.of(tensor(type, false, 2, 2)), List.of(seq, weight));
            var backend = new BackendId("length-scan-query-spy");
            var snapshots = List.of(new BackendAvailabilitySnapshot(backend,
                    Map.of(new BackendDeviceId(backend, "0"), DeviceClass.CPU)));
            var queries = new ArrayList<OperationCapabilityQuery>();
            BackendCapabilityProvider accept = new BackendCapabilityProvider() {
                public BackendId backendId() { return backend; }
                public boolean supports(OperationCapabilityQuery query) { queries.add(query); return true; }
            };
            BackendCapabilityProvider rejectBinder = new BackendCapabilityProvider() {
                public BackendId backendId() { return backend; }
                public boolean supports(OperationCapabilityQuery query) { return query.scanBody().isEmpty(); }
            };
            for (var opt : List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
                queries.clear();
                var artifacts = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, roots, Optional.of(req),
                        CompileTimeConstantGraph.Ingress.empty(), opt, BackendIntent.unconstrained(),
                        PartitionScoringConfig.neutral(), List.of(accept), snapshots);
                var scans = artifacts.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
                var scanQueries = queries.stream().filter(q -> q.scanBody().isPresent()).toList();
                for (int i = 0; i < scans.size(); i++) {
                    var n = scans.get(i); var q = scanQueries.get(i); var a = (ScanAttrs) q.operation().attrs();
                    assertEquals(n.operation(), q.operation()); assertEquals(n.scanBody(), q.scanBody());
                    assertEquals(a.inputCount(), q.inputs().size());
                    assertEquals(a.inputCount() - 1, q.scanBody().orElseThrow().graph().inputs().size());
                    assertEquals(DataType.INT64, q.inputs().getLast().dataType());
                    assertFalse(q.inputs().getLast().requiresGrad());
                    assertEquals(a.sequenceCount(), a.sequenceRowAxes().size());
                }
                assertEquals(scans.getFirst().inputs().getLast(), scans.getLast().inputs().getLast());
                var a = (ScanAttrs) scans.getLast().operation().attrs();
                assertTrue(scans.getLast().scanBody().orElseThrow().demandedOutputIndices().stream()
                        .allMatch(i -> i < a.carryCount() + a.emissionCount()));
                assertThrows(IllegalStateException.class, () -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                        roots, Optional.of(req), CompileTimeConstantGraph.Ingress.empty(), opt, BackendIntent.unconstrained(),
                        PartitionScoringConfig.neutral(), List.of(rejectBinder), snapshots));
            }
            // Publication is the only history demand; deleting a dead consumer removes its demand.
            var raw = GraphCapture.capture(List.of(roots.getFirst(), source.residuals().getLast().neg()));
            raw = new CompiledGraphModel(raw.values(), raw.nodes(), raw.inputs(), List.of(raw.outputs().getFirst()), raw.nodePhases());
            var demanded = ScanBodyInference.refreshDemand(CapturedGraphInference.inferAndValidate(raw));
            var before = demanded.graph().nodes().getFirst().scanBody().orElseThrow();
            var dce = ForwardDeadCodeElimination.DerivativeAware.eliminate(demanded.constantGraph(), demanded.derivatives());
            var after = ScanBodyInference.refreshDemand(CapturedGraphInference.inferAndValidate(dce.constantGraph(), dce.derivatives()))
                    .graph().nodes().getFirst().scanBody().orElseThrow();
            assertEquals(List.of(0), after.demandedOutputIndices()); assertSame(before.graph(), after.graph());
            assertEquals(before.residuals(), after.residuals());
        }
    }


    @Test void demandShrinksAfterDeadConsumerRemovalWithoutLosingSavedSiteIdentity() {
        Tensor h = tensor(DataType.FLOAT64, false), x = tensor(DataType.FLOAT64, false);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of());
        var scan = Scan.apply(List.of(tensor(DataType.FLOAT64, false)), List.of(tensor(DataType.FLOAT64, false, 3)), List.of(), body, false);
        var graph = GraphCapture.capture(List.of(scan.finalCarry().getFirst(), scan.residuals().getLast().neg()));
        graph = new CompiledGraphModel(graph.values(), graph.nodes(), graph.inputs(), List.of(graph.outputs().getFirst()), graph.nodePhases());
        var demanded = ScanBodyInference.refreshDemand(CapturedGraphInference.inferAndValidate(graph));
        var before = demanded.graph().nodes().getFirst().scanBody().orElseThrow();
        var eliminated = ForwardDeadCodeElimination.DerivativeAware.eliminate(demanded.constantGraph(), demanded.derivatives());
        var refreshed = ScanBodyInference.refreshDemand(CapturedGraphInference.inferAndValidate(eliminated.constantGraph(), eliminated.derivatives()));
        var after = refreshed.graph().nodes().getFirst().scanBody().orElseThrow();
        assertEquals(List.of(0), after.demandedOutputIndices());
        assertSame(before.graph(), after.graph());
        assertEquals(before.residuals(), after.residuals());
    }

    @Test void finalQueryCarriesWholeValidatedBodyAndUnsupportedBinderCannotBorrowChildSupport() {
        Tensor h = tensor(DataType.FLOAT32, false), x = tensor(DataType.FLOAT32, true);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
        Tensor seq = tensor(DataType.FLOAT32, true, 3);
        var scan = Scan.apply(List.of(tensor(DataType.FLOAT32, false)), List.of(seq), List.of(), body, false);
        var roots = List.of(scan.finalCarry().getFirst());
        var request = FunctionalGradientTestSupport.request(roots.getFirst(), List.of(seq));
        var backend = new BackendId("scan-compiler-test");
        var snapshots = List.of(new BackendAvailabilitySnapshot(backend, Map.of(new BackendDeviceId(backend, "0"), DeviceClass.CPU)));
        var queries = new ArrayList<OperationCapabilityQuery>();
        BackendCapabilityProvider accept = new BackendCapabilityProvider() {
            public BackendId backendId() { return backend; }
            public boolean supports(OperationCapabilityQuery query) { queries.add(query); return true; }
        };
        var artifacts = GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD, roots, Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard(), BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(), List.of(accept), snapshots);
        var scanQueries = queries.stream().filter(q -> q.scanBody().isPresent()).toList();
        for (OperationCapabilityQuery query : scanQueries) {
            var compiledBody = query.scanBody().orElseThrow();
            assertTrue(CapturedGraphInference.inferAndValidate(compiledBody.graph()).constraints().isEmpty());
            assertFalse(compiledBody.demandedOutputIndices().isEmpty());
            assertTrue(compiledBody.demandedOutputIndices().stream().noneMatch(i -> i >= ((ScanAttrs) query.operation().attrs()).carryCount() + ((ScanAttrs) query.operation().attrs()).emissionCount())
                    || query == scanQueries.getFirst());
        }
        assertEquals(roots.getFirst().id(), artifacts.publication().forwardBindings().getFirst().tensorId());
        assertEquals(seq.id(), artifacts.publication().gradientBindings().getFirst().target());
        BackendCapabilityProvider rejectBinder = new BackendCapabilityProvider() {
            public BackendId backendId() { return backend; }
            public boolean supports(OperationCapabilityQuery query) { return query.scanBody().isEmpty(); }
        };
        assertThrows(IllegalStateException.class, () -> GraphCompiler.compile(CompileMode.FORWARD_AND_BACKWARD,
                roots, Optional.of(request), CompileTimeConstantGraph.Ingress.empty(), GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(rejectBinder), snapshots));
    }
}
