package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.zip.CRC32;
import org.junit.jupiter.api.Test;

class MetalNegRouteCandidateGeneratorTest {
    @Test
    void candidateDomainsAndBudgetPrefixesPreserveTheCurrentSafeChoice() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload singleton = workload(device, 100, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            Generated generated = generated(singleton, 2);
            assertEquals(List.of(
                    MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                    MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated.batch().candidates());
            assertEquals(List.of(MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG),
                    new MetalNegRouteCandidateGenerator().generate(
                            singleton.context(), generated.analysis().plan(), 1).candidates());

            Workload oversized = workload(device, 200, Shape.of(0x1_0000_0000L), false,
                    Optional.empty(), true, false, 1);
            assertEquals(List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(oversized, 2).batch().candidates());
            Workload chain = workload(device, 300, Shape.of(4), false,
                    Optional.empty(), true, false, 2);
            assertEquals(List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(chain, 2).batch().candidates());
            assertThrows(IllegalArgumentException.class,
                    () -> new MetalNegRouteCandidateGenerator().generate(
                            singleton.context(), generated.analysis().plan(), 0));
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    @Test
    void signaturesUseStructuralPositionsAndCoverIndependentCurrentFacts() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload baseline = workload(device, 1, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            Workload differentIds = workload(device, 50_000, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            var baselineCompatibility = generated(baseline, 2).batch().compatibility();
            assertEquals(baselineCompatibility,
                    generated(differentIds, 2).batch().compatibility());

            List<Workload> changed = List.of(
                    workload(device, 2, Shape.of(5), false,
                            Optional.empty(), true, false, 1),
                    workload(device, 3, Shape.of(4), true,
                            Optional.empty(), true, false, 1),
                    workload(device, 4, Shape.of(4), false,
                            Optional.of(ScalarValue.float32(-0.0f)), true, false, 1),
                    workload(device, 5, Shape.of(4), false,
                            Optional.empty(), false, true, 1),
                    workload(device, 6, Shape.of(4), false,
                            Optional.empty(), true, false, 2));
            for (Workload changedWorkload : changed) {
                assertNotEquals(baselineCompatibility,
                        generated(changedWorkload, 2).batch().compatibility());
            }

            Workload binaryAdd = binaryWorkload(
                    device, 10_000, BinaryArithmeticKind.ADD, false);
            Workload binarySubtract = binaryWorkload(
                    device, 20_000, BinaryArithmeticKind.SUB, false);
            Workload binaryReversed = binaryWorkload(
                    device, 30_000, BinaryArithmeticKind.ADD, true);
            assertEquals(
                    List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(binaryAdd, 2).batch().candidates());
            assertNotEquals(
                    workloadSignature(binaryAdd),
                    workloadSignature(binarySubtract),
                    "ordered binary operation kind participates in workload identity");
            assertNotEquals(
                    workloadSignature(binaryAdd),
                    workloadSignature(binaryReversed),
                    "ordered binary operand edges participate in workload identity");
            Workload unaryAbs = unaryWorkload(
                    device, 40_000, NumericalProfile.STRICT_IEEE, UnaryElementwiseKind.ABS);
            assertEquals(
                    List.of(MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generated(unaryAbs, 2).batch().candidates(),
                    "ABS must never enter the custom singleton NEG route");
            assertNotEquals(
                    workloadSignature(baseline),
                    workloadSignature(unaryAbs),
                    "ordered ABS topology participates in workload identity");
            Workload acceleratorAbs = unaryWorkload(
                    device, 50_000, NumericalProfile.ACCELERATOR, UnaryElementwiseKind.ABS);
            assertNotEquals(
                    workloadSignature(unaryAbs),
                    workloadSignature(acceleratorAbs),
                    "requested profile participates in ABS workload identity");
        }
    }

    @Test
    void affineFingerprintsCoverKindsTypedAttrsLayoutsDenseGeometryAndOrderedTopology() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            TensorDescriptor matrix = canonical(Shape.of(2, 3));
            TensorDescriptor identityView = view(Shape.of(2, 3), 3, 1);
            Workload reshapeIdentity = affineWorkload(
                    device,
                    1_000,
                    new Operation(
                            ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(2, 3))),
                    matrix,
                    identityView);
            Workload expandIdentity = affineWorkload(
                    device,
                    2_000,
                    new Operation(
                            ShapeTransformKind.EXPAND,
                            new TargetShapeAttrs(Shape.of(2, 3))),
                    matrix,
                    identityView);
            assertNotEquals(workloadSignature(reshapeIdentity), workloadSignature(expandIdentity),
                    "operation kind is part of the affine compatibility identity");

            Workload reshapeTranspose = affineWorkload(
                    device,
                    3_000,
                    new Operation(
                            ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(3, 2))),
                    matrix,
                    view(Shape.of(3, 2), 2, 1));
            Workload reshapeFlat = affineWorkload(
                    device,
                    4_000,
                    new Operation(
                            ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(1, 6))),
                    matrix,
                    view(Shape.of(1, 6), 6, 1));
            assertNotEquals(workloadSignature(reshapeTranspose), workloadSignature(reshapeFlat),
                    "target-shape attributes are part of the affine compatibility identity");

            TensorDescriptor symmetricRankThree = canonical(Shape.of(1, 1, 1));
            TensorDescriptor symmetricRankThreeView = view(Shape.of(1, 1, 1), 1, 1, 1);
            Workload identityPermutation = affineWorkload(
                    device,
                    5_000,
                    new Operation(
                            AxisTransformKind.PERMUTE,
                            new PermutationAttrs(List.of(0, 1, 2))),
                    symmetricRankThree,
                    symmetricRankThreeView);
            Workload swappedPermutation = affineWorkload(
                    device,
                    6_000,
                    new Operation(
                            AxisTransformKind.PERMUTE,
                            new PermutationAttrs(List.of(1, 0, 2))),
                    symmetricRankThree,
                    symmetricRankThreeView);
            assertNotEquals(
                    workloadSignature(identityPermutation),
                    workloadSignature(swappedPermutation),
                    "permutation attributes change the fingerprint even when geometry does not");

            TensorDescriptor symmetricRankTwo = canonical(Shape.of(1, 1));
            TensorDescriptor insertedSingletonView = view(Shape.of(1, 1, 1), 1, 1, 1);
            Workload frontAxis = affineWorkload(
                    device,
                    7_000,
                    new Operation(
                            AxisTransformKind.EXPAND_DIMS,
                            new AxisTransformAttrs(0)),
                    symmetricRankTwo,
                    insertedSingletonView);
            Workload middleAxis = affineWorkload(
                    device,
                    8_000,
                    new Operation(
                            AxisTransformKind.EXPAND_DIMS,
                            new AxisTransformAttrs(1)),
                    symmetricRankTwo,
                    insertedSingletonView);
            assertNotEquals(workloadSignature(frontAxis), workloadSignature(middleAxis),
                    "normalized axis attributes change the fingerprint independently");

            Generated baseline = generated(reshapeTranspose, 2);
            MetalNegPreparationPlan plan = baseline.analysis().plan();
            MetalMpsGraphProgram changedTargetAttributes = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.targetShape(
                            MetalMpsGraphProgram.NodeKind.RESHAPE,
                            0,
                            1,
                            new long[] {1, 6})));
            MetalNegPreparationPlan changedTargetShape = copyPlan(
                    plan,
                    plan.descriptors(),
                    changedTargetAttributes,
                    plan.targetRequiredBytes());
            assertNotEquals(
                    baseline.batch().compatibility().workload(),
                    generate(reshapeTranspose, changedTargetShape).compatibility().workload(),
                    "target-shape attributes are authenticated independently of descriptors");
            var changedDescriptors = new ArrayList<>(plan.descriptors());
            changedDescriptors.set(1, view(Shape.of(3, 2), 1, 3));
            MetalNegPreparationPlan changedLayout = copyPlan(
                    plan,
                    changedDescriptors,
                    plan.graphProgram(),
                    plan.targetRequiredBytes());
            assertNotEquals(
                    baseline.batch().compatibility().workload(),
                    generate(reshapeTranspose, changedLayout).compatibility().workload(),
                    "logical view layout and strides are authenticated independently");

            long[] changedTargetBytes = plan.targetRequiredBytes();
            changedTargetBytes[0] = Math.addExact(changedTargetBytes[0], Float.BYTES);
            MetalNegPreparationPlan changedDenseGeometry = copyPlan(
                    plan,
                    plan.descriptors(),
                    plan.graphProgram(),
                    changedTargetBytes);
            assertNotEquals(
                    baseline.batch().compatibility().workload(),
                    generate(reshapeTranspose, changedDenseGeometry).compatibility().workload(),
                    "dense represented-order target geometry is authenticated independently");

            assertNotEquals(
                    workloadSignature(twoAffineLeaves(device, 9_000, false)),
                    workloadSignature(twoAffineLeaves(device, 10_000, true)),
                    "ordered affine topology changes the fingerprint");
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    @Test
    void typedNodeFingerprintsCoverMatmulAndReductionKindFormAxesOrderKeepAndTarget() {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            TensorDescriptor input = canonical(Shape.of(2, 3, 4));
            Workload baselineWorkload = reductionWorkload(
                    device,
                    60_000,
                    new Operation(
                            AggregateReductionKind.SUM,
                            new AxisReductionAttrs(1, false)),
                    input,
                    canonical(Shape.of(2, 4)));
            Generated baseline = generated(baselineWorkload, 2);
            MetalNegPreparationPlan plan = baseline.analysis().plan();
            MetalNegTuningBatch.WorkloadSignature identity =
                    baseline.batch().compatibility().workload();

            List<MetalMpsGraphProgram> changedRecords = List.of(
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.reduction(
                                    MetalMpsGraphProgram.NodeKind.MEAN,
                                    0,
                                    1,
                                    MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                                    List.of(1),
                                    false))),
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.reduction(
                                    MetalMpsGraphProgram.NodeKind.SUM,
                                    0,
                                    1,
                                    MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                                    List.of(2),
                                    false))),
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.reduction(
                                    MetalMpsGraphProgram.NodeKind.SUM,
                                    0,
                                    1,
                                    MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                                    List.of(1),
                                    true))),
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.reduction(
                                    MetalMpsGraphProgram.NodeKind.SUM,
                                    0,
                                    1,
                                    MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                                    List.of(1),
                                    false))),
                    new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.matmul(0, 0, 1))));
            for (MetalMpsGraphProgram changedRecord : changedRecords) {
                MetalNegPreparationPlan changed = copyPlan(
                        plan,
                        plan.descriptors(),
                        changedRecord,
                        plan.targetRequiredBytes());
                assertNotEquals(
                        identity,
                        generate(baselineWorkload, changed).compatibility().workload());
            }

            assertNotEquals(
                    workloadSignature(reductionWorkload(
                            device,
                            61_000,
                            new Operation(
                                    AggregateReductionKind.MEAN,
                                    new MultiAxisReductionAttrs(List.of(2, 0), false)),
                            input,
                            canonical(Shape.of(3)))),
                    workloadSignature(reductionWorkload(
                            device,
                            62_000,
                            new Operation(
                                    AggregateReductionKind.MEAN,
                                    new MultiAxisReductionAttrs(List.of(0, 2), false)),
                            input,
                            canonical(Shape.of(3)))),
                    "ordered reduction axes are authenticated");
            assertNotEquals(
                    workloadSignature(reductionWorkload(
                            device,
                            63_000,
                            new Operation(
                                    AggregateReductionKind.SUM,
                                    new MultiAxisReductionAttrs(List.of(), false)),
                            input,
                            input)),
                    workloadSignature(reductionWorkload(
                            device,
                            64_000,
                            new Operation(
                                    AggregateReductionKind.SUM,
                                    NoOperationAttrs.INSTANCE),
                            input,
                            canonical(Shape.scalar()))),
                    "empty multi-axis identity and full reduction remain distinct");
            assertNotEquals(
                    workloadSignature(reductionWorkload(
                            device,
                            65_000,
                            new Operation(
                                    AggregateReductionKind.SUM,
                                    new SumToShapeAttrs(Shape.of(1, 4))),
                            input,
                            canonical(Shape.of(1, 4)))),
                    workloadSignature(reductionWorkload(
                            device,
                            66_000,
                            new Operation(
                                    AggregateReductionKind.SUM,
                                    new SumToShapeAttrs(Shape.of(2, 1, 4))),
                            input,
                            canonical(Shape.of(2, 1, 4)))),
                    "sum-to-Shape target dimensions are authenticated");
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    @Test
    void codecIsCanonicalBoundedAndRejectsEveryDefensiveMismatch() {
        TestNativeApi api = new TestNativeApi();
        TestNativeApi otherApi = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            Generated current = generated(workload(device, 10, Shape.of(4), false,
                    Optional.empty(), true, false, 1), 2);
            var decision = new MetalNegTuningDecision(
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    current.batch().compatibility(), MetalNegTuningBatch.Candidate.MPSGRAPH);
            var codec = new MetalNegTuningCodec();
            assertEquals(9, MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION);
            assertEquals(9, MetalNegTuningBatch.COMPATIBILITY_SCHEMA_VERSION);
            assertEquals(9, MetalNegTuningBatch.ROUTE_POLICY_VERSION);
            byte[] first = codec.encodeDecision(decision);
            assertEquals(9, java.nio.ByteBuffer.wrap(first).getInt(Integer.BYTES));
            assertEquals(9, current.batch().compatibility().schemaVersion());
            assertEquals(9, current.batch().compatibility().candidateSchemaVersion());
            assertEquals(9, current.batch().compatibility().routePolicyVersion());
            assertArrayEquals(first, codec.encodeDecision(decision));
            assertTrue(first.length <= MetalNegTuningCodec.MAX_DECISION_BYTES);
            assertEquals(decision, codec.decodeDecision(first, current.batch()).orElseThrow());
            var strictCompatibility = current.batch().compatibility();
            var acceleratorCompatibility = new MetalNegTuningBatch.Compatibility(
                    MetalNegTuningBatch.COMPATIBILITY_SCHEMA_VERSION,
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    MetalNegTuningBatch.ROUTE_POLICY_VERSION,
                    NumericalProfile.ACCELERATOR,
                    strictCompatibility.workload(),
                    strictCompatibility.target());
            var acceleratorBatch = new MetalNegTuningBatch(
                    acceleratorCompatibility, current.batch().candidates());
            var acceleratorDecision = new MetalNegTuningDecision(
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    acceleratorCompatibility,
                    MetalNegTuningBatch.Candidate.MPSGRAPH);
            byte[] acceleratorBytes = codec.encodeDecision(acceleratorDecision);
            assertFalse(Arrays.equals(first, acceleratorBytes));
            assertTrue(codec.decodeDecision(acceleratorBytes, current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(first, acceleratorBatch).isEmpty());

            byte[] corrupt = first.clone();
            corrupt[20] ^= 1;
            assertTrue(codec.decodeDecision(corrupt, current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(Arrays.copyOf(first, first.length - 1),
                    current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(Arrays.copyOf(first, first.length + 1),
                    current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(rewriteInt(first, 0, 0), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(rewriteInt(first, 4, 99), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, 4, 1), current.batch()).isEmpty(),
                    "checksummed codec-v1 decisions must fail closed");
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, 4, 2), current.batch()).isEmpty(),
                    "checksummed schema-v2 decisions must fail closed");
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, 4, 5), current.batch()).isEmpty(),
                    "checksummed codec-v5 decisions must fail closed");
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, 4, 6), current.batch()).isEmpty(),
                    "checksummed version-six decisions must fail closed");
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, 4, 7), current.batch()).isEmpty(),
                    "checksummed version-seven decisions must fail closed");
            assertTrue(codec.decodeDecision(rewriteInt(first, 8, 99), current.batch()).isEmpty());
            assertTrue(codec.decodeDecision(
                    rewriteInt(first, first.length - 8, 99), current.batch()).isEmpty());

            Generated foreignSession = generated(workload(otherDevice, 20, Shape.of(4), false,
                    Optional.empty(), true, false, 1), 2);
            assertTrue(codec.decodeDecision(first, foreignSession.batch()).isEmpty());
            Generated changedWorkload = generated(workload(device, 30, Shape.of(5), false,
                    Optional.empty(), true, false, 1), 2);
            assertTrue(codec.decodeDecision(first, changedWorkload.batch()).isEmpty());
        }
    }

    @Test
    void freshAnalysisAuthenticatesSelectionsAndDeclaresOnlyTheirResources() {
        TestNativeApi api = new TestNativeApi();
        TestNativeApi otherApi = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api);
                MetalDeviceContext otherDevice = MetalDeviceContext.open(otherApi)) {
            Workload workload = workload(device, 1000, Shape.of(8), false,
                    Optional.empty(), true, false, 1);
            Generated original = generated(workload, 2);
            var generator = new MetalNegRouteCandidateGenerator();
            BackendPartitionTuningHandoff<MetalNegTuningBatch, MetalNegTuningDecision> absent =
                    generator.absentHandoff(workload.context(), original.analysis().plan(), 2);
            var absentAnalysis = analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(absent))).context());
            assertEquals(MetalNegPreparationPlan.Route.CUSTOM_SINGLE_NEG,
                    absentAnalysis.plan().route());
            assertEquals(2, absentAnalysis.requirements().size());

            var selected = generator.presentHandoff(
                    workload.context().partition(), original.batch(),
                    MetalNegTuningBatch.Candidate.MPSGRAPH);
            var selectedAnalysis = analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(selected))).context());
            assertEquals(MetalNegPreparationPlan.Route.MPSGRAPH, selectedAnalysis.plan().route());
            assertEquals(3, selectedAnalysis.requirements().size());
            assertTrue(selectedAnalysis.plan().addressWorkspace().isPresent());
            assertEquals(List.of(
                    MetalNegTuningBatch.Candidate.CUSTOM_SINGLE_NEG,
                    MetalNegTuningBatch.Candidate.MPSGRAPH),
                    generator.generate(
                            workload.context(), selectedAnalysis.plan(), 2).candidates());

            Workload foreignFacts = workload(otherDevice, 1000, Shape.of(8), false,
                    Optional.empty(), true, false, 1);
            Generated foreign = generated(foreignFacts, 2);
            var foreignDecision = generator.presentHandoff(
                    workload.context().partition(), foreign.batch(),
                    MetalNegTuningBatch.Candidate.MPSGRAPH);
            assertThrows(IllegalArgumentException.class, () -> analyze(withInputs(workload,
                    new MetalNegAnalysisInputs(device, Optional.of(foreignDecision))).context()));

            Workload changed = workload(device, 1000, Shape.of(9), false,
                    Optional.empty(), true, false, 1);
            assertThrows(IllegalArgumentException.class, () -> analyze(withInputs(changed,
                    new MetalNegAnalysisInputs(device, Optional.of(selected))).context()));
            assertEquals(0, api.nativeAllocations.get());
            assertEquals(0, otherApi.nativeAllocations.get());
        }
    }

    @Test
    void invalidSemanticsFailBeforeCandidateGenerationAndColdUseIsConcurrent() throws Exception {
        TestNativeApi api = new TestNativeApi();
        try (MetalDeviceContext device = MetalDeviceContext.open(api)) {
            Workload valid = workload(device, 700, Shape.of(4), false,
                    Optional.empty(), true, false, 1);
            CompiledNode invalidNode = new CompiledNode(valid.context().nodes().getFirst().id(),
                    new Operation(UnaryElementwiseKind.EXP, NoOperationAttrs.INSTANCE),
                    valid.context().nodes().getFirst().inputs(),
                    valid.context().nodes().getFirst().outputs());
            var invalidDag = new PartitionDag(valid.context().partition(), List.of(invalidNode));
            var accelerator = new PrepareContext<>(
                    NumericalProfile.ACCELERATOR,
                    valid.context().partitionDag(),
                    valid.context().values(),
                    valid.context().memoryRequirements(),
                    valid.context().constants(),
                    new MetalNegAnalysisInputs(device));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new MetalNegPartitionPreparer().analyze(accelerator));
            var invalid = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, invalidDag, valid.context().values(), valid.context().memoryRequirements(), valid.context().constants(), new MetalNegAnalysisInputs(device));
            assertThrows(IllegalArgumentException.class, () -> new MetalNegPartitionPreparer()
                    .analyze(invalid));

            Generated generated = generated(valid, 2);
            var executor = Executors.newFixedThreadPool(6);
            try {
                var calls = new ArrayList<java.util.concurrent.Callable<MetalNegTuningBatch>>();
                for (int index = 0; index < 48; index++) {
                    calls.add(() -> new MetalNegRouteCandidateGenerator().generate(
                            valid.context(), generated.analysis().plan(), 2));
                }
                for (var future : executor.invokeAll(calls)) {
                    assertEquals(generated.batch(), future.get(5, TimeUnit.SECONDS));
                }
            } finally {
                executor.shutdownNow();
                assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
            }
            assertEquals(0, api.nativeAllocations.get());
        }
    }

    private static Generated generated(Workload workload, int budget) {
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(workload.context());
        MetalNegTuningBatch batch = new MetalNegRouteCandidateGenerator().generate(
                workload.context(), analysis.plan(), budget);
        return new Generated(analysis, batch);
    }
    private static MetalNegTuningBatch.WorkloadSignature workloadSignature(Workload workload) {
        return generated(workload, 2).batch().compatibility().workload();
    }

    private static MetalNegTuningBatch generate(
            Workload workload, MetalNegPreparationPlan plan) {
        return new MetalNegRouteCandidateGenerator().generate(
                workload.context(), plan, 2);
    }

    private static MetalNegPreparationPlan copyPlan(
            MetalNegPreparationPlan source,
            List<TensorDescriptor> descriptors,
            MetalMpsGraphProgram graphProgram,
            long[] targetRequiredBytes) {
        return new MetalNegPreparationPlan(source.numericalProfile(), source.partition(),
        source.partitionDag(),
        source.context(),
        source.route(),
        source.valueIds(),
        descriptors,
        source.valueStates(),
        source.valueRanks(),
        source.valueDimensions(),
        graphProgram,
        source.feedValueIds(),
        source.feedValueIndices(),
        source.targetValueIds(),
        source.targetValueIndices(),
        source.declarations(),
        source.feedSplats(),
        source.addressWorkspace(),
        source.feedRequiredBytes(),
        targetRequiredBytes);
    }

    private static Workload affineWorkload(
            MetalDeviceContext device,
            long identityBase,
            Operation operation,
            TensorDescriptor inputDescriptor,
            TensorDescriptor outputDescriptor) {
        return operationWorkload(
                device,
                identityBase,
                NumericalProfile.STRICT_IEEE,
                operation,
                inputDescriptor,
                outputDescriptor);
    }

    private static Workload reductionWorkload(
            MetalDeviceContext device,
            long identityBase,
            Operation operation,
            TensorDescriptor inputDescriptor,
            TensorDescriptor outputDescriptor) {
        return operationWorkload(
                device,
                identityBase,
                NumericalProfile.ACCELERATOR,
                operation,
                inputDescriptor,
                outputDescriptor);
    }

    private static Workload operationWorkload(
            MetalDeviceContext device,
            long identityBase,
            NumericalProfile profile,
            Operation operation,
            TensorDescriptor inputDescriptor,
            TensorDescriptor outputDescriptor) {
        ValueId feed = new ValueId(identityBase);
        ValueId target = new ValueId(identityBase + 1);
        CompiledNode node = new CompiledNode(
                new NodeId(identityBase),
                operation,
                List.of(feed),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                List.of(node.id()));
        var context = new PrepareContext<>(profile, new PartitionDag(partition, List.of(node)),
                List.of(
                        new GraphValue(feed, inputDescriptor),
                        new GraphValue(target, outputDescriptor)),
                List.of(
                        new LogicalMemoryRequirement(
                                feed,
                                inputDescriptor,
                                Optional.empty(),
                                List.of(partition),
                                false),
                        new LogicalMemoryRequirement(
                                target,
                                outputDescriptor,
                                Optional.of(partition),
                                List.of(),
                                true)),
                Map.of(),
                new MetalNegAnalysisInputs(device));
        return new Workload(context);
    }

    private static Workload binaryWorkload(
            MetalDeviceContext device,
            long identityBase,
            BinaryArithmeticKind kind,
            boolean reverseInputs) {
        TensorDescriptor leftDescriptor = canonical(Shape.of(2, 1));
        TensorDescriptor rightDescriptor = canonical(Shape.of(1, 3));
        TensorDescriptor outputDescriptor = canonical(Shape.of(2, 3));
        ValueId left = new ValueId(identityBase);
        ValueId right = new ValueId(identityBase + 1);
        ValueId output = new ValueId(identityBase + 2);
        List<ValueId> inputs = reverseInputs
                ? List.of(right, left)
                : List.of(left, right);
        CompiledNode node = new CompiledNode(
                new NodeId(identityBase),
                new Operation(kind, NoOperationAttrs.INSTANCE),
                inputs,
                List.of(output));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                List.of(node.id()));
        return new Workload(new PrepareContext<>(
                NumericalProfile.ACCELERATOR,
                new PartitionDag(partition, List.of(node)),
                List.of(
                        new GraphValue(left, leftDescriptor),
                        new GraphValue(right, rightDescriptor),
                        new GraphValue(output, outputDescriptor)),
                List.of(
                        new LogicalMemoryRequirement(
                                left,
                                leftDescriptor,
                                Optional.empty(),
                                List.of(partition),
                                false),
                        new LogicalMemoryRequirement(
                                right,
                                rightDescriptor,
                                Optional.empty(),
                                List.of(partition),
                                false),
                        new LogicalMemoryRequirement(
                                output,
                                outputDescriptor,
                                Optional.of(partition),
                                List.of(),
                                true)),
                Map.of(),
                new MetalNegAnalysisInputs(device)));
    }

    private static Workload twoAffineLeaves(
            MetalDeviceContext device, long identityBase, boolean reverse) {
        TensorDescriptor inputDescriptor = canonical(Shape.of(2, 3));
        TensorDescriptor outputDescriptor = view(Shape.of(2, 3), 3, 1);
        ValueId firstFeed = new ValueId(identityBase);
        ValueId firstTarget = new ValueId(identityBase + 1);
        ValueId secondFeed = new ValueId(identityBase + 2);
        ValueId secondTarget = new ValueId(identityBase + 3);
        CompiledNode reshape = new CompiledNode(
                new NodeId(identityBase),
                new Operation(
                        ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(2, 3))),
                List.of(firstFeed),
                List.of(firstTarget));
        CompiledNode expand = new CompiledNode(
                new NodeId(identityBase + 1),
                new Operation(
                        ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(2, 3))),
                List.of(secondFeed),
                List.of(secondTarget));
        List<CompiledNode> nodes = reverse
                ? List.of(expand, reshape)
                : List.of(reshape, expand);
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        return new Workload(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, new PartitionDag(partition, nodes), List.of(
                new GraphValue(firstFeed, inputDescriptor),
                new GraphValue(firstTarget, outputDescriptor),
                new GraphValue(secondFeed, inputDescriptor),
                new GraphValue(secondTarget, outputDescriptor)), List.of(
                new LogicalMemoryRequirement(
                        firstFeed,
                        inputDescriptor,
                        Optional.empty(),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        firstTarget,
                        outputDescriptor,
                        Optional.of(partition),
                        List.of(),
                        true),
                new LogicalMemoryRequirement(
                        secondFeed,
                        inputDescriptor,
                        Optional.empty(),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        secondTarget,
                        outputDescriptor,
                        Optional.of(partition),
                        List.of(),
                        true)), Map.of(), new MetalNegAnalysisInputs(device)));
    }

    private static TensorDescriptor canonical(Shape shape) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static TensorDescriptor view(Shape shape, long... strides) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false);
    }


    private static BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            PrepareContext<MetalNegAnalysisInputs> context) {
        return new MetalNegPartitionPreparer().analyze(context);
    }

    private static Workload withInputs(Workload workload, MetalNegAnalysisInputs inputs) {
        PrepareContext<MetalNegAnalysisInputs> context = workload.context();
        return new Workload(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, context.partitionDag(), context.values(), context.memoryRequirements(), context.constants(), inputs));
    }

    private static Workload unaryWorkload(
            MetalDeviceContext device,
            long identityBase,
            NumericalProfile profile,
            UnaryElementwiseKind kind) {
        Workload source = workload(
                device, identityBase, Shape.of(4), false,
                Optional.empty(), true, false, 1);
        PrepareContext<MetalNegAnalysisInputs> context = source.context();
        CompiledNode original = context.nodes().getFirst();
        CompiledNode unary = new CompiledNode(
                original.id(),
                new Operation(kind, NoOperationAttrs.INSTANCE),
                original.inputs(),
                original.outputs());
        return new Workload(new PrepareContext<>(
                profile,
                new PartitionDag(context.partition(), List.of(unary)),
                context.values(),
                context.memoryRequirements(),
                context.constants(),
                context.backendInputs()));
    }

    private static Workload workload(
            MetalDeviceContext device,
            long identityBase,
            Shape shape,
            boolean requiresGrad,
            Optional<ScalarValue> splat,
            boolean graphOutput,
            boolean externalConsumer,
            int nodeCount) {
        TensorDescriptor descriptor = new TensorDescriptor(
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                shape, Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
        var nodes = new ArrayList<CompiledNode>();
        var values = new ArrayList<GraphValue>();
        ValueId feed = new ValueId(identityBase);
        values.add(new GraphValue(feed, descriptor));
        ValueId previous = feed;
        for (int index = 0; index < nodeCount; index++) {
            ValueId output = new ValueId(identityBase + index + 1);
            nodes.add(new CompiledNode(new NodeId(identityBase + index),
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(previous), List.of(output)));
            values.add(new GraphValue(output, descriptor));
            previous = output;
        }
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        PlannedPartition external = new PlannedPartition(
                new io.github.pho001.synaptik.backend.contract.BackendId("external-test"),
                List.of(new NodeId(identityBase + 90_000)));
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        requirements.add(new LogicalMemoryRequirement(feed, descriptor, Optional.empty(),
                List.of(partition), false));
        for (int index = 0; index < nodeCount; index++) {
            ValueId output = nodes.get(index).outputs().getFirst();
            boolean last = index == nodeCount - 1;
            List<PlannedPartition> consumers = last
                    ? externalConsumer ? List.of(external) : List.of()
                    : List.of(partition);
            requirements.add(new LogicalMemoryRequirement(output, descriptor,
                    Optional.of(partition), consumers, last && graphOutput));
        }
        Map<ValueId, ScalarValue> constants = splat
                .<Map<ValueId, ScalarValue>>map(value -> Map.of(feed, value))
                .orElseGet(Map::of);
        var dag = new PartitionDag(partition, nodes);
        return new Workload(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, dag, values, requirements, constants, new MetalNegAnalysisInputs(device)));
    }

    private static byte[] rewriteInt(byte[] source, int offset, int value) {
        byte[] changed = source.clone();
        changed[offset] = (byte) (value >>> 24);
        changed[offset + 1] = (byte) (value >>> 16);
        changed[offset + 2] = (byte) (value >>> 8);
        changed[offset + 3] = (byte) value;
        CRC32 checksum = new CRC32();
        checksum.update(changed, 0, changed.length - Integer.BYTES);
        int crc = (int) checksum.getValue();
        int checksumOffset = changed.length - Integer.BYTES;
        changed[checksumOffset] = (byte) (crc >>> 24);
        changed[checksumOffset + 1] = (byte) (crc >>> 16);
        changed[checksumOffset + 2] = (byte) (crc >>> 8);
        changed[checksumOffset + 3] = (byte) crc;
        return changed;
    }

    private record Workload(PrepareContext<MetalNegAnalysisInputs> context) { }

    private record Generated(
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis,
            MetalNegTuningBatch batch) { }

    private static final class TestNativeApi extends MetalNativeApi {
        private final AtomicInteger nativeAllocations = new AtomicInteger();
        private long nextHandle = 1;

        @Override synchronized Handle createContext() { return handle(); }
        @Override void releaseContext(Handle context) { }
        @Override Handle createBuffer(Handle context, long logicalByteSize) {
            nativeAllocations.incrementAndGet();
            return handle();
        }
        @Override void releaseBuffer(Handle buffer) { }
        @Override void upload(Handle buffer, long offset, MemorySegment source, long count) { }
        @Override void download(Handle buffer, long offset, MemorySegment target, long count) { }
        @Override NativeCreateResult createMpsGraphExecutableNative(
                Handle context, int[] ranks, long[] dimensions,
                MetalMpsGraphProgram graphProgram, int[] feeds, int[] targets) {
            nativeAllocations.incrementAndGet();
            return new NativeCreateResult(0, handle());
        }
        @Override int releaseExecutableNative(Handle executable) { return 0; }
        @Override int runExecutableNative(
                Handle executable, int inputCount, MemorySegment inputs,
                int outputCount, MemorySegment outputs) { return 0; }
        @Override NativeCreateResult createNegKernelPipelineNative(
                Handle context, long elementCount) {
            nativeAllocations.incrementAndGet();
            return new NativeCreateResult(0, handle());
        }
        @Override int releaseNegKernelPipelineNative(Handle pipeline) { return 0; }
        @Override int runNegKernelPipelineNative(
                Handle pipeline, Handle input, Handle output) { return 0; }
        @Override public void close() { }

        private synchronized Handle handle() {
            return new Handle(MemorySegment.ofAddress(nextHandle++));
        }
    }
}
