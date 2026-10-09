package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class CpuOpenBlasRouteSelectorTest {
    @Test void selectsCompleteFittingThreadCandidateByCostThenLowerCount() {
        var config = new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(
                Optional.of(qualification()),
                List.of(candidate(8, 0), candidate(4, 5), candidate(2, 5), candidate(1, 9)),
                CpuPartitionAnalysisInputs.CostTerms.complete(100, 2, 10),
                CpuPartitionAnalysisInputs.RepresentationCostTerms.ZERO,
                java.util.OptionalLong.of(0), java.util.OptionalInt.of(0));
        var base = qualified(DataType.FLOAT32, false, 1, config);
        var inputs = new CpuPartitionAnalysisInputs(base.loweringManifestEnabled(),
                base.carrierPattern(), new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                        CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR,
                        4, 4, 1), base.materializationPolicy(), false,
                base.partialReductionEvidence(), base.boundaryStorageFacts(), config);
        var route = analyze(inputs).openBlasPlan().orElseThrow();
        assertAll(() -> assertEquals(2, route.threadCount()),
                () -> assertEquals(2, route.permitDemand()),
                () -> assertEquals(4, route.analysisCapacity()),
                () -> assertEquals(2, route.candidateOrder()),
                () -> assertEquals(route.threadCandidate(), config.threadCandidates().get(2)),
                () -> assertSame(config.qualification().orElseThrow(),
                        route.qualification().orElseThrow()));
    }

    @Test void validatesBoundedUniqueCompleteCandidateFacts() {
        var duplicate = List.of(candidate(1, 1), candidate(1, 2));
        assertAll(() -> assertThrows(IllegalArgumentException.class,
                        () -> candidate(0, 1)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(
                                Optional.of(qualification()),
                                duplicate, CpuPartitionAnalysisInputs.CostTerms.complete(1, 1, 1),
                                CpuPartitionAnalysisInputs.RepresentationCostTerms.ZERO,
                                java.util.OptionalLong.of(0), java.util.OptionalInt.of(0))),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(
                                Optional.of(qualification()),
                                java.util.stream.IntStream.rangeClosed(1, 33)
                                        .mapToObj(value -> candidate(value, value)).toList(),
                                CpuPartitionAnalysisInputs.CostTerms.complete(1, 1, 1),
                                CpuPartitionAnalysisInputs.RepresentationCostTerms.ZERO,
                                java.util.OptionalLong.of(0), java.util.OptionalInt.of(0))));
    }

    @Test void selectsExactDirectFloatRoutesAndRetainsPortableOracle() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            var analysis = new CpuPartitionPreparer().analyze(context(type, dense(Shape.of(2, 3)),
                    dense(Shape.of(3, 4)), dense(Shape.of(2, 4)), qualified(type, false, 1)));
            var plan = analysis.plan();
            var route = plan.openBlasPlan().orElseThrow();
            assertAll(() -> assertEquals(CpuPartitionPreparationPlan.Route.OPENBLAS, plan.route()),
                    () -> assertEquals(type, route.dataType()),
                    () -> assertEquals(2, route.m()), () -> assertEquals(4, route.n()),
                    () -> assertEquals(3, route.k()),
                    () -> assertEquals(CpuOpenBlasRoutePlan.Representation.DIRECT,
                            route.representation()),
                    () -> assertTrue(route.materialization().isEmpty()),
                    () -> assertTrue(route.openBlasCost() < route.portableCost()),
                    () -> assertTrue(plan.units().getFirst().portablePlan().kernelIr() != null),
                    () -> assertEquals(3, analysis.requirements().size()));
        }
    }

    @Test void defaultsAndIncompleteOrInsufficientCostsFailClosedToPortable() {
        var defaultPlan = new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                dense(Shape.of(2, 3)), dense(Shape.of(3, 4)), dense(Shape.of(2, 4)),
                CpuPartitionAnalysisInputs.DEFAULT)).plan();
        var tied = qualified(DataType.FLOAT32, false, 1,
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        qualification(), 1, 0, 0, 1, 0, 0, 0, 0));
        var threshold = qualified(DataType.FLOAT32, false, 1,
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        qualification(), 100, 0, 0, 99, 0, 0, 2, 0));
        assertAll(() -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        defaultPlan.route()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        analyze(tied).route()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        analyze(threshold).route()));
    }

    @Test void selectsExactlyOneAffineCopyAndDeclaresItsExactWorkspace() {
        Shape leftShape = Shape.of(2, 3);
        var inputs = qualified(DataType.FLOAT32, true, 3);
        var analysis = new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                new TensorDescriptor(DataType.FLOAT32, leftShape, Optional.of(
                        LayoutDescriptor.of(leftShape, new long[] {1, 2}, 0, true)), false),
                dense(Shape.of(3, 4)), dense(Shape.of(2, 4)), inputs));
        var route = analysis.plan().openBlasPlan().orElseThrow();
        var copy = route.materialization().orElseThrow();
        var workspace = route.workspaceRequirement().orElseThrow();
        assertAll(() -> assertEquals(CpuOpenBlasRoutePlan.Representation.COPY_LEFT,
                        route.representation()),
                () -> assertEquals(0, copy.sourceBoundaryIndex()),
                () -> assertEquals(6L * Float.BYTES, workspace.byteSize()),
                () -> assertEquals(Float.BYTES, workspace.byteAlignment()),
                () -> assertSame(workspace, analysis.requirements().getLast()),
                () -> assertEquals(4, analysis.requirements().size()));
    }

    @Test void selectsEveryBoundedCopyMaskWithDistinctOrderedWorkspaces() {
        record Case(boolean left, boolean right, boolean output,
                CpuOpenBlasRoutePlan.Representation expected) { }
        var cases = List.of(
                new Case(false, false, false, CpuOpenBlasRoutePlan.Representation.DIRECT),
                new Case(true, false, false, CpuOpenBlasRoutePlan.Representation.COPY_LEFT),
                new Case(false, true, false, CpuOpenBlasRoutePlan.Representation.COPY_RIGHT),
                new Case(false, false, true, CpuOpenBlasRoutePlan.Representation.COPY_OUTPUT),
                new Case(true, true, false, CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT),
                new Case(true, false, true, CpuOpenBlasRoutePlan.Representation.COPY_LEFT_OUTPUT),
                new Case(false, true, true, CpuOpenBlasRoutePlan.Representation.COPY_RIGHT_OUTPUT),
                new Case(true, true, true,
                        CpuOpenBlasRoutePlan.Representation.COPY_LEFT_RIGHT_OUTPUT));
        for (Case value : cases) {
            var route = analyze(maskInputs(DataType.FLOAT32, value.left(), value.right(),
                    value.output(), 1, defaultConfig())).openBlasPlan().orElseThrow();
            int copies = (value.left() ? 1 : 0) + (value.right() ? 1 : 0)
                    + (value.output() ? 1 : 0);
            assertAll(value.expected().name(),
                    () -> assertEquals(value.expected(), route.representation()),
                    () -> assertEquals(copies, route.workspaceRequirements().size()),
                    () -> assertEquals(java.util.stream.LongStream.range(8, 8L + copies).boxed()
                            .toList(), route.workspaceRequirements().stream()
                                    .map(io.github.pho001.synaptik.prepare.analysis
                                            .PreparationResourceRequirement.Workspace::requirementId)
                                    .toList()),
                    () -> assertEquals(value.left(), route.leftMaterialization().isPresent()),
                    () -> assertEquals(value.right(), route.rightMaterialization().isPresent()),
                    () -> assertEquals(value.output(), route.outputCopy().isPresent()));
        }
    }

    @Test void chargesCompleteRepresentationCostAcrossExpectedRunsAndFailsClosedWhenMissing() {
        var config = CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                qualification(),
                CpuPartitionAnalysisInputs.CostTerms.complete(1_000, 0, 0),
                CpuPartitionAnalysisInputs.CostTerms.complete(1, 1, 1),
                CpuPartitionAnalysisInputs.RepresentationCostTerms.complete(1, 1, 1, 1, 1, 1),
                0, 0);
        var route = analyze(maskInputs(DataType.FLOAT32, true, true, true, 2, config))
                .openBlasPlan().orElseThrow();
        assertAll(() -> assertEquals(2_000, route.portableCost()),
                () -> assertEquals(338, route.openBlasCost()),
                () -> assertEquals(104, route.workspaceBytes()),
                () -> assertEquals(18, route.inputCopiedElements()),
                () -> assertEquals(8, route.outputCopiedElements()),
                () -> assertEquals(2, route.expectedRunCount()));
        var missing = new CpuPartitionAnalysisInputs.OpenBlasRouteConfig(
                Optional.of(qualification()),
                List.of(new CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate(1,
                        CpuPartitionAnalysisInputs.CostTerms.complete(0, 0, 0))),
                CpuPartitionAnalysisInputs.CostTerms.complete(1_000, 0, 0),
                CpuPartitionAnalysisInputs.RepresentationCostTerms.MISSING,
                java.util.OptionalLong.of(0), java.util.OptionalInt.of(0));
        assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                analyze(maskInputs(DataType.FLOAT32, true, true, true, 2, missing)).route());
    }

    @Test void chargesEachRepresentationCoefficientIndependently() {
        long[] expected = {3, 104, 2, 18, 1, 8};
        for (int term = 0; term < expected.length; term++) {
            long[] values = new long[6];
            values[term] = 1;
            var config = CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                    qualification(),
                    CpuPartitionAnalysisInputs.CostTerms.complete(1_000, 0, 0),
                    CpuPartitionAnalysisInputs.CostTerms.complete(0, 0, 0),
                    CpuPartitionAnalysisInputs.RepresentationCostTerms.complete(values[0],
                            values[1], values[2], values[3], values[4], values[5]), 0, 0);
            var route = analyze(maskInputs(DataType.FLOAT32, true, true, true, 1, config))
                    .openBlasPlan().orElseThrow();
            assertEquals(expected[term], route.openBlasCost(), "term " + term);
        }
    }

    @Test void selectsRightCopyAndPrefersDirectWhenCopyCostsTie() {
        DataType type = DataType.FLOAT64;
        int width = type.byteWidth();
        var copyPolicy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 0, 10, 0, 1, 1_000_000, 0, 0);
        var rightCopyInputs = new CpuPartitionAnalysisInputs(false, List.of(
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, copyPolicy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, List.of(
                        new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width),
                        CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN,
                        new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width)),
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        qualification(), 100, 2, 10, 1, 1, 1, 1, 1));
        var right = new CpuPartitionPreparer().analyze(context(type, dense(Shape.of(2, 3)),
                strided(Shape.of(3, 4)), dense(Shape.of(2, 4)), rightCopyInputs)).plan()
                .openBlasPlan().orElseThrow();
        var directInputs = new CpuPartitionAnalysisInputs(false,
                java.util.Collections.nCopies(3,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, copyPolicy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE,
                java.util.Collections.nCopies(3,
                        new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width)),
                rightCopyInputs.openBlasRoute());
        var direct = new CpuPartitionPreparer().analyze(context(type, dense(Shape.of(2, 3)),
                dense(Shape.of(3, 4)), dense(Shape.of(2, 4)), directInputs)).plan()
                .openBlasPlan().orElseThrow();
        assertAll(() -> assertEquals(CpuOpenBlasRoutePlan.Representation.COPY_RIGHT,
                        right.representation()),
                () -> assertEquals(1, right.copiedBoundaryPosition()),
                () -> assertEquals(CpuOpenBlasRoutePlan.Representation.DIRECT,
                        direct.representation()));
    }

    @Test void rejectsUnprovedStorageWrongShapeCopyCeilingAndArithmeticOverflow() {
        var nonNative = qualified(DataType.FLOAT64, false, 1);
        var facts = new ArrayList<>(nonNative.boundaryStorageFacts());
        facts.set(2, CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN);
        CpuPartitionAnalysisInputs rejectedStorage = replace(nonNative, facts,
                nonNative.openBlasRoute());
        var batched = context(DataType.FLOAT64, dense(Shape.of(1, 2, 3)),
                dense(Shape.of(1, 3, 4)), dense(Shape.of(1, 2, 4)), qualified(DataType.FLOAT64,
                        false, 1));
        var copyCapped = qualified(DataType.FLOAT32, true, 1);
        CpuPartitionAnalysisInputs rejectedCopy = new CpuPartitionAnalysisInputs(false,
                copyCapped.carrierPattern(),
                copyCapped.portableExecution(), new CpuPartitionAnalysisInputs.MaterializationPolicy(
                        true, 1, 1, 10, 0, 1, 1, 0, 0), false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE,
                copyCapped.boundaryStorageFacts(), copyCapped.openBlasRoute());
        var overflowConfig = CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                qualification(), Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE,
                0, 0, 0, 0, 0);
        assertAll(() -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        analyze(rejectedStorage).route()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        new CpuPartitionPreparer().analyze(batched).plan().route()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                                strided(Shape.of(2, 3)), dense(Shape.of(3, 4)),
                                dense(Shape.of(2, 4)), rejectedCopy)).plan().route()),
                () -> assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE,
                        analyze(qualified(DataType.FLOAT32, false, 1, overflowConfig)).route()));
    }

    static CpuPartitionPreparationPlan analyze(CpuPartitionAnalysisInputs inputs) {
        return new CpuPartitionPreparer().analyze(context(DataType.FLOAT32,
                dense(Shape.of(2, 3)), dense(Shape.of(3, 4)), dense(Shape.of(2, 4)), inputs)).plan();
    }

    private static CpuPartitionAnalysisInputs qualified(DataType type, boolean copy, long runs) {
        return qualified(type, copy, runs,
                CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                        qualification(), 100, 2, 10, 1, 1, 1, 1, 1));
    }

    static CpuPartitionAnalysisInputs.OpenBlasRouteConfig defaultConfig() {
        return CpuPartitionAnalysisInputs.OpenBlasRouteConfig.qualifiedSingleThread(
                qualification(), 100, 2, 10, 1, 1, 1, 0, 0);
    }

    static CpuOpenBlasQualification qualification() {
        return new CpuOpenBlasQualification(CpuOpenBlasQualification.Scope.SESSION_ONLY,
                CpuOpenBlasQualifier.target("Linux", "x86_64", 64,
                        java.nio.ByteOrder.LITTLE_ENDIAN), Optional.empty(),
                new CpuOpenBlasQualification.SessionKey());
    }

    static CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate candidate(
            int threads, long fixed) {
        return new CpuPartitionAnalysisInputs.OpenBlasRouteConfig.ThreadCandidate(threads,
                CpuPartitionAnalysisInputs.CostTerms.complete(fixed, 0, 0));
    }

    static CpuPartitionAnalysisInputs maskInputs(DataType type, boolean left,
            boolean right, boolean output, long runs,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config) {
        int width = type.byteWidth();
        CpuKernelSpecialization.CarrierAccess array = type == DataType.FLOAT32
                ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
        var carriers = List.of(left ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                right ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                output ? array : CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        var nativeFact = new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width);
        var facts = List.of(left ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact,
                right ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact,
                output ? CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN : nativeFact);
        var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                0, 0, 10, 0, runs, 1_000_000, 0, 0);
        return new CpuPartitionAnalysisInputs(false, carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, facts, config);
    }

    private static CpuPartitionAnalysisInputs qualified(DataType type, boolean copy, long runs,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config) {
        int width = type.byteWidth();
        var carriers = copy ? List.of(type == DataType.FLOAT32
                        ? CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY
                        : CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT)
                : java.util.Collections.nCopies(3,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT);
        var facts = copy ? List.of(CpuPartitionAnalysisInputs.BoundaryStorageFact.UNKNOWN,
                new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width),
                new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width))
                : java.util.Collections.nCopies(3,
                        new CpuPartitionAnalysisInputs.BoundaryStorageFact(true, width));
        var policy = copy ? new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                1, 1, 10, 0, runs, 1_000_000, 0, 0)
                : CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED;
        return new CpuPartitionAnalysisInputs(false, carriers,
                CpuPartitionAnalysisInputs.PortableExecutionConfig.DEFAULT, policy, false,
                CpuPartitionAnalysisInputs.PartialReductionEvidence.NONE, facts, config);
    }

    private static CpuPartitionAnalysisInputs replace(CpuPartitionAnalysisInputs source,
            List<CpuPartitionAnalysisInputs.BoundaryStorageFact> facts,
            CpuPartitionAnalysisInputs.OpenBlasRouteConfig config) {
        return new CpuPartitionAnalysisInputs(source.loweringManifestEnabled(),
                source.carrierPattern(), source.portableExecution(),
                source.materializationPolicy(), source.conv2dMaterializedSuffixUnit(),
                source.partialReductionEvidence(), facts, config);
    }

    private static TensorDescriptor dense(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor strided(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1,
                        ((StaticDimension) shape.dimension(0)).size()}, 0, true)),
                false);
    }

    static PrepareContext<CpuPartitionAnalysisInputs> context(DataType type,
            TensorDescriptor left, TensorDescriptor right, TensorDescriptor output,
            CpuPartitionAnalysisInputs inputs) {
        left = typed(type, left); right = typed(type, right); output = typed(type, output);
        var node = new CompiledNode(new NodeId(0),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2)));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                List.of(node.id()));
        List<TensorDescriptor> descriptors = List.of(left, right, output);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < descriptors.size(); index++) {
            ValueId id = new ValueId(index);
            TensorDescriptor descriptor = descriptors.get(index);
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    index == 2 ? Optional.of(partition) : Optional.empty(),
                    index < 2 ? List.of(partition) : List.of(), index == 2));
        }
        return new PrepareContext<>(partition, List.of(node), values, memory, Map.of(), inputs);
    }

    private static TensorDescriptor typed(DataType type, TensorDescriptor descriptor) {
        return new TensorDescriptor(type, descriptor.shape(), descriptor.layout(),
                descriptor.requiresGrad());
    }
}
