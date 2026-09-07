package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRepresentationDecision;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.security.MessageDigest;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/** Source-derived test fixture factory.  It deliberately has no production callers. */
final class CpuScalarImmediateClampMatrixOracle {
    enum Category {
        POSITIVE_ZERO, NEGATIVE_ZERO, POSITIVE_ONE, POSITIVE_TWO, NEGATIVE_ONE, OTHER,
        POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO, IDENTITY, SQUARE, RECIPROCAL,
        DIRECT_FRACTIONAL, DIRECT_INTEGRAL,
        CLAMP_NEGATIVE_ONE_POSITIVE_ONE, CLAMP_POSITIVE_ZERO_POSITIVE_TWO,
        CLAMP_NEGATIVE_ZERO_POSITIVE_ZERO, CLAMP_POSITIVE_ZERO_NEGATIVE_ZERO,
        CLAMP_EQUAL_OTHER, CLAMP_POSITIVE_ONE_POSITIVE_TWO,
        CLAMP_NEGATIVE_TWO_NEGATIVE_ONE, CLAMP_POSITIVE_TWO_DIRECT_THREE
    }
    record Fixture(String id, ScalarElementwiseKind operation, DataType type, Category category,
                   ScalarValue immediate, ScalarValue upper,
                   List<CpuKernelSpecialization.CarrierAccess> carriers,
                   CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference preference,
                   int parallelism) { }
    record Artifact(Fixture fixture, byte[] bytes, String hash, String descriptor,
                    String strategy, String generatorSchema, String classIdentitySchema,
                    String structuralKey) { }
    /** A finite source-reachable representative, including every code-shaping witness. */
    record Form(String id, Fixture fixture, Shape inputShape, LayoutDescriptor inputLayout,
                Shape outputShape, LayoutDescriptor outputLayout, String shape, String layout,
                CpuAccessPlan.Regime accessRegime,
                CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) { }
    record FormArtifact(Form form, Artifact artifact, int materializationCandidates,
                        int materializationSelected) { }

    static List<Fixture> fixtures() {
        var result = new java.util.ArrayList<Fixture>();
        for (DataType type : numericTypes()) {
            for (ScalarElementwiseKind operation : List.of(ScalarElementwiseKind.ADD,
                    ScalarElementwiseKind.SUB, ScalarElementwiseKind.MUL,
                    ScalarElementwiseKind.MIN, ScalarElementwiseKind.MAX)) {
                result.add(scalar(operation, type, Category.POSITIVE_ZERO, positiveZero(type)));
                if (type.isFloating()) {
                    result.add(scalar(operation, type, Category.NEGATIVE_ZERO, negativeZero(type)));
                }
                result.add(scalar(operation, type, Category.POSITIVE_ONE, one(type)));
                result.add(scalar(operation, type, Category.POSITIVE_TWO, two(type)));
                result.add(scalar(operation, type, Category.NEGATIVE_ONE, minusOne(type)));
                result.add(scalar(operation, type, Category.OTHER, other(type)));
            }
            if (type.isFloating()) {
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.POSITIVE_ZERO, positiveZero(type)));
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.NEGATIVE_ZERO, negativeZero(type)));
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.POSITIVE_ONE, one(type)));
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.POSITIVE_TWO, two(type)));
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.NEGATIVE_ONE, minusOne(type)));
                result.add(scalar(ScalarElementwiseKind.DIV, type, Category.OTHER, other(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.POW_POSITIVE_ZERO, positiveZero(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.POW_NEGATIVE_ZERO, negativeZero(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.IDENTITY, one(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.SQUARE, two(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.RECIPROCAL, minusOne(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.DIRECT_FRACTIONAL, other(type)));
                result.add(scalar(ScalarElementwiseKind.POW, type, Category.DIRECT_INTEGRAL, three(type)));
                result.addAll(clamps(type));
            }
        }
        return List.copyOf(result);
    }

    static Artifact artifact(Fixture fixture) {
        var context = context(fixture);
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        byte[] bytes = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        return new Artifact(fixture, bytes, sha256(bytes), route.specialization().entryType().descriptorString(),
                route.specialization().executionStrategy().toString(),
                Integer.toString(CpuGeneratorSchema.CURRENT_VERSION),
                Integer.toString(route.specialization().classIdentitySchema()),
                route.kernelIr().structuralKey());
    }

    /** Returns the exact lowered instruction so tests can check source-owned immediate facts. */
    static CpuKernelIr.Instruction instruction(Fixture fixture) {
        return new CpuPartitionPreparer().analyze(context(fixture)).plan().units().getFirst()
                .portablePlan().kernelIr().instructions().getFirst();
    }

    static boolean capability(Fixture fixture) {
        Shape shape = Shape.of(32);
        var descriptor = new TensorDescriptor(fixture.type(), shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        Operation operation = fixture.operation() == ScalarElementwiseKind.CLAMP
                ? new Operation(fixture.operation(), new ClampRangeAttrs(
                        fixture.immediate(), fixture.upper()))
                : new Operation(fixture.operation(), new ScalarValueAttrs(fixture.immediate()));
        return new CpuCapabilityProvider().supports(new OperationCapabilityQuery(
                operation, List.of(descriptor), List.of(descriptor)));
    }

    /**
     * Returns the finite compositional ledger basis. Semantic/category forms retain their own
     * artifacts; the added rows close every typed carrier pair and every source-normalized access
     * regime, degenerate shape, caller-orchestration identity, and candidate-only materialization
     * distinction without pretending those dimensions form an unproved Cartesian projection.
     */
    static List<Form> forms() {
        var forms = new java.util.ArrayList<Form>();
        for (var fixture : fixtures()) forms.add(form("FIXTURE-" + fixture.id(), fixture,
                Shape.of(32), LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS",
                CpuAccessPlan.Regime.DENSE_LINEAR));
        for (DataType type : numericTypes()) addTypedForms(forms, type);
        var base = fixture("ADD-FLOAT32-OTHER");
        var array = array(DataType.FLOAT32);
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        forms.add(form("ORCHESTRATION-SCALAR", with(base, "ORCHESTRATION-SCALAR", List.of(array, array),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR, 1), Shape.of(32),
                LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS", CpuAccessPlan.Regime.DENSE_LINEAR));
        forms.add(form("ORCHESTRATION-PARALLEL_SCALAR", with(base, "ORCHESTRATION-PARALLEL_SCALAR", List.of(array, array),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR, 4), Shape.of(32),
                LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS", CpuAccessPlan.Regime.DENSE_LINEAR));
        forms.add(form("ORCHESTRATION-VECTOR", with(base, "ORCHESTRATION-VECTOR", List.of(array, array),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE, 1), Shape.of(32),
                LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS", CpuAccessPlan.Regime.DENSE_LINEAR));
        forms.add(form("ORCHESTRATION-PARALLEL_VECTOR", with(base, "ORCHESTRATION-PARALLEL_VECTOR", List.of(array, array),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE, 4), Shape.of(32),
                LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS", CpuAccessPlan.Regime.DENSE_LINEAR));
        return List.copyOf(forms);
    }

    private static void addTypedForms(List<Form> forms, DataType type) {
        var base = fixture("ADD-" + type + "-OTHER");
        var array = array(type);
        var segment = CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        for (var carriers : List.of(List.of(segment, segment), List.of(array, segment),
                List.of(segment, array))) {
            String carrier = carriers.getFirst() == segment
                    ? (carriers.getLast() == segment ? "SEGMENT-SEGMENT" : "SEGMENT-ARRAY")
                    : "ARRAY-SEGMENT";
            forms.add(form("CARRIER-" + type + '-' + carrier, with(base,
                    "CARRIER-" + type + '-' + carrier, carriers, base.preference(), base.parallelism()),
                    Shape.of(32), LayoutDescriptor.contiguous(Shape.of(32)), "R1_32", "CONTIGUOUS",
                    CpuAccessPlan.Regime.DENSE_LINEAR));
        }
        var matrix = Shape.of(2, 3);
        forms.add(form("ACCESS-" + type + "-SCALAR_ALL_ZERO", base, matrix,
                LayoutDescriptor.of(matrix, new long[] {0, 0}, 0, true), "R2_2x3", "ZERO_STRIDES",
                CpuAccessPlan.Regime.SCALAR_ALL_ZERO));
        forms.add(form("ACCESS-" + type + "-LAST_AXIS_BIAS", base, matrix,
                LayoutDescriptor.of(matrix, new long[] {0, 1}, 0, true), "R2_2x3", "LAST_AXIS_BIAS",
                CpuAccessPlan.Regime.LAST_AXIS_BIAS));
        forms.add(form("ACCESS-" + type + "-BLOCK_OUTER", base, matrix,
                LayoutDescriptor.of(matrix, new long[] {5, 1}, 0, true), "R2_2x3", "BLOCK_OUTER",
                CpuAccessPlan.Regime.BLOCK_OUTER));
        forms.add(form("ACCESS-" + type + "-GENERAL_ODOMETER", base, matrix,
                LayoutDescriptor.of(matrix, new long[] {5, 2}, 0, true), "R2_2x3", "GENERAL_ODOMETER",
                CpuAccessPlan.Regime.GENERAL_ODOMETER));
        forms.add(form("SHAPE-" + type + "-RANK_ZERO", base, Shape.scalar(),
                LayoutDescriptor.contiguous(Shape.scalar()), "R0", "CONTIGUOUS",
                CpuAccessPlan.Regime.SCALAR_ALL_ZERO));
        forms.add(form("SHAPE-" + type + "-ZERO_WORK", base, Shape.of(0, 3),
                LayoutDescriptor.contiguous(Shape.of(0, 3)), "R2_0x3", "CONTIGUOUS",
                CpuAccessPlan.Regime.DENSE_LINEAR));
        if (type != DataType.BFLOAT16) {
            var policy = new CpuPartitionAnalysisInputs.MaterializationPolicy(true,
                    0, 1, 20, 1, 3, 1_000_000, 1, 1);
            forms.add(form("MATERIALIZATION-" + type + "-CANDIDATE_ONLY", base, matrix,
                    LayoutDescriptor.of(matrix, new long[] {5, 2}, 0, true), "R2_2x3",
                    "GENERAL_ODOMETER", CpuAccessPlan.Regime.GENERAL_ODOMETER, policy));
        }
    }

    static FormArtifact formArtifact(Form form) {
        var plan = new CpuPartitionPreparer().analyze(contextFor(form.fixture(), form.inputShape(), form.inputLayout(),
                form.outputShape(), form.outputLayout(), form.materializationPolicy())).plan();
        var route = plan.units().getFirst().portablePlan();
        var artifact = new Artifact(form.fixture(), new CpuClassFileKernelGenerator().generateClassBytes(
                route.specialization(), route.kernelIr()), "", route.specialization().entryType().descriptorString(),
                route.specialization().executionStrategy().toString(),
                Integer.toString(CpuGeneratorSchema.CURRENT_VERSION),
                Integer.toString(route.specialization().classIdentitySchema()),
                route.kernelIr().structuralKey());
        artifact = new Artifact(artifact.fixture(), artifact.bytes(), sha256(artifact.bytes()), artifact.descriptor(),
                artifact.strategy(), artifact.generatorSchema(), artifact.classIdentitySchema(),
                artifact.structuralKey());
        int materializationCandidates = Math.toIntExact(plan.representationDecisions().stream()
                .filter(CpuRepresentationDecision.Variant.class::isInstance)
                .map(CpuRepresentationDecision.Variant.class::cast)
                .filter(candidate -> !candidate.identity().materializations().isEmpty())
                .count());
        return new FormArtifact(form, artifact, materializationCandidates,
                plan.materializations().size());
    }

    private static Form form(String id, Fixture fixture, Shape shape, LayoutDescriptor layout, String shapeName,
            String layoutName, CpuAccessPlan.Regime regime) {
        return form(id, fixture, shape, layout, shapeName, layoutName, regime,
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
    }
    private static Form form(String id, Fixture fixture, Shape shape, LayoutDescriptor layout, String shapeName,
            String layoutName, CpuAccessPlan.Regime regime,
            CpuPartitionAnalysisInputs.MaterializationPolicy policy) {
        return new Form(id, fixture, shape, layout, shape, LayoutDescriptor.contiguous(shape),
                shapeName, layoutName, regime, policy);
    }
    private static Fixture with(Fixture base, String id, List<CpuKernelSpecialization.CarrierAccess> carriers,
            CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference preference, int parallelism) {
        return new Fixture(id, base.operation(), base.type(), base.category(), base.immediate(), base.upper(), carriers, preference, parallelism);
    }

    /** Executes only through the generated hidden class; this never invokes a reference kernel. */
    static Object generated(Fixture fixture) throws Throwable {
        return generated(fixture, 0L, 32L);
    }
    static Object generated(Fixture fixture, long start, long end) throws Throwable {
        var artifact = artifact(fixture);
        Object inputArray = input(fixture.type());
        Object outputArray = empty(fixture.type());
        Object input = fixture.carriers().getFirst() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? segment(inputArray) : inputArray;
        Object output = fixture.carriers().getLast() == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? segment(outputArray) : outputArray;
        MethodHandle handle = new CpuClassFileKernelGenerator().defineClassBytes(
                new CpuPartitionPreparer().analyze(context(fixture)).plan().units().getFirst().portablePlan()
                        .specialization(), artifact.bytes()).entryPoint();
        handle.invokeWithArguments(input, output, geometry(start), start, end);
        return outputArray;
    }

    /** Executes the exact prepared artifact for a compositional form and its resolved layout. */
    static Object generated(Form form) throws Throwable {
        var context = contextFor(form.fixture(), form.inputShape(), form.inputLayout(),
                form.outputShape(), form.outputLayout(), form.materializationPolicy());
        var route = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().portablePlan();
        var artifact = formArtifact(form).artifact();
        Object inputArray = input(form.fixture().type());
        Object outputArray = empty(form.fixture().type());
        Object input = form.fixture().carriers().getFirst()
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? segment(inputArray) : inputArray;
        Object output = form.fixture().carriers().getLast()
                == CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT
                ? segment(outputArray) : outputArray;
        new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), artifact.bytes())
                .entryPoint().invokeWithArguments(input, output, geometryFor(context), 0L,
                        form.outputShape().knownElementCount().orElseThrow());
        return outputArray;
    }

    /** Independent oracle for each extra compositional form; fixture forms use the full oracle. */
    static Object cleanJava(Form form) {
        if (form.id().startsWith("FIXTURE-")) return cleanJava(form.fixture());
        if (form.fixture().operation() != ScalarElementwiseKind.ADD
                || form.fixture().category() != Category.OTHER)
            throw new AssertionError("missing compositional oracle for " + form.id());
        Object source = input(form.fixture().type());
        Object output = empty(form.fixture().type());
        long count = form.outputShape().knownElementCount().orElseThrow();
        for (int ordinal = 0; ordinal < count; ordinal++) {
            int address = Math.toIntExact(address(form.inputShape(), form.inputLayout(), ordinal));
            switch (form.fixture().type()) {
                case BFLOAT16 -> {
                    short[] input = (short[]) source; short[] result = (short[]) output;
                    result[ordinal] = CpuScalarImmediateClampEquivalenceOracle.bfloat(
                            CpuScalarImmediateClampEquivalenceOracle.floatValue(input[address]) + .5f);
                }
                case FLOAT32 -> ((float[]) output)[ordinal] = ((float[]) source)[address] + .5f;
                case FLOAT64 -> ((double[]) output)[ordinal] = ((double[]) source)[address] + .5d;
                case INT32 -> ((int[]) output)[ordinal] = ((int[]) source)[address] + 3;
                case INT64 -> ((long[]) output)[ordinal] = ((long[]) source)[address] + 3L;
                default -> throw new AssertionError(form.fixture().type());
            }
        }
        return output;
    }

    private static long address(Shape shape, LayoutDescriptor layout, long ordinal) {
        long address = layout.storageOffset();
        long remaining = ordinal;
        long[] extents = shape.toLongArray();
        long[] strides = layout.strides();
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            long coordinate = remaining % extents[axis];
            remaining /= extents[axis];
            address += coordinate * strides[axis];
        }
        return address;
    }

    /** Independent typed clean-Java specialization; it deliberately knows nothing about CPU IR/codegen. */
    static Object cleanJava(Fixture fixture) {
        return switch (fixture.type()) {
            case BFLOAT16 -> bf16(fixture); case FLOAT32 -> f32(fixture); case FLOAT64 -> f64(fixture);
            case INT32 -> i32(fixture); case INT64 -> i64(fixture); default -> throw new AssertionError(fixture.type());
        };
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Fixture fixture) {
        Shape shape = Shape.of(32); var descriptor = new TensorDescriptor(fixture.type(), shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var input = new ValueId(0); var output = new ValueId(1); var nodeId = new NodeId(0);
        Operation operation = fixture.operation() == ScalarElementwiseKind.CLAMP
                ? new Operation(fixture.operation(), new ClampRangeAttrs(fixture.immediate(), fixture.upper()))
                : new Operation(fixture.operation(), new ScalarValueAttrs(fixture.immediate()));
        var node = new CompiledNode(nodeId, operation, List.of(input), List.of(output));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(nodeId));
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(fixture.preference(), fixture.parallelism(), fixture.parallelism(), 1);
        return new PrepareContext<>(partition, List.of(node), List.of(new GraphValue(input, descriptor), new GraphValue(output, descriptor)),
                List.of(new LogicalMemoryRequirement(input, descriptor, Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(output, descriptor, Optional.of(partition), List.of(), true)),
                java.util.Map.of(), new CpuPartitionAnalysisInputs(true, fixture.carriers(), config));
    }
    /** Exact source-derived preparation context, exposed only to the matrix carrier execution test. */
    static PrepareContext<CpuPartitionAnalysisInputs> contextFor(Fixture fixture) { return context(fixture); }
    /**
     * Builds the same one-occurrence context with deliberately chosen resolved layout witnesses.
     * This is test-only evidence for the lowering normalizer; it is not a second implementation
     * of access classification.
     */
    static PrepareContext<CpuPartitionAnalysisInputs> contextFor(Fixture fixture, Shape inputShape,
            LayoutDescriptor inputLayout, Shape outputShape, LayoutDescriptor outputLayout) {
        return contextFor(fixture, inputShape, inputLayout, outputShape, outputLayout,
                CpuPartitionAnalysisInputs.MaterializationPolicy.DISABLED);
    }
    static PrepareContext<CpuPartitionAnalysisInputs> contextFor(Fixture fixture, Shape inputShape,
            LayoutDescriptor inputLayout, Shape outputShape, LayoutDescriptor outputLayout,
            CpuPartitionAnalysisInputs.MaterializationPolicy materializationPolicy) {
        var input = new ValueId(0); var output = new ValueId(1); var nodeId = new NodeId(0);
        Operation operation = fixture.operation() == ScalarElementwiseKind.CLAMP
                ? new Operation(fixture.operation(), new ClampRangeAttrs(fixture.immediate(), fixture.upper()))
                : new Operation(fixture.operation(), new ScalarValueAttrs(fixture.immediate()));
        var node = new CompiledNode(nodeId, operation, List.of(input), List.of(output));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(nodeId));
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(fixture.preference(),
                fixture.parallelism(), fixture.parallelism(), 1);
        var inputDescriptor = new TensorDescriptor(fixture.type(), inputShape,
                Optional.of(inputLayout), false);
        var outputDescriptor = new TensorDescriptor(fixture.type(), outputShape,
                Optional.of(outputLayout), false);
        return new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(input, inputDescriptor), new GraphValue(output, outputDescriptor)),
                List.of(new LogicalMemoryRequirement(input, inputDescriptor, Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(output, outputDescriptor, Optional.of(partition), List.of(), true)),
                java.util.Map.of(), new CpuPartitionAnalysisInputs(true, fixture.carriers(), config,
                        materializationPolicy));
    }

    /** Packs direct pointwise geometry exactly as the prepared executable does for a full range. */
    static long[] geometryFor(PrepareContext<CpuPartitionAnalysisInputs> context) {
        var bindings = new CpuPartitionPreparer().analyze(context).plan().units().getFirst().accessBindings();
        int rank = bindings.getFirst().plan().iterationRank(); int count = bindings.size();
        long[] geometry = new long[2 * rank + count + count * rank + 2 * count];
        for (int axis = 0; axis < rank; axis++) geometry[axis] = bindings.getFirst().extents().get(axis);
        for (int value = 0; value < count; value++) {
            var binding = bindings.get(value);
            geometry[2 * rank + value] = binding.startAddress();
            for (int axis = 0; axis < rank; axis++)
                geometry[2 * rank + count + value * rank + axis] = binding.effectiveStrides().get(axis);
            long innerSize = 1;
            for (int axis = rank - binding.plan().contiguousSuffix(); axis < rank; axis++)
                innerSize = Math.multiplyExact(innerSize, binding.extents().get(axis));
            geometry[2 * rank + count + count * rank + count + value] = innerSize;
        }
        return geometry;
    }
    private static Object input(DataType type) { return switch (type) {
        case BFLOAT16 -> {
            short[] edges = {(short) 0x0000, (short) 0x8000, (short) 0x3fc0,
                    (short) 0xc020, (short) 0x7fc1, (short) 0x7f80, (short) 0xff80,
                    (short) 0x0001, (short) 0x8001, (short) 0x7f7f, (short) 0xff7f};
            short[] result = new short[32];
            for (int index = 0; index < result.length; index++) result[index] = edges[index % edges.length];
            yield result;
        }
        case FLOAT32 -> new float[] {0f,-0f,1.5f,-2.5f,Float.NaN,Float.POSITIVE_INFINITY,Float.NEGATIVE_INFINITY,Float.MIN_VALUE,-Float.MIN_VALUE,Float.MAX_VALUE,-Float.MAX_VALUE,3f,4f,5f,6f,7f,8f,9f,10f,11f,12f,13f,14f,15f,16f,17f,18f,19f,20f,21f,22f,23f};
        case FLOAT64 -> new double[] {0d,-0d,1.5d,-2.5d,Double.NaN,Double.POSITIVE_INFINITY,Double.NEGATIVE_INFINITY,Double.MIN_VALUE,-Double.MIN_VALUE,Double.MAX_VALUE,-Double.MAX_VALUE,3d,4d,5d,6d,7d,8d,9d,10d,11d,12d,13d,14d,15d,16d,17d,18d,19d,20d,21d,22d,23d};
        case INT32 -> new int[] {Integer.MIN_VALUE, Integer.MAX_VALUE, Integer.MIN_VALUE + 1,
                Integer.MAX_VALUE - 1, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7,
                -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        case INT64 -> new long[] {Long.MIN_VALUE, Long.MAX_VALUE, Long.MIN_VALUE + 1,
                Long.MAX_VALUE - 1, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7,
                -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        default -> throw new AssertionError(type); }; }
    private static Object empty(DataType type) { return switch(type) { case BFLOAT16->new short[32]; case FLOAT32->new float[32]; case FLOAT64->new double[32]; case INT32->new int[32]; case INT64->new long[32]; default->throw new AssertionError(type); }; }
    private static MemorySegment segment(Object array) {
        return switch (array) {
            case short[] value -> MemorySegment.ofArray(value);
            case float[] value -> MemorySegment.ofArray(value);
            case double[] value -> MemorySegment.ofArray(value);
            case int[] value -> MemorySegment.ofArray(value);
            case long[] value -> MemorySegment.ofArray(value);
            default -> throw new AssertionError(array.getClass());
        };
    }
    /* Package-visible because the carrier matrix deliberately invokes the exact generated entry. */
    static long[] geometry(){ return geometry(0); }
    private static long[] geometry(long base){ long[] g=new long[10];g[0]=32;g[2]=base;g[3]=base;g[4]=1;g[5]=1;g[8]=32;g[9]=32;return g; }
    private static short[] bf16(Fixture fixture) {
        short[] input = (short[]) input(DataType.BFLOAT16), output = new short[32];
        float immediate = CpuScalarImmediateClampEquivalenceOracle.floatValue(fixture.immediate().bfloat16Bits());
        float upper = fixture.upper() == null ? 0f
                : CpuScalarImmediateClampEquivalenceOracle.floatValue(fixture.upper().bfloat16Bits());
        switch (fixture.operation()) {
            case ADD -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(value(input[i]) + immediate); }
            case SUB -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(value(input[i]) - immediate); }
            case MUL -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(value(input[i]) * immediate); }
            case DIV -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(value(input[i]) / immediate); }
            case MIN -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(Math.min(value(input[i]), immediate)); }
            case MAX -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(Math.max(value(input[i]), immediate)); }
            case POW -> power(fixture, input, output, immediate);
            case CLAMP -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(Math.min(Math.max(value(input[i]), immediate), upper)); }
        }
        return output;
    }
    private static float[] f32(Fixture fixture) {
        float[] input = (float[]) input(DataType.FLOAT32), output = new float[32];
        float immediate = fixture.immediate().float32Value();
        float upper = fixture.upper() == null ? 0f : fixture.upper().float32Value();
        switch (fixture.operation()) {
            case ADD -> { for (int i = 0; i < output.length; i++) output[i] = input[i] + immediate; }
            case SUB -> { for (int i = 0; i < output.length; i++) output[i] = input[i] - immediate; }
            case MUL -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * immediate; }
            case DIV -> { for (int i = 0; i < output.length; i++) output[i] = input[i] / immediate; }
            case MIN -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(input[i], immediate); }
            case MAX -> { for (int i = 0; i < output.length; i++) output[i] = Math.max(input[i], immediate); }
            case POW -> power(fixture, input, output, immediate);
            case CLAMP -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(Math.max(input[i], immediate), upper); }
        }
        return output;
    }
    private static double[] f64(Fixture fixture) {
        double[] input = (double[]) input(DataType.FLOAT64), output = new double[32];
        double immediate = fixture.immediate().float64Value();
        double upper = fixture.upper() == null ? 0d : fixture.upper().float64Value();
        switch (fixture.operation()) {
            case ADD -> { for (int i = 0; i < output.length; i++) output[i] = input[i] + immediate; }
            case SUB -> { for (int i = 0; i < output.length; i++) output[i] = input[i] - immediate; }
            case MUL -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * immediate; }
            case DIV -> { for (int i = 0; i < output.length; i++) output[i] = input[i] / immediate; }
            case MIN -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(input[i], immediate); }
            case MAX -> { for (int i = 0; i < output.length; i++) output[i] = Math.max(input[i], immediate); }
            case POW -> power(fixture, input, output, immediate);
            case CLAMP -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(Math.max(input[i], immediate), upper); }
        }
        return output;
    }
    private static int[] i32(Fixture fixture) {
        int[] input = (int[]) input(DataType.INT32), output = new int[32];
        int immediate = fixture.immediate().int32Value();
        switch (fixture.operation()) {
            case ADD -> { for (int i = 0; i < output.length; i++) output[i] = input[i] + immediate; }
            case SUB -> { for (int i = 0; i < output.length; i++) output[i] = input[i] - immediate; }
            case MUL -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * immediate; }
            case MIN -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(input[i], immediate); }
            case MAX -> { for (int i = 0; i < output.length; i++) output[i] = Math.max(input[i], immediate); }
            default -> throw new AssertionError(fixture.operation());
        }
        return output;
    }
    private static long[] i64(Fixture fixture) {
        long[] input = (long[]) input(DataType.INT64), output = new long[32];
        long immediate = fixture.immediate().int64Value();
        switch (fixture.operation()) {
            case ADD -> { for (int i = 0; i < output.length; i++) output[i] = input[i] + immediate; }
            case SUB -> { for (int i = 0; i < output.length; i++) output[i] = input[i] - immediate; }
            case MUL -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * immediate; }
            case MIN -> { for (int i = 0; i < output.length; i++) output[i] = Math.min(input[i], immediate); }
            case MAX -> { for (int i = 0; i < output.length; i++) output[i] = Math.max(input[i], immediate); }
            default -> throw new AssertionError(fixture.operation());
        }
        return output;
    }
    private static void power(Fixture fixture, float[] input, float[] output, float immediate) {
        switch (fixture.category()) {
            case POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO -> { for (int i = 0; i < output.length; i++) output[i] = 1.0f; }
            case IDENTITY -> { for (int i = 0; i < output.length; i++) output[i] = input[i]; }
            case SQUARE -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * input[i]; }
            case RECIPROCAL -> { for (int i = 0; i < output.length; i++) output[i] = 1.0f / input[i]; }
            case DIRECT_FRACTIONAL, DIRECT_INTEGRAL -> { for (int i = 0; i < output.length; i++) output[i] = (float) StrictMath.pow(input[i], immediate); }
            default -> throw new AssertionError(fixture.category());
        }
    }
    private static void power(Fixture fixture, short[] input, short[] output, float immediate) {
        switch (fixture.category()) {
            case POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(1.0f); }
            case IDENTITY -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(value(input[i])); }
            case SQUARE -> { for (int i = 0; i < output.length; i++) { float value = value(input[i]); output[i] = bfloat(value * value); } }
            case RECIPROCAL -> { for (int i = 0; i < output.length; i++) output[i] = bfloat(1.0f / value(input[i])); }
            case DIRECT_FRACTIONAL, DIRECT_INTEGRAL -> { for (int i = 0; i < output.length; i++) output[i] = bfloat((float) StrictMath.pow(value(input[i]), immediate)); }
            default -> throw new AssertionError(fixture.category());
        }
    }
    private static void power(Fixture fixture, double[] input, double[] output, double immediate) {
        switch (fixture.category()) {
            case POW_POSITIVE_ZERO, POW_NEGATIVE_ZERO -> { for (int i = 0; i < output.length; i++) output[i] = 1.0d; }
            case IDENTITY -> { for (int i = 0; i < output.length; i++) output[i] = input[i]; }
            case SQUARE -> { for (int i = 0; i < output.length; i++) output[i] = input[i] * input[i]; }
            case RECIPROCAL -> { for (int i = 0; i < output.length; i++) output[i] = 1.0d / input[i]; }
            case DIRECT_FRACTIONAL, DIRECT_INTEGRAL -> { for (int i = 0; i < output.length; i++) output[i] = StrictMath.pow(input[i], immediate); }
            default -> throw new AssertionError(fixture.category());
        }
    }
    private static float value(short bits) {
        return CpuScalarImmediateClampEquivalenceOracle.floatValue(bits);
    }
    private static short bfloat(float value) {
        return CpuScalarImmediateClampEquivalenceOracle.bfloat(value);
    }

    private static Fixture scalar(ScalarElementwiseKind operation, DataType type, Category category, ScalarValue immediate) {
        return new Fixture(operation + "-" + type + "-" + category, operation, type, category, immediate, null,
                List.of(array(type), array(type)),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE, 4);
    }
    private static List<Fixture> clamps(DataType type) {
        return List.of(
                clamp(type, Category.CLAMP_NEGATIVE_ONE_POSITIVE_ONE,
                        minusOne(type), one(type)),
                clamp(type, Category.CLAMP_POSITIVE_ZERO_POSITIVE_TWO,
                        positiveZero(type), two(type)),
                clamp(type, Category.CLAMP_NEGATIVE_ZERO_POSITIVE_ZERO,
                        negativeZero(type), positiveZero(type)),
                clamp(type, Category.CLAMP_POSITIVE_ZERO_NEGATIVE_ZERO,
                        positiveZero(type), negativeZero(type)),
                clamp(type, Category.CLAMP_EQUAL_OTHER, other(type), other(type)),
                clamp(type, Category.CLAMP_POSITIVE_ONE_POSITIVE_TWO,
                        one(type), two(type)),
                clamp(type, Category.CLAMP_NEGATIVE_TWO_NEGATIVE_ONE,
                        minusTwo(type), minusOne(type)),
                clamp(type, Category.CLAMP_POSITIVE_TWO_DIRECT_THREE,
                        two(type), three(type)));
    }
    private static Fixture clamp(DataType type, Category category, ScalarValue lower, ScalarValue upper) {
        return new Fixture("CLAMP-" + type + '-' + category, ScalarElementwiseKind.CLAMP, type,
                category, lower, upper, List.of(array(type), array(type)),
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE, 4);
    }
    private static CpuKernelSpecialization.CarrierAccess array(DataType type) {
        return switch (type) {
            case BFLOAT16 -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case FLOAT32 -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case FLOAT64 -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case INT32 -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case INT64 -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            default -> throw new AssertionError(type);
        };
    }
    private static List<DataType> numericTypes() {
        return List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64,
                DataType.INT32, DataType.INT64);
    }
    private static Fixture fixture(String id) {
        return fixtures().stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow();
    }
    private static ScalarValue positiveZero(DataType type) { return scalar(type, 0); }
    private static ScalarValue negativeZero(DataType type) { return scalar(type, -0.0d); }
    private static ScalarValue one(DataType type) { return scalar(type, 1); }
    private static ScalarValue two(DataType type) { return scalar(type, 2); }
    private static ScalarValue minusOne(DataType type) { return scalar(type, -1); }
    private static ScalarValue minusTwo(DataType type) { return scalar(type, -2); }
    private static ScalarValue three(DataType type) { return scalar(type, 3); }
    private static ScalarValue other(DataType type) { return scalar(type, type == DataType.INT32 || type == DataType.INT64 ? 3 : 0.5); }
    private static ScalarValue scalar(DataType type, double value) {
        return switch (type) {
            case BFLOAT16 -> ScalarValue.bfloat16((float) value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case FLOAT64 -> ScalarValue.float64(value);
            case INT32 -> ScalarValue.int32((int) value);
            case INT64 -> ScalarValue.int64((long) value);
            default -> throw new AssertionError(type);
        };
    }
    static String sha256(byte[] value) {
        try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value)); }
        catch (Exception failure) { throw new AssertionError(failure); }
    }

    /** Regenerates all checked matrix resources from actual current preparation and Class-Files. */
    static void writeResources(Path directory) throws Exception {
        Files.createDirectories(directory);
        var fixtureLines = new java.util.ArrayList<String>();
        fixtureLines.add("id\toperation\ttype\tcategory\tmodel-capability\tlowering-admitted\tpreparation-eligible\tcarrier-input\tcarrier-output\tpreference\tparallelism\tentry-descriptor\tstrategy\tgenerator-schema\tclass-identity-schema\tstructural-key\tclass-file-sha256");
        for (var fixture : fixtures()) {
            var artifact = artifact(fixture);
            fixtureLines.add(String.join("\t", fixture.id(), fixture.operation().name(), fixture.type().name(),
                    fixture.category().name(), Boolean.toString(capability(fixture)), "true", "true",
                    fixture.carriers().getFirst().name(), fixture.carriers().getLast().name(),
                    fixture.preference().name(), Integer.toString(fixture.parallelism()), artifact.descriptor(),
                    artifact.strategy(), artifact.generatorSchema(), artifact.classIdentitySchema(),
                    artifact.structuralKey(), artifact.hash()));
        }
        writeSealed(directory.resolve("scalar-immediate-clamp-matrix-fixtures.tsv"), fixtureLines);

        var formLines = new java.util.ArrayList<String>();
        formLines.add("id\toperation\ttype\tcategory\tmodel-capability\tlowering-admitted\tpreparation-eligible\tcarrier-input\tcarrier-output\tshape\tlayout\taccess-regime\tpreference\tparallelism\tselected-strategy\tartifact-strategy\tentry-descriptor\tmatrix-version\tgenerator-schema\tclass-identity-schema\tstructural-key\tclass-file-sha256\tmaterialization-candidates\tmaterialization-selected\tdisposition");
        var formArtifacts = new java.util.LinkedHashMap<String, Artifact>();
        for (var form : forms()) {
            var result = formArtifact(form);
            var artifact = result.artifact();
            var plan = new CpuPartitionPreparer().analyze(contextFor(form.fixture(), form.inputShape(),
                    form.inputLayout(), form.outputShape(), form.outputLayout(), form.materializationPolicy())).plan();
            String selected = plan.units().getFirst().executionStrategy().toString().toUpperCase(java.util.Locale.ROOT)
                    .replace("COMPUTE=", "").replace("ORCHESTRATION=", "");
            selected = plan.units().getFirst().executionStrategy().equals(
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR)
                    ? "SCALAR" : plan.units().getFirst().executionStrategy().equals(
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.PARALLEL_SCALAR)
                    ? "PARALLEL_SCALAR" : plan.units().getFirst().executionStrategy().equals(
                    io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan.ExecutionStrategy.VECTOR)
                    ? "VECTOR" : "PARALLEL_VECTOR";
            String disposition = result.materializationCandidates() == 0 ? "DIRECT_ONLY"
                    : "CANDIDATE_AVAILABLE_DIRECT_SELECTED";
            formLines.add(String.join("\t", form.id(), form.fixture().operation().name(),
                    form.fixture().type().name(), form.fixture().category().name(),
                    Boolean.toString(capability(form.fixture())), "true", "true",
                    form.fixture().carriers().getFirst().name(), form.fixture().carriers().getLast().name(),
                    form.shape(), form.layout(), form.accessRegime().name(), form.fixture().preference().name(),
                    Integer.toString(form.fixture().parallelism()), selected, artifact.strategy(), artifact.descriptor(),
                    "matrix-v3", artifact.generatorSchema(), artifact.classIdentitySchema(),
                    artifact.structuralKey(), artifact.hash(),
                    Integer.toString(result.materializationCandidates()),
                    Integer.toString(result.materializationSelected()), disposition));
            formArtifacts.put(form.id(), artifact);
        }
        writeSealed(directory.resolve("scalar-immediate-clamp-matrix-forms.tsv"), formLines);

        var projectionLines = new java.util.ArrayList<String>();
        projectionLines.add("unit\tnormalizer-version\tdisposition\tsource-category\tspecialization-facts\tmembers\tmember-sha256\tconstant-locations");
        addIdentityProjection(projectionLines, formArtifacts, "array-scalar-orchestration",
                "ADD/FLOAT32/FLOAT_ARRAY-FLOAT_ARRAY/DENSE_LINEAR/SCALAR",
                "ORCHESTRATION-SCALAR", "ORCHESTRATION-PARALLEL_SCALAR");
        addIdentityProjection(projectionLines, formArtifacts, "array-vector-orchestration",
                "ADD/FLOAT32/FLOAT_ARRAY-FLOAT_ARRAY/DENSE_LINEAR/VECTOR",
                "ORCHESTRATION-VECTOR", "ORCHESTRATION-PARALLEL_VECTOR");
        writeSealed(directory.resolve("scalar-immediate-clamp-matrix-projections.tsv"), projectionLines);
    }

    private static void addIdentityProjection(List<String> lines, java.util.Map<String, Artifact> artifacts,
            String unit, String facts, String first, String second) {
        Artifact left = artifacts.get(first); Artifact right = artifacts.get(second);
        if (left == null || right == null || !java.util.Arrays.equals(left.bytes(), right.bytes())) {
            throw new AssertionError(unit + " is not byte-identical");
        }
        lines.add(String.join("\t", unit,
                CpuScalarImmediateClampMatrixStructuralTest.NORMALIZER_VERSION, "IDENTITY",
                "CALLER_ORCHESTRATION", facts,
                first + ',' + second, left.hash() + ',' + right.hash(), "NONE"));
    }

    private static void writeSealed(Path path, List<String> lines) throws Exception {
        String body = String.join("\n", lines) + "\n";
        String metadata = "# matrix-resource-sha256=" + sha256(body.getBytes(StandardCharsets.UTF_8))
                + "; canonical UTF-8 bytes of every header/data row plus LF, excluding this metadata line\n";
        Files.writeString(path, metadata + body, StandardCharsets.UTF_8);
    }
    private CpuScalarImmediateClampMatrixOracle() { }
}
