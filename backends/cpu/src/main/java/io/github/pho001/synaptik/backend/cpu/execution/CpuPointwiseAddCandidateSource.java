package io.github.pho001.synaptik.backend.cpu.execution;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import java.lang.classfile.ClassFile;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;

/**
 * Builds the sole CPU-0005 ordered scalar native-segment {@code ADD} partition candidate.
 *
 * <p>Every node is checked against {@link CpuCapabilityProvider} before a candidate is returned.
 * Buffer requirements are interned by graph value in first encounter order, use exact checked
 * dense byte geometry, and are reused by identity across node recipes. The source performs no
 * slot assignment, artifact work, allocation, binding, or execution.</p>
 */
final class CpuPointwiseAddCandidateSource implements CpuPortableCandidateSource {
    private final CpuCapabilityProvider capabilityProvider = new CpuCapabilityProvider();

    /**
     * Validates every node through public capability truth and returns one complete candidate.
     *
     * @param context non-null complete partition projection
     * @return singleton deterministic candidate list
     * @throws IllegalArgumentException if any partition node is outside the exact ADD matrix or
     *     aliases an input from its output, if the partition is empty, or if checked static
     *     geometry overflows
     */
    @Override
    public List<CpuPortablePartitionCandidate> candidates(
            PrepareContext<CpuPortableAnalysisInputs> context) {
        if (context.nodes().isEmpty()) throw new IllegalArgumentException(
                "CPU portable partition must contain at least one node");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        for (GraphValue value : context.values()) values.put(value.id(), value);
        var declarations = new LinkedHashMap<ValueId, PreparationResourceRequirement.Buffer>();
        var kernels = new ArrayList<CpuPortableKernelCandidate>(context.nodes().size());
        for (int nodeIndex = 0; nodeIndex < context.nodes().size(); nodeIndex++) {
            var node = context.nodes().get(nodeIndex);
            var inputDescriptors = node.inputs().stream()
                    .map(values::get).map(GraphValue::descriptor).toList();
            var outputDescriptors = node.outputs().stream()
                    .map(values::get).map(GraphValue::descriptor).toList();
            var query = new OperationCapabilityQuery(
                    node.operation(), inputDescriptors, outputDescriptors);
            if (!capabilityProvider.supports(query)) throw new IllegalArgumentException(
                    "nodes[" + nodeIndex + "] is not supported by CPU pointwise ADD");
            var left = values.get(node.inputs().get(0));
            var right = values.get(node.inputs().get(1));
            var output = values.get(node.outputs().get(0));
            if (output.id().equals(left.id()) || output.id().equals(right.id())) {
                throw new IllegalArgumentException(
                        "nodes[" + nodeIndex + "] output must not alias an input");
            }
            long elementCount = left.descriptor().shape().knownElementCount().orElseThrow();
            DataType dataType = left.descriptor().dataType();
            var lowering = new CpuPointwiseAddLowering(left.id(), right.id(), output.id(),
                    dataType, elementCount);
            var nodeRequirements = new LinkedHashSet<PreparationResourceRequirement>();
            var bufferUses = new ArrayList<CpuPortableKernelCandidate.BufferUse>(3);
            addUse(declarations, nodeRequirements, bufferUses, left, dataType, elementCount);
            addUse(declarations, nodeRequirements, bufferUses, right, dataType, elementCount);
            addUse(declarations, nodeRequirements, bufferUses, output, dataType, elementCount);
            var specialization = specialization(lowering);
            kernels.add(new CpuPortableKernelCandidate(specialization,
                    new CpuPointwiseAddKernelEmitter(lowering), List.copyOf(nodeRequirements),
                    bufferUses, List.of(),
                    (state, entryPoint, ignored, parallel, workers, arguments, workspaces) ->
                            new CpuPointwiseAddInvocation(entryPoint,
                                    segment(arguments[0]), segment(arguments[1]),
                                    segment(arguments[2]), lowering.elementCount())));
        }
        return List.of(new CpuPortablePartitionCandidate(
                new ArrayList<>(declarations.values()), kernels));
    }

    private static void addUse(
            LinkedHashMap<ValueId, PreparationResourceRequirement.Buffer> declarations,
            LinkedHashSet<PreparationResourceRequirement> nodeRequirements,
            List<CpuPortableKernelCandidate.BufferUse> uses,
            GraphValue value,
            DataType dataType,
            long elementCount) {
        long byteSize = Math.multiplyExact(elementCount, dataType.byteWidth());
        var requirement = declarations.computeIfAbsent(value.id(), ignored ->
                new PreparationResourceRequirement.Buffer(
                        value.id(), byteSize, dataType.byteWidth()));
        if (requirement.byteSize() != byteSize
                || requirement.byteAlignment() != dataType.byteWidth()) {
            throw new IllegalArgumentException("inconsistent shared buffer geometry for " + value.id());
        }
        nodeRequirements.add(requirement);
        uses.add(new CpuPortableKernelCandidate.BufferUse(requirement, 0));
    }

    private static CpuKernelSpecialization specialization(CpuPointwiseAddLowering lowering) {
        DataType type = lowering.dataType();
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION,
                lowering.fingerprint(), CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(argument(type, BufferAccess.READ_ONLY),
                        argument(type, BufferAccess.READ_ONLY),
                        argument(type, BufferAccess.WRITE_ONLY)),
                List.of(), 0, null, ByteOrder.nativeOrder(), 1, 1,
                CpuKernelSpecialization.Tail.NONE,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    private static CpuKernelSpecialization.Argument argument(DataType type, BufferAccess access) {
        return new CpuKernelSpecialization.Argument(type,
                CpuKernelSpecialization.Carrier.MEMORY_SEGMENT, access, true, 0, List.of(1L));
    }

    private static java.lang.foreign.MemorySegment segment(CpuBufferArgument argument) {
        return ((CpuBufferArgument.Segment) argument).segment();
    }
}
