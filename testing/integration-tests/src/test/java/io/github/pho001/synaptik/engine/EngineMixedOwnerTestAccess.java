package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Test-only access to focused Engine ownership, lifecycle, and physical-identity evidence. */
public final class EngineMixedOwnerTestAccess {
    private EngineMixedOwnerTestAccess() {}

    /** Makes any later registry owner lookup fail without closing captured backend adapters. */
    public static void poisonBackendLookup(Engine engine) {
        Objects.requireNonNull(engine, "engine").poisonBackendLookupForTesting();
    }

    /** Returns exact planned owner names in partition order for one compiled graph. */
    public static List<String> partitionOwners(CompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        return partitionOwners(compileArtifacts(graph));
    }

    /**
     * Reads actual planned owner names without adding a Planning test dependency.
     *
     * @param artifacts non-null current compile result
     * @return immutable backend names in partition order
     */
    public static List<String> partitionOwners(CompileArtifacts artifacts) {
        List<?> partitions = (List<?>) invoke(artifacts, "partitions");
        var owners = new ArrayList<String>(partitions.size());
        for (Object partition : partitions) {
            Object owner = invoke(partition, "owner");
            owners.add((String) invoke(owner, "value"));
        }
        return List.copyOf(owners);
    }

    /**
     * Uses real capability predicates with test availability, without loading a native device.
     * Reflection avoids broadening this integration module's dependencies to Planning.
     *
     * @param outputs ordered non-null expressions compiled with ordinary Engine defaults
     * @return actual immutable Compiler artifacts; not device or prepared-transfer evidence
     * @throws AssertionError if the compile boundary fails
     */
    public static CompileArtifacts compileWithoutDevice(List<Tensor> outputs) {
        var metal = new BackendId("metal");
        var cpu = new BackendId("cpu");
        try {
            return (CompileArtifacts) GraphCompilationPort.class.getMethod("compile",
                    CompileMode.class, List.class, Optional.class, GraphOptimizationConfig.class,
                    BackendIntent.class, PartitionScoringConfig.class, List.class, List.class)
                    .invoke(null, CompileMode.FORWARD_ONLY, outputs, Optional.empty(),
                            GraphOptimizationConfig.standard(), BackendIntent.unconstrained(),
                            PartitionScoringConfig.neutral(),
                            List.<Object>of(new MetalCapabilityProvider(), new CpuCapabilityProvider()),
                            List.of(new BackendAvailabilitySnapshot(metal, Map.of(
                                            new BackendDeviceId(metal, "default"),
                                            DeviceClass.ACCELERATOR)),
                                    new BackendAvailabilitySnapshot(cpu, Map.of(
                                            new BackendDeviceId(cpu, "test-cpu"), DeviceClass.CPU))));
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError("cannot compile device-free ownership fixture", failure);
        }
    }

    /**
     * Reads logical cross-owner obligations, including producerless input fan-out, in consumer order.
     * This is deliberately separate from inspecting actual prepared transfer recipes.
     *
     * @param artifacts non-null compiled fixture
     * @return immutable deduplicated source/destination direction names in consumer order
     */
    public static List<String> logicalTransferDirections(CompileArtifacts artifacts) {
        List<?> partitions = (List<?>) invoke(artifacts, "partitions");
        List<?> requirements = (List<?>) invoke(invoke(artifacts, "memory"), "requirements");
        var directions = new ArrayList<String>();
        var visited = new HashSet<List<Object>>();
        for (Object partition : partitions) {
            for (Object requirement : requirements) {
                List<?> consumers = (List<?>) invoke(requirement, "consumerPartitions");
                if (!consumers.contains(partition)) continue;
                Optional<?> producer = (Optional<?>) invoke(requirement, "producerPartition");
                Object sourcePartition = producer.isPresent() ? producer.get() : consumers.getFirst();
                Object source = invoke(sourcePartition, "owner");
                Object destination = invoke(partition, "owner");
                if (!source.equals(destination)
                        && visited.add(List.of(invoke(requirement, "valueId"), destination))) {
                    directions.add(source.equals(new BackendId("cpu"))
                            ? "CPU_TO_METAL" : "METAL_TO_CPU");
                }
            }
        }
        return List.copyOf(directions);
    }

    /** Returns exact planned owner names in partition order for one advanced compiled graph. */
    public static List<String> partitionOwners(AdvancedCompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        Object artifacts = field(graph, "artifacts");
        List<?> partitions = (List<?>) invoke(artifacts, "partitions");
        var owners = new ArrayList<String>(partitions.size());
        for (Object partition : partitions) {
            Object owner = invoke(partition, "owner");
            owners.add((String) invoke(owner, "value"));
        }
        return List.copyOf(owners);
    }

    /** Returns the exact immutable Compiler artifacts retained by one compiled graph. */
    public static CompileArtifacts compileArtifacts(CompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        return (CompileArtifacts) field(graph, "artifacts");
    }

    /**
     * Snapshots actual CPU/Metal transfer recipes in schedule order, not inferred graph edges.
     * Reflection remains test-local and only reads the recipe's retained direction and descriptor.
     *
     * @param session non-null open prepared owner whose immutable recipe is inspected without running it
     * @return immutable ordered facts, empty for a schedule with no transfers
     * @throws AssertionError if a transfer is not the expected CPU/Metal recipe
     * @throws IllegalStateException if the prepared owner is closed
     */
    public static List<TransferFact> preparedTransfers(PreparedExecution session) {
        PreparedExecution prepared = Objects.requireNonNull(session, "session");
        var facts = new ArrayList<TransferFact>();
        int partitionIndex = 0;
        for (var step : prepared.execution().schedule().steps()) {
            if (step instanceof PreparedSchedule.ExecutionStep) {
                partitionIndex++;
            } else if (step instanceof PreparedSchedule.BufferTransferStep transferStep) {
                var transfer = transferStep.transfer();
                if (!(transfer instanceof CpuMetalPreparedBufferTransfer)) {
                    throw new AssertionError("unexpected transfer recipe: " + transfer.getClass());
                }
                facts.add(new TransferFact(
                        ((Enum<?>) field(transfer, "direction")).name(),
                        (TensorDescriptor) field(transfer, "descriptor"),
                        transfer.memoryPlan().buffers().get(transfer.bufferIndex()).byteSize(),
                        transfer.bufferIndex(),
                        transfer.sourceRepresentationIndex(),
                        transfer.destinationRepresentationIndex(),
                        partitionIndex));
            }
        }
        return List.copyOf(facts);
    }

    /**
     * Immutable evidence from one actual prepared transfer, including its destination ordering.
     *
     * @param direction exact retained CPU_TO_METAL or METAL_TO_CPU direction
     * @param descriptor exact retained logical descriptor; transfer performs no conversion
     * @param byteSize prepared physical buffer span in bytes
     * @param bufferIndex dense prepared logical-buffer position
     * @param sourceRepresentationIndex source owner's representation position
     * @param destinationRepresentationIndex destination owner's representation position
     * @param beforePartition zero-based execution position immediately following this transfer group
     */
    public record TransferFact(
            String direction, TensorDescriptor descriptor, long byteSize, int bufferIndex,
            int sourceRepresentationIndex, int destinationRepresentationIndex,
            int beforePartition) {}

    /** Returns the advanced owner behind an ordinary Engine for staged-gradient integration tests. */
    public static AdvancedEngine advancedEngine(Engine engine) {
        return Objects.requireNonNull(engine, "engine").lifecycleOwner();
    }

    /** Returns the exact immutable Compiler artifacts retained by one advanced compiled graph. */
    public static CompileArtifacts compileArtifacts(AdvancedCompiledGraph graph) {
        return Objects.requireNonNull(graph, "graph").artifacts();
    }

    /**
     * Returns identity-bearing inward publications and workspaces while the result remains open.
     */
    public static RunOwnedIdentities runOwnedIdentities(RunResult result) {
        Objects.requireNonNull(result, "result");
        AdvancedRunResult advanced = (AdvancedRunResult) field(result, "owner");
        io.github.pho001.synaptik.runtime.run.RunResult runtime =
                (io.github.pho001.synaptik.runtime.run.RunResult) field(advanced, "delegate");
        RunState state = (RunState) field(runtime, "runState");
        var publications = new ArrayList<Object>(runtime.resultCount());
        for (int index = 0; index < runtime.resultCount(); index++) {
            publications.add(runtime.publicationRepresentation(index));
        }
        var workspaces = new ArrayList<Object>(state.workspaceSlotCount());
        for (int index = 0; index < state.workspaceSlotCount(); index++) {
            workspaces.add(state.workspaceRepresentation(index));
        }
        return new RunOwnedIdentities(publications, workspaces);
    }

    private static Object field(Object owner, String name) {
        try {
            Field field = owner.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return field.get(owner);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "cannot inspect " + owner.getClass().getName() + "." + name,
                    failure);
        }
    }

    private static Object invoke(Object owner, String name) {
        try {
            return owner.getClass().getMethod(name).invoke(owner);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "cannot invoke " + owner.getClass().getName() + "." + name,
                    failure);
        }
    }

    /** Identity-bearing snapshots; list elements are never copied or closed by this value. */
    public record RunOwnedIdentities(List<Object> publications, List<Object> workspaces) {
        /** Defensively snapshots non-null identity lists. */
        public RunOwnedIdentities {
            publications = List.copyOf(publications);
            workspaces = List.copyOf(workspaces);
        }
    }
}
