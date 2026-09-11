package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuMaterializationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferRepresentation;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable reusable OpenBLAS copy-in/GEMM/copy-out recipe. Cold binding validates the complete
 * provider, buffer, workspace, affine-copy, and overlap contract before returning an invocation.
 * Hot execution performs selected input copies left then right, exactly one typed provider call,
 * and the optional output copy, with no route decision or late fallback.
 */
public final class CpuOpenBlasPreparedExecutable extends PreparedExecutable {
    private final CpuOpenBlasRoutePlan plan;
    private final CpuOpenBlasInvocation invocation;
    private final List<CpuGeneratedKernel> inputCopyArtifacts;
    private final Optional<CpuGeneratedKernel> outputCopyArtifact;

    /**
     * Creates one exact reusable finalized native sequence. The recipe snapshots its artifact
     * collections and borrows, but does not close or configure, the provider invocation. Physical
     * workspaces are supplied separately by each {@link RunState}.
     *
     * @param memoryPlan exact shared memory plan containing every selected buffer and workspace
     * @param buffers exactly three ordered left, right, and output selections
     * @param workspaces ordered selections for every route workspace declaration
     * @param plan immutable analysis-selected OpenBLAS route facts
     * @param invocation borrowed provider invocation that must remain open and externally
     *     coordinated as single-threaded throughout use
     * @param inputCopyArtifacts ordered left-then-right generated artifacts for selected copies
     * @param outputCopyArtifact generated copy-out artifact exactly when output copy is selected
     * @throws NullPointerException if a required reference or collection element is {@code null}
     * @throws IllegalArgumentException if resource cardinality or generated specialization does
     *     not match the selected route
     */
    public CpuOpenBlasPreparedExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            CpuOpenBlasRoutePlan plan, CpuOpenBlasInvocation invocation,
            List<CpuGeneratedKernel> inputCopyArtifacts,
            Optional<CpuGeneratedKernel> outputCopyArtifact) {
        super(memoryPlan, buffers, workspaces, List.of(BufferAccess.READ_ONLY,
                BufferAccess.READ_ONLY, BufferAccess.WRITE_ONLY));
        this.plan = Objects.requireNonNull(plan, "plan");
        this.invocation = Objects.requireNonNull(invocation, "invocation");
        this.inputCopyArtifacts = List.copyOf(inputCopyArtifacts);
        this.outputCopyArtifact = Objects.requireNonNull(outputCopyArtifact,
                "outputCopyArtifact");
        if (buffers.size() != 3 || workspaces.size() != plan.workspaceRequirements().size()
                || this.inputCopyArtifacts.size() != plan.inputMaterializations().size()
                || this.outputCopyArtifact.isPresent() != plan.outputCopy().isPresent()) {
            throw new IllegalArgumentException("OpenBLAS resource cardinality disagrees");
        }
        for (int index = 0; index < this.inputCopyArtifacts.size(); index++) {
            if (!this.inputCopyArtifacts.get(index).specialization().equals(
                    plan.inputMaterializations().get(index).copySpecialization())) {
                throw new IllegalArgumentException("OpenBLAS input-copy artifact disagrees");
            }
        }
        if (this.outputCopyArtifact.isPresent() && !this.outputCopyArtifact.orElseThrow()
                .specialization().equals(plan.outputCopy().orElseThrow().copySpecialization())) {
            throw new IllegalArgumentException("OpenBLAS output-copy artifact disagrees");
        }
    }

    /**
     * Creates the compatibility recipe for a direct route with no generated copies.
     *
     * @param memoryPlan exact shared memory plan
     * @param buffers exactly three ordered direct matrix selections
     * @param workspaces empty workspace selections
     * @param plan immutable direct OpenBLAS route facts
     * @param invocation borrowed provider invocation that must outlive prepared use
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if the supplied plan or resources are not direct and exact
     */
    public CpuOpenBlasPreparedExecutable(PreparedMemoryPlan memoryPlan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            CpuOpenBlasRoutePlan plan, CpuOpenBlasInvocation invocation) {
        this(memoryPlan, buffers, workspaces, plan, invocation, List.of(), Optional.empty());
    }

    /**
     * Returns the analysis-selected facts used by every later binding.
     *
     * @return the retained immutable route plan; never {@code null}
     */
    public CpuOpenBlasRoutePlan routePlan() { return plan; }

    /**
     * Checks the cold-bound boundary's type, accessibility, mutability, and selected direct or
     * copied carrier form without retaining it.
     *
     * @param index zero-based left, right, or output boundary position
     * @param representation candidate runtime representation; may be {@code null}
     * @return whether the representation satisfies the route's preliminary boundary contract
     */
    @Override protected boolean acceptsBufferRepresentation(int index,
            BufferRepresentation representation) {
        if (!(representation instanceof CpuBufferRepresentation cpu) || !cpu.isAccessible()
                || cpu.dataType() != plan.dataType()) return false;
        try {
            CpuBufferArgument argument = cpu.argument();
            if (index == plan.outputBoundaryPosition() && argument.readOnly()) return false;
            if (copied(index)) return carrierAccess(argument) == expectedCarrier(index);
            return argument instanceof CpuBufferArgument.Segment segment
                    && segment.segment().isNative();
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    /**
     * Checks one selected workspace for CPU ownership, accessibility, exact size, and alignment.
     *
     * @param index zero-based position in the route's ordered workspace declarations
     * @param representation candidate runtime representation; may be {@code null}
     * @return whether the workspace satisfies its selected declaration
     */
    @Override protected boolean acceptsWorkspaceRepresentation(int index,
            WorkspaceRepresentation representation) {
        if (!(representation instanceof CpuContiguousWorkspace workspace)
                || !workspace.isAccessible()) return false;
        try {
            var requirement = plan.workspaceRequirements().get(index);
            MemorySegment segment = workspace.writableSegment();
            return segment.byteSize() == requirement.byteSize()
                    && Math.floorMod(segment.address(), requirement.byteAlignment()) == 0;
        } catch (IllegalArgumentException | IllegalStateException incompatible) {
            return false;
        }
    }

    /**
     * Completes cold validation and binds one invocation only after validating all provider,
     * buffer, workspace, generated-copy, writability, span, alignment, and overlap preconditions.
     * No copy or provider mutation occurs before this method returns successfully.
     *
     * @param state live per-run state retained by the returned invocation
     * @param buffers compatible ordered left, right, and output representations
     * @param workspaces compatible distinct run-owned route workspaces
     * @return a non-null invocation that performs left copy, right copy, one GEMM, then output
     *     copy according to the immutable route plan
     * @throws IllegalArgumentException if any exact representation, geometry, writability, or
     *     overlap invariant disagrees
     * @throws IllegalStateException if the borrowed provider is closed or not single-threaded
     * @throws ArithmeticException if exact bound address or span arithmetic overflows
     */
    @Override protected BoundInvocation bindCompatible(RunState state,
            BufferRepresentation[] buffers, WorkspaceRepresentation[] workspaces) {
        if (!invocation.isOpen()) throw new IllegalStateException(
                "borrowed OpenBLAS provider is closed");
        if (invocation.threadCount() != 1) throw new IllegalStateException(
                "borrowed OpenBLAS provider must remain single-threaded");
        var arguments = new CpuBufferArgument[3];
        for (int index = 0; index < arguments.length; index++) {
            CpuBufferRepresentation cpu = (CpuBufferRepresentation) buffers[index];
            arguments[index] = cpu.argument();
            if (cpu.dataType() != plan.dataType()) throw new IllegalArgumentException(
                    "OpenBLAS bound data type changed");
        }
        if (overlaps(arguments[2], arguments[0]) || overlaps(arguments[2], arguments[1])) {
            throw new IllegalArgumentException("OpenBLAS output must not overlap an input");
        }
        var workspaceSegments = new ArrayList<MemorySegment>(workspaces.length);
        for (int index = 0; index < workspaces.length; index++) {
            MemorySegment segment = ((CpuContiguousWorkspace) workspaces[index]).writableSegment();
            validateSegment(segment, plan.workspaceRequirements().get(index).byteSize(),
                    plan.dataType(), true);
            for (CpuBufferArgument argument : arguments) if (overlaps(segment, segment(argument)))
                throw new IllegalArgumentException("OpenBLAS workspace overlaps a buffer");
            for (MemorySegment other : workspaceSegments) if (overlaps(segment, other))
                throw new IllegalArgumentException("OpenBLAS workspaces must be disjoint");
            workspaceSegments.add(segment);
        }

        MemorySegment left = effectiveInput(arguments[0], plan.leftMaterialization(),
                workspaceSegments);
        MemorySegment right = effectiveInput(arguments[1], plan.rightMaterialization(),
                workspaceSegments);
        MemorySegment output = plan.outputCopy().isPresent()
                ? workspace(plan.outputCopy().orElseThrow().workspaceRequirementId(),
                        workspaceSegments)
                : direct(arguments[2]);
        validateSegment(left, matrixBytes(plan.m(), plan.k(), plan.dataType()), plan.dataType(),
                false);
        validateSegment(right, matrixBytes(plan.k(), plan.n(), plan.dataType()), plan.dataType(),
                false);
        validateSegment(output, matrixBytes(plan.m(), plan.n(), plan.dataType()), plan.dataType(),
                true);

        var inputCalls = new ArrayList<GeneratedCopyCall>(inputCopyArtifacts.size());
        List<CpuMaterializationPlan> inputPlans = plan.inputMaterializations();
        for (int index = 0; index < inputPlans.size(); index++) {
            CpuMaterializationPlan copy = inputPlans.get(index);
            CpuBufferArgument source = arguments[copy.sourceBoundaryIndex()];
            validateCopyArgument(source, copy.sourceBinding(), copy.dataType(), false);
            long[] geometry = copy.affineAddressPairs();
            addBase(geometry, 0, source, copy.dataType());
            inputCalls.add(copyCall(inputCopyArtifacts.get(index), carrier(source),
                    workspace(copy.workspaceRequirementId(), workspaceSegments), geometry,
                    copy.elementCount()));
        }
        Optional<GeneratedCopyCall> outputCall = plan.outputCopy().map(copy -> {
            validateCopyArgument(arguments[2], copy.destinationBinding(), copy.dataType(), true);
            long[] geometry = copy.affineAddressPairs();
            addBase(geometry, 1, arguments[2], copy.dataType());
            return copyCall(outputCopyArtifact.orElseThrow(),
                    workspace(copy.workspaceRequirementId(), workspaceSegments),
                    carrier(arguments[2]), geometry, copy.elementCount());
        });
        List<GeneratedCopyCall> copies = List.copyOf(inputCalls);
        return new BoundInvocation(state) {
            @Override protected void executeBound() {
                try {
                    for (GeneratedCopyCall copy : copies) copy.invoke();
                    if (plan.dataType() == DataType.FLOAT32) {
                        invocation.sgemm(plan.m(), plan.n(), plan.k(), 1.0f,
                                left, right, 0.0f, output);
                    } else {
                        invocation.dgemm(plan.m(), plan.n(), plan.k(), 1.0d,
                                left, right, 0.0d, output);
                    }
                    if (outputCall.isPresent()) outputCall.orElseThrow().invoke();
                } catch (RuntimeException | Error failure) { throw failure; }
                catch (Throwable failure) { throw new IllegalStateException(
                        "generated OpenBLAS representation copy failed", failure); }
            }
        };
    }

    private boolean copied(int position) {
        return position == 0 && plan.representation().copiesLeft()
                || position == 1 && plan.representation().copiesRight()
                || position == 2 && plan.representation().copiesOutput();
    }

    private CpuKernelSpecialization.CarrierAccess expectedCarrier(int position) {
        if (position == 0) return plan.leftMaterialization().orElseThrow().sourceCarrier();
        if (position == 1) return plan.rightMaterialization().orElseThrow().sourceCarrier();
        return plan.outputCopy().orElseThrow().destinationCarrier();
    }

    private MemorySegment effectiveInput(CpuBufferArgument direct,
            Optional<CpuMaterializationPlan> copy, List<MemorySegment> workspaces) {
        return copy.isPresent() ? workspace(copy.orElseThrow().workspaceRequirementId(), workspaces)
                : direct(direct);
    }

    private MemorySegment workspace(int requirementId, List<MemorySegment> workspaces) {
        for (int index = 0; index < plan.workspaceRequirements().size(); index++) {
            if (plan.workspaceRequirements().get(index).requirementId() == requirementId)
                return workspaces.get(index);
        }
        throw new IllegalArgumentException("OpenBLAS workspace identity is absent");
    }

    private static MemorySegment direct(CpuBufferArgument argument) {
        if (!(argument instanceof CpuBufferArgument.Segment direct)
                || !direct.segment().isNative()) throw new IllegalArgumentException(
                        "OpenBLAS direct boundary must remain native");
        return direct.segment();
    }

    private static void validateSegment(MemorySegment segment, long requiredBytes, DataType type,
            boolean writable) {
        if (!segment.isNative() || !segment.scope().isAlive()
                || !segment.isAccessibleBy(Thread.currentThread())
                || segment.byteSize() != requiredBytes
                || Math.floorMod(segment.address(), type.byteWidth()) != 0
                || writable && segment.isReadOnly()) {
            throw new IllegalArgumentException("OpenBLAS native segment geometry disagrees");
        }
    }

    private static long matrixBytes(long rows, long columns, DataType type) {
        return Math.multiplyExact(Math.multiplyExact(rows, columns), type.byteWidth());
    }

    private static void addBase(long[] geometry, int parity, CpuBufferArgument argument,
            DataType type) {
        long base = argument.byteOffset() / type.byteWidth();
        for (int index = parity; index < geometry.length; index += 2)
            geometry[index] = Math.addExact(geometry[index], base);
    }

    private static void validateCopyArgument(CpuBufferArgument argument,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding binding,
            DataType type, boolean writable) {
        long required = Math.multiplyExact(binding.referencedElementSpan(), type.byteWidth());
        if (argument.byteOffset() % type.byteWidth() != 0 || argument.byteSize() < required
                || writable && argument.readOnly()) {
            throw new IllegalArgumentException("OpenBLAS affine-copy buffer geometry disagrees");
        }
    }

    @FunctionalInterface private interface GeneratedCopyCall { void invoke() throws Throwable; }

    private static GeneratedCopyCall copyCall(CpuGeneratedKernel artifact, Object source,
            Object destination, long[] geometry, long elements) {
        MethodHandle handle = MethodHandles.insertArguments(artifact.entryPoint(), 0,
                source, destination, geometry);
        return () -> invokeVoid(handle, 0L, elements);
    }

    private static void invokeVoid(MethodHandle handle, long start, long end) throws Throwable {
        handle.invokeExact(start, end);
    }

    private static Object carrier(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> value.carrier();
            case CpuBufferArgument.Floats value -> value.carrier();
            case CpuBufferArgument.Shorts value -> value.carrier();
            case CpuBufferArgument.Ints value -> value.carrier();
            case CpuBufferArgument.Longs value -> value.carrier();
            case CpuBufferArgument.Bytes value -> value.carrier();
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }

    private static CpuKernelSpecialization.CarrierAccess carrierAccess(
            CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles ignored -> CpuKernelSpecialization.CarrierAccess.DOUBLE_ARRAY;
            case CpuBufferArgument.Floats ignored -> CpuKernelSpecialization.CarrierAccess.FLOAT_ARRAY;
            case CpuBufferArgument.Shorts ignored -> CpuKernelSpecialization.CarrierAccess.SHORT_ARRAY;
            case CpuBufferArgument.Ints ignored -> CpuKernelSpecialization.CarrierAccess.INT_ARRAY;
            case CpuBufferArgument.Longs ignored -> CpuKernelSpecialization.CarrierAccess.LONG_ARRAY;
            case CpuBufferArgument.Bytes ignored -> CpuKernelSpecialization.CarrierAccess.BYTE_ARRAY;
            case CpuBufferArgument.Segment ignored -> CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
        };
    }

    private static boolean overlaps(CpuBufferArgument left, CpuBufferArgument right) {
        return overlaps(segment(left), segment(right));
    }

    private static boolean overlaps(MemorySegment left, MemorySegment right) {
        return left.byteSize() != 0 && right.byteSize() != 0
                && left.asOverlappingSlice(right).isPresent();
    }

    private static MemorySegment segment(CpuBufferArgument argument) {
        return switch (argument) {
            case CpuBufferArgument.Doubles value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Floats value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Shorts value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Ints value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Longs value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Bytes value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Segment value -> value.segment();
        };
    }
}
