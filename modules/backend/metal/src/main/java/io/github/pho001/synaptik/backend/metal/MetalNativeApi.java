package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Package-private typed seam for the ABI-v7 Metal foundation and bounded graph/custom-program C
 * ABI.
 *
 * <p>Handles remain opaque carrier segments inside this package. Implementations consume each
 * successful context or buffer handle exactly once through its matching release call. The
 * schema-twenty-one creator accepts zero feeds and compact materialized slots, while execution
 * still binds every ordered live slot buffer. Java preflight independently authenticates all seven
 * carriers, the 49 ordered casts, logical-versus-physical layouts, saved gradient roles, exact
 * replacement safety and fixed custom routing before native entry. FLOAT32 image folds may
 * accumulate overlap, including Compiler-generated Pool2d/Pool3d cotangents; FLOAT64/low pooling
 * cotangents, image folds and FOLD_AXIS require non-overlapping windows.
 * Every BFLOAT16/FLOAT16 value is custom-program-only. Direct mixed BFLOAT16/FLOAT16 WHERE and
 * comparison pairs reject independently of provider admission; explicit FLOAT32 casts remain legal.
 * Native status failures are unchecked and retain both the operation name and raw status value.
 */
abstract class MetalNativeApi implements AutoCloseable {
    static final int ABI_VERSION = 7;
    static final String EXECUTABLE_CREATE_OPERATION =
            "synaptik_metal_mpsgraph_executable_create";
    static final String EXECUTABLE_RELEASE_OPERATION =
            "synaptik_metal_mpsgraph_executable_release";
    static final String EXECUTABLE_RUN_OPERATION =
            "synaptik_metal_mpsgraph_executable_run";
    static final String NEG_KERNEL_PIPELINE_CREATE_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_create";
    static final String NEG_KERNEL_PIPELINE_RELEASE_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_release";
    static final String NEG_KERNEL_PIPELINE_RUN_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_run";

    /**
     * Opens and completely validates the production ABI without creating a Metal context.
     *
     * @param absoluteLibraryPath caller-selected absolute library path; not retained as mutable state
     * and must not be {@code null}
     * @return a new open API owner whose symbol-lookup lifetime must be closed; never {@code null}
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if the path is not absolute
     * @throws RuntimeException if loading, symbol resolution, or ABI validation fails
     * @throws Error if native linking or failure cleanup reports an error
     */
    static MetalNativeApi open(Path absoluteLibraryPath) {
        Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath");
        if (!absoluteLibraryPath.isAbsolute()) {
            throw new IllegalArgumentException("absoluteLibraryPath must be absolute");
        }
        return Ffm.open(absoluteLibraryPath);
    }

    /**
     * Creates one native default-device and command-queue context.
     *
     * @return a fresh non-null opaque handle owned by the caller
     * @throws RuntimeException if native creation fails or violates the ABI output contract
     */
    abstract Handle createContext();



    /**
     * Consumes one live context handle exactly once.
     *
     * @param context non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    abstract void releaseContext(Handle context);

    /**
     * Creates one fresh shared-storage buffer for the supplied logical byte size.
     *
     * @param context non-null live context handle; ownership remains with the caller
     * @param logicalByteSize exact non-negative logical byte extent
     * @return a fresh non-null opaque buffer handle owned by the caller
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails or violates the ABI output contract
     */
    abstract Handle createBuffer(Handle context, long logicalByteSize);

    /**
     * Consumes one live buffer handle exactly once.
     *
     * @param buffer non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    abstract void releaseBuffer(Handle buffer);

    /**
     * Copies one already validated byte range from caller-owned memory into a live buffer.
     *
     * @param buffer non-null live buffer handle; ownership remains with the caller
     * @param bufferOffset non-negative logical destination offset
     * @param source non-null native source slice, or the null-address segment for a zero-byte copy
     * @param byteCount non-negative number of bytes to copy
     * @throws RuntimeException if the native copy fails
     */
    abstract void upload(Handle buffer, long bufferOffset, MemorySegment source, long byteCount);

    /**
     * Copies one already validated byte range from a live buffer into caller-owned memory.
     *
     * @param buffer non-null live buffer handle; ownership remains with the caller
     * @param bufferOffset non-negative logical source offset
     * @param destination non-null writable native destination slice, or the null-address segment for
     * a zero-byte copy
     * @param byteCount non-negative number of bytes to copy
     * @throws RuntimeException if the native copy fails
     */
    abstract void download(
            Handle buffer, long bufferOffset, MemorySegment destination, long byteCount);

    /**
     * Compiles one shape-specialized whole-partition typed Metal program executable. The selected
     * route is fixed for the whole partition; a custom program uses typed BFLOAT16/FLOAT16 EXP
     * or SIGMOID steps and may contain an internal FLOAT32-only MPSGraph EXP or sign-guarded SIGMOID
     * boundary step, including after an explicit low-to-FLOAT32 cast. The caller must
     * release the returned executable exactly once, including when later preparation fails.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param values non-null explicit schema-twenty-one value descriptors in program-value order
     * @param graphProgram non-null schema-twenty-one typed node program in partition order
     * @param feedValueIndices non-null ordered boundary-feed value indices, not retained
     * @param targetValueIndices non-null ordered boundary-target value indices, not retained
     * @param route non-null selected {@code MPSGRAPH} or {@code CUSTOM_PROGRAM} partition route;
     *     singleton custom NEG has no program executable
     * @return a fresh non-null opaque executable handle owned by the caller and requiring release
     * @throws NullPointerException if a required reference or value descriptor is {@code null}
     * @throws IllegalArgumentException if the route, typed program, descriptors, indices, or
     *     encoded plan violate the Java preflight contract
     * @throws NativeFailure if native creation reports a non-success status
     * @throws IllegalStateException if native reports success without an executable handle
     */
    final Handle createProgramExecutable(
            Handle context,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram graphProgram,
            int[] feedValueIndices,
            int[] targetValueIndices,
            MetalPreparedRoute route) {
        return createProgramExecutable(
                context,
                values,
                graphProgram,
                feedValueIndices,
                targetValueIndices,
                route,
                null);
    }

    /**
     * Compiles a typed Metal program with an optional authenticated custom-step plan. A
     * {@code null} plan is computed for {@code CUSTOM_PROGRAM} and omitted for {@code MPSGRAPH}.
     * The caller owns the fresh handle and must release it exactly once, including after later
     * preparation failure.
     *
     * @param context non-null live context retained by the caller
     * @param values non-null schema-twenty-one descriptors in program-value order
     * @param graphProgram non-null typed node program in partition order
     * @param feedValueIndices non-null ordered boundary-feed indices, not retained
     * @param targetValueIndices non-null ordered boundary-target indices, not retained
     * @param route non-null selected {@code MPSGRAPH} or {@code CUSTOM_PROGRAM} partition route
     * @param fusionPlan optional plan for {@code CUSTOM_PROGRAM}; {@code null} requests planning,
     *     while {@code MPSGRAPH} requires {@code null}
     * @return fresh non-null executable handle owned by the caller and requiring release
     * @throws NullPointerException if a required reference or value descriptor is {@code null}
     * @throws IllegalArgumentException if route, program, descriptors, indices, or supplied plan
     *     violate Java preflight, including any fused step in a low-containing partition
     * @throws NativeFailure if native creation reports a non-success status
     * @throws IllegalStateException if native reports success without an executable handle
     */
    final Handle createProgramExecutable(
            Handle context,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram graphProgram,
            int[] feedValueIndices,
            int[] targetValueIndices,
            MetalPreparedRoute route,
            MetalPointwiseFusionPlan fusionPlan) {
        Objects.requireNonNull(context, "context");
        Objects.requireNonNull(graphProgram, "graphProgram");
        MetalPointwiseFusionPlan effectiveFusionPlan = graphProgram.resolvePlan(
                values,
                feedValueIndices,
                targetValueIndices,
                route,
                fusionPlan);
        ProgramExecutableAbi.validateCreate(
                values, graphProgram, feedValueIndices, targetValueIndices,
                route, effectiveFusionPlan);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment image = graphProgram.encodeNative(
                    arena,
                    values,
                    feedValueIndices,
                    targetValueIndices,
                    route,
                    effectiveFusionPlan);
            NativeCreateResult result = Objects.requireNonNull(
                    createProgramExecutableNative(context, image),
                    "native executable create result");
            return finishExecutableCreate(result);
        }
    }

    /**
     * Performs one already validated ABI-v7 program-image create invocation synchronously.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param programImage exact readable schema-twenty-one image, valid only for this call
     * @return non-null raw status/output-cell result for checked interpretation
     */
    abstract NativeCreateResult createProgramExecutableNative(
            Handle context, MemorySegment programImage);

    /**
     * Consumes one live native executable handle exactly once.
     *
     * @param executable non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    final void releaseExecutable(Handle executable) {
        Objects.requireNonNull(executable, "executable");
        checkExecutableStatus(EXECUTABLE_RELEASE_OPERATION,
                releaseExecutableNative(executable));
    }

    /**
     * Returns the raw status from one already validated executable release invocation.
     *
     * @param executable non-null live executable handle to consume
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int releaseExecutableNative(Handle executable);

    /**
     * Executes one compiled region with stable ordered direct buffer bindings.
     *
     * @param executable non-null live executable whose ownership remains with the caller
     * @param inputBuffers non-null stable input handles: feed order for MPSGraph, complete stable
     *     value-table order for shared custom programs
     * @param outputBuffers non-null stable target-ordered live buffer handles
     * @throws RuntimeException if validation or synchronous execution fails
     */
    final void runExecutable(
            Handle executable, int inputCount, MemorySegment inputBuffers,
            int outputCount, MemorySegment outputBuffers) {
        Objects.requireNonNull(executable, "executable");
        Objects.requireNonNull(inputBuffers, "inputBuffers");
        Objects.requireNonNull(outputBuffers, "outputBuffers");
        checkExecutableStatus(EXECUTABLE_RUN_OPERATION, runExecutableNative(
                executable, inputCount, inputBuffers, outputCount, outputBuffers));
    }

    /**
     * Returns the raw status from one already validated executable run invocation.
     *
     * @param executable non-null live executable whose ownership remains with the caller
     * @param inputCount positive validated input count
     * @param inputBuffers exact readable input-address segment
     * @param outputCount positive validated target count
     * @param outputBuffers exact readable target-address segment
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int runExecutableNative(
            Handle executable, int inputCount, MemorySegment inputBuffers,
            int outputCount, MemorySegment outputBuffers);

    /**
     * Compiles one fixed custom FLOAT32 NEG pipeline specialized to an element count.
     *
     * <p>The ABI carries {@code elementCount} as {@code uint64_t}. The custom kernel accepts only
     * {@code 1..UINT32_MAX}; zero or a larger non-negative carrier value fails with native {@link
   * Status#UNSUPPORTED_SHAPE} rather than narrowing or selecting another route.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param elementCount unsigned 64-bit ABI value represented by a non-negative Java     {@code long};
   * the successful custom domain is {@code 1..UINT32_MAX}
     * @return a fresh non-null opaque custom-pipeline handle owned by the caller
     * @throws IllegalArgumentException if {@code elementCount} is negative
     * @throws RuntimeException if the count is outside the native custom domain, construction     fails,
   * or the output-cell contract is violated
     */
    final Handle createNegKernelPipeline(Handle context, long elementCount) {
        Objects.requireNonNull(context, "context");
        if (elementCount < 0L) {
            throw new IllegalArgumentException("elementCount must be non-negative");
        }
        NativeCreateResult result = Objects.requireNonNull(
                createNegKernelPipelineNative(context, elementCount),
                "native NEG kernel pipeline create result");
        return finishCreate(
                NEG_KERNEL_PIPELINE_CREATE_OPERATION,
                result,
                this::releaseNegKernelPipelineNative);
    }

    /**
     * Returns the raw status/output-cell result of custom-pipeline creation.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param elementCount exact {@code uint64_t} carrier value represented by a non-negative Java
     *     {@code long}; native code enforces the successful {@code 1..UINT32_MAX} domain
     * @return non-null raw status and nullable output-cell handle
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract NativeCreateResult createNegKernelPipelineNative(Handle context, long elementCount);

    /**
     * Consumes one live custom-NEG pipeline handle exactly once.
     *
     * @param pipeline non-null live custom-pipeline handle owned by the caller
     * @throws NullPointerException if {@code pipeline} is {@code null}
     * @throws RuntimeException if native release fails
     */
    final void releaseNegKernelPipeline(Handle pipeline) {
        Objects.requireNonNull(pipeline, "pipeline");
        checkExecutableStatus(NEG_KERNEL_PIPELINE_RELEASE_OPERATION,
                releaseNegKernelPipelineNative(pipeline));
    }

    /**
     * Returns the raw status of one custom-pipeline release invocation.
     *
     * @param pipeline non-null live custom-pipeline handle to consume
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int releaseNegKernelPipelineNative(Handle pipeline);

    /**
     * Executes one custom NEG dispatch through direct input and assigned output buffer handles.
     *
     * <p>The matching native function submits one compute command and waits synchronously. It writes
     * the supplied output buffer directly and performs no explicit host staging or intermediate
     * output copy.
     *
     * @param pipeline non-null live custom pipeline whose ownership remains with the caller
     * @param inputBuffer non-null live direct input buffer
     * @param outputBuffer non-null live direct output buffer
     * @throws NullPointerException if an argument is {@code null}
     * @throws RuntimeException if native validation or execution fails
     */
    final void runNegKernelPipeline(Handle pipeline, Handle inputBuffer, Handle outputBuffer) {
        Objects.requireNonNull(pipeline, "pipeline");
        Objects.requireNonNull(inputBuffer, "inputBuffer");
        Objects.requireNonNull(outputBuffer, "outputBuffer");
        checkExecutableStatus(NEG_KERNEL_PIPELINE_RUN_OPERATION,
                runNegKernelPipelineNative(pipeline, inputBuffer, outputBuffer));
    }

    /**
     * Returns the raw status of one custom NEG dispatch invocation.
     *
     * @param pipeline non-null live custom pipeline
     * @param inputBuffer non-null live direct input buffer
     * @param outputBuffer non-null live direct output buffer
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int runNegKernelPipelineNative(
            Handle pipeline, Handle inputBuffer, Handle outputBuffer);

    /**
     * Ends the production symbol-lookup lifetime after every native handle has been released.
     *
     * <p>Production close is thread-safe and idempotent. Test implementations must preserve the same
     * ownership boundary but may expose deterministic injected failures.
     *
     * @throws RuntimeException if lookup cleanup fails
     * @throws Error if lookup cleanup reports an error
     */
    @Override
    public abstract void close();

    /** Opaque non-null FFM carrier for one context, buffer, executable, or pipeline handle. */
    static final class Handle {
        private final MemorySegment carrier;

        /**
         * Retains one non-null, non-zero opaque carrier.
         *
         * @param carrier the native address carrier; must not be {@code null} or zero
         * @throws NullPointerException if {@code carrier} is {@code null}
         * @throws IllegalArgumentException if {@code carrier} has address zero
         */
        Handle(MemorySegment carrier) {
            this.carrier = Objects.requireNonNull(carrier, "carrier");
            if (carrier.address() == 0L) {
                throw new IllegalArgumentException("carrier must not be null");
            }
        }

        /**
         * Returns the package-private FFM carrier for a downcall.
         *
         * @return the exact non-null, non-zero carrier retained by this handle
         */
        MemorySegment carrier() {
            return carrier;
        }
    }

    /**
     * Carries the exact native create status and nullable output-cell handle.
     *
     * @param statusCode exact signed native status
     * @param handle output-cell handle, or {@code null} when native left the cell empty
     */
    record NativeCreateResult(int statusCode, Handle handle) {}

    private Handle finishExecutableCreate(NativeCreateResult result) {
        return finishCreate(EXECUTABLE_CREATE_OPERATION, result, this::releaseExecutableNative);
    }

    private Handle finishCreate(
            String operation, NativeCreateResult result, NativeHandleRelease release) {
        int status = result.statusCode();
        Handle handle = result.handle();
        if (status == 0 && handle != null) return handle;
        RuntimeException failure = status == 0
                ? new IllegalStateException(
                        operation + " returned OK with a null handle")
                : new NativeFailure(operation, status);
        if (handle != null) {
            try {
                checkExecutableStatus(
                        operation + " malformed-handle cleanup", release.release(handle));
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            if (status != 0) {
                IllegalStateException contractFailure = new IllegalStateException(
                        operation + " returned failure with a non-null handle",
                        failure);
                for (Throwable suppressed : failure.getSuppressed()) {
                    contractFailure.addSuppressed(suppressed);
                }
                throw contractFailure;
            }
        }
        throw failure;
    }

    @FunctionalInterface
    private interface NativeHandleRelease {
        int release(Handle handle);
    }

    private static void checkExecutableStatus(String operation, int status) {
        if (status != 0) throw new NativeFailure(operation, status);
    }

    /** Stable ABI-v5 non-success status meanings. */
    enum Status {
        INVALID_ARGUMENT(1),
        NO_DEVICE(2),
        NO_COMMAND_QUEUE(3),
        ALLOCATION_FAILED(4),
        RANGE_OUT_OF_BOUNDS(5),
        COPY_FAILED(6),
        INTERNAL_ERROR(7),
        UNSUPPORTED_SHAPE(8),
        GRAPH_COMPILATION_FAILED(9),
        INCOMPATIBLE_RESOURCE(10),
        EXECUTION_FAILED(11),
        KERNEL_COMPILATION_FAILED(12),
        UNSUPPORTED_OPERATION(13);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        static Status fromCode(int code) {
            for (Status status : values()) {
                if (status.code == code) {
                    return status;
                }
            }
            return null;
        }
    }

    /** Unchecked failure retaining stable native operation and status facts. */
    static final class NativeFailure extends RuntimeException {
        private final String operation;
        private final int statusCode;
        private final Status status;

        /**
         * Creates a known- or unknown-status native failure.
         *
         * @param operation stable non-null ABI operation name
         * @param statusCode exact signed status value returned by native code
         * @throws NullPointerException if {@code operation} is {@code null}
         */
        NativeFailure(String operation, int statusCode) {
            super(message(operation, statusCode));
            this.operation = Objects.requireNonNull(operation, "operation");
            this.statusCode = statusCode;
            this.status = Status.fromCode(statusCode);
        }

        /**
     * @return the stable non-null ABI operation name */
        String operation() {
            return operation;
        }

        /*** @return the exact signed status integer returned by native code */
        int statusCode() {
            return statusCode;
        }

        /*** @return the known status, or {@code null} when native code returned an unknown value */
        Status status() {
            return status;
        }

        private static String message(String operation, int statusCode) {
            Status status = Status.fromCode(statusCode);
            return operation + " failed with "
                    + (status == null ? "unknown native status " + statusCode
                            : "native status " + status + " (" + statusCode + ")");
        }
    }

    /**
     * Exact Java preflight for the schema-twenty-one typed Metal program create contract, including
     * EXP and SIGMOID route-specific FLOAT32 or homogeneous low types,
     * shape, layout, gradient, and byte-span gates. Low EXP and SIGMOID are custom only. FLOAT32 EXP or
     * SIGMOID in a custom partition remains an internal MPSGraph boundary step, and SIGMOID's
     * stable sign-guarded formula is fixed by the authenticated native source.
     * FLOAT32 scalar occurrences use their independently admitted rank-zero domains; low carriers
     * retain their original scalar restrictions. Neither changes generated-fusion eligibility or
     * relaxes native authentication of actual fused execution records.
     * MATMUL checks original typed roles, exact logical provenance, and independently projected
     * represented storage. Logical density after SELECT/SLICE does not imply physical density.
     * Affine producers retain their exact source offsets/strides. RESHAPE independently selects
     * an exact view or an ordinary canonical materialization; a canonical claim never replaces
     * authentication of the source producer. Only native's separate physical source proof may
     * elide an already-qualified transpose.
     */
    static final class ProgramExecutableAbi {
        private static final int MAX_RANK = 16;
        private static final long UINT32_MAX = 0xffff_ffffL;
        private static final long TASK0064_MAX_POOL_KERNEL_POSITIONS = 65_536L;

        private ProgramExecutableAbi() {}

        /**
         * Rejects a schema-twenty-one program before native creation if its fixed route, graph
         * geometry, types, or occurrence metadata disagrees. In particular, wire-64 SIGMOID
         * requires canonical FLOAT32 with matching gradient metadata or homogeneous no-gradient
         * BFLOAT16/FLOAT16 for a typed custom step. A standalone FLOAT32 SIGMOID cannot invent
         * a custom partition route.
         *
         * @param values non-null ordered value descriptors, not mutated
         * @param graphProgram non-null typed program, not mutated
         * @param feeds non-null ordered feed indices, not mutated
         * @param targets non-null ordered target indices, not mutated
         * @param route non-null fixed whole-partition route
         * @throws NullPointerException if any argument is null
         * @throws IllegalArgumentException if any encoded occurrence or route is invalid
         */
        static void validateCreate(
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalMpsGraphProgram graphProgram,
                int[] feeds,
                int[] targets,
                MetalPreparedRoute route) {
            validateCreate(
                    values, graphProgram, feeds, targets, route, null);
        }

        /**
         * Independently validates one physical ordinary Scan unit with its original dimensions.
         * Only referenced descriptors are compacted; no batch or contraction extent is replaced.
         *
         * @param node decoded ordinary unit, with no containing Scan instruction
         * @param values complete authenticated physical local-value table
         * @throws IllegalArgumentException if its grammar, metadata or carrier equations disagree
         */
        static void validateScanBody(MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            MetalScanDecoding.validateOrdinaryKind(node.kind());
            var mapping = new java.util.LinkedHashMap<Integer, Integer>();
            for (int input : node.inputs()) { requireIndex(input, values.size(), "body input"); mapping.putIfAbsent(input, mapping.size()); }
            for (int output : node.outputs()) { requireIndex(output, values.size(), "body output"); mapping.putIfAbsent(output, mapping.size()); }
            var local = new java.util.ArrayList<MetalMpsGraphProgram.ValueDescriptor>(mapping.size());
            for (int source : mapping.keySet()) {
                var value = values.get(source);
                local.add(new MetalMpsGraphProgram.ValueDescriptor(
                        value.dataType(), value.dimensions(), value.requiresGrad()));
            }
            int[] inputs = java.util.Arrays.stream(node.inputs()).map(mapping::get).toArray();
            int[] outputs = java.util.Arrays.stream(node.outputs()).map(mapping::get).toArray();
            var compact = MetalMpsGraphProgram.Node.genericForScanBody(
                    node.kind(), inputs, outputs, node.attributeKind(), node.attributeWords());
            validateCreate(local, new MetalMpsGraphProgram(List.of(compact)),
                    java.util.Arrays.stream(inputs).distinct().toArray(), outputs,
                    MetalPreparedRoute.CUSTOM_PROGRAM, null, true);
        }

        /**
         * Validates the same fixed program contract with an optional authenticated custom
         * execution plan. Low-valued SIGMOID requires a custom whole-partition route and a
         * separately typed kernel; it cannot use the FLOAT32 MPSGraph boundary step.
         *
         * @param values non-null ordered value descriptors, not mutated
         * @param graphProgram non-null typed program, not mutated
         * @param feeds non-null ordered feed indices, not mutated
         * @param targets non-null ordered target indices, not mutated
         * @param route non-null fixed whole-partition route
         * @param fusionPlan nullable cold custom execution plan, not mutated
         * @throws NullPointerException if a required argument is null
         * @throws IllegalArgumentException if the encoded occurrence, route, or plan is invalid
         */
        static void validateCreate(
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalMpsGraphProgram graphProgram,
                int[] feeds,
                int[] targets,
                MetalPreparedRoute route,
                MetalPointwiseFusionPlan fusionPlan) {
            validateCreate(values, graphProgram, feeds, targets, route, fusionPlan, false);
        }

        private static void validateCreate(
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalMpsGraphProgram graphProgram, int[] feeds, int[] targets,
                MetalPreparedRoute route, MetalPointwiseFusionPlan fusionPlan, boolean scanBody) {
            Objects.requireNonNull(values, "values");
            Objects.requireNonNull(graphProgram, "graphProgram");
            Objects.requireNonNull(feeds, "feedValueIndices");
            Objects.requireNonNull(targets, "targetValueIndices");
            Objects.requireNonNull(route, "route");
            if (route == MetalPreparedRoute.CUSTOM_SINGLE_NEG) {
                throw new IllegalArgumentException(
                        "singleton custom NEG does not use the program executable");
            }
            int valueCount = values.size();
            if (valueCount == 0) {
                throw new IllegalArgumentException("Metal MPSGraph value count must be positive");
            }
            if (graphProgram.nodes().isEmpty()) {
                throw new IllegalArgumentException("Metal MPSGraph node count must be positive");
            }
            if (targets.length == 0) {
                throw new IllegalArgumentException("Metal MPSGraph target count must be positive");
            }
            int[] valueRanks = new int[valueCount];
            long[] valueDimensions =
                    new long[Math.multiplyExact(valueCount, MAX_RANK)];
            ValueType[] types = new ValueType[valueCount];
            var modelValues = new ArrayList<TensorDescriptor>(valueCount);
            for (int value = 0; value < valueCount; value++) {
                MetalMpsGraphProgram.ValueDescriptor descriptor =
                        Objects.requireNonNull(values.get(value), "values[" + value + "]");
                valueRanks[value] = descriptor.rank();
                long[] dimensions = descriptor.dimensions();
                System.arraycopy(dimensions, 0, valueDimensions, value * MAX_RANK,
                        dimensions.length);
                types[value] = ValueType.from(descriptor.dataType());
                descriptor.byteCount();
                modelValues.add(new TensorDescriptor(descriptor.dataType(), Shape.of(dimensions),
                        descriptor.layout(), descriptor.requiresGrad()));
            }

            MetalMpsGraphProgram.ValueState[] states =
                    new MetalMpsGraphProgram.ValueState[valueCount];
            java.util.Arrays.fill(states, MetalMpsGraphProgram.ValueState.UNAVAILABLE);
            boolean[] used = new boolean[valueCount];
            boolean[] fed = new boolean[valueCount];
            boolean[] produced = new boolean[valueCount];
            boolean[] localTranspose = new boolean[valueCount];
            boolean[] localSingletonHeight = new boolean[valueCount];
            LayoutDescriptor[] physicalLayouts = new LayoutDescriptor[valueCount];
            for (int feed : feeds) {
                requireIndex(feed, valueCount, "feed");
                if (states[feed] != MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph feed indices must be unique");
                }
                if (values.get(feed).densePhysical()) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph affine views must be produced inside the partition");
                }
                MetalMpsGraphProgram.ValueDescriptor descriptor = values.get(feed);
                if (!descriptor.supportsStorageLayout(0)) {
                    throw new IllegalArgumentException(
                            "Metal graph feed storage must have authenticated resolved read geometry");
                }
                states[feed] = descriptor.layout().filter(layout -> layout.equals(
                                LayoutDescriptor.contiguous(Shape.of(descriptor.dimensions()))))
                        .isPresent()
                        ? MetalMpsGraphProgram.ValueState.CANONICAL
                        : MetalMpsGraphProgram.ValueState.MATERIALIZED_LAYOUT;
                physicalLayouts[feed] = descriptor.layout().orElseThrow();
                used[feed] = true;
                fed[feed] = true;
            }
            for (MetalMpsGraphProgram.Node node : graphProgram.nodes()) {
                if (!node.kind().executable()) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph node kind is not executable");
                }
            }
            boolean containsCustomOperation = graphProgram.nodes().stream()
                            .anyMatch(node -> MetalPointwiseFusionPlanner.fixedCustom(node, values, feeds)
                                    || task0066Selected(node.kind())
                                    || usesCustomMatmul(node, types, valueRanks)
                                    || usesLowPrecisionStorage(node, types))
                    || fusionPlan != null && fusionPlan.steps().stream().anyMatch(
                            step -> step.kind()
                                    == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE);
            if (route == MetalPreparedRoute.MPSGRAPH
                    && values.stream().anyMatch(value ->
                            value.dataType() == DataType.BFLOAT16
                                    || value.dataType() == DataType.FLOAT16)) {
                throw new IllegalArgumentException(
                        "BFLOAT16/FLOAT16 values require the custom-program route");
            }
            if (route == MetalPreparedRoute.MPSGRAPH
                    && graphProgram.nodes().stream()
                            .anyMatch(node -> node.kind().isNumericElementary()
                                    || MetalReductionSupport.handles(node, values)
                                    || MetalNormalizationSupport.handles(node))
                    || route == MetalPreparedRoute.MPSGRAPH
                            && graphProgram.nodes().stream()
                            .anyMatch(node -> task0052CustomOnly(node.kind())
                                    || task0063CustomOnly(node.kind())
                                    || task0064CustomOnly(node.kind())
                                    || task0065CustomOnly(node.kind())
                                    || task0066Selected(node.kind())
                                    || task0069CustomOnly(
                                            node, valueRanks, valueDimensions)
                                    || usesCustomMatmul(node, types, valueRanks)
                                    || MetalPointwiseFusionPlanner.float32BinaryRequiresPhysicalLayout(node, values)
                                    || MetalPointwiseFusionPlanner.numericPhysicalRequiresCustom(node, values, feeds)
                                    || usesLowPrecisionStorage(node, types))) {
                throw new IllegalArgumentException(
                        "custom-only operations have no approved direct MPSGraph route");
            }
            for (int nodePosition = 0;
                    nodePosition < graphProgram.nodes().size();
                    nodePosition++) {
                MetalMpsGraphProgram.Node node = graphProgram.nodes().get(nodePosition);
                int[] nodeInputs = node.inputs();
                int[] nodeOutputs = node.outputs();
                int left = node.firstInputIndex();
                int right = node.secondInputIndex();
                int output = node.outputIndex();
                int auxiliary = node.auxiliary();
                if (nodeInputs.length != 0) {
                    requireIndex(left, valueCount, "first node input");
                }
                for (int nodeOutput : nodeOutputs) {
                    requireIndex(nodeOutput, valueCount, "node output");
                    if (produced[nodeOutput] || states[nodeOutput]
                            != MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                        throw new IllegalArgumentException(
                                "Metal MPSGraph node outputs must be unique and not feeds");
                    }
                    produced[nodeOutput] = true;
                }
                for (int input : nodeInputs) {
                    requireIndex(input, valueCount, "node input");
                    if (!node.kind().accepts(states[input], values.get(input))) {
                        throw new IllegalArgumentException(
                                "Metal MPSGraph node input value state is unavailable or"
                                    + " incompatible");
                    }
                    if (node.kind() == MetalMpsGraphProgram.NodeKind.GENERAL_SCAN
                            && !LayoutDescriptor.contiguous(Shape.of(values.get(input).dimensions()))
                                    .equals(physicalLayouts[input])
                            && !(MetalScanDecoding.mappedInput(values.get(input))
                                    && values.get(input).supportsStorageLayout(
                                            physicalLayouts[input], 0))) {
                        throw new IllegalArgumentException(
                                "Metal Scan requires authenticated dense or mapped numeric input storage");
                    }
                    if (numericFloating(types[input]) && node.kind().isNumericUnary()
                            && !values.get(input).supportsStorageLayout(physicalLayouts[input], 0)) {
                        throw new IllegalArgumentException(
                                "numeric unary input requires authenticated represented read storage");
                    }
                    used[input] = true;
                }
                if (node.kind() != MetalMpsGraphProgram.NodeKind.GENERAL_SCAN
                        && node.kind() != MetalMpsGraphProgram.NodeKind.INITIAL_STATE
                        && node.kind() != MetalMpsGraphProgram.NodeKind.DROPOUT
                        && !MetalRecurrentSupport.handles(node)) {
                    validateModelOrdinaryOutputs(node, nodeInputs, nodeOutputs, modelValues);
                }
                if (!scanBody && node.kind() == MetalMpsGraphProgram.NodeKind.RESHAPE) {
                    requireShape(states[left] != MetalMpsGraphProgram.ValueState.CANONICAL
                                    || canonicalExpValue(values.get(left)),
                            "RESHAPE canonical source state must retain canonical logical geometry");
                    requireShape(node.kind().outputState(values.get(left), values.get(output))
                                    != MetalMpsGraphProgram.ValueState.UNAVAILABLE,
                            "RESHAPE must declare its exact view or canonical materialization");
                } else if (!scanBody && node.kind().isAffine()) {
                    requireShape(affineLayoutMatches(
                                    node, values.get(left), values.get(output)),
                            node.kind() + " must preserve its source logical mapping,"
                                    + " carrier, and gradient metadata");
                }
                if (!validateExtendedOperation(node, values, scanBody)) {
                if (task0066Selected(node.kind())) {
                    validateTask0066Gradients(node, values, types);
                } else if (exactNoGradientProduction(node.kind())
                        && (left < 0 || !numericFloating(types[left]))
                        && (node.kind() != MetalMpsGraphProgram.NodeKind.VARIANCE
                                || task0069VarianceMatches(
                                        node, left, output, valueRanks, valueDimensions))) {
                    for (int input : node.inputs()) {
                        if (values.get(input).requiresGrad()) {
                            throw new IllegalArgumentException(
                                    "exact production routes require no-grad inputs");
                        }
                    }
                    if (values.get(output).requiresGrad()) {
                        throw new IllegalArgumentException(
                                "exact production routes require no-grad outputs");
                    }
                }
                if (node.kind().isNumericElementary()) {
                    for (int input : nodeInputs)
                        requireShape(numericFloating(types[input]), "elementary input must use a numerical carrier");
                    requireShape(numericFloating(types[output]), "elementary output must use a numerical carrier");
                }
                if ((node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_ADD
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_SUB
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_MUL
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_DIV
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_POW
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_MIN
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_MAX
                                || node.kind() == MetalMpsGraphProgram.NodeKind.CLAMP)
                        && (node.attributeWords()[0]
                                        != MetalMpsGraphProgram.dataTypeWire(
                                                values.get(left).dataType())
                                || node.kind() == MetalMpsGraphProgram.NodeKind.CLAMP
                                        && node.attributeWords()[2]
                                                != node.attributeWords()[0])) {
                    throw new IllegalArgumentException(
                            "Metal scalar operation requires exact matching scalar types");
                }
                if ((node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_ADD
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_SUB
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_MUL
                                || node.kind() == MetalMpsGraphProgram.NodeKind.SCALAR_DIV
                                || node.kind() == MetalMpsGraphProgram.NodeKind.RECIPROCAL)
                        && values.get(left).requiresGrad()
                                != values.get(output).requiresGrad()) {
                    throw new IllegalArgumentException(
                            "Metal scalar arithmetic and reciprocal must preserve gradient metadata");
                }
                if (node.kind() == MetalMpsGraphProgram.NodeKind.SORT
                        || node.kind() == MetalMpsGraphProgram.NodeKind.TOP_K) {
                    if (values.get(left).requiresGrad()
                            != values.get(output).requiresGrad()) {
                        throw new IllegalArgumentException(
                                "value ordering output must preserve input gradient metadata");
                    }
                } else if ((node.kind() == MetalMpsGraphProgram.NodeKind.ARGSORT
                                || node.kind() == MetalMpsGraphProgram.NodeKind.ARG_MAX
                                || node.kind() == MetalMpsGraphProgram.NodeKind.ARG_MIN)
                        && values.get(output).requiresGrad()) {
                    throw new IllegalArgumentException(
                            "index ordering outputs must not require gradients");
                }
                if (node.kind() == MetalMpsGraphProgram.NodeKind.TOP_K
                        && values.get(node.outputs()[1]).requiresGrad()) {
                    throw new IllegalArgumentException(
                            "TOP_K index output must not require gradients");
                }
                switch (node.kind()) {
                    case INITIAL_STATE -> {
                        requireShape(
                                nodeInputs.length == 0
                                        && node.outputs().length == 1
                                        && task0065State(values.get(output)),
                                "INITIAL_STATE must publish one canonical INT64[2] value");
                        long[] words = node.attributeWords();
                        requireShape(
                                words.length == 2,
                                "INITIAL_STATE must retain exactly key and counter words");
                    }
                    case DROPOUT -> {
                        requireIndex(right, valueCount, "dropout state input");
                        if (nodeOutputs.length != 3) {
                            throw new IllegalArgumentException(
                                    "DROPOUT must publish value, mask, and next state");
                        }
                        int mask = nodeOutputs[1];
                        int nextState = nodeOutputs[2];
                        requireIndex(mask, valueCount, "dropout mask output");
                        requireIndex(nextState, valueCount, "dropout next-state output");
                        if (left == right || left == output || left == mask || left == nextState
                                || right == output || right == mask || right == nextState
                                || output == mask || output == nextState || mask == nextState) {
                            throw new IllegalArgumentException(
                                    "DROPOUT inputs and outputs must be distinct values");
                        }
                        long[] words = node.attributeWords();
                        if (words.length != 1) {
                            throw new IllegalArgumentException(
                                    "DROPOUT must retain one raw probability word");
                        }
                        MetalMpsGraphProgram.dropoutThreshold(words[0]);
                        MetalMpsGraphProgram.dropoutComplementBits(words[0]);
                        ValueType valueType = types[left];
                        requireShape(
                                (valueType == ValueType.FLOAT32 || lowPrecision(valueType))
                                        && task0065Value(values.get(left), valueType, true, scanBody)
                                        && task0065State(values.get(right))
                                        && task0065Value(
                                                values.get(output), valueType, true, scanBody)
                                        && task0065Value(
                                                values.get(mask), ValueType.BOOL, true, scanBody)
                                        && task0065State(values.get(nextState))
                                        && sameShape(
                                                left, output, valueRanks, valueDimensions)
                                        && sameShape(
                                                left, mask, valueRanks, valueDimensions),
                                "DROPOUT value, mask, and state geometry is incompatible");
                        used[right] = true;
                    }
                    case NEG, ABS, CONTIGUOUS, SCALAR_MIN, SCALAR_MAX, CLAMP, CUM_SUM, CUM_PROD,
                            SCALAR_POW, LOG, LOG1P, EXPM1, ERF, SQRT, RSQRT,
                            TANH, GELU, GELU_TANH_APPROXIMATION, SILU ->
                            requireShape(
                                    sameShape(left, output, valueRanks, valueDimensions),
                                    node.kind() + " input/output shapes must match exactly");
                    case SCALAR_ADD, SCALAR_SUB, SCALAR_DIV, RECIPROCAL ->
                            requireShape(
                                    (numericFloating(types[left])
                                            || valueRanks[left] >= 1 && valueRanks[output] >= 1)
                                            && sameShape(
                                                    left, output, valueRanks, valueDimensions),
                                    node.kind()
                                            + " input/output must have matching positive ranks");
                    case SCALAR_MUL ->
                            requireShape(
                                    sameShape(left, output, valueRanks, valueDimensions),
                                    "SCALAR_MUL input/output shape is unsupported");
                    case EXP, SIGMOID -> requireShape(
                            (route == MetalPreparedRoute.MPSGRAPH
                                    || route == MetalPreparedRoute.CUSTOM_PROGRAM)
                                    && types[left] == types[output]
                                    && numericFloating(types[left])
                                    && (types[left] == ValueType.FLOAT32
                                            || route == MetalPreparedRoute.CUSTOM_PROGRAM)
                                    && sameShape(left, output, valueRanks, valueDimensions)
                                    && values.get(left).supportsStorageLayout(0)
                                    && canonicalExpValue(values.get(output))
                                    && values.get(left).requiresGrad() == values.get(output).requiresGrad()
                                    && values.get(left).byteCount() <= UINT32_MAX,
                            node.kind() + " requires physical same-type reads and canonical output with matching gradient metadata");
                    case FLOOR, CEIL, SIGN ->
                            requireShape(
                                    (numericFloating(types[left])
                                            || valueRanks[left] >= 1 && valueRanks[output] >= 1)
                                            && sameShape(
                                                    left, output, valueRanks, valueDimensions),
                                    node.kind()
                                            + " input/output must have matching positive ranks");
                    case RELU ->
                            requireShape(
                                    sameShape(left, output, valueRanks, valueDimensions),
                                    "RELU input/output shape is unsupported");
                    case IS_FINITE, IS_NAN, IS_INF, LOGICAL_NOT ->
                            requireShape( sameShape(left, output, valueRanks, valueDimensions),
                                    node.kind()
                                            + " input/output shapes must match exactly");
                    case ADD, SUB, MUL, DIV, TENSOR_POW, GT, GE, LT, LE, EQ, NE,
                            TENSOR_MIN, TENSOR_MAX, LOGICAL_AND, LOGICAL_OR -> {
                        requireIndex(right, valueCount, "second node input");
                        if (!node.kind().accepts(states[right], values.get(right))) {
                            throw new IllegalArgumentException(
                  "Metal second node input has an unsupported value state");
                        }
                        requireShape(
                                broadcastsTo(left, right, output, valueRanks, valueDimensions),
                                "binary output must equal exact right-aligned broadcast");
                        used[right] = true;
                    }
                    case MEAN_SQUARED_ERROR -> {
                        requireIndex(right, valueCount, "MSE target input");
                        if (!node.kind().accepts(states[right])
                                || !sameShape(left, right, valueRanks, valueDimensions)) {
                            throw new IllegalArgumentException(
                                    "MSE inputs must be matching canonical values");
                        }
                        long reduction = node.attributeWords()[0];
                        requireShape(
                                reduction == 1L
                                        ? sameShape(
                                                left, output, valueRanks, valueDimensions)
                                        : valueRanks[output] == 0,
                                "MSE output shape must match its reduction");
                        if (values.get(output).requiresGrad()
                                != (values.get(left).requiresGrad()
                                        || values.get(right).requiresGrad())) {
                            throw new IllegalArgumentException(
                                    "MSE output gradient metadata must equal the input logical OR");
                        }
                        used[right] = true;
                    }
                    case WHERE -> {
                        requireIndex(right, valueCount, "true branch input");
                        requireIndex(auxiliary, valueCount, "false branch input");
                        if (!node.kind().accepts(states[left])
                                || !node.kind().accepts(states[right])
                                || !node.kind().accepts(states[auxiliary])) {
              throw new IllegalArgumentException("WHERE inputs must be canonical or affine");
                        }
            requireShape(
                broadcastsTo(right, auxiliary, output, valueRanks, valueDimensions)
                    && broadcastsTo(left, output, output, valueRanks, valueDimensions),
                "WHERE output must equal branch-first and condition-second broadcast");
                        used[right] = true;
                        used[auxiliary] = true;
                    }
                    case RESHAPE -> {
                        requireShape(
                                sameElementCount(left, output, valueRanks, valueDimensions),
                                "RESHAPE input/output element counts must match");
                        requireShape(
                                targetMatches(node, output, valueRanks, valueDimensions),
                                "RESHAPE target attributes must equal the output shape");
                    }
                    case EXPAND -> {
                        requireShape(
                                expandsTo(left, output, valueRanks, valueDimensions),
                                "EXPAND output must be an exact right-aligned expansion");
                        requireShape(
                                targetMatches(node, output, valueRanks, valueDimensions),
                                "EXPAND target attributes must equal the output shape");
                    }
                    case PERMUTE -> {
                        boolean matches = permutationMatches(
                                node, left, output, valueRanks, valueDimensions);
                        requireShape(matches, "PERMUTE attributes and output shape disagree");
                        localTranspose[output] = exactLocalTranspose(
                                node, left, output, states, valueRanks, values);
                    }
                    case EXPAND_DIMS -> {
                        requireShape(
                                expandDimsMatches(node, left, output, valueRanks, valueDimensions),
                                "EXPAND_DIMS axis and output shape disagree");
                        localSingletonHeight[output] = exactLocalSingletonHeight(
                                node, left, output, states, valueRanks, values);
                    }
                    case SQUEEZE -> requireShape(
                            squeezeMatches(node, left, output, valueRanks, valueDimensions),
                            "SQUEEZE axis and output shape disagree");
                    case SUM, MEAN, REDUCTION_MIN, REDUCTION_MAX -> {
                        requireShape(valueRanks[left] > 0,
                                "reduction input must have positive rank");
                        requireShape(
                                reductionMatches(node, left, output, valueRanks, valueDimensions),
                                "reduction attributes, count, and output shape disagree");
                    }
                    case PROD, ALL, ANY, LOG_SUM_EXP, L2_NORM ->
                            requireShape(
                                    reductionMatches(
                                            node, left, output, valueRanks, valueDimensions),
                                    "reduction attributes, count, and output shape disagree");
                    case L1_NORM ->
                            requireShape(
                                    task0069L1Matches(
                                            node,
                                            values.get(left),
                                            values.get(output),
                                            left,
                                            output,
                                            valueRanks,
                                            valueDimensions),
                                    "L1_NORM attributes, storage, and output shape disagree");
                    case VARIANCE ->
                            requireShape(
                                    route == MetalPreparedRoute.CUSTOM_PROGRAM
                                            ? task0069VarianceMatches(
                                                    node,
                                                    left,
                                                    output,
                                                    valueRanks,
                                                    valueDimensions)
                                            : statisticalReductionMatches(
                                                    node,
                                                    left,
                                                    output,
                                                    valueRanks,
                                                    valueDimensions),
                                    "VARIANCE attributes, storage, and output shape disagree");
                    case STANDARD_DEVIATION ->
                            requireShape(
                                    statisticalReductionMatches(
                                            node, left, output, valueRanks, valueDimensions),
                                    "STANDARD_DEVIATION attributes and output shape disagree");
                    case SORT, ARGSORT, TOP_K, ARG_MAX, ARG_MIN ->
                            requireShape(
                                    task0063ShapeMatches(node, values, scanBody),
                                    node.kind()
                                            + " attributes, unsigned geometry, and output shape"
                                            + " disagree");
                    case MATMUL -> {
                        requireIndex(right, valueCount, "second node input");
                        if (valueRanks[left] == 0 || valueRanks[right] == 0
                                || !node.kind().accepts(states[right], values.get(right))
                                || !values.get(left).supportsStorageLayout(physicalLayouts[left], 1)
                                || !values.get(right).supportsStorageLayout(physicalLayouts[right], 1)
                                || !numericFloating(types[left])
                                        && ((!values.get(left).matmulDense() && !localTranspose[left])
                                                || (!values.get(right).matmulDense()
                                                        && !localTranspose[right]))) {
                            throw new IllegalArgumentException(
                                    "Metal MATMUL inputs require authenticated represented read storage");
                        }
                        requireShape(
                                !values.get(output).densePhysical()
                                        && values.get(output).layout().filter(layout ->
                                                layout.equals(LayoutDescriptor.contiguous(
                                                        Shape.of(values.get(output).dimensions()))))
                                                .isPresent(),
                                "MATMUL output must be canonical");
                        requireShape(
                                matmulMatches(left, right, output, valueRanks, valueDimensions),
                                "MATMUL shapes must be an exact static contraction");
                        used[right] = true;
                    }
                    case GATHER -> {
                        requireIndex(right, valueCount, "indices node input");
                        if (!node.kind().accepts(states[right])) {
                            throw new IllegalArgumentException(
                                    "Metal GATHER indices have an incompatible value state");
                        }
                        requireShape(
                                gatherMatches(
                                        node, left, right, output, valueRanks, valueDimensions),
                                "GATHER axis and output shape disagree");
                        used[right] = true;
                    }
                    case SCATTER_ADD -> {
                        requireIndex(right, valueCount, "scatter indices input");
                        requireIndex(auxiliary, valueCount, "scatter updates input");
                        requireShape(
                                left != right
                                        && left != auxiliary
                                        && left != output
                                        && right != auxiliary
                                        && right != output
                                        && auxiliary != output
                                        && fed[right]
                                        && task0069ScatterAddMatches(
                                                node,
                                                values.get(left),
                                                values.get(right),
                                                values.get(auxiliary),
                                                values.get(output)),
                                "SCATTER_ADD axis, storage, feed, or shape contract disagrees");
                    }
                    case SCATTER_ELEMENTS -> {
                        requireIndex(right, valueCount, "scatter indices input");
                        requireIndex(auxiliary, valueCount, "scatter updates input");
                        if (valueRanks[right] == 0
                                || valueRanks[auxiliary] == 0
                                || !node.kind().accepts(states[right])
                                || !node.kind().accepts(states[auxiliary])) {
              throw new IllegalArgumentException(
                  "Metal SCATTER_ELEMENTS inputs must be positive-rank canonical");
                        }
                        requireShape(
                                scatterElementsMatches(
                                        node, left, right, auxiliary, output,
                                        valueRanks, valueDimensions),
                                "SCATTER_ELEMENTS axis and shapes disagree");
                        used[right] = true;
                        used[auxiliary] = true;
                    }
                    case ONE_HOT -> requireShape(
                            oneHotMatches(node, left, output, valueRanks, valueDimensions),
                            "ONE_HOT depth and output shape disagree");
                    case CAST, GATHER_ELEMENTS, GATHER_ND, SCATTER_ND, SELECT, PAD,
                            SLICE, SLICE_UPDATE, CONCAT, STACK, TILE, FOLD_AXIS, UNFOLD2D, FOLD2D,
                            UNFOLD3D, FOLD3D -> {
                        requireShape(
                                task0059ShapeMatches(node, values, route, scanBody),
                                node.kind() + " attributes and output shape disagree");
                        if (route == MetalPreparedRoute.CUSTOM_PROGRAM
                                && node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ND
                                && node.attributeWords()[1] != 1L) {
                            throw new IllegalArgumentException(
                                    "custom SCATTER_ND requires replacement reduction");
                        }
                        if (route == MetalPreparedRoute.CUSTOM_PROGRAM
                                && (node.kind() == MetalMpsGraphProgram.NodeKind.FOLD_AXIS
                                        || node.kind() == MetalMpsGraphProgram.NodeKind.FOLD2D
                                        || node.kind() == MetalMpsGraphProgram.NodeKind.FOLD3D)
                                && !supportsFoldGeometry(
                                        node, types[node.firstInputIndex()])) {
                            throw new IllegalArgumentException(
                                    "custom FOLD overlap requires a FLOAT32 image fold");
                        }
                        if (!scanBody && (node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                                        || node.kind() == MetalMpsGraphProgram.NodeKind.SLICE)
                                && !task0059LayoutMatches(node, values)) {
                            throw new IllegalArgumentException(
                                    node.kind() + " attributes and output layout disagree");
                        }
                    }
                    case CONV2D, CONV3D, MAX_POOL2D, AVERAGE_POOL2D,
                            MAX_POOL3D, AVERAGE_POOL3D -> {
                        if (!task0064ShapeMatches(node, values, scanBody)) {
                            throw new IllegalArgumentException(
                                    node.kind() + " attributes, unsigned geometry, and output"
                                            + " shape disagree");
                        }
                        if (node.kind() == MetalMpsGraphProgram.NodeKind.CONV2D
                                || node.kind() == MetalMpsGraphProgram.NodeKind.MAX_POOL2D
                                || node.kind()
                                        == MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D) {
                            for (int input : node.inputs()) {
                                if (states[input]
                                                == MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                                        && !localSingletonHeight[input]) {
                                    throw new IllegalArgumentException(
                                            "Conv2d/Pool2d affine input must be an authenticated"
                                                    + " local singleton-height expansion");
                                }
                            }
                        }
                    }
                    case UNFOLD_AXIS -> requireShape(
                            unfoldAxisMatches(
                                    node, left, output, valueRanks, valueDimensions),
                            "UNFOLD_AXIS attributes and output shape disagree");
                }
                switch (node.kind()) {
                    case INITIAL_STATE -> requireType(types, output, ValueType.INT64);
                    case DROPOUT -> {
                        int mask = nodeOutputs[1];
                        int nextState = nodeOutputs[2];
                        if (types[left] != ValueType.FLOAT32
                                && !lowPrecision(types[left])) {
                            throw new IllegalArgumentException(
                                    "DROPOUT value carrier is unsupported");
                        }
                        requireType(types, right, ValueType.INT64);
                        if (types[output] != types[left]) {
                            throw new IllegalArgumentException(
                                    "DROPOUT must preserve its value carrier");
                        }
                        requireType(types, mask, ValueType.BOOL);
                        requireType(types, nextState, ValueType.INT64);
                        if (values.get(left).requiresGrad()
                                        != values.get(output).requiresGrad()
                                || values.get(right).requiresGrad()
                                || values.get(mask).requiresGrad()
                                || values.get(nextState).requiresGrad()) {
                            throw new IllegalArgumentException(
                                    "DROPOUT gradient metadata is incompatible");
                        }
                    }
                    case NEG,
                            ABS,
                            SUM,
                            MEAN,
                            REDUCTION_MIN,
                            REDUCTION_MAX,
                            SCALAR_MIN,
                            SCALAR_MAX,
                            CLAMP,
                            SCALAR_ADD,
                            SCALAR_SUB,
                            SCALAR_MUL,
                            SCALAR_DIV,
                            SCALAR_POW,
                            RECIPROCAL,
                            EXP,
                            SIGMOID,
                            LOG,
                            LOG1P,
                            EXPM1,
                            ERF,
                            SQRT,
                            RSQRT,
                            FLOOR,
                            CEIL,
                            SIGN,
                            RELU,
                            TANH,
                            GELU,
                            GELU_TANH_APPROXIMATION,
                            SILU -> {
                        requireShape(numericFloating(types[left]) && types[output] == types[left],
                                node.kind() + " must preserve its numerical carrier");
                    }
                    case CUM_SUM, CUM_PROD -> requireShape(
                            (numericFloating(types[left]) || types[left] == ValueType.INT32
                                    || types[left] == ValueType.INT64) && types[output] == types[left],
                            "cumulative operations require one numerical or signed-integral carrier");
                    case CONTIGUOUS,
                            RESHAPE,
                            EXPAND,
                            PERMUTE,
                            EXPAND_DIMS,
                            SQUEEZE,
                            UNFOLD_AXIS -> {
                        if (!task0066Carrier(types[left]) || types[left] != types[output]) {
                            throw new IllegalArgumentException(
                                    node.kind() + " must preserve one exact carrier type");
                        }
                    }
                    case SORT -> {
                        if (types[left] != types[output]) {
                            throw new IllegalArgumentException(
                                    "SORT must preserve its exact carrier type");
                        }
                    }
                    case ARGSORT -> requireType(types, output, ValueType.INT64);
                    case TOP_K -> {
                        if (types[left] != types[nodeOutputs[0]]) {
                            throw new IllegalArgumentException(
                                    "TOP_K values must preserve the exact input carrier");
                        }
                        requireType(types, nodeOutputs[1], ValueType.INT64);
                    }
                    case ARG_MAX, ARG_MIN -> {
                        if (types[left] == ValueType.BOOL) {
                            throw new IllegalArgumentException(
                                    "arg extrema require a numeric input carrier");
                        }
                        requireType(types, output, ValueType.INT64);
                    }
                    case PROD -> {
                        if (types[left] != types[output]
                                || types[left] != ValueType.INT32
                                        && types[left] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "PROD requires one exact integer carrier");
                        }
                    }
                    case ALL, ANY -> {
                        requireType(types, left, ValueType.BOOL);
                        requireType(types, output, ValueType.BOOL);
                    }
                    case LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM -> {
                        requireShape(numericFloating(types[left]) && types[output] == types[left],
                                node.kind() + " must preserve its numerical carrier");
                    }
                    case IS_FINITE, IS_NAN, IS_INF -> {
                        if (!task0066Floating(types[left])) {
                            throw new IllegalArgumentException(
                                    "floating classification requires a floating input");
                        }
                        requireType(types, output, ValueType.BOOL);
                    }
                    case LOGICAL_NOT -> {
                        requireType(types, left, ValueType.BOOL);
                        requireType(types, output, ValueType.BOOL);
                    }
                    case LOGICAL_AND, LOGICAL_OR -> {
                        requireType(types, left, ValueType.BOOL);
                        requireType(types, right, ValueType.BOOL);
                        requireType(types, output, ValueType.BOOL);
                    }
                    case WHERE -> {
                        requireType(types, left, ValueType.BOOL);
                        ValueType promoted = task0066Promote(types[right], types[auxiliary]);
                        if (promoted == null || types[output] != promoted
                                || types[right] != types[auxiliary]
                                        && (types[right] == ValueType.BFLOAT16 || types[right] == ValueType.FLOAT16)
                                        && (types[auxiliary] == ValueType.BFLOAT16 || types[auxiliary] == ValueType.FLOAT16)) {
                            throw new IllegalArgumentException(
                                    "WHERE branches and output must use exact floating promotion");
                        }
                    }
                    case ADD, SUB, MUL, DIV, TENSOR_POW, TENSOR_MIN, TENSOR_MAX,
                            MEAN_SQUARED_ERROR -> {
                        requireShape(numericFloating(types[left]) && numericFloating(types[right])
                                        && types[output] == task0066Promote(types[left], types[right]),
                                node.kind() + " must retain exact floating promotion");
                    }
                    case MATMUL ->
                            validateMatmulTypesAndGradients(
                                    left, right, output, types, values);
                    case GT, GE, LT, LE, EQ, NE -> {
                        if (!numericFloating(types[left]) || !numericFloating(types[right])
                                || types[left] != types[right]
                                        && (types[left] == ValueType.BFLOAT16 || types[left] == ValueType.FLOAT16)
                                        && (types[right] == ValueType.BFLOAT16 || types[right] == ValueType.FLOAT16)) {
                            throw new IllegalArgumentException(
                                    "comparison input carriers disagree");
                        }
                        requireType(types, output, ValueType.BOOL);
                    }
                    case GATHER -> {
                        if (!task0066Carrier(types[left])
                                || types[left] != types[output]
                                || types[right] != ValueType.INT32
                                        && types[right] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "GATHER has incompatible data or index types");
                        }
                    }
                    case SCATTER_ELEMENTS -> {
                        if (!task0066Carrier(types[left])
                                || types[left] != types[auxiliary]
                                || types[left] != types[output]
                                || types[right] != ValueType.INT32
                                        && types[right] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "SCATTER_ELEMENTS has incompatible data, update, or index"
                                        + " types");
                        }
                    }
                    case ONE_HOT -> {
                        if (types[left] != ValueType.INT32 && types[left] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "ONE_HOT indices must use INT32 or INT64");
                        }
                        requireType(types, output, ValueType.BOOL);
                    }
                    case CAST -> {
                        if (!task0066Carrier(types[left])
                                || !task0066Carrier(types[output])
                                || node.attributeWords()[0]
                                        != MetalMpsGraphProgram.dataTypeWire(
                                                values.get(output).dataType())) {
                            throw new IllegalArgumentException(
                                    "CAST pair is outside the exact production matrix");
                        }
                    }
                    case GATHER_ELEMENTS, GATHER_ND -> {
                        if (types[left] != types[output]
                                || types[right] != ValueType.INT32
                                        && types[right] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    node.kind() + " has incompatible data or index types");
                        }
                    }
                    case SCATTER_ADD -> {
                        if (types[left] != ValueType.FLOAT32 && !lowPrecision(types[left])
                                || types[auxiliary] != types[left]
                                || types[output] != types[left]
                                || types[right] != ValueType.INT32
                                        && types[right] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "SCATTER_ADD requires one supported floating data carrier and"
                                            + " INT32 or INT64 indices");
                        }
                    }
                    case SCATTER_ND -> {
                        if (types[left] != types[auxiliary]
                                || types[left] != types[output]
                                || types[right] != ValueType.INT32
                                && types[right] != ValueType.INT64) {
                            throw new IllegalArgumentException(
                                    "SCATTER_ND has incompatible data, update, or index types");
                        }
                    }
                    case SLICE_UPDATE -> {
                        if (types[left] != types[right] || types[left] != types[output]) {
                            throw new IllegalArgumentException(
                                    "SLICE_UPDATE must preserve the exact carrier type");
                        }
                    }
                    case FOLD_AXIS -> {
                        if (types[left] != types[output]
                                || types[left] == ValueType.BOOL) {
              throw new IllegalArgumentException("FOLD_AXIS requires one exact non-BOOL carrier");
            }
          }
          case FOLD2D, FOLD3D -> {
            if (types[left] != types[output] || !task0066Floating(types[left])) {
                            throw new IllegalArgumentException(
                                    node.kind() + " requires one exact floating carrier");
                        }
                    }
                    case SELECT, SLICE, TILE -> {
                        if (types[left] != types[output]) {
                            throw new IllegalArgumentException(
                                    node.kind() + " must preserve the exact carrier type");
                        }
                    }
                    case PAD -> {
                        long[] words = node.attributeWords();
                        int rank = Math.toIntExact(words[0]);
                        if (types[left] != types[output]
                                || words[1 + rank * 2]
                                != MetalMpsGraphProgram.dataTypeWire(
                                        values.get(left).dataType())) {
                            throw new IllegalArgumentException(
                                    "PAD must preserve the carrier and padding scalar type");
                        }
                    }
                    case CONCAT, STACK -> {
                        for (int input : node.inputs()) {
                            if (types[input] != types[output]) {
                                throw new IllegalArgumentException(
                                        node.kind() + " must preserve every carrier type");
                            }
                        }
                    }
                    case CONV2D, CONV3D ->
                            validateTask0064ConvolutionTypesAndGradients(
                                    node, types, values);
                    case MAX_POOL2D, MAX_POOL3D -> {
                        if (types[left] != types[output]
                                || types[left] != ValueType.FLOAT64
                                        && types[left] != ValueType.FLOAT32
                                        && types[left] != ValueType.BFLOAT16
                                        && types[left] != ValueType.FLOAT16
                                || values.get(left).requiresGrad()
                                        != values.get(output).requiresGrad()) {
                            throw new IllegalArgumentException(
                                    "max pool carrier or gradient metadata is incompatible");
                        }
                    }
                    case AVERAGE_POOL2D, AVERAGE_POOL3D -> {
                        if (types[left] != ValueType.FLOAT32 && !lowPrecision(types[left])
                                || types[output] != types[left]) {
                            throw new IllegalArgumentException(
                                    "average pool carrier is incompatible");
                        }
                        if (values.get(left).requiresGrad()
                                != values.get(output).requiresGrad()) {
                            throw new IllegalArgumentException(
                                    "average pool gradient metadata is incompatible");
                        }
                    }
                    case UNFOLD2D, UNFOLD3D -> {
                        if (types[left] != types[output]
                                || types[left] != ValueType.FLOAT32
                                && types[left] != ValueType.FLOAT64
                                && types[left] != ValueType.BFLOAT16
                                && types[left] != ValueType.FLOAT16) {
                            throw new IllegalArgumentException(
                                    node.kind() + " has an unsupported carrier type");
                        }
                        int dimensions =
                                node.kind() == MetalMpsGraphProgram.NodeKind.UNFOLD2D ? 2 : 3;
                        long[] words = node.attributeWords();
                        if (node.attributeKind()
                                        == (dimensions == 2
                                                ? MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_2D
                                                : MetalMpsGraphProgram.AttributeKind.PADDED_WINDOW_3D)
                                && words[dimensions * 4 + 1]
                                != MetalMpsGraphProgram.dataTypeWire(
                                        values.get(left).dataType())) {
                            throw new IllegalArgumentException(
                                    node.kind() + " padding scalar type must match its carrier");

                        }
                    }
                }
                }
                for (int nodeOutput : nodeOutputs) {
                    MetalMpsGraphProgram.ValueState outputState =
                            scanBody ? MetalMpsGraphProgram.ValueState.CANONICAL
                                    : (node.kind() == MetalMpsGraphProgram.NodeKind.RESHAPE
                                            || node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                                            || node.kind() == MetalMpsGraphProgram.NodeKind.SLICE)
                                    ? node.kind().outputState(values.get(left), values.get(nodeOutput))
                                    : node.kind().outputState();
                    if (!scanBody && (node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                                    || node.kind() == MetalMpsGraphProgram.NodeKind.SLICE)
                            && values.get(nodeOutput).elementCount() != 0L
                            && values.get(nodeOutput).layout()
                                    .filter(LayoutDescriptor::hasZeroStride).isPresent()) {
                        outputState = MetalMpsGraphProgram.ValueState.AFFINE_VIEW;
                    }
                    if (values.get(nodeOutput).densePhysical()
                            != (outputState == MetalMpsGraphProgram.ValueState.AFFINE_VIEW)) {
                        throw new IllegalArgumentException(
                                "Metal value affine declaration disagrees with its producer");
                    }
                    states[nodeOutput] = outputState;
                    // Ordinary affine steps represent their own canonical buffers. Native may
                    // elide a transpose only after independently proving its actual source.
                    LayoutDescriptor physical =
                            !scanBody && (node.kind() == MetalMpsGraphProgram.NodeKind.SELECT
                                            || node.kind() == MetalMpsGraphProgram.NodeKind.SLICE)
                                    ? extractionLayout(node, physicalLayouts[left],
                                            values.get(nodeOutput).dimensions())
                                    : LayoutDescriptor.contiguous(
                                            Shape.of(values.get(nodeOutput).dimensions()));
                    requireShape(physical != null, "Metal physical extraction layout is invalid");
                    physicalLayouts[nodeOutput] = physical;
                    used[nodeOutput] = true;
                }
                if (nodeInputs.length != 0) used[left] = true;
            }

            if (!scanBody && route == MetalPreparedRoute.CUSTOM_PROGRAM && !containsCustomOperation) {
                throw new IllegalArgumentException(
                        "custom Metal program route requires a custom operation");
            }
            boolean[] targeted = new boolean[valueCount];
            for (int target : targets) {
                requireIndex(target, valueCount, "target");
                if (!produced[target]
                        || states[target] == MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph target must be a node-produced value");
                }
                if (targeted[target]) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph target indices must be unique");
                }
                targeted[target] = true;
            }
            for (int value = 0; value < valueCount; value++) {
                if (!used[value] || types[value] == null) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph value table contains an unused or untyped index: "
                                    + value);
                }
            }
            for (int value = 0; value < valueCount; value++) {
                try {
                    Math.multiplyExact(
                            elementCount(value, valueRanks, valueDimensions),
                            types[value].byteWidth);
                } catch (ArithmeticException overflow) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph typed byte geometry overflows at value " + value,
                            overflow);
                }
            }
        }

        /**
         * Reconstructs the existing Model signature and independently infers every canonical
         * output's carrier, Shape and gradient flag. This is descriptor-only validation, never
         * evaluation or provider admission. Completed physical layouts are checked separately:
         * they must not be compared to Model's deliberately unresolved expression layouts.
         */
        private static void validateModelOrdinaryOutputs(MetalMpsGraphProgram.Node node,
                int[] inputs, int[] outputs, List<TensorDescriptor> values) {
            var operands = new ArrayList<TensorDescriptor>(inputs.length);
            for (int input : inputs) operands.add(values.get(input));
            var operation = MetalScanOrdinaryOperation.decode(node, values);
            var inferred = ScanBody.describeOrdinary(operation, operands, outputs.length, Optional.empty());
            requireShape(inferred.size() == outputs.length, "Model canonical output cardinality disagrees");
            for (int port = 0; port < outputs.length; port++) {
                TensorDescriptor actual = values.get(outputs[port]), expected = inferred.get(port);
                requireShape(actual.dataType() == expected.dataType()
                                && actual.shape().equals(expected.shape())
                                && actual.requiresGrad() == expected.requiresGrad(),
                        node.kind() + " canonical output metadata disagrees with Model at port " + port);
            }
        }

        /**
         * Independently checks new wire domains, without consulting provider admission.
         *
         * @param node exact decoded node
         * @param values complete encoded descriptors
         * @return whether one extended domain owns and validated this occurrence
         * @throws IllegalArgumentException if an owned encoded occurrence is malformed
         */
        private static boolean validateExtendedOperation(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values, boolean scanBody) {
            if (node.kind() == MetalMpsGraphProgram.NodeKind.GENERAL_SCAN) {
                MetalScanLowering.validate(node, values);
            } else if (MetalNormalizationSupport.handles(node)) {
                MetalNormalizationSupport.validate(node, values);
            } else if (MetalAttentionSupport.handles(node)) {
                MetalAttentionSupport.validate(node, values);
            } else if (MetalRecurrentSupport.handles(node)) {
                MetalRecurrentSupport.validate(node, values);
            } else if (MetalExactSupport.handles(node, values)) {
                MetalExactSupport.validate(node, values);
            } else if (MetalScatterSupport.handles(node, values)) {
                MetalScatterSupport.validate(node, values);
            } else if (MetalReductionSupport.handles(node, values)) {
                if (scanBody) MetalReductionSupport.validateScan(node, values);
                else MetalReductionSupport.validate(node, values);
            } else {
                return false;
            }
            return true;
        }
        private static boolean task0064ShapeMatches(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values, boolean scanBody) {
            try {
                boolean convolution = node.kind() == MetalMpsGraphProgram.NodeKind.CONV2D
                        || node.kind() == MetalMpsGraphProgram.NodeKind.CONV3D;
                int spatial = node.kind() == MetalMpsGraphProgram.NodeKind.CONV2D
                                || node.kind() == MetalMpsGraphProgram.NodeKind.MAX_POOL2D
                                || node.kind()
                                        == MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D
                        ? 2 : 3;
                int rank = spatial + 2;
                int[] inputs = node.inputs();
                MetalMpsGraphProgram.ValueDescriptor source = values.get(inputs[0]);
                MetalMpsGraphProgram.ValueDescriptor output =
                        values.get(node.outputIndex());
                if (!task0064Image(source, rank, spatial == 2, scanBody)
                        || !task0064Canonical(output, rank, scanBody)) {
                    return false;
                }
                long[] sourceShape = source.dimensions();
                long[] outputShape = output.dimensions();
                long[] words = node.attributeWords();
                if (sourceShape[0] != outputShape[0]) return false;
                if (convolution) {
                    if (inputs.length < 2 || inputs.length > 3
                            || words.length != spatial * 3 + 1) {
                        return false;
                    }
                    MetalMpsGraphProgram.ValueDescriptor weight =
                            values.get(inputs[1]);
                    if (!task0064Image(weight, rank, spatial == 2, scanBody)) return false;
                    long[] weightShape = weight.dimensions();
                    long groups = words[spatial * 3];
                    if (!task0064Positive(groups)
                            || sourceShape[1] % groups != 0L
                            || weightShape[0] % groups != 0L
                            || weightShape[1] != sourceShape[1] / groups
                            || outputShape[1] != weightShape[0]) {
                        return false;
                    }
                    if (inputs.length == 3) {
                        MetalMpsGraphProgram.ValueDescriptor bias =
                                values.get(inputs[2]);
                        if (!task0064Canonical(bias, 1, scanBody)
                                || bias.dimensions()[0] != weightShape[0]) {
                            return false;
                        }
                    }
                    long contributors = weightShape[1];
                    for (int axis = 0; axis < spatial; axis++) {
                        long stride = words[axis];
                        long padding = words[spatial + axis];
                        long dilation = words[spatial * 2 + axis];
                        if (outputShape[axis + 2] != task0064WindowExtent(
                                sourceShape[axis + 2],
                                weightShape[axis + 2],
                                stride,
                                padding,
                                dilation,
                                false)) {
                            return false;
                        }
                        contributors =
                                Math.multiplyExact(contributors, weightShape[axis + 2]);
                    }
                    return contributors <= UINT32_MAX;
                }
                if (inputs.length != 1 || words.length != spatial * 4 + 1
                        || sourceShape[1] != outputShape[1]
                        || words[spatial * 4] < 0L
                        || words[spatial * 4] > 1L) {
                    return false;
                }
                long divisor = 1L;
                for (int axis = 0; axis < spatial; axis++) {
                    long kernel = words[axis];
                    long stride = words[spatial + axis];
                    long padding = words[spatial * 2 + axis];
                    long dilation = words[spatial * 3 + axis];
                    if (outputShape[axis + 2] != task0064WindowExtent(
                            sourceShape[axis + 2],
                            kernel,
                            stride,
                            padding,
                            dilation,
                            words[spatial * 4] != 0L)) {
                        return false;
                    }
                    if (divisor > TASK0064_MAX_POOL_KERNEL_POSITIONS / kernel) {
                        return false;
                    }
                    divisor *= kernel;
                    if (source.requiresGrad() && source.dataType() != DataType.FLOAT32
                            && source.dataType() != DataType.BFLOAT16
                            && source.dataType() != DataType.FLOAT16
                            && stride < Math.addExact(
                                    Math.multiplyExact(dilation, kernel - 1L), 1L)) {
                        return false;
                    }
                }
                return true;
            } catch (ArithmeticException | IndexOutOfBoundsException exception) {
                return false;
            }
        }

        private static boolean task0064Canonical(
                MetalMpsGraphProgram.ValueDescriptor descriptor, int rank, boolean allowEmpty) {
            return descriptor.rank() == rank
                    && task0064Bounded(descriptor, allowEmpty)
                    && descriptor.layout().filter(
                            LayoutDescriptor.contiguous(Shape.of(descriptor.dimensions()))::equals)
                            .isPresent();
        }

        private static boolean task0064Image(
                MetalMpsGraphProgram.ValueDescriptor descriptor,
                int rank,
                boolean allowSingletonHeight, boolean allowEmpty) {
            if (descriptor.rank() != rank || !task0064Bounded(descriptor, allowEmpty)) return false;
            LayoutDescriptor layout = descriptor.layout().orElseThrow();
            if (layout.equals(LayoutDescriptor.contiguous(
                    Shape.of(descriptor.dimensions())))) {
                return true;
            }
            long[] shape = descriptor.dimensions();
            return allowSingletonHeight
                    && rank == 4
                    && shape[2] == 1L
                    && layout.storageOffset() == 0L
                    && layout.isView()
                    && Arrays.equals(
                            layout.strides(),
                            new long[] {
                                Math.multiplyExact(shape[1], shape[3]),
                                shape[3],
                                shape[3],
                                1L
                            })
                    && layout.referencedElementSpan()
                            == Math.multiplyExact(
                                    Math.multiplyExact(shape[0], shape[1]), shape[3]);
        }

        private static boolean task0064Bounded(
                MetalMpsGraphProgram.ValueDescriptor descriptor, boolean allowEmpty) {
            long elements = 1L;
            for (long extent : descriptor.dimensions()) {
                if (extent < (allowEmpty ? 0L : 1L) || extent > UINT32_MAX) return false;
                elements = Math.multiplyExact(elements, extent);
                if (elements > UINT32_MAX) return false;
            }
            return descriptor.layout().isPresent()
                    && descriptor.layout().orElseThrow().referencedElementSpan() <= UINT32_MAX;
        }

        private static long task0064WindowExtent(
                long input,
                long kernel,
                long stride,
                long padding,
                long dilation,
                boolean ceil) {
            if (!task0064Positive(kernel)
                    || !task0064Positive(stride)
                    || padding < 0L || padding > UINT32_MAX
                    || !task0064Positive(dilation)) {
                throw new ArithmeticException("window word exceeds uint32");
            }
            long effective = Math.addExact(Math.multiplyExact(dilation, kernel - 1L), 1L);
            long padded = Math.addExact(input, Math.multiplyExact(2L, padding));
            if (effective > UINT32_MAX || padded > UINT32_MAX || padded < effective) {
                throw new ArithmeticException("window does not fit uint32");
            }
            long numerator = padded - effective;
            long result = Math.addExact(
                    numerator / stride + (ceil && numerator % stride != 0L ? 1L : 0L),
                    1L);
            if (result > UINT32_MAX) throw new ArithmeticException("grid exceeds uint32");
            if (result - 1L > UINT32_MAX / stride) {
                throw new ArithmeticException("window origin exceeds uint32");
            }
            return result;
        }

        private static boolean task0064Positive(long value) {
            return value > 0L && value <= UINT32_MAX;
        }

        private static boolean task0063ShapeMatches(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values, boolean scanBody) {
            try {
                int inputIndex = node.firstInputIndex();
                int[] outputs = node.outputs();
                long[] input = values.get(inputIndex).dimensions();
                long[] words = node.attributeWords();
                boolean zeroK = node.kind() == MetalMpsGraphProgram.NodeKind.TOP_K
                        && words.length == 4 && words[1] == 0L;
                if (!task0063Uint32Bounded(input, false, scanBody || zeroK)) return false;
                return switch (node.kind()) {
                    case SORT, ARGSORT -> outputs.length == 1
                            && words.length == 2
                            && words[0] >= 0L && words[0] < input.length
                            && words[1] >= 0L && words[1] <= 1L
                            && task0063Uint32Bounded(
                                    values.get(outputs[0]).dimensions(), false, scanBody)
                            && Arrays.equals(
                                    input, values.get(outputs[0]).dimensions());
                    case TOP_K -> {
                        if (outputs.length != 2
                                || words.length != 4
                                || words[0] < 0L || words[0] >= input.length
                                || words[1] < 0L || words[1] > UINT32_MAX
                                || words[1] > input[Math.toIntExact(words[0])]
                                || words[2] < 0L || words[2] > 1L
                                || words[3] < 0L || words[3] > 1L) {
                            yield false;
                        }
                        long[] expected = input.clone();
                        expected[Math.toIntExact(words[0])] = words[1];
                        yield task0063Uint32Bounded(
                                        values.get(outputs[0]).dimensions(), false, scanBody || zeroK)
                                && task0063Uint32Bounded(
                                        values.get(outputs[1]).dimensions(), false, scanBody || zeroK)
                                && Arrays.equals(
                                        expected, values.get(outputs[0]).dimensions())
                                && Arrays.equals(
                                        expected, values.get(outputs[1]).dimensions());
                    }
                    case ARG_MAX, ARG_MIN -> {
                        if (outputs.length != 1
                                || words.length != 3
                                || words[0] < 0L || words[0] >= input.length
                                || words[1] < 0L || words[1] > 1L
                                || words[2] < 1L || words[2] > 2L
                                || input[Math.toIntExact(words[0])] == 0L) {
                            yield false;
                        }
                        int axis = Math.toIntExact(words[0]);
                        boolean keep = words[1] != 0L;
                        long[] expected = new long[keep ? input.length : input.length - 1];
                        for (int source = 0, destination = 0;
                                source < input.length;
                                source++) {
                            if (source == axis) {
                                if (keep) expected[destination++] = 1L;
                            } else {
                                expected[destination++] = input[source];
                            }
                        }
                        yield task0063Uint32Bounded(
                                        values.get(outputs[0]).dimensions(), true, scanBody)
                                && Arrays.equals(
                                        expected, values.get(outputs[0]).dimensions());
                    }
                    default -> false;
                };
            } catch (ArithmeticException | IndexOutOfBoundsException exception) {
                return false;
            }
        }

        private static boolean task0063Uint32Bounded(
                long[] dimensions, boolean allowScalar, boolean allowEmpty) {
            if ((!allowScalar && dimensions.length == 0)
                    || dimensions.length > MAX_RANK) {
                return false;
            }
            long elements = 1L;
            for (long dimension : dimensions) {
                if (dimension < (allowEmpty ? 0L : 1L) || dimension > UINT32_MAX
                        || dimension != 0L && elements > UINT32_MAX / dimension) {
                    return false;
                }
                elements *= dimension;
            }
            return true;
        }

        private static boolean supportsFoldGeometry(
                MetalMpsGraphProgram.Node node, ValueType type) {
            long[] words = node.attributeWords();
            if (numericFloating(type)) return true;
            if (node.kind() == MetalMpsGraphProgram.NodeKind.FOLD_AXIS) {
                return words[2] >= words[1];
            }
            int dimensions = node.kind() == MetalMpsGraphProgram.NodeKind.FOLD2D ? 2 : 3;
            int offset = dimensions + 3;
            try {
                for (int spatial = 0; spatial < dimensions; spatial++) {
                    long effective =
                            Math.addExact(
                                    Math.multiplyExact(
                                            words[offset + dimensions * 3 + spatial],
                                            words[offset + spatial] - 1L),
                                    1L);
                    if (words[offset + dimensions + spatial] < effective) return false;
                }
                return true;
            } catch (ArithmeticException | IndexOutOfBoundsException exception) {
                return false;
            }
        }

        private static boolean task0066Selected(MetalMpsGraphProgram.NodeKind kind) {
            return switch (kind) {
                case RESHAPE,
                        EXPAND,
                        PERMUTE,
                        EXPAND_DIMS,
                        SQUEEZE,
                        CONTIGUOUS,
                        GATHER,
                        ONE_HOT,
                        SCATTER_ELEMENTS,
                        UNFOLD_AXIS,
                        CAST,
                        IS_FINITE,
                        IS_NAN,
                        IS_INF,
                        LOGICAL_AND,
                        LOGICAL_OR,
                        LOGICAL_NOT,
                        WHERE,
                        GATHER_ELEMENTS,
                        GATHER_ND,
                        SCATTER_ND,
                        SELECT,
                        PAD,
                        SLICE,
                        SLICE_UPDATE,
                        CONCAT,
                        STACK,
                        TILE,
                        FOLD_AXIS,
                        UNFOLD2D,
                        FOLD2D,
                        UNFOLD3D,
                        FOLD3D ->
                        true;
                default -> false;
            };
        }

        private static boolean task0066Carrier(ValueType type) {
            return type == ValueType.FLOAT64
                    || type == ValueType.FLOAT32
                    || type == ValueType.BFLOAT16
                    || type == ValueType.FLOAT16
                    || type == ValueType.INT64
                    || type == ValueType.INT32
                    || type == ValueType.BOOL;
        }

        private static boolean lowPrecision(ValueType type) {
            return type == ValueType.BFLOAT16 || type == ValueType.FLOAT16;
        }

        private static boolean numericFloating(ValueType type) {
            return type == ValueType.FLOAT32 || lowPrecision(type);
        }

        private static boolean task0066Floating(ValueType type) {
            return type == ValueType.FLOAT64
                    || type == ValueType.FLOAT32
                    || type == ValueType.BFLOAT16
                    || type == ValueType.FLOAT16;
        }

        private static ValueType task0066Promote(ValueType left, ValueType right) {
            if (!task0066Floating(left) || !task0066Floating(right)) return null;
            if (left == ValueType.FLOAT64 || right == ValueType.FLOAT64) {
                return ValueType.FLOAT64;
            }
            if (left == ValueType.FLOAT32 || right == ValueType.FLOAT32 || left != right) {
                return ValueType.FLOAT32;
            }
            return left;
        }

        private static void validateTask0066Gradients(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                ValueType[] types) {
            int[] inputs = node.inputs();
            boolean[] gradients = new boolean[inputs.length];
            boolean any = false;
            for (int index = 0; index < inputs.length; index++) {
                gradients[index] = values.get(inputs[index]).requiresGrad();
                if (gradients[index] && !task0066Floating(types[inputs[index]])) {
                    throw new IllegalArgumentException(
                            "Task 0066 integral and BOOL roles are non-differentiable");
                }
                any |= gradients[index];
            }
            int output = node.outputIndex();
            boolean outputGradient = values.get(output).requiresGrad();
            if (outputGradient && !task0066Floating(types[output])) {
                throw new IllegalArgumentException(
                        "Task 0066 integral and BOOL outputs are non-differentiable");
            }
            boolean expected = gradients.length != 0 && gradients[0];
            switch (node.kind()) {
                case CAST ->
                        expected =
                                gradients[0]
                                        && task0066Floating(types[inputs[0]])
                                        && task0066Floating(types[output]);
                case IS_FINITE, IS_NAN, IS_INF, GT, GE, LT, LE, EQ, NE,
                        LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT ->
                        expected = false;
                case WHERE -> expected = gradients[1] || gradients[2];
                case ONE_HOT -> expected = false;
                case GATHER, GATHER_ELEMENTS, GATHER_ND -> expected = gradients[0];
                case SCATTER_ELEMENTS, SCATTER_ND -> expected = gradients[0] || gradients[2];
                case SLICE_UPDATE -> expected = gradients[0] || gradients[1];
                case CONCAT, STACK -> expected = any;
                default -> {}
            }
            if (outputGradient != expected) {
                throw new IllegalArgumentException(
                        "Task 0066 input and output gradient metadata disagree");
            }
        }

        private static boolean exactNoGradientProduction(MetalMpsGraphProgram.NodeKind kind) {
            return switch (kind) {
                case INITIAL_STATE,
                        CAST,
                        GATHER_ELEMENTS,
                        GATHER_ND,
                        SCATTER_ND,
                        SCATTER_ADD,
                        SELECT,
                        PAD,
                        SLICE,
                        SLICE_UPDATE,
                        CONCAT,
                        STACK,
                        TILE,
                        FOLD_AXIS,
                        UNFOLD2D,
                        FOLD2D,
                        UNFOLD3D,
                        FOLD3D,
                        PROD,
                        ALL,
                        ANY,
                        L1_NORM,
                        VARIANCE ->
                        true;
                default -> false;
            };
        }

        private static boolean task0065State(MetalMpsGraphProgram.ValueDescriptor descriptor) {
            return task0065Value(descriptor, ValueType.INT64, false, false)
                    && Arrays.equals(descriptor.dimensions(), new long[] {2L})
                    && !descriptor.requiresGrad();
        }

        private static boolean task0065Value(
                MetalMpsGraphProgram.ValueDescriptor descriptor,
                ValueType type,
                boolean allowScalar, boolean allowEmpty) {
            if (ValueType.from(descriptor.dataType()) != type
                    || descriptor.rank() < (allowScalar ? 0 : 1)
                    || descriptor.rank() > MAX_RANK
                    || descriptor.layout().isEmpty()
                    || !descriptor.layout().orElseThrow().equals(
                            LayoutDescriptor.contiguous(Shape.of(descriptor.dimensions())))) {
                return false;
            }
            long elements = 1L;
            for (long extent : descriptor.dimensions()) {
                if (extent < (allowEmpty ? 0L : 1L) || extent > UINT32_MAX) return false;
                elements = Math.multiplyExact(elements, extent);
                if (elements > UINT32_MAX) return false;
            }
            return descriptor.layout().orElseThrow().referencedElementSpan() <= UINT32_MAX;
        }

        private static boolean task0059ShapeMatches(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values,
                MetalPreparedRoute route, boolean scanBody) {
            try {
                int[] inputs = node.inputs();
                long[] input = values.get(inputs[0]).dimensions();
                long[] output = values.get(node.outputIndex()).dimensions();
                long[] words = node.attributeWords();
                return switch (node.kind()) {
                    case CAST -> Arrays.equals(input, output);
                    case GATHER_ELEMENTS -> {
                        long[] indices = values.get(inputs[1]).dimensions();
                        int axis = node.axis();
                        boolean matches = input.length > 0
                                && axis >= 0
                                && axis < input.length
                                && input.length == indices.length
                                && Arrays.equals(indices, output);
                        for (int dimension = 0;
                                matches && dimension < input.length; dimension++) {
                            matches = dimension == axis
                                    || input[dimension] == indices[dimension];
                        }
                        yield matches;
                    }
                    case GATHER_ND -> {
                        long[] indices = values.get(inputs[1]).dimensions();
                        int batch = Math.toIntExact(words[0]);
                        boolean matches = indices.length > 0
                                && batch >= 0
                                && batch < indices.length
                                && batch <= input.length;
                        for (int axis = 0; matches && axis < batch; axis++) {
                            matches = input[axis] == indices[axis];
                        }
                        int tuple = matches
                                ? Math.toIntExact(indices[indices.length - 1]) : 0;
                        matches = matches && tuple >= 1 && tuple <= input.length - batch;
                        if (!matches) yield false;
                        long[] expected = new long[
                                indices.length - 1 + input.length - batch - tuple];
                        System.arraycopy(indices, 0, expected, 0, indices.length - 1);
                        System.arraycopy(input, batch + tuple, expected, indices.length - 1,
                                input.length - batch - tuple);
                        yield Arrays.equals(expected, output);
                    }
                    case SCATTER_ADD -> {
                        long[] indices = values.get(inputs[1]).dimensions();
                        long[] updates = values.get(inputs[2]).dimensions();
                        int axis = node.axis();
                        boolean matches = input.length > 0
                                && axis >= 0
                                && axis < input.length
                                && Arrays.equals(input, output)
                                && Arrays.equals(indices, updates)
                                && indices.length == input.length;
                        for (int dimension = 0;
                                matches && dimension < input.length; dimension++) {
                            matches = dimension == axis
                                    || input[dimension] == indices[dimension];
                        }
                        yield matches;
                    }
                    case SCATTER_ND -> {
                        long[] indices = values.get(inputs[1]).dimensions();
                        long[] updates = values.get(inputs[2]).dimensions();
                        int batch = Math.toIntExact(words[0]);
                        boolean matches = Arrays.equals(input, output)
                                && indices.length > 0
                                && batch >= 0
                                && batch < indices.length
                                && batch <= input.length;
                        for (int axis = 0; matches && axis < batch; axis++) {
                            matches = input[axis] == indices[axis];
                        }
                        int tuple = matches
                                ? Math.toIntExact(indices[indices.length - 1]) : 0;
                        matches = matches && tuple >= 1 && tuple <= input.length - batch;
                        if (!matches) yield false;
                        long[] expected = new long[
                                indices.length - 1 + input.length - batch - tuple];
                        System.arraycopy(indices, 0, expected, 0, indices.length - 1);
                        System.arraycopy(input, batch + tuple, expected, indices.length - 1,
                                input.length - batch - tuple);
                        yield Arrays.equals(expected, updates);
                    }
                    case SELECT -> {
                        int axis = Math.toIntExact(words[0]);
                        long index = words[1];
                        if (input.length == 0
                                || axis < 0
                                || axis >= input.length
                                || index < 0
                                || index >= input[axis]) yield false;
                        long[] expected = new long[input.length - 1];
                        System.arraycopy(input, 0, expected, 0, axis);
                        System.arraycopy(input, axis + 1, expected, axis,
                                input.length - axis - 1);
                        yield Arrays.equals(expected, output);
                    }
                    case PAD -> {
                        int rank = Math.toIntExact(words[0]);
                        if (rank != input.length || output.length != rank) yield false;
                        long[] expected = input.clone();
                        for (int axis = 0; axis < rank; axis++) {
                            expected[axis] = Math.addExact(
                                    Math.addExact(input[axis], words[1 + axis]),
                                    words[1 + rank + axis]);
                        }
                        yield Arrays.equals(expected, output);
                    }
                    case SLICE -> {
                        long[] region = task0059SliceRegion(node, input, true);
                        yield region != null
                                && Arrays.equals(region, output);
                    }
                    case SLICE_UPDATE -> {
                        long[] region = task0059SliceRegion(node, input, scanBody);
                        yield region != null
                                && Arrays.equals(input, output)
                                && Arrays.equals(region, values.get(inputs[1]).dimensions());
                    }
                    case CONCAT -> {
                        int axis = node.axis();
                        if (input.length == 0 || axis < 0 || axis >= input.length
                                || output.length != input.length) yield false;
                        long[] expected = input.clone();
                        expected[axis] = 0;
                        boolean matches = true;
                        for (int value : inputs) {
                            long[] part = values.get(value).dimensions();
                            matches &= part.length == input.length;
                            for (int dimension = 0;
                                    matches && dimension < input.length; dimension++) {
                                matches = dimension == axis
                                        || part[dimension] == input[dimension];
                            }
                            if (matches) {
                                expected[axis] =
                                        Math.addExact(expected[axis], part[axis]);
                            }
                        }
                        yield matches && Arrays.equals(expected, output);
                    }
                    case STACK -> {
                        int axis = node.axis();
                        if (input.length >= MAX_RANK || axis < 0 || axis > input.length
                                || output.length != input.length + 1
                                || output[axis] != inputs.length) yield false;
                        boolean matches = true;
                        for (int value : inputs) {
                            matches &= Arrays.equals(
                                    input, values.get(value).dimensions());
                        }
                        for (int dimension = 0;
                                matches && dimension < input.length; dimension++) {
                            matches = output[dimension < axis ? dimension : dimension + 1]
                                    == input[dimension];
                        }
                        yield matches;
                    }
                    case TILE -> {
                        int rank = Math.toIntExact(words[0]);
                        if (rank != input.length || output.length != rank) yield false;
                        long[] expected = new long[rank];
                        for (int axis = 0; axis < rank; axis++) {
                            expected[axis] =
                                    Math.multiplyExact(input[axis], words[1 + axis]);
                        }
                        yield Arrays.equals(expected, output);
                    }
                    case FOLD_AXIS -> {
                        int axis = node.axis();
                        long size = words[1];
                        long step = words[2];
                        if (output.length + 1 != input.length
                                || axis < 0
                                || axis >= output.length
                                || input[input.length - 1] != size
                                || output[axis] < size) yield false;
                        long positions = (output[axis] - size) / step + 1;
                        boolean matches = true;
                        for (int dimension = 0;
                                matches && dimension < output.length; dimension++) {
                            matches = input[dimension]
                                    == (dimension == axis ? positions : output[dimension]);
                        }
                        yield matches;
                    }
                    case UNFOLD2D, UNFOLD3D -> {
                        int dimensions =
                                node.kind() == MetalMpsGraphProgram.NodeKind.UNFOLD2D ? 2 : 3;
                        if (input.length != dimensions + 2 || output.length != 3) yield false;
                        long kernelVolume = 1;
                        long positions = 1;
                        for (int spatial = 0; spatial < dimensions; spatial++) {
                            kernelVolume =
                                    Math.multiplyExact(kernelVolume, words[spatial]);
                            positions = Math.multiplyExact(
                                    positions,
                                    task0059WindowExtent(
                                            input[spatial + 2],
                                            words[spatial],
                                            words[dimensions * 2 + spatial],
                                            words[dimensions + spatial],
                                            words[dimensions * 3 + spatial],
                                            words[dimensions * 4] != 0,
                                            route == MetalPreparedRoute.MPSGRAPH
                                                    && (dimensions == 2 || spatial > 0)));
                        }
                        long[] expected = {
                                input[0],
                                Math.multiplyExact(input[1], kernelVolume),
                                positions
                        };
                        yield Arrays.equals(expected, output);
                    }
                    case FOLD2D, FOLD3D -> {
                        int dimensions =
                                node.kind() == MetalMpsGraphProgram.NodeKind.FOLD2D ? 2 : 3;
                        int targetRank = Math.toIntExact(words[0]);
                        if (targetRank != dimensions + 2
                                || output.length != targetRank
                                || input.length != 3) yield false;
                        boolean matches = true;
                        for (int axis = 0; matches && axis < targetRank; axis++) {
                            matches = words[1 + axis] == output[axis];
                        }
                        int offset = 1 + targetRank;
                        long kernelVolume = 1;
                        long positions = 1;
                        for (int spatial = 0; matches && spatial < dimensions; spatial++) {
                            kernelVolume =
                                    Math.multiplyExact(kernelVolume, words[offset + spatial]);
                            positions = Math.multiplyExact(
                                    positions,
                                    task0059WindowExtent(
                                            output[spatial + 2],
                                            words[offset + spatial],
                                            words[offset + dimensions * 2 + spatial],
                                            words[offset + dimensions + spatial],
                                            words[offset + dimensions * 3 + spatial],
                                            words[offset + dimensions * 4] != 0,
                                            route == MetalPreparedRoute.MPSGRAPH
                                                    && (dimensions == 2 || spatial > 0)));
                        }
                        yield matches
                                && input[0] == output[0]
                                && input[1] == Math.multiplyExact(output[1], kernelVolume)
                                && input[2] == positions;
                    }
                    default -> false;
                };
            } catch (ArithmeticException | IndexOutOfBoundsException exception) {
                return false;
            }
        }

        private static boolean task0059LayoutMatches(
                MetalMpsGraphProgram.Node node,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            MetalMpsGraphProgram.ValueDescriptor input = values.get(node.firstInputIndex());
            MetalMpsGraphProgram.ValueDescriptor output = values.get(node.outputIndex());
            if (input.layout().isEmpty() || output.layout().isEmpty()) return false;
            if (node.kind() != MetalMpsGraphProgram.NodeKind.SELECT
                    && node.kind() != MetalMpsGraphProgram.NodeKind.SLICE) return true;
            LayoutDescriptor expected = extractionLayout(
                    node, input.layout().orElseThrow(), output.dimensions());
            return expected != null && output.layout().filter(expected::equals).isPresent();
        }

        /** The same extraction map applies separately to logical and represented source layouts. */
        private static LayoutDescriptor extractionLayout(
                MetalMpsGraphProgram.Node node, LayoutDescriptor source, long[] outputShape) {
            try {
                Shape resultShape = Shape.of(outputShape);
                if (resultShape.knownElementCount().orElseThrow() == 0L)
                    return LayoutDescriptor.contiguous(resultShape);
                long[] words = node.attributeWords();
                long[] strides = source.strides();
                long offset = source.storageOffset();
                if (node.kind() == MetalMpsGraphProgram.NodeKind.SELECT) {
                    int axis = Math.toIntExact(words[0]);
                    if (axis < 0 || axis >= strides.length) return null;
                    offset = Math.addExact(offset, Math.multiplyExact(words[1], strides[axis]));
                    long[] selected = new long[strides.length - 1];
                    System.arraycopy(strides, 0, selected, 0, axis);
                    System.arraycopy(strides, axis + 1, selected, axis, strides.length - axis - 1);
                    strides = selected;
                } else if (node.kind() == MetalMpsGraphProgram.NodeKind.SLICE) {
                    if (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.CROP_TO_SHAPE) {
                        int rank = Math.toIntExact(words[0]);
                        if (rank != strides.length) return null;
                        int prefixOffset = 2 + rank;
                        for (int axis = 0; axis < rank; axis++) {
                            offset = Math.addExact(offset,
                                    Math.multiplyExact(words[prefixOffset + axis], strides[axis]));
                        }
                    } else {
                        int count = Math.toIntExact(words[0]);
                        boolean materialized = false;
                        for (int item = 0; item < count; item++) {
                            int axis = Math.toIntExact(words[1 + count * 2 + item]);
                            long step = words[1 + count * 3 + item];
                            if (axis < 0 || axis >= strides.length || step == 0L) return null;
                            materialized |= step < 0L;
                        }
                        if (materialized) return LayoutDescriptor.contiguous(resultShape);
                        for (int item = 0; item < count; item++) {
                            int axis = Math.toIntExact(words[1 + count * 2 + item]);
                            long step = words[1 + count * 3 + item];
                            offset = Math.addExact(offset,
                                    Math.multiplyExact(words[1 + item], strides[axis]));
                            strides[axis] = Math.multiplyExact(strides[axis], step);
                        }
                    }
                } else {
                    return null;
                }
                return LayoutDescriptor.of(resultShape, strides, offset, true);
            } catch (ArithmeticException | IndexOutOfBoundsException incompatible) {
                return null;
            }
        }

        private static long[] task0059SliceRegion(
                MetalMpsGraphProgram.Node node, long[] input, boolean allowEmpty) {
            long[] words = node.attributeWords();
            if (node.attributeKind() == MetalMpsGraphProgram.AttributeKind.CROP_TO_SHAPE) {
                int targetRank = Math.toIntExact(words[0]);
                int prefixOffset = 1 + targetRank;
                int prefixRank = Math.toIntExact(words[prefixOffset]);
                if (targetRank != input.length || prefixRank != input.length) return null;
                long[] target = Arrays.copyOfRange(words, 1, prefixOffset);
                for (int axis = 0; axis < input.length; axis++) {
                    long prefix = words[prefixOffset + 1 + axis];
                    if (Math.addExact(prefix, target[axis]) > input[axis]) return null;
                }
                return target;
            }
            int count = Math.toIntExact(words[0]);
            if (count > input.length) return null;
            long[] expected = input.clone();
            for (int item = 0; item < count; item++) {
                long start = words[1 + item];
                long length = words[1 + count + item];
                int axis = Math.toIntExact(words[1 + count * 2 + item]);
                long step = words[1 + count * 3 + item];
                if (axis < 0
                        || axis >= input.length
                        || start < 0L
                        || length < 0L || length == 0L && !allowEmpty
                        || step == 0L
                        || (length == 0L ? start != 0L : start >= input[axis])) {
                    return null;
                }
                if (length == 0L) { expected[axis] = 0L; continue; }
                long last = Math.addExact(start, Math.multiplyExact(length - 1L, step));
                if (last < 0L || last >= input[axis]) return null;
                expected[axis] = length;
            }
            return expected;
        }

        private static long task0059WindowExtent(
                long input, long kernel, long padding, long stride,
                long dilation, boolean ceil, boolean boundedOrigin) {
            long effective =
                    Math.addExact(Math.multiplyExact(dilation, kernel - 1), 1);
            long numerator = Math.subtractExact(
                    Math.addExact(input, Math.multiplyExact(2, padding)), effective);
            if (numerator < 0) throw new ArithmeticException("window does not fit");
            long result = Math.addExact(
                    numerator / stride + (ceil && numerator % stride != 0 ? 1 : 0), 1);
            if (boundedOrigin && result - 1L > UINT32_MAX / stride) {
                throw new ArithmeticException("MPSGraph window origin exceeds uint32");
            }
            return result;
        }

        static void validateRun(
                int expectedInputs,
                int inputCount,
                MemorySegment inputBuffers,
                int expectedOutputs,
                int outputCount,
                MemorySegment outputBuffers,
                boolean allowInputOutputAlias) {
            Objects.requireNonNull(inputBuffers, "inputBuffers");
            Objects.requireNonNull(outputBuffers, "outputBuffers");
            if (inputCount != expectedInputs || outputCount != expectedOutputs) {
                throw new IllegalArgumentException(
                        "Metal program run counts must match the compiled inputs and targets");
            }
            validateAddressSegment(inputBuffers, inputCount, "inputBuffers");
            validateAddressSegment(outputBuffers, outputCount, "outputBuffers");
            for (int output = 0; output < outputCount; output++) {
                long outputAddress = outputBuffers.getAtIndex(ADDRESS, output).address();
                for (int previous = 0; previous < output; previous++) {
                    if (outputAddress
                            == outputBuffers.getAtIndex(ADDRESS, previous).address()) {
                        throw new IllegalArgumentException(
                                "Metal program output native buffers must not alias");
                    }
                }
            }
            if (!allowInputOutputAlias) {
                for (int input = 0; input < inputCount; input++) {
                    long inputAddress = inputBuffers.getAtIndex(ADDRESS, input).address();
                    for (int output = 0; output < outputCount; output++) {
                        if (inputAddress == outputBuffers.getAtIndex(ADDRESS, output).address()) {
              throw new IllegalArgumentException(
                  "Metal MPSGraph input and output native buffers must not alias");
                        }
                    }
                }
            }
        }

        private static void validateAddressSegment(
                MemorySegment segment, int count, String name) {
            if (!segment.scope().isAlive() || !segment.isAccessibleBy(Thread.currentThread())) {
                throw new IllegalStateException(name + " is not alive and accessible");
            }
            if (!segment.isNative()) {
                throw new IllegalArgumentException(name + " must be native memory");
            }
            long expectedBytes = Math.multiplyExact((long) count, ADDRESS.byteSize());
            if (segment.byteSize() != expectedBytes) {
                throw new IllegalArgumentException(
                        name + " byte size must exactly match its handle count");
            }
            for (int index = 0; index < count; index++) {
                if (segment.getAtIndex(ADDRESS, index).address() == 0L) {
                    throw new IllegalArgumentException(name + " contains a null handle");
                }
            }
        }

        private static void requireIndex(int index, int valueCount, String role) {
            if (index < 0 || index >= valueCount) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph " + role + " index is out of range");
            }
        }

        private static boolean task0052CustomOnly(MetalMpsGraphProgram.NodeKind kind) {
            int wire = kind.wireIdentity();
            return wire >= MetalMpsGraphProgram.NodeKind.GT.wireIdentity()
                    && wire <= MetalMpsGraphProgram.NodeKind.CUM_PROD.wireIdentity();
        }

        private static boolean task0063CustomOnly(MetalMpsGraphProgram.NodeKind kind) {
            return switch (kind) {
                case SORT, ARGSORT, TOP_K, ARG_MAX, ARG_MIN -> true;
                default -> false;
            };
        }
        private static boolean task0064CustomOnly(MetalMpsGraphProgram.NodeKind kind) {
            return switch (kind) {
                case CONV2D, CONV3D, MAX_POOL2D, AVERAGE_POOL2D,
                        MAX_POOL3D, AVERAGE_POOL3D -> true;
                default -> false;
            };
        }
        private static boolean task0065CustomOnly(MetalMpsGraphProgram.NodeKind kind) {
            return kind == MetalMpsGraphProgram.NodeKind.DROPOUT
                    || kind == MetalMpsGraphProgram.NodeKind.INITIAL_STATE;
        }

        private static boolean task0069CustomOnly(
                MetalMpsGraphProgram.Node node, int[] ranks, long[] dimensions) {
            if (node.kind() == MetalMpsGraphProgram.NodeKind.L1_NORM
                    || node.kind() == MetalMpsGraphProgram.NodeKind.SCATTER_ADD) {
                return true;
            }
            if (node.kind() != MetalMpsGraphProgram.NodeKind.VARIANCE) return false;
            int input = node.firstInputIndex();
            int output = node.outputIndex();
            return input >= 0
                    && output >= 0
                    && input < ranks.length
                    && output < ranks.length
                    && task0069VarianceMatches(node, input, output, ranks, dimensions);
        }

        private static boolean usesCustomMatmul(
                MetalMpsGraphProgram.Node node, ValueType[] types, int[] ranks) {
            if (node.kind() != MetalMpsGraphProgram.NodeKind.MATMUL) return false;
            int left = node.firstInputIndex();
            int right = node.secondInputIndex();
            int output = node.outputIndex();
            return types[left] != ValueType.FLOAT32
                    || types[right] != ValueType.FLOAT32
                    || types[output] != ValueType.FLOAT32
                    || ranks[left] != 2
                    || ranks[right] != 2
                    || ranks[output] != 2;
        }
        private static boolean usesLowPrecisionStorage(
                MetalMpsGraphProgram.Node node, ValueType[] types) {
            for (int input : node.inputs()) if (lowPrecision(types[input])) return true;
            for (int output : node.outputs()) if (lowPrecision(types[output])) return true;
            return false;
        }




        private static boolean affineLayoutMatches(
                MetalMpsGraphProgram.Node node,
                MetalMpsGraphProgram.ValueDescriptor input,
                MetalMpsGraphProgram.ValueDescriptor output) {
            if (input.dataType() != output.dataType()
                    || input.requiresGrad() != output.requiresGrad()
                    || input.layout().isEmpty() || output.layout().isEmpty()) return false;
            LayoutDescriptor source = input.layout().orElseThrow();
            LayoutDescriptor result = output.layout().orElseThrow();
            boolean emptyExpand = node.kind() == MetalMpsGraphProgram.NodeKind.EXPAND
                    && output.elementCount() == 0;
            if (!result.isView() || source.storageOffset() != result.storageOffset()
                    || !emptyExpand && source.referencedElementSpan() != result.referencedElementSpan()) return false;
            long[] sourceStrides = source.strides();
            long[] resultStrides = result.strides();
            long[] sourceShape = input.dimensions();
            long[] resultShape = output.dimensions();
            try {
                switch (node.kind()) {
                    case EXPAND -> {
                        int padding = resultShape.length - sourceShape.length;
                        if (padding < 0) return false;
                        for (int axis = 0; axis < resultShape.length; axis++) {
                            int sourceAxis = axis - padding;
                            long stride = sourceAxis < 0 ? 0L
                                    : sourceShape[sourceAxis] == 1L && resultShape[axis] != 1L
                                            ? 0L : sourceStrides[sourceAxis];
                            if (resultStrides[axis] != stride) return false;
                        }
                    }
                    case PERMUTE -> {
                        long[] axes = node.attributeValues();
                        if (axes.length != sourceStrides.length
                                || axes.length != resultStrides.length) return false;
                        for (int axis = 0; axis < axes.length; axis++) {
                            if (axes[axis] < 0 || axes[axis] >= axes.length
                                    || resultStrides[axis] != sourceStrides[(int) axes[axis]]) {
                                return false;
                            }
                        }
                    }
                    case EXPAND_DIMS -> {
                        int inserted = node.axis();
                        if (inserted < 0 || inserted > sourceShape.length
                                || resultShape.length != sourceShape.length + 1) return false;
                        for (int axis = 0; axis < resultShape.length; axis++) {
                            long stride = axis == inserted
                                    ? inserted == sourceShape.length ? 1L
                                            : Math.multiplyExact(
                                                    sourceStrides[inserted], sourceShape[inserted])
                                    : sourceStrides[axis < inserted ? axis : axis - 1];
                            if (resultStrides[axis] != stride) return false;
                        }
                    }
                    case SQUEEZE -> {
                        int removed = node.axis();
                        if (removed < 0 || removed >= sourceShape.length
                                || sourceShape[removed] != 1L
                                || resultShape.length != sourceShape.length - 1) return false;
                        for (int axis = 0; axis < resultShape.length; axis++) {
                            if (resultStrides[axis]
                                    != sourceStrides[axis < removed ? axis : axis + 1]) return false;
                        }
                    }
                    default -> { return false; }
                }
                return true;
            } catch (ArithmeticException overflow) {
                return false;
            }
        }

        private static boolean exactLocalTranspose(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                MetalMpsGraphProgram.ValueState[] states,
                int[] ranks,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            int rank = ranks[input];
            if (states[input] == MetalMpsGraphProgram.ValueState.UNAVAILABLE
                    || !values.get(input).matmulDense()
                    || values.get(input).dataType() != values.get(output).dataType()
                    || values.get(input).requiresGrad() != values.get(output).requiresGrad()
                    || rank < 2
                    || ranks[output] != rank
                    || node.attributeCount() != rank) {
                return false;
            }
            long[] axes = node.attributeValues();
            for (int axis = 0; axis < rank - 2; axis++) {
                if (axes[axis] != axis) return false;
            }
            if (axes[rank - 2] != rank - 1 || axes[rank - 1] != rank - 2) return false;
            MetalMpsGraphProgram.ValueDescriptor descriptor = values.get(output);
            if (!descriptor.densePhysical()) return false;
            long[] sourceShape = descriptor.dimensions();
            long swap = sourceShape[rank - 2];
            sourceShape[rank - 2] = sourceShape[rank - 1];
            sourceShape[rank - 1] = swap;
            long[] sourceStrides = LayoutDescriptor.contiguous(Shape.of(sourceShape)).strides();
            long[] outputStrides = sourceStrides.clone();
            outputStrides[rank - 2] = sourceStrides[rank - 1];
            outputStrides[rank - 1] = sourceStrides[rank - 2];
            LayoutDescriptor expected =
                    LayoutDescriptor.of(
                            Shape.of(descriptor.dimensions()), outputStrides, 0L, true);
            return descriptor.layout().filter(expected::equals).isPresent();
        }

        private static boolean exactLocalSingletonHeight(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                MetalMpsGraphProgram.ValueState[] states,
                int[] ranks,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            if (node.kind() != MetalMpsGraphProgram.NodeKind.EXPAND_DIMS
                    || node.axis() != 2
                    || states[input] != MetalMpsGraphProgram.ValueState.CANONICAL
                    || ranks[input] != 3 || ranks[output] != 4) {
                return false;
            }
            MetalMpsGraphProgram.ValueDescriptor descriptor = values.get(output);
            long[] shape = descriptor.dimensions();
            if (!descriptor.densePhysical() || shape[2] != 1L) return false;
            LayoutDescriptor expected = LayoutDescriptor.of(
                    Shape.of(shape),
                    new long[] {
                        Math.multiplyExact(shape[1], shape[3]),
                        shape[3],
                        shape[3],
                        1L
                    },
                    0L,
                    true);
            return descriptor.layout().filter(expected::equals).isPresent();
        }

        private static boolean task0069VarianceMatches(
                MetalMpsGraphProgram.Node node,
                int inputIndex,
                int outputIndex,
                int[] ranks,
                long[] dimensions) {
            if (node.kind() != MetalMpsGraphProgram.NodeKind.VARIANCE
                    || node.attributeKind()
                            != MetalMpsGraphProgram.AttributeKind.STATISTICAL_REDUCTION) {
                return false;
            }
            long[] words = node.attributeWords();
            if (words.length != 4
                    || words[0] != 1L
                    || words[1] != 0L
                    || (words[2] != 0L && words[2] != 1L)
                    || words[3] != 0L
                    || ranks[inputIndex] != 1
                    || dimensions[inputIndex * MAX_RANK] != 1L) {
                return false;
            }
            if (words[2] == 0L) {
                return ranks[outputIndex] == 0;
            }
            return ranks[outputIndex] == 1
                    && dimensions[outputIndex * MAX_RANK] == 1L;
        }

        private static void validateTask0064ConvolutionTypesAndGradients(
                MetalMpsGraphProgram.Node node,
                ValueType[] types,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            ValueType promoted = null;
            boolean anyGradient = false;
            for (int input : node.inputs()) {
                ValueType type = types[input];
                if (!numericFloating(type))
                    throw new IllegalArgumentException("convolution input carrier is incompatible");
                promoted = promoted == null ? type : task0066Promote(promoted, type);
                anyGradient |= values.get(input).requiresGrad();
            }
            int output = node.outputIndex();
            requireShape(types[output] == promoted && values.get(output).requiresGrad() == anyGradient,
                    "convolution must retain original-role promotion and gradient metadata");
        }

        private static void validateMatmulTypesAndGradients(
                int left,
                int right,
                int output,
                ValueType[] types,
                List<MetalMpsGraphProgram.ValueDescriptor> values) {
            ValueType leftType = types[left];
            ValueType rightType = types[right];
            ValueType outputType = types[output];
            boolean leftGrad = values.get(left).requiresGrad();
            boolean rightGrad = values.get(right).requiresGrad();
            boolean outputGrad = values.get(output).requiresGrad();
            boolean integral = (leftType == ValueType.INT32 || leftType == ValueType.INT64)
                    && (rightType == ValueType.INT32 || rightType == ValueType.INT64);
            if (integral) {
                ValueType promoted =
                        leftType == ValueType.INT64 || rightType == ValueType.INT64
                                ? ValueType.INT64 : ValueType.INT32;
                if (outputType == promoted && !leftGrad && !rightGrad && !outputGrad) return;
            } else if (numericFloating(leftType) && numericFloating(rightType)
                    && outputType == task0066Promote(leftType, rightType)
                    && outputGrad == (leftGrad || rightGrad)) {
                return;
            }
            throw new IllegalArgumentException(
                    "MATMUL type and gradient metadata are incompatible");
        }

        private static boolean matmulMatches(
                int left,
                int right,
                int output,
                int[] ranks,
                long[] dimensions) {
            int leftRank = ranks[left];
            int rightRank = ranks[right];
            int outputRank = ranks[output];
            if (leftRank < 1 || rightRank < 1) return false;
            int leftBatchRank = Math.max(0, leftRank - 2);
            int rightBatchRank = Math.max(0, rightRank - 2);
            int batchRank = Math.max(leftBatchRank, rightBatchRank);
            int expectedOutputRank = batchRank
                    + (leftRank > 1 ? 1 : 0)
                    + (rightRank > 1 ? 1 : 0);
            if (outputRank != expectedOutputRank) return false;
            int leftRow = left * MAX_RANK;
            int rightRow = right * MAX_RANK;
            int outputRow = output * MAX_RANK;
            long leftContract = dimensions[leftRow + leftRank - 1];
            long rightContract =
                    dimensions[rightRow + (rightRank == 1 ? 0 : rightRank - 2)];
            if (leftContract != rightContract) return false;
            for (int axis = 0; axis < batchRank; axis++) {
                int leftAxis = axis - (batchRank - leftBatchRank);
                int rightAxis = axis - (batchRank - rightBatchRank);
                long leftExtent =
                        leftAxis < 0 ? 1L : dimensions[leftRow + leftAxis];
                long rightExtent =
                        rightAxis < 0 ? 1L : dimensions[rightRow + rightAxis];
                if (leftExtent != rightExtent && leftExtent != 1L && rightExtent != 1L) {
                    return false;
                }
                if (dimensions[outputRow + axis] != (leftExtent == 1L ? rightExtent : leftExtent)) {
                    return false;
                }
            }
            int outputAxis = batchRank;
            if (leftRank > 1
                    && dimensions[outputRow + outputAxis++]
                            != dimensions[leftRow + leftRank - 2]) {
                return false;
            }
            return rightRank == 1
                    || dimensions[outputRow + outputAxis]
                            == dimensions[rightRow + rightRank - 1];
        }

        private static boolean gatherMatches(
                MetalMpsGraphProgram.Node node,
                int data,
                int indices,
                int output,
                int[] ranks,
                long[] dimensions) {
            int dataRank = ranks[data];
            int indexRank = ranks[indices];
            int axis = node.axis();
            int outputRank = ranks[output];
            if (axis < 0 || axis >= dataRank
                    || outputRank != dataRank - 1 + indexRank) {
                return false;
            }
            int dataRow = data * MAX_RANK;
            int indexRow = indices * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int source = 0; source < axis; source++) {
                if (dimensions[outputRow + source] != dimensions[dataRow + source]) return false;
            }
            for (int source = 0; source < indexRank; source++) {
                if (dimensions[outputRow + axis + source]
                        != dimensions[indexRow + source]) return false;
            }
            for (int source = axis + 1; source < dataRank; source++) {
                if (dimensions[outputRow + indexRank + source - 1]
                        != dimensions[dataRow + source]) return false;
            }
            return true;
        }

        private static boolean scatterElementsMatches(
                MetalMpsGraphProgram.Node node,
                int data,
                int indices,
                int updates,
                int output,
                int[] ranks,
                long[] dimensions) {
            int rank = ranks[data];
            int axis = node.axis();
            if (rank < 1
                    || ranks[indices] != rank
                    || ranks[updates] != rank
                    || ranks[output] != rank
                    || axis < 0
                    || axis >= rank) {
                return false;
            }
            int dataRow = data * MAX_RANK;
            int indexRow = indices * MAX_RANK;
            int updateRow = updates * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int dimension = 0; dimension < rank; dimension++) {
                if (dimensions[indexRow + dimension] != dimensions[updateRow + dimension]
                        || dimensions[outputRow + dimension]
                                != dimensions[dataRow + dimension]
                        || (dimension != axis
                                && dimensions[indexRow + dimension]
                                        != dimensions[dataRow + dimension])) {
                    return false;
                }
            }
            return true;
        }

        private static boolean unfoldAxisMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int rank = ranks[input];
            int axis = node.axis();
            long[] attributes = node.attributeValues();
            if (rank < 1
                    || rank > 15
                    || ranks[output] != rank + 1
                    || axis < 0
                    || axis >= rank
                    || node.attributeCount() != 3
                    || node.auxiliary() != 0
                    || attributes.length != 2) {
                return false;
            }
            long size = attributes[0];
            long step = attributes[1];
            if (size < 1L || step <= 0L) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            long selected = dimensions[inputRow + axis];
            if (size > selected) {
                return false;
            }
            long positions = Math.addExact(Math.subtractExact(selected, size) / step, 1L);
            long lastEnd =
                    Math.addExact(
                            Math.addExact(size - 1L, Math.multiplyExact(positions - 1L, step)), 1L);
            if (lastEnd > selected) {
                return false;
            }
            for (int dimension = 0; dimension < rank; dimension++) {
                long expected = dimension == axis ? positions : dimensions[inputRow + dimension];
                if (dimensions[outputRow + dimension] != expected) {
                    return false;
                }
            }
            return dimensions[outputRow + rank] == size;
        }

        private static boolean oneHotMatches(
                MetalMpsGraphProgram.Node node,
                int indices,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[indices];
            if (ranks[output] != inputRank + 1) return false;
            long[] attributes = node.attributeValues();
            if (attributes.length != 1 || attributes[0] <= 0L) return false;
            int inputRow = indices * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int axis = 0; axis < inputRank; axis++) {
                if (dimensions[inputRow + axis] != dimensions[outputRow + axis]) return false;
            }
            return dimensions[outputRow + inputRank] == attributes[0];
        }

        private static boolean task0069ScatterAddMatches(
                MetalMpsGraphProgram.Node node,
                MetalMpsGraphProgram.ValueDescriptor data,
                MetalMpsGraphProgram.ValueDescriptor indices,
                MetalMpsGraphProgram.ValueDescriptor updates,
                MetalMpsGraphProgram.ValueDescriptor output) {
            if (node.axis() != 0
                    || node.attributeCount() != 1
                    || data.rank() != 1
                    || indices.rank() != 1
                    || updates.rank() != 1
                    || output.rank() != 1
                    || data.requiresGrad()
                    || indices.requiresGrad()
                    || updates.requiresGrad()
                    || output.requiresGrad()
                    || !canonicalTask0069(data)
                    || !canonicalTask0069Index(indices)
                    || !canonicalTask0069(updates)
                    || !canonicalTask0069(output)) {
                return false;
            }
            long dataExtent = data.dimensions()[0];
            long updateExtent = updates.dimensions()[0];
            int valueWidth = data.dataType().byteWidth();
            if (dataExtent != output.dimensions()[0]
                    || updateExtent != indices.dimensions()[0]
                    || dataExtent < 1L
                    || updateExtent < 1L
                    || dataExtent > UINT32_MAX / valueWidth
                    || updateExtent > UINT32_MAX / valueWidth
                    || updateExtent > UINT32_MAX / indices.dataType().byteWidth()) {
                return false;
            }
            return indices.layout().orElseThrow().referencedElementSpan()
                    <= UINT32_MAX / indices.dataType().byteWidth();
        }

        /**
         * Checks the explicit canonical storage descriptor required by direct FLOAT32 EXP and
         * SIGMOID and custom low EXP and SIGMOID images; none accepts an affine view or absent
         * layout.
         *
         * @param value non-null encoded value descriptor
         * @return whether its physical layout is the non-view canonical contiguous layout
         */
        private static boolean canonicalExpValue(MetalMpsGraphProgram.ValueDescriptor value) {
            return !value.densePhysical()
                    && value.layout().filter(layout -> layout.equals(
                            LayoutDescriptor.contiguous(Shape.of(value.dimensions()))))
                            .isPresent();
        }

        private static boolean canonicalTask0069(
                MetalMpsGraphProgram.ValueDescriptor descriptor) {
            DataType type = descriptor.dataType();
            return (type == DataType.FLOAT32
                            || type == DataType.BFLOAT16
                            || type == DataType.FLOAT16)
                    && descriptor.layout().isPresent()
                    && descriptor.layout().orElseThrow().equals(
                            LayoutDescriptor.contiguous(
                                    Shape.of(descriptor.dimensions())));
        }

        private static boolean canonicalTask0069Index(
                MetalMpsGraphProgram.ValueDescriptor descriptor) {
            return (descriptor.dataType() == DataType.INT32
                            || descriptor.dataType() == DataType.INT64)
                    && descriptor.layout().isPresent()
                    && descriptor.layout().orElseThrow().equals(
                            LayoutDescriptor.contiguous(
                                    Shape.of(descriptor.dimensions())));
        }

        private static boolean task0069L1Matches(
                MetalMpsGraphProgram.Node node,
                MetalMpsGraphProgram.ValueDescriptor inputDescriptor,
                MetalMpsGraphProgram.ValueDescriptor outputDescriptor,
                int input,
                int output,
                int[] valueRanks,
                long[] valueDimensions) {
            long[] words = node.attributeValues();
            long extent = valueRanks[input] == 1
                    ? valueDimensions[input * MAX_RANK] : 0L;
            return node.axis()
                            == MetalMpsGraphProgram.ReductionForm.MULTI_AXIS.wireIdentity()
                    && node.attributeCount() == 1
                    && words.length == 1
                    && words[0] == 0L
                    && valueRanks[input] == 1
                    && extent >= 1L
                    && extent <= UINT32_MAX / inputDescriptor.dataType().byteWidth()
                    && valueRanks[output] == (node.auxiliary() == 1 ? 1 : 0)
                    && (node.auxiliary() != 1
                            || valueDimensions[output * MAX_RANK] == 1L)
                    && inputDescriptor.layout().isPresent()
                    && inputDescriptor.layout().orElseThrow().equals(
                            LayoutDescriptor.contiguous(
                                    Shape.of(inputDescriptor.dimensions())))
                    && outputDescriptor.layout().isPresent()
                    && outputDescriptor.layout().orElseThrow().equals(
                            LayoutDescriptor.contiguous(
                                    Shape.of(outputDescriptor.dimensions())))
                    && !inputDescriptor.requiresGrad()
                    && !outputDescriptor.requiresGrad()
                    && reductionMatches(
                            node, input, output, valueRanks, valueDimensions);
        }

        private static boolean reductionMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int outputRank = ranks[output];
            long[] values = node.attributeValues();
            MetalMpsGraphProgram.ReductionForm form;
            try {
                form = MetalMpsGraphProgram.ReductionForm.fromWireIdentity(node.axis());
            } catch (IllegalArgumentException malformed) {
                return false;
            }
            if (form == MetalMpsGraphProgram.ReductionForm.SUM_TO_SHAPE) {
                if (node.kind() != MetalMpsGraphProgram.NodeKind.SUM
                        || node.auxiliary() != 0
                        || node.attributeCount() != outputRank
                        || values.length != outputRank
                        || outputRank > inputRank) {
                    return false;
                }
                int inputRow = input * MAX_RANK;
                int outputRow = output * MAX_RANK;
                int padding = inputRank - outputRank;
                long count = 1L;
                try {
                    for (int axis = 0; axis < inputRank; axis++) {
                        long source = dimensions[inputRow + axis];
                        if (axis < padding) {
                            count = Math.multiplyExact(count, source);
                        } else {
                            long target = values[axis - padding];
                            if (target != dimensions[outputRow + axis - padding]
                                    || (target != 1L && target != source)) {
                                return false;
                            }
                            if (target == 1L && source != 1L) {
                                count = Math.multiplyExact(count, source);
                            }
                        }
                    }
                } catch (ArithmeticException overflow) {
                    return false;
                }
                return count > 0L;
            }

            if (form == MetalMpsGraphProgram.ReductionForm.FULL) {
                if (node.attributeCount() != 0 || values.length != 0
                        || node.auxiliary() != 0 || outputRank != 0) {
                    return false;
                }
                try {
                    return elementCount(input, ranks, dimensions) > 0L;
                } catch (ArithmeticException overflow) {
                    return false;
                }
            }

            if (form == MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS
                    && node.attributeCount() != 1) {
                return false;
            }
            if (form != MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS
                    && form != MetalMpsGraphProgram.ReductionForm.MULTI_AXIS) {
                return false;
            }
            boolean keep = node.auxiliary() == 1;
            if (node.auxiliary() < 0 || node.auxiliary() > 1
                    || node.attributeCount() != values.length) {
                return false;
            }
            boolean[] reduced = new boolean[inputRank];
            long count = 1L;
            int inputRow = input * MAX_RANK;
            try {
                for (long value : values) {
                    if (value < 0L || value >= inputRank || reduced[(int) value]) {
                        return false;
                    }
                    reduced[(int) value] = true;
                    count = Math.multiplyExact(count, dimensions[inputRow + (int) value]);
                }
            } catch (ArithmeticException overflow) {
                return false;
            }
            int expectedRank = keep ? inputRank : inputRank - values.length;
            if (outputRank != expectedRank || count <= 0L) return false;
            int outputRow = output * MAX_RANK;
            for (int source = 0, target = 0; source < inputRank; source++) {
                if (reduced[source]) {
                    if (keep && dimensions[outputRow + target++] != 1L) return false;
                } else if (dimensions[outputRow + target++]
                        != dimensions[inputRow + source]) {
                    return false;
                }
            }
            return true;
        }
        private static boolean statisticalReductionMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            long[] words = node.attributeWords();
            if (words.length < 3L) return false;
            int count;
            try {
                count = Math.toIntExact(words[0]);
            } catch (ArithmeticException malformed) {
                return false;
            }
            if (count < 0 || count > ranks[input] || words.length != count + 3) return false;
            boolean keep = words[count + 1] == 1L;
            if (!keep && words[count + 1] != 0L || words[count + 2] < 0L) return false;
            boolean[] reduced = new boolean[ranks[input]];
            long selected = 1L;
            int inputRow = input * MAX_RANK;
            try {
                for (int index = 0; index < count; index++) {
                    int axis = Math.toIntExact(words[index + 1]);
                    if (axis < 0 || axis >= ranks[input] || reduced[axis]) return false;
                    reduced[axis] = true;
                    selected = Math.multiplyExact(selected, dimensions[inputRow + axis]);
                }
            } catch (ArithmeticException malformed) {
                return false;
            }
            if (selected <= words[count + 2]) return false;
            int expectedRank = keep ? ranks[input] : ranks[input] - count;
            if (ranks[output] != expectedRank) return false;
            int outputRow = output * MAX_RANK;
            for (int source = 0, target = 0; source < ranks[input]; source++) {
                if (reduced[source]) {
                    if (keep && dimensions[outputRow + target++] != 1L) return false;
                } else if (dimensions[outputRow + target++]
                        != dimensions[inputRow + source]) {
                    return false;
                }
            }
            return true;
        }


        private static boolean sameShape(
                int first, int second, int[] ranks, long[] dimensions) {
            int rank = ranks[first];
            if (rank != ranks[second]) return false;
            int firstRow = first * MAX_RANK;
            int secondRow = second * MAX_RANK;
            for (int axis = 0; axis < rank; axis++) {
                if (dimensions[firstRow + axis] != dimensions[secondRow + axis]) return false;
            }
            return true;
        }

        private static void requireShape(boolean condition, String message) {
            if (!condition) {
                throw new IllegalArgumentException("Metal MPSGraph " + message);
            }
        }

        private static boolean sameElementCount(
                int first, int second, int[] ranks, long[] dimensions) {
            return elementCount(first, ranks, dimensions)
                    == elementCount(second, ranks, dimensions);
        }

        private static long elementCount(int value, int[] ranks, long[] dimensions) {
            long elements = 1L;
            int row = value * MAX_RANK;
            for (int axis = 0; axis < ranks[value]; axis++) if (dimensions[row + axis] == 0) return 0;
            for (int axis = 0; axis < ranks[value]; axis++) {
                elements = Math.multiplyExact(elements, dimensions[row + axis]);
            }
            return elements;
        }

        private static boolean targetMatches(
                MetalMpsGraphProgram.Node node,
                int output,
                int[] ranks,
                long[] dimensions) {
            if (node.attributeCount() != ranks[output]) {
                return false;
            }
            long[] target = node.attributeValues();
            int row = output * MAX_RANK;
            for (int axis = 0; axis < target.length; axis++) {
                if (target[axis] != dimensions[row + axis]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean expandsTo(
                int input, int output, int[] ranks, long[] dimensions) {
            int inputRank = ranks[input];
            int outputRank = ranks[output];
            if (inputRank > outputRank) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            int padding = outputRank - inputRank;
            for (int axis = 0; axis < inputRank; axis++) {
                long source = dimensions[inputRow + axis];
                long target = dimensions[outputRow + axis + padding];
                if (source != target && source != 1L) {
                    return false;
                }
            }
            return true;
        }

        private static boolean permutationMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int rank = ranks[input];
            if (ranks[output] != rank || node.attributeCount() != rank) {
                return false;
            }
            long[] permutation = node.attributeValues();
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int axis = 0; axis < rank; axis++) {
                int source = Math.toIntExact(permutation[axis]);
                if (dimensions[outputRow + axis] != dimensions[inputRow + source]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean expandDimsMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int axis = node.axis();
            if (ranks[output] != inputRank + 1 || axis > inputRank) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int outputAxis = 0; outputAxis < ranks[output]; outputAxis++) {
                long expected = outputAxis == axis
                        ? 1L
                        : dimensions[inputRow + (outputAxis < axis
                                ? outputAxis : outputAxis - 1)];
                if (dimensions[outputRow + outputAxis] != expected) {
                    return false;
                }
            }
            return true;
        }

        private static boolean squeezeMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int axis = node.axis();
            if (ranks[output] != inputRank - 1
                    || axis >= inputRank
                    || dimensions[input * MAX_RANK + axis] != 1L) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int source = 0, target = 0; source < inputRank; source++) {
                if (source != axis
                        && dimensions[inputRow + source] != dimensions[outputRow + target++]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean broadcastsTo(
                int left,
                int right,
                int output,
                int[] ranks,
                long[] dimensions) {
            int leftRank = ranks[left];
            int rightRank = ranks[right];
            int outputRank = ranks[output];
            if (outputRank != Math.max(leftRank, rightRank)) {
                return false;
            }
            int leftRow = left * MAX_RANK;
            int rightRow = right * MAX_RANK;
            int outputRow = output * MAX_RANK;
            int leftPadding = outputRank - leftRank;
            int rightPadding = outputRank - rightRank;
            for (int axis = 0; axis < outputRank; axis++) {
                long leftDimension = axis < leftPadding
                        ? 1L : dimensions[leftRow + axis - leftPadding];
                long rightDimension = axis < rightPadding
                        ? 1L : dimensions[rightRow + axis - rightPadding];
                if (leftDimension != rightDimension
                        && leftDimension != 1L && rightDimension != 1L) {
                    return false;
                }
                if (dimensions[outputRow + axis]
                        != (leftDimension == 1L ? rightDimension : leftDimension)) {
                    return false;
                }
            }
            return true;
        }

        private static void requireType(ValueType[] types, int value, ValueType expected) {
            ValueType existing = types[value];
            if (existing != null && existing != expected) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph value has incompatible typed uses");
            }
            types[value] = expected;
        }

        private enum ValueType {
            FLOAT32(Float.BYTES),
            INT32(Integer.BYTES),
            BOOL(Byte.BYTES),
            FLOAT64(Double.BYTES),
            BFLOAT16(Short.BYTES),
            INT64(Long.BYTES),
            FLOAT16(Short.BYTES);

            private final int byteWidth;

            ValueType(int byteWidth) {
                this.byteWidth = byteWidth;
            }

            static ValueType from(DataType dataType) {
                return switch (dataType) {
                    case FLOAT32 -> FLOAT32;
                    case INT32 -> INT32;
                    case BOOL -> BOOL;
                    case FLOAT64 -> FLOAT64;
                    case BFLOAT16 -> BFLOAT16;
                    case INT64 -> INT64;
                    case FLOAT16 -> FLOAT16;
                    case FLOAT8_E4M3, FLOAT8_E5M2 ->
                            throw new IllegalArgumentException("Metal does not support FP8 carriers");
                };
            }
        }

    }

    /** Production JDK Foreign Function and Memory binding of the exact thirteen-symbol ABI. */
    private static final class Ffm extends MetalNativeApi {
        private static final String VERSION = "synaptik_metal_foundation_abi_version";
        private static final String CONTEXT_CREATE = "synaptik_metal_context_create";
        private static final String CONTEXT_RELEASE = "synaptik_metal_context_release";
        private static final String BUFFER_CREATE = "synaptik_metal_buffer_create";
        private static final String BUFFER_RELEASE = "synaptik_metal_buffer_release";
        private static final String BUFFER_UPLOAD = "synaptik_metal_buffer_upload";
        private static final String BUFFER_DOWNLOAD = "synaptik_metal_buffer_download";
        private static final String EXECUTABLE_CREATE =
                EXECUTABLE_CREATE_OPERATION;
        private static final String EXECUTABLE_RELEASE =
                EXECUTABLE_RELEASE_OPERATION;
        private static final String EXECUTABLE_RUN =
                EXECUTABLE_RUN_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_CREATE =
                NEG_KERNEL_PIPELINE_CREATE_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_RELEASE =
                NEG_KERNEL_PIPELINE_RELEASE_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_RUN =
                NEG_KERNEL_PIPELINE_RUN_OPERATION;

        private final Arena lookupArena;
        private final MethodHandle contextCreate;
        private final MethodHandle contextRelease;
        private final MethodHandle bufferCreate;
        private final MethodHandle bufferRelease;
        private final MethodHandle bufferUpload;
        private final MethodHandle bufferDownload;
        private final MethodHandle executableCreate;
        private final MethodHandle executableRelease;
        private final MethodHandle executableRun;
        private final MethodHandle negKernelPipelineCreate;
        private final MethodHandle negKernelPipelineRelease;
        private final MethodHandle negKernelPipelineRun;
        private final AtomicBoolean closed = new AtomicBoolean();

        private Ffm(
                Arena lookupArena,
                MethodHandle contextCreate,
                MethodHandle contextRelease,
                MethodHandle bufferCreate,
                MethodHandle bufferRelease,
                MethodHandle bufferUpload,
                MethodHandle bufferDownload,
                MethodHandle executableCreate,
                MethodHandle executableRelease,
                MethodHandle executableRun,
                MethodHandle negKernelPipelineCreate,
                MethodHandle negKernelPipelineRelease,
                MethodHandle negKernelPipelineRun) {
            this.lookupArena = lookupArena;
            this.contextCreate = contextCreate;
            this.contextRelease = contextRelease;
            this.bufferCreate = bufferCreate;
            this.bufferRelease = bufferRelease;
            this.bufferUpload = bufferUpload;
            this.bufferDownload = bufferDownload;
            this.executableCreate = executableCreate;
            this.executableRelease = executableRelease;
            this.executableRun = executableRun;
            this.negKernelPipelineCreate = negKernelPipelineCreate;
            this.negKernelPipelineRelease = negKernelPipelineRelease;
            this.negKernelPipelineRun = negKernelPipelineRun;
        }

        static Ffm open(Path path) {
            Arena arena = Arena.ofShared();
            try {
                SymbolLookup lookup = SymbolLookup.libraryLookup(path, arena);
                Linker linker = Linker.nativeLinker();

                MethodHandle version = linker.downcallHandle(
                        require(lookup, VERSION), FunctionDescriptor.of(JAVA_INT));
                int actualVersion = invokeInt(version, VERSION);
                if (Integer.toUnsignedLong(actualVersion) != ABI_VERSION) {
                    throw new IllegalStateException(
                            "Metal foundation ABI version must be " + ABI_VERSION
                                    + " but was " + Integer.toUnsignedLong(actualVersion));
                }

                MethodHandle contextCreate = linker.downcallHandle(
                        require(lookup, CONTEXT_CREATE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle contextRelease = linker.downcallHandle(
                        require(lookup, CONTEXT_RELEASE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle bufferCreate = linker.downcallHandle(
                        require(lookup, BUFFER_CREATE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
                MethodHandle bufferRelease = linker.downcallHandle(
                        require(lookup, BUFFER_RELEASE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle bufferUpload = linker.downcallHandle(
                        require(lookup, BUFFER_UPLOAD),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
                MethodHandle bufferDownload = linker.downcallHandle(
                        require(lookup, BUFFER_DOWNLOAD),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
                MethodHandle executableCreate = linker.downcallHandle(
                        require(lookup, EXECUTABLE_CREATE),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
                MethodHandle executableRelease = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RELEASE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle executableRun = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RUN),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
                MethodHandle negKernelPipelineCreate = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_CREATE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
                MethodHandle negKernelPipelineRelease = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_RELEASE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle negKernelPipelineRun = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_RUN),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
                return new Ffm(arena, contextCreate,
                        contextRelease, bufferCreate, bufferRelease, bufferUpload, bufferDownload,
                        executableCreate, executableRelease, executableRun,
                        negKernelPipelineCreate, negKernelPipelineRelease, negKernelPipelineRun);
            } catch (RuntimeException | Error failure) {
                closeAfterFailure(arena, failure);
                throw failure;
            }
        }

        @Override
        Handle createContext() {
            return create(CONTEXT_CREATE, contextCreate, null, 0L, contextRelease);
        }



        @Override
        void releaseContext(Handle context) {
            requireOpen();
            checkStatus(CONTEXT_RELEASE,
                    invokeIntAddress(contextRelease, CONTEXT_RELEASE, context.carrier()));
        }

        @Override
        Handle createBuffer(Handle context, long logicalByteSize) {
            requireOpen();
            if (logicalByteSize < 0L) {
                throw new IllegalArgumentException("logicalByteSize must be non-negative");
            }
            return create(BUFFER_CREATE, bufferCreate, context, logicalByteSize, bufferRelease);
        }

        @Override
        void releaseBuffer(Handle buffer) {
            requireOpen();
            checkStatus(BUFFER_RELEASE,
                    invokeIntAddress(bufferRelease, BUFFER_RELEASE, buffer.carrier()));
        }

        @Override
        void upload(Handle buffer, long bufferOffset, MemorySegment source, long byteCount) {
            requireOpen();
            int status = invokeIntCopy(bufferUpload, BUFFER_UPLOAD, buffer.carrier(), bufferOffset,
                    source, byteCount);
            checkStatus(BUFFER_UPLOAD, status);
        }

        @Override
        void download(
                Handle buffer, long bufferOffset, MemorySegment destination, long byteCount) {
            requireOpen();
            int status = invokeIntCopy(bufferDownload, BUFFER_DOWNLOAD, buffer.carrier(),
                    bufferOffset, destination, byteCount);
            checkStatus(BUFFER_DOWNLOAD, status);
        }

        @Override
        NativeCreateResult createProgramExecutableNative(
                Handle context, MemorySegment programImage) {
            requireOpen();
            Objects.requireNonNull(programImage, "programImage");
            if (programImage.byteSize() <= 0L
                    || programImage.byteSize() > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Metal program image byte size must fit positive signed int");
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = invokeCreateExecutable(
                        executableCreate,
                        context.carrier(),
                        programImage,
                        Math.toIntExact(programImage.byteSize()),
                        output);
                MemorySegment carrier = output.get(ADDRESS, 0L);
                return new NativeCreateResult(status,
                        carrier.address() == 0L ? null : new Handle(carrier));
            }
        }

        @Override
        int releaseExecutableNative(Handle executable) {
            requireOpen();
            return invokeIntAddress(
                    executableRelease, EXECUTABLE_RELEASE, executable.carrier());
        }

        @Override
        int runExecutableNative(Handle executable, int inputCount, MemorySegment inputBuffers,
                int outputCount, MemorySegment outputBuffers) {
            requireOpen();
            return invokeRunExecutable(executableRun, executable.carrier(),
                    inputCount, inputBuffers, outputCount, outputBuffers);
        }

        @Override
        NativeCreateResult createNegKernelPipelineNative(Handle context, long elementCount) {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = invokeIntCreateBuffer(
                        negKernelPipelineCreate, NEG_KERNEL_PIPELINE_CREATE,
                        context.carrier(), elementCount, output);
                MemorySegment carrier = output.get(ADDRESS, 0L);
                return new NativeCreateResult(
                        status, carrier.address() == 0L ? null : new Handle(carrier));
            }
        }

        @Override
        int releaseNegKernelPipelineNative(Handle pipeline) {
            requireOpen();
            return invokeIntAddress(negKernelPipelineRelease,
                    NEG_KERNEL_PIPELINE_RELEASE, pipeline.carrier());
        }

        @Override
        int runNegKernelPipelineNative(
                Handle pipeline, Handle inputBuffer, Handle outputBuffer) {
            requireOpen();
            return invokeIntThreeAddresses(negKernelPipelineRun, NEG_KERNEL_PIPELINE_RUN,
                    pipeline.carrier(), inputBuffer.carrier(), outputBuffer.carrier());
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                lookupArena.close();
            }
        }

        private Handle create(
                String operation,
                MethodHandle create,
                Handle context,
                long logicalByteSize,
                MethodHandle malformedHandleRelease) {
            requireOpen();
            try (Arena outputArena = Arena.ofConfined()) {
                MemorySegment output = outputArena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = context == null
                        ? invokeIntAddress(create, operation, output)
                        : invokeIntCreateBuffer(
                                create, operation, context.carrier(), logicalByteSize, output);
                return finishCreate(operation, status, output.get(ADDRESS, 0L),
                        malformedHandleRelease);
            }
        }

        private static Handle finishCreate(String operation, int status, MemorySegment carrier,
                MethodHandle malformedHandleRelease) {
            if (status == 0 && carrier.address() != 0L) return new Handle(carrier);
            RuntimeException failure = status == 0
                    ? new IllegalStateException(operation + " returned OK with a null handle")
                    : new NativeFailure(operation, status);
            if (carrier.address() != 0L) {
                try {
                    checkStatus(operation + " malformed-handle cleanup",
                            invokeIntAddress(malformedHandleRelease,
                                    operation + " malformed-handle cleanup", carrier));
                } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) failure.addSuppressed(cleanup);
                }
                if (status != 0) {
                    IllegalStateException contractFailure = new IllegalStateException(
                            operation + " returned failure with a non-null handle", failure);
                    for (Throwable suppressed : failure.getSuppressed()) {
                        contractFailure.addSuppressed(suppressed);
                    }
                    throw contractFailure;
                }
            }
            throw failure;
        }

        private static MemorySegment copyInts(Arena arena, int[] values) {
            MemorySegment result = arena.allocate(JAVA_INT, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_INT, index, values[index]);
            }
            return result;
        }

        private static MemorySegment copyLongs(Arena arena, long[] values) {
            MemorySegment result = arena.allocate(JAVA_LONG, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_LONG, index, values[index]);
            }
            return result;
        }

        private void requireOpen() {
            if (closed.get()) {
                throw new IllegalStateException("Metal native API is closed");
            }
        }

        private static MemorySegment require(SymbolLookup lookup, String symbol) {
            return lookup.find(symbol)
                    .orElseThrow(() -> new IllegalStateException(
                            "Missing required Metal foundation symbol: " + symbol));
        }

        private static void checkStatus(String operation, int status) {
            if (status != 0) {
                throw new NativeFailure(operation, status);
            }
        }

        private static int invokeInt(MethodHandle handle, String operation) {
            try {
                return (int) handle.invokeExact();
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntAddress(
                MethodHandle handle, String operation, MemorySegment value) {
            try {
                return (int) handle.invokeExact(value);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntCreateBuffer(
                MethodHandle handle,
                String operation,
                MemorySegment context,
                long logicalByteSize,
                MemorySegment output) {
            try {
                return (int) handle.invokeExact(context, logicalByteSize, output);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntCopy(
                MethodHandle handle,
                String operation,
                MemorySegment buffer,
                long bufferOffset,
                MemorySegment bytes,
                long byteCount) {
            try {
                return (int) handle.invokeExact(buffer, bufferOffset, bytes, byteCount);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntThreeAddresses(
                MethodHandle handle,
                String operation,
                MemorySegment first,
                MemorySegment second,
                MemorySegment third) {
            try {
                return (int) handle.invokeExact(first, second, third);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }


        private static int invokeCreateExecutable(
                MethodHandle handle,
                MemorySegment context,
                MemorySegment program,
                int programBytes,
                MemorySegment output) {
            try {
                return (int) handle.invokeExact(context, program, programBytes, output);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(EXECUTABLE_CREATE + " invocation failed", failure);
            }
        }

        private static int invokeRunExecutable(MethodHandle handle, MemorySegment executable,
                int inputCount, MemorySegment inputs, int outputCount, MemorySegment outputs) {
            try {
                return (int) handle.invokeExact(
                        executable, inputCount, inputs, outputCount, outputs);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(EXECUTABLE_RUN + " invocation failed", failure);
            }
        }

        private static void closeAfterFailure(Arena arena, Throwable primary) {
            try {
                arena.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != primary) {
                    primary.addSuppressed(cleanup);
                }
            }
        }
    }
}
