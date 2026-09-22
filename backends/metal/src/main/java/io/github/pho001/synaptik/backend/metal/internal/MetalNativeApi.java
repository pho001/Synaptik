package io.github.pho001.synaptik.backend.metal.internal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Package-private typed seam for the version-two Metal foundation and MPSGraph C ABI.
 *
 * <p>Handles remain opaque carrier segments inside this package. Implementations consume each
 * successful context or buffer handle exactly once through its matching release call. Native
 * status failures are unchecked and retain both the operation name and raw status value.</p>
 */
abstract class MetalNativeApi implements AutoCloseable {
    static final int ABI_VERSION = 2;
    static final String NEG_EXECUTABLE_CREATE_OPERATION =
            "synaptik_metal_mpsgraph_neg_executable_create";
    static final String EXECUTABLE_RELEASE_OPERATION =
            "synaptik_metal_mpsgraph_executable_release";
    static final String EXECUTABLE_RUN_OPERATION =
            "synaptik_metal_mpsgraph_executable_run";

    /**
     * Opens and completely validates the production ABI without creating a Metal context.
     *
     * @param absoluteLibraryPath caller-selected absolute library path; not retained as mutable
     *     state and must not be {@code null}
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
     * @param destination non-null writable native destination slice, or the null-address segment
     *     for a zero-byte copy
     * @param byteCount non-negative number of bytes to copy
     * @throws RuntimeException if the native copy fails
     */
    abstract void download(
            Handle buffer, long bufferOffset, MemorySegment destination, long byteCount);

    /**
     * Compiles one shape-specialized whole-partition unary-NEG executable.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param valueRanks non-null value-aligned ranks
     * @param valueDimensions non-null row-major value-count by sixteen dimension table
     * @param nodeInputValueIndices non-null ordered unary-node input indices
     * @param nodeOutputValueIndices non-null ordered unary-node output indices
     * @param feedValueIndices non-null stable feed value indices
     * @param targetValueIndices non-null stable target value indices
     * @return a fresh non-null opaque executable handle owned by the caller
     * @throws RuntimeException if native construction, validation, or compilation fails
     */
    final Handle createNegExecutable(
            Handle context,
            int[] valueRanks,
            long[] valueDimensions,
            int[] nodeInputValueIndices,
            int[] nodeOutputValueIndices,
            int[] feedValueIndices,
            int[] targetValueIndices) {
        Objects.requireNonNull(context, "context");
        NegExecutableAbi.validateCreate(
                valueRanks, valueDimensions, nodeInputValueIndices,
                nodeOutputValueIndices, feedValueIndices, targetValueIndices);
        NativeCreateResult result = Objects.requireNonNull(createNegExecutableNative(
                context, valueRanks, valueDimensions, nodeInputValueIndices,
                nodeOutputValueIndices, feedValueIndices, targetValueIndices),
                "native executable create result");
        return finishExecutableCreate(result);
    }

    /**
     * Performs the already validated native create invocation.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param valueRanks validated value-aligned ranks
     * @param valueDimensions validated padded dimension table
     * @param nodeInputValueIndices validated topological node inputs
     * @param nodeOutputValueIndices validated unique node outputs
     * @param feedValueIndices validated unique feeds
     * @param targetValueIndices validated unique produced targets
     * @return non-null raw status/output-cell result for checked interpretation
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract NativeCreateResult createNegExecutableNative(
            Handle context,
            int[] valueRanks,
            long[] valueDimensions,
            int[] nodeInputValueIndices,
            int[] nodeOutputValueIndices,
            int[] feedValueIndices,
            int[] targetValueIndices);

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
     * @param inputBuffers non-null stable feed-ordered live buffer handles
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
     * @param inputCount positive validated feed count
     * @param inputBuffers exact readable feed-address segment
     * @param outputCount positive validated target count
     * @param outputBuffers exact readable target-address segment
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int runExecutableNative(
            Handle executable, int inputCount, MemorySegment inputBuffers,
            int outputCount, MemorySegment outputBuffers);

    /**
     * Ends the production symbol-lookup lifetime after every native handle has been released.
     *
     * <p>Production close is thread-safe and idempotent. Test implementations must preserve the
     * same ownership boundary but may expose deterministic injected failures.</p>
     *
     * @throws RuntimeException if lookup cleanup fails
     * @throws Error if lookup cleanup reports an error
     */
    @Override
    public abstract void close();

    /** Opaque non-null FFM carrier for one context or buffer handle. */
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
        int status = result.statusCode();
        Handle handle = result.handle();
        if (status == 0 && handle != null) return handle;
        RuntimeException failure = status == 0
                ? new IllegalStateException(
                        NEG_EXECUTABLE_CREATE_OPERATION + " returned OK with a null handle")
                : new NativeFailure(NEG_EXECUTABLE_CREATE_OPERATION, status);
        if (handle != null) {
            try {
                checkExecutableStatus(
                        NEG_EXECUTABLE_CREATE_OPERATION + " malformed-handle cleanup",
                        releaseExecutableNative(handle));
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            if (status != 0) {
                IllegalStateException contractFailure = new IllegalStateException(
                        NEG_EXECUTABLE_CREATE_OPERATION
                                + " returned failure with a non-null handle",
                        failure);
                for (Throwable suppressed : failure.getSuppressed()) {
                    contractFailure.addSuppressed(suppressed);
                }
                throw contractFailure;
            }
        }
        throw failure;
    }

    private static void checkExecutableStatus(String operation, int status) {
        if (status != 0) throw new NativeFailure(operation, status);
    }

    /** Stable version-two non-success status meanings. */
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
        EXECUTION_FAILED(11);

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

        /** @return the stable non-null ABI operation name */
        String operation() {
            return operation;
        }

        /** @return the exact signed status integer returned by native code */
        int statusCode() {
            return statusCode;
        }

        /** @return the known status, or {@code null} when native code returned an unknown value */
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

    /** Exact Java preflight for the unary-NEG executable-create ABI. */
    static final class NegExecutableAbi {
        private static final int MAX_RANK = 16;

        private NegExecutableAbi() {}

        /**
         * Rejects every malformed create description before native storage or a downcall exists.
         *
         * @param valueRanks non-null rank table defining a positive value count
         * @param valueDimensions non-null value-count by sixteen zero-padded dimension table
         * @param nodeInputs non-null positive node-count input-index table
         * @param nodeOutputs non-null node-count output-index table
         * @param feeds non-null positive unique external-feed index table
         * @param targets non-null positive unique produced-target index table
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if cardinality, rank, padding, topology, indices,
         *     per-node shape equality, or checked positive FLOAT32 geometry is invalid
         */
        static void validateCreate(
                int[] valueRanks,
                long[] valueDimensions,
                int[] nodeInputs,
                int[] nodeOutputs,
                int[] feeds,
                int[] targets) {
            Objects.requireNonNull(valueRanks, "valueRanks");
            Objects.requireNonNull(valueDimensions, "valueDimensions");
            Objects.requireNonNull(nodeInputs, "nodeInputValueIndices");
            Objects.requireNonNull(nodeOutputs, "nodeOutputValueIndices");
            Objects.requireNonNull(feeds, "feedValueIndices");
            Objects.requireNonNull(targets, "targetValueIndices");
            int valueCount = valueRanks.length;
            if (valueCount == 0) {
                throw new IllegalArgumentException("Metal NEG value count must be positive");
            }
            int dimensionCount;
            try {
                dimensionCount = Math.multiplyExact(valueCount, MAX_RANK);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException("Metal NEG dimension-table cardinality overflows", overflow);
            }
            if (valueDimensions.length != dimensionCount) {
                throw new IllegalArgumentException(
                        "Metal NEG dimensions must contain exactly valueCount * 16 cells");
            }
            if (nodeInputs.length == 0 || nodeInputs.length != nodeOutputs.length) {
                throw new IllegalArgumentException(
                        "Metal NEG node input/output cardinalities must be equal and positive");
            }
            if (feeds.length == 0 || targets.length == 0) {
                throw new IllegalArgumentException(
                        "Metal NEG feed and target counts must be positive");
            }

            for (int value = 0; value < valueCount; value++) {
                int rank = valueRanks[value];
                if (rank < 1 || rank > MAX_RANK) {
                    throw new IllegalArgumentException(
                            "Metal NEG value rank must be in 1..16 at index " + value);
                }
                long elements = 1L;
                int row = value * MAX_RANK;
                try {
                    for (int axis = 0; axis < rank; axis++) {
                        long dimension = valueDimensions[row + axis];
                        if (dimension <= 0L) {
                            throw new IllegalArgumentException(
                                    "Metal NEG used dimensions must be positive");
                        }
                        elements = Math.multiplyExact(elements, dimension);
                    }
                    Math.multiplyExact(elements, Float.BYTES);
                } catch (ArithmeticException overflow) {
                    throw new IllegalArgumentException(
                            "Metal NEG FLOAT32 geometry overflows at value " + value, overflow);
                }
                for (int axis = rank; axis < MAX_RANK; axis++) {
                    if (valueDimensions[row + axis] != 0L) {
                        throw new IllegalArgumentException(
                                "Metal NEG unused dimension cells must be zero");
                    }
                }
            }

            byte[] available = new byte[valueCount];
            boolean[] used = new boolean[valueCount];
            for (int feed : feeds) {
                requireIndex(feed, valueCount, "feed");
                if (available[feed] != 0) {
                    throw new IllegalArgumentException("Metal NEG feed indices must be unique");
                }
                available[feed] = 1;
                used[feed] = true;
            }
            for (int node = 0; node < nodeInputs.length; node++) {
                int input = nodeInputs[node];
                int output = nodeOutputs[node];
                requireIndex(input, valueCount, "node input");
                requireIndex(output, valueCount, "node output");
                if (available[input] == 0) {
                    throw new IllegalArgumentException(
                            "Metal NEG node input must be a feed or earlier node output");
                }
                if (available[output] != 0) {
                    throw new IllegalArgumentException(
                            "Metal NEG node outputs must be unique and not feeds");
                }
                if (!sameShape(input, output, valueRanks, valueDimensions)) {
                    throw new IllegalArgumentException(
                            "Metal NEG node input/output shapes must match exactly");
                }
                available[output] = 2;
                used[input] = true;
                used[output] = true;
            }
            boolean[] targeted = new boolean[valueCount];
            for (int target : targets) {
                requireIndex(target, valueCount, "target");
                if (available[target] != 2) {
                    throw new IllegalArgumentException(
                            "Metal NEG target must be a node-produced value");
                }
                if (targeted[target]) {
                    throw new IllegalArgumentException("Metal NEG target indices must be unique");
                }
                targeted[target] = true;
            }
            for (int value = 0; value < valueCount; value++) {
                if (!used[value]) {
                    throw new IllegalArgumentException(
                            "Metal NEG value table contains an unused index: " + value);
                }
            }
        }

        /**
         * Validates exact run cardinality, address-workspace geometry, and cross-role aliasing.
         *
         * @param expectedInputs positive compiled feed count
         * @param inputCount supplied feed count
         * @param inputBuffers exact readable feed-address segment
         * @param expectedOutputs positive compiled target count
         * @param outputCount supplied target count
         * @param outputBuffers exact readable target-address segment
         * @throws NullPointerException if an address segment is {@code null}
         * @throws IllegalStateException if a segment is not alive or thread-accessible
         * @throws IllegalArgumentException if counts, segment geometry, a handle, or input/output
         *     aliasing violates the compiled invocation contract
         */
        static void validateRun(
                int expectedInputs,
                int inputCount,
                MemorySegment inputBuffers,
                int expectedOutputs,
                int outputCount,
                MemorySegment outputBuffers) {
            Objects.requireNonNull(inputBuffers, "inputBuffers");
            Objects.requireNonNull(outputBuffers, "outputBuffers");
            if (inputCount != expectedInputs || outputCount != expectedOutputs) {
                throw new IllegalArgumentException(
                        "Metal NEG run counts must match the compiled feeds and targets");
            }
            validateAddressSegment(inputBuffers, inputCount, "inputBuffers");
            validateAddressSegment(outputBuffers, outputCount, "outputBuffers");
            for (int input = 0; input < inputCount; input++) {
                long inputAddress = inputBuffers.getAtIndex(ADDRESS, input).address();
                for (int output = 0; output < outputCount; output++) {
                    if (inputAddress == outputBuffers.getAtIndex(ADDRESS, output).address()) {
                        throw new IllegalArgumentException(
                                "Metal NEG input and output native buffers must not alias");
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
                throw new IllegalArgumentException("Metal NEG " + role + " index is out of range");
            }
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
    }

    /** Production JDK Foreign Function and Memory binding of the exact ten-symbol ABI. */
    private static final class Ffm extends MetalNativeApi {
        private static final String VERSION = "synaptik_metal_foundation_abi_version";
        private static final String CONTEXT_CREATE = "synaptik_metal_context_create";
        private static final String CONTEXT_RELEASE = "synaptik_metal_context_release";
        private static final String BUFFER_CREATE = "synaptik_metal_buffer_create";
        private static final String BUFFER_RELEASE = "synaptik_metal_buffer_release";
        private static final String BUFFER_UPLOAD = "synaptik_metal_buffer_upload";
        private static final String BUFFER_DOWNLOAD = "synaptik_metal_buffer_download";
        private static final String NEG_EXECUTABLE_CREATE =
                NEG_EXECUTABLE_CREATE_OPERATION;
        private static final String EXECUTABLE_RELEASE =
                EXECUTABLE_RELEASE_OPERATION;
        private static final String EXECUTABLE_RUN =
                EXECUTABLE_RUN_OPERATION;

        private final Arena lookupArena;
        private final MethodHandle contextCreate;
        private final MethodHandle contextRelease;
        private final MethodHandle bufferCreate;
        private final MethodHandle bufferRelease;
        private final MethodHandle bufferUpload;
        private final MethodHandle bufferDownload;
        private final MethodHandle negExecutableCreate;
        private final MethodHandle executableRelease;
        private final MethodHandle executableRun;
        private final AtomicBoolean closed = new AtomicBoolean();

        private Ffm(
                Arena lookupArena,
                MethodHandle contextCreate,
                MethodHandle contextRelease,
                MethodHandle bufferCreate,
                MethodHandle bufferRelease,
                MethodHandle bufferUpload,
                MethodHandle bufferDownload,
                MethodHandle negExecutableCreate,
                MethodHandle executableRelease,
                MethodHandle executableRun) {
            this.lookupArena = lookupArena;
            this.contextCreate = contextCreate;
            this.contextRelease = contextRelease;
            this.bufferCreate = bufferCreate;
            this.bufferRelease = bufferRelease;
            this.bufferUpload = bufferUpload;
            this.bufferDownload = bufferDownload;
            this.negExecutableCreate = negExecutableCreate;
            this.executableRelease = executableRelease;
            this.executableRun = executableRun;
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
                MethodHandle negExecutableCreate = linker.downcallHandle(
                        require(lookup, NEG_EXECUTABLE_CREATE),
                        FunctionDescriptor.of(JAVA_INT,
                                ADDRESS, JAVA_INT, ADDRESS, ADDRESS,
                                JAVA_INT, ADDRESS, ADDRESS,
                                JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, ADDRESS));
                MethodHandle executableRelease = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RELEASE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle executableRun = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RUN),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
                return new Ffm(arena, contextCreate, contextRelease, bufferCreate, bufferRelease,
                        bufferUpload, bufferDownload, negExecutableCreate, executableRelease,
                        executableRun);
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
        NativeCreateResult createNegExecutableNative(
                Handle context,
                int[] valueRanks,
                long[] valueDimensions,
                int[] nodeInputValueIndices,
                int[] nodeOutputValueIndices,
                int[] feedValueIndices,
                int[] targetValueIndices) {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment ranks = copyInts(arena, valueRanks);
                MemorySegment dimensions = copyLongs(arena, valueDimensions);
                MemorySegment nodeInputs = copyInts(arena, nodeInputValueIndices);
                MemorySegment nodeOutputs = copyInts(arena, nodeOutputValueIndices);
                MemorySegment feeds = copyInts(arena, feedValueIndices);
                MemorySegment targets = copyInts(arena, targetValueIndices);
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = invokeCreateExecutable(
                        negExecutableCreate, context.carrier(), valueRanks.length, ranks,
                        dimensions, nodeInputValueIndices.length, nodeInputs, nodeOutputs,
                        feedValueIndices.length, feeds, targetValueIndices.length, targets, output);
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

        private static int invokeCreateExecutable(MethodHandle handle, MemorySegment context,
                int valueCount, MemorySegment ranks, MemorySegment dimensions, int nodeCount,
                MemorySegment nodeInputs, MemorySegment nodeOutputs, int feedCount,
                MemorySegment feeds, int targetCount, MemorySegment targets,
                MemorySegment output) {
            try {
                return (int) handle.invokeExact(context, valueCount, ranks, dimensions,
                        nodeCount, nodeInputs, nodeOutputs, feedCount, feeds,
                        targetCount, targets, output);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(NEG_EXECUTABLE_CREATE + " invocation failed", failure);
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
