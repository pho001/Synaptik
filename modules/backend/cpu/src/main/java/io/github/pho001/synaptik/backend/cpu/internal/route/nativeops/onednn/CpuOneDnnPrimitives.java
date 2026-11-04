package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.check;
import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.propagate;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.NativeScope;
import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnKernelSpec.Tensor;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Fixed typed primitive C bindings with run-owned memory objects and user scratch. Literal
 * pipelines reuse these same descriptor and invocation owners, borrowing slices and one ordered
 * stream. Diagnostic contraction forms remain separate from production qualification.
 * Persistent descriptors/code and invocation state have distinct leases, so closing a kernel
 * does not invalidate an already-issued workspace.
 */
public final class CpuOneDnnPrimitives {
    private static final int SRC = 1, WEIGHTS = 33, BIAS = 41, DST = 17, SCRATCHPAD = 80;
    private static final int QUERY_EXEC_ARG_MD = 255, QUERY_IMPL_INFO_STR = 8;
    private static final int F32 = 3, FORWARD_INFERENCE = 96, CONVOLUTION_DIRECT = 1;
    private static final int STRICT_FPMATH = 0, USER_SCRATCHPAD = 1, STREAM_IN_ORDER = 1;
    private static final long EXEC_ARG_BYTES = 16;

    private CpuOneDnnPrimitives() { }

    /**
     * Queries an actual typed F32-working primitive descriptor without creating an executable primitive.
     * All descriptor, attribute and ABI-memory resources are closed before returning; the copied
     * facts retain no native handle or library lease. A query or cleanup failure prevents publication,
     * and cleanup failures are suppressed on an earlier failure.
     *
     * @param library non-null, open library borrowed for the duration of this query
     * @param spec non-null numeric descriptors; tensor contents are never inspected or retained
     * @return non-null, handle-free scratchpad size and copied implementation name
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if descriptors are inconsistent or unsupported by oneDNN
     * @throws IllegalStateException if the library is closed or native querying or cleanup fails
     */
    public static CpuOneDnnKernelMetadata describe(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec) {
        try (DescriptorPreparation preparation = DescriptorPreparation.prepare(library, spec)) {
            return new CpuOneDnnKernelMetadata(preparation.scratchBytes, preparation.implementation);
        }
    }

    /**
     * Creates one executable primitive from the same strict F32-working descriptor used for metadata queries.
     * The returned kernel owns the descriptor scope and a library lease, retained by issued
     * workspaces until their close. Failed preparation, primitive creation or publication closes all
     * acquired resources, suppressing cleanup failures on the original failure.
     *
     * @param library non-null, open library whose lifetime the resulting kernel retains
     * @param spec non-null numeric descriptors; tensor contents are never inspected or retained
     * @return non-null kernel whose owner must close it
     * @throws NullPointerException if either argument is null
     * @throws IllegalArgumentException if descriptors are inconsistent or unsupported by oneDNN
     * @throws IllegalStateException if the library is closed or native creation or cleanup fails
     */
    public static CpuOneDnnKernel compile(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec) {
        try (DescriptorPreparation preparation = DescriptorPreparation.prepare(library, spec)) {
            return preparation.compile(library, spec);
        } catch (Throwable problem) {
            throw propagate(problem);
        }
    }

    /**
     * Transactional cold descriptor owner, never an executable. Temporary attributes must close
     * successfully before this owner can escape preparation; only successful kernel construction
     * transfers its descriptor scope. Queried descriptor references borrow that scope.
     */
    private static final class DescriptorPreparation implements AutoCloseable {
        private final Api api;
        private final NativeScope scope;
        private final MemorySegment descriptor;
        private final MemorySegment[] executionDescriptors;
        private final long[] requiredBytes;
        private final MemorySegment scratchDescriptor;
        private final long scratchBytes;
        private final String implementation;
        private boolean transferred;

        private DescriptorPreparation(Api api, NativeScope scope, MemorySegment descriptor,
                MemorySegment[] executionDescriptors, long[] requiredBytes,
                MemorySegment scratchDescriptor, long scratchBytes, String implementation) {
            this.api = api;
            this.scope = scope;
            this.descriptor = descriptor;
            this.executionDescriptors = executionDescriptors;
            this.requiredBytes = requiredBytes;
            this.scratchDescriptor = scratchDescriptor;
            this.scratchBytes = scratchBytes;
            this.implementation = implementation;
        }

        private static DescriptorPreparation prepare(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec) {
            Objects.requireNonNull(library, "library");
            Objects.requireNonNull(spec, "spec");
            NativeScope owned;
            synchronized (library) {
                library.requireOpen();
                owned = new NativeScope(library);
            }
            try (NativeScope temporary = new NativeScope(library)) {
                Api api = new Api(library);
                Arena arena = owned.arena();
                MemorySegment result = arena.allocate(ADDRESS);
                Tensor[] tensors = nativeTensors(spec);
                MemorySegment[] descriptors = new MemorySegment[tensors.length];
                for (int index = 0; index < tensors.length; index++) {
                    descriptors[index] = descriptor(api, owned, result, tensors[index]);
                }
                MemorySegment attrResult = temporary.arena().allocate(ADDRESS);
                MemorySegment attributes = created((int) api.attrCreate.invokeExact(attrResult),
                        attrResult, temporary, api.attrDestroy, "create primitive attributes");
                check((int) api.attrFpmath.invokeExact(attributes, STRICT_FPMATH), "set strict fpmath");
                if (spec.inputs().stream().anyMatch(input -> CpuOneDnnProjection.lowType(input.dataType()))) {
                    // Default/strict is F32 accumulation; never allow partial low spills.
                    check((int) api.attrAccumulation.invokeExact(attributes, 0), "set strict accumulation");
                }
                check((int) api.attrScratch.invokeExact(attributes, USER_SCRATCHPAD), "set user scratchpad");
                var contraction = spec instanceof CpuOneDnnKernelSpec.Contraction value ? value : null;
                if (contraction != null && contraction.activation() != CpuOneDnnKernelSpec.Activation.NONE) {
                    attrResult.set(ADDRESS, 0, MemorySegment.NULL);
                    MemorySegment postOps = created((int) api.postCreate.invokeExact(attrResult),
                            attrResult, temporary, api.postDestroy, "create post-ops");
                    int algorithm = switch (contraction.activation()) {
                        case RELU -> 32;
                        case TANH -> 33;
                        case SIGMOID -> 41;
                        case NONE -> throw new AssertionError("activation already excluded");
                    };
                    check((int) api.postEltwise.invokeExact(postOps, algorithm, 0.0f, 0.0f), "append activation");
                    check((int) api.attrPostOps.invokeExact(attributes, postOps), "set activation post-ops");
                }
                int outputIndex = tensors.length - 1;
                result.set(ADDRESS, 0, MemorySegment.NULL);
                int descriptorStatus = createDescriptor(api, library, spec, descriptors, result, attributes, arena);
                MemorySegment pd = created(descriptorStatus, result, owned, api.pdDestroy,
                        "create contraction descriptor");
                // Query references remain valid for the complete descriptor/run-lease lifetime.
                MemorySegment[] executionDescriptors = new MemorySegment[descriptors.length];
                long[] requiredBytes = new long[descriptors.length];
                for (int index = 0; index < descriptors.length; index++) {
                    int argument = index == outputIndex ? DST : inputArgument(spec, index);
                    executionDescriptors[index] = (MemorySegment) api.queryMd.invokeExact(pd, QUERY_EXEC_ARG_MD, argument);
                    requireHandle(executionDescriptors[index], "execution memory descriptor");
                    long bytes = (long) api.mdSize.invokeExact(executionDescriptors[index]);
                    if (bytes < 0) throw new IllegalArgumentException("oneDNN requires a static memory descriptor");
                    if (bytes > tensors[index].byteSize()) {
                        throw new CpuOneDnnCompiler.UnsupportedRealizationException(
                                "descriptor exceeds exact common tensor span");
                    }
                    requiredBytes[index] = Math.max(bytes, tensors[index].byteSize());
                }
                MemorySegment scratch = (MemorySegment) api.queryMd.invokeExact(pd, QUERY_EXEC_ARG_MD, SCRATCHPAD);
                requireHandle(scratch, "scratchpad descriptor");
                long scratchBytes = (long) api.mdSize.invokeExact(scratch);
                if (scratchBytes < 0) throw new IllegalArgumentException("oneDNN requires a static scratchpad");
                result.set(ADDRESS, 0, MemorySegment.NULL);
                check((int) api.query.invokeExact(pd, QUERY_IMPL_INFO_STR, 0, result), "query implementation");
                MemorySegment implementation = result.get(ADDRESS, 0);
                requireHandle(implementation, "implementation name");
                String name = implementation.reinterpret(Long.MAX_VALUE).getString(0);
                qualifyImplementation(spec, name);
                return new DescriptorPreparation(api, owned, pd, executionDescriptors,
                        requiredBytes, scratch, scratchBytes, name);
            } catch (Throwable problem) {
                try { owned.close(); } catch (Throwable cleanup) {
                    if (cleanup != problem) problem.addSuppressed(cleanup);
                }
                throw propagate(problem);
            }
        }

        private Kernel compile(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec) throws Throwable {
            MemorySegment result = scope.arena().allocate(ADDRESS);
            result.set(ADDRESS, 0, MemorySegment.NULL);
            MemorySegment primitive = created((int) api.primitiveCreate.invokeExact(result, descriptor),
                    result, scope, api.primitiveDestroy, "create primitive");
            Kernel kernel = new Kernel(library, spec, api, scope, primitive, executionDescriptors,
                    requiredBytes, scratchDescriptor, scratchBytes, implementation);
            transferred = true;
            return kernel;
        }

        @Override public void close() {
            if (!transferred) scope.close();
        }
    }

    private static int physicalType(Tensor tensor) {
        return switch (tensor.dataType()) {
            case FLOAT32 -> F32;
            case FLOAT16 -> 1;
            case BFLOAT16 -> 2;
            case FLOAT8_E5M2 -> 9;
            case FLOAT8_E4M3 -> 10;
            case BOOL -> 6; // Primitive storage is u8, not Graph's boolean enum.
            default -> throw new IllegalArgumentException("unsupported oneDNN represented dtype");
        };
    }

    private static int createDescriptor(Api api, CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec,
            MemorySegment[] descriptors, MemorySegment result, MemorySegment attributes, Arena arena)
            throws Throwable {
        MemorySegment output = descriptors[descriptors.length - 1];
        if (spec instanceof CpuOneDnnKernelSpec.Contraction contraction) {
            MemorySegment bias = contraction.bias() == CpuOneDnnKernelSpec.Bias.NONE
                    ? MemorySegment.NULL : descriptors[2];
            if (contraction.operation() == CpuOneDnnKernelSpec.Operation.MATMUL) {
                return (int) api.matmulCreate.invokeExact(result, library.engine(), descriptors[0],
                        descriptors[1], bias, output, attributes);
            }
            return (int) api.convCreate.invokeExact(result, library.engine(), FORWARD_INFERENCE,
                    CONVOLUTION_DIRECT, descriptors[0], descriptors[1], bias, output,
                    vector(arena, contraction.strides(), 0), vector(arena, contraction.dilations(), -1),
                    vector(arena, contraction.paddingLeft(), 0), vector(arena, contraction.paddingRight(), 0),
                    attributes);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Unary unary) {
            int algorithm = switch (unary.algorithm()) {
                case ABS -> 36;
                case SQRT -> 37;
                case TANH -> 33;
                case EXP -> 42;
            };
            return (int) api.unaryCreate.invokeExact(result, library.engine(), FORWARD_INFERENCE,
                    algorithm, descriptors[0], output, 0.0f, 0.0f, attributes);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Binary binary) {
            int algorithm = switch (binary.algorithm()) {
                case ADD -> 0x1fff0;
                case MUL -> 0x1fff1;
                case MAX -> 0x1fff2;
                case MIN -> 0x1fff3;
                case DIV -> 0x1fff4;
                case SUB -> 0x1fff5;
                case LE -> 0x1fff8;
                case LT -> 0x1fff9;
                case EQ -> 0x1fffa;
                case NE -> 0x1fffb;
            };
            return (int) api.binaryCreate.invokeExact(result, library.engine(), algorithm,
                    descriptors[0], descriptors[1], output, attributes);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Reduction reduction) {
            int algorithm = reduction.algorithm() == CpuOneDnnKernelSpec.ReductionAlgorithm.SUM
                    ? 0x2fff4 : 0x2fff5;
            return (int) api.reductionCreate.invokeExact(result, library.engine(), algorithm,
                    descriptors[0], output, 0.0f, 0.0f, attributes);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Reorder) {
            return (int) api.reorderCreate.invokeExact(result, descriptors[0], library.engine(),
                    output, library.engine(), attributes);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Pooling pooling) {
            return (int) api.poolingCreate.invokeExact(result, library.engine(), FORWARD_INFERENCE,
                    0x2ff, descriptors[0], output, vector(arena, pooling.strides(), 0),
                    vector(arena, pooling.kernel(), 0), vector(arena, pooling.dilations(), -1),
                    vector(arena, pooling.paddingLeft(), 0), vector(arena, pooling.paddingRight(), 0), attributes);
        }
        throw new IllegalArgumentException("pipeline requires fixed atomic descriptor preparation");
    }

    /**
     * Qualifies the selected implementation, not just the primitive operation.
     *
     * <p>Raw reorder is restricted to AArch64 {@code jit:uni}, homogeneous F32, plain
     * dense/permuted-dense descriptors, and the default scales/zero-points/no-post-ops attributes
     * created above. For pinned oneDNN 3.14 commit
     * {@code 18378c8475643c4172fe059fefba855b14c07345},
     * {@code src/cpu/aarch64/reorder/jit_uni_reorder_utils.cpp:293-324,450-451} makes scales
     * NONE, zero-points absent, beta zero and scale-adjust one. In
     * {@code jit_uni_reorder_kernel.cpp}, direct-copy (593-719), generic (723-1396), and
     * F32 transpose (382-575) then only load, shuffle and store the represented bits; F32
     * conversion is explicitly a no-op. The other optimized transpose is F32-to-BF16 only.
     * This proof does not qualify x64 implementations sharing the same name, converted,
     * repeated-address or gapped descriptors, or any other implementation. In particular
     * {@code simple:any}'s reference path in {@code src/cpu/reorder/simple_reorder.hpp:2728-2732}
     * performs floating arithmetic even with default attributes, changing negative zero and NaNs.
     *
     * @param spec exact descriptor recipe using only the attributes created by this class
     * @param implementation actual implementation identity returned by its primitive descriptor
     * @throws CpuOneDnnCompiler.UnsupportedRealizationException if this implementation and
     *     descriptor combination has not been qualified for the represented semantics
     */
    private static void qualifyImplementation(CpuOneDnnKernelSpec spec, String implementation) {
        boolean lowInputs = spec.inputs().stream().anyMatch(input -> CpuOneDnnProjection.lowType(input.dataType()));
        if (lowInputs && spec instanceof CpuOneDnnKernelSpec.Contraction contraction
                && !qualifiedLowMatmul(contraction, implementation)) {
            throw new CpuOneDnnCompiler.UnsupportedRealizationException(
                    "unqualified typed F32-accumulating MATMUL implementation: " + implementation);
        }
        if (lowInputs && !(spec instanceof CpuOneDnnKernelSpec.Contraction)
                && !implementation.equals("ref:any")) {
            // Pinned ref_binary.cpp and ref_reduction.cpp load represented operands to float
            // and compute/store F32. Their proof does not qualify an unrelated optimized body.
            throw new CpuOneDnnCompiler.UnsupportedRealizationException(
                    "unqualified typed F32-working implementation: " + implementation);
        }
        // Reference ABS changes +0 to -0; x64 EXP flushes at an unauthorized elementary site.
        if (spec instanceof CpuOneDnnKernelSpec.Unary unary
                && (unary.algorithm() == CpuOneDnnKernelSpec.UnaryAlgorithm.ABS
                        || unary.algorithm() == CpuOneDnnKernelSpec.UnaryAlgorithm.EXP)
                && !implementation.equals("jit:asimd")) {
            throw new CpuOneDnnCompiler.UnsupportedRealizationException("unqualified elementary implementation: " + implementation);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Binary binary
                && (binary.algorithm() == CpuOneDnnKernelSpec.BinaryAlgorithm.MIN
                        || binary.algorithm() == CpuOneDnnKernelSpec.BinaryAlgorithm.MAX)
                && !(implementation.equals("jit:uni")
                        && System.getProperty("os.arch").equals("aarch64"))) {
            throw new CpuOneDnnCompiler.UnsupportedRealizationException("unqualified stored-extrema implementation: " + implementation);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Reorder reorder
                && !(implementation.equals("jit:uni")
                        && System.getProperty("os.arch").equals("aarch64")
                        && reorder.input().dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                        && reorder.output().dataType() == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                        && rawMovementDense(reorder.input()) && rawMovementDense(reorder.output()))) {
            throw new CpuOneDnnCompiler.UnsupportedRealizationException(
                    "unqualified raw-movement implementation or descriptors: " + implementation);
        }
        if (spec instanceof CpuOneDnnKernelSpec.Pooling && !implementation.equals("simple_nchw:any")) {
            throw new CpuOneDnnCompiler.UnsupportedRealizationException("unqualified padded-average implementation: " + implementation);
        }
    }

    /**
     * Binds the low MATMUL working-precision proof to pinned implementations, not descriptor
     * availability. oneDNN 3.14 ref_matmul.cpp uses float operands/accumulators. AArch64 kai_utils.hpp
     * dispatches homogeneous BF16/F16 to gemm&lt;low,low,float&gt; with fast mode disabled under
     * strict attributes. The two x64 FP8-via-exact-F16-conversion brgemm paths accumulate with
     * TDPFP16PS/VDPPHPS into F32. Native AMX-FP8 (avx10_2_amx_2) uses a different fixed-point
     * accumulator and is deliberately not admitted by a broad implementation-name prefix.
     * All recipes have F32 destinations, default scales and no post-operations.
     * @param spec exact typed bare contraction recipe
     * @param implementation actual native primitive-descriptor identity
     * @return whether the selected source implementation preserves F32 working precision
     */
    private static boolean qualifiedLowMatmul(CpuOneDnnKernelSpec.Contraction spec, String implementation) {
        if (spec.operation() != CpuOneDnnKernelSpec.Operation.MATMUL
                || spec.bias() != CpuOneDnnKernelSpec.Bias.NONE
                || spec.activation() != CpuOneDnnKernelSpec.Activation.NONE
                || spec.output().dataType() != DataType.FLOAT32) return false;
        DataType type = spec.inputs().getFirst().dataType();
        if (spec.inputs().stream().anyMatch(input -> input.dataType() != type)) return false;
        if (implementation.equals("ref:any")) return true;
        String arch = System.getProperty("os.arch");
        if ((type == DataType.FLOAT16 || type == DataType.BFLOAT16) && arch.equals("aarch64")) {
            return implementation.equals("kai");
        }
        if ((type == DataType.FLOAT8_E4M3 || type == DataType.FLOAT8_E5M2)
                && (arch.equals("amd64") || arch.equals("x86_64"))) {
            return implementation.equals("brg_matmul:avx10_1_512_amx_fp16")
                    || implementation.equals("brg_matmul:avx10_2");
        }
        return false;
    }

    /**
     * Proves a plain descriptor has one dense physical permutation without repeated addresses.
     *
     * @param tensor positive-rank plain strided native descriptor
     * @return whether all nonsingleton axes form a complete dense stride chain
     */
    private static boolean rawMovementDense(Tensor tensor) {
        int remaining = 0;
        for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
            if (tensor.dimensions().get(axis) > 1) remaining |= 1 << axis;
        }
        long stride = 1;
        while (remaining != 0) {
            int next = -1;
            for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
                if ((remaining & (1 << axis)) != 0 && tensor.strides().get(axis) == stride) {
                    next = axis;
                    break;
                }
            }
            if (next < 0) return false;
            remaining &= ~(1 << next);
            stride = Math.multiplyExact(stride, tensor.dimensions().get(next));
        }
        return true;
    }

    private static MemorySegment descriptor(Api api, NativeScope scope, MemorySegment result,
            Tensor tensor) throws Throwable {
        result.set(ADDRESS, 0, MemorySegment.NULL);
        int status = (int) api.mdCreate.invokeExact(result, tensor.dimensions().size(),
                vector(scope.arena(), tensor.dimensions(), 0), physicalType(tensor),
                vector(scope.arena(), tensor.strides(), 0));
        return created(status, result, scope, api.mdDestroy, "create strided memory descriptor");
    }

    private static MemorySegment created(int status, MemorySegment result, NativeScope scope,
            MethodHandle destroy, String operation) {
        MemorySegment handle = result.get(ADDRESS, 0);
        if (handle.address() != 0) scope.own(handle, destroy);
        check(status, operation);
        requireHandle(handle, operation);
        return handle;
    }

    private static MemorySegment vector(Arena arena, List<Long> values, long adjustment) {
        MemorySegment result = arena.allocate(Math.multiplyExact((long) values.size(), Long.BYTES), Long.BYTES);
        for (int index = 0; index < values.size(); index++) {
            result.setAtIndex(JAVA_LONG, index, Math.addExact(values.get(index), adjustment));
        }
        return result;
    }

    private static void requireHandle(MemorySegment handle, String role) {
        if (handle.address() == 0) throw new IllegalStateException("oneDNN returned null " + role);
    }

    private static int inputArgument(CpuOneDnnKernelSpec spec, int index) {
        if (spec instanceof CpuOneDnnKernelSpec.Binary) return index + 1;
        return switch (index) {
            case 0 -> SRC;
            case 1 -> WEIGHTS;
            case 2 -> BIAS;
            default -> throw new IllegalArgumentException("invalid primitive input index");
        };
    }

    private static Tensor[] nativeTensors(CpuOneDnnKernelSpec recipe) {
        if (!(recipe instanceof CpuOneDnnKernelSpec.Contraction)) {
            int count = recipe.inputs().size();
            Tensor[] result = new Tensor[count + 1];
            for (int index = 0; index < count; index++) {
                Tensor input = recipe.inputs().get(index);
                result[index] = recipe instanceof CpuOneDnnKernelSpec.Binary
                        ? leadingSingletons(input, recipe.output().dimensions().size()) : input;
            }
            result[count] = recipe.output();
            return result;
        }
        var spec = (CpuOneDnnKernelSpec.Contraction) recipe;
        int inputCount = spec.inputs().size();
        Tensor[] tensors = new Tensor[inputCount + 1];
        Tensor output = spec.output();
        int rank = output.dimensions().size();
        if (spec.operation() == CpuOneDnnKernelSpec.Operation.MATMUL) {
            if (rank < 2) throw new IllegalArgumentException("matmul output rank must be at least two");
            for (int index = 0; index < inputCount; index++) {
                Tensor tensor = spec.inputs().get(index);
                if (index < 2 && tensor.dimensions().size() < 2) {
                    throw new IllegalArgumentException("matmul operands must have rank at least two");
                }
                tensors[index] = leadingSingletons(tensor, rank);
            }
            List<Long> left = tensors[0].dimensions(), right = tensors[1].dimensions();
            if (!left.get(rank - 1).equals(right.get(rank - 2))
                    || !left.get(rank - 2).equals(output.dimensions().get(rank - 2))
                    || !right.get(rank - 1).equals(output.dimensions().get(rank - 1))) {
                throw new IllegalArgumentException("matmul contraction dimensions disagree");
            }
            for (int axis = 0; axis < rank - 2; axis++) {
                long a = left.get(axis), b = right.get(axis), c = output.dimensions().get(axis);
                if ((a != 1 && a != c) || (b != 1 && b != c) || Math.max(a, b) != c) {
                    throw new IllegalArgumentException("matmul batch dimensions disagree");
                }
            }
            if (inputCount == 3) {
                for (int axis = 0; axis < rank; axis++) {
                    long dimension = tensors[2].dimensions().get(axis);
                    if (dimension != 1 && dimension != output.dimensions().get(axis)) {
                        throw new IllegalArgumentException("matmul bias cannot broadcast to output");
                    }
                }
            }
        } else {
            tensors[0] = spec.inputs().get(0);
            Tensor weights = spec.inputs().get(1);
            if (tensors[0].dimensions().size() != rank || weights.dimensions().size() != rank) {
                throw new IllegalArgumentException("convolution tensor ranks disagree");
            }
            long channels = weights.dimensions().get(0);
            if (channels % spec.groups() != 0
                    || channels != output.dimensions().get(1)
                    || Math.multiplyExact(weights.dimensions().get(1), spec.groups())
                            != tensors[0].dimensions().get(1)) {
                throw new IllegalArgumentException("convolution group channels disagree");
            }
            if (spec.groups() == 1) {
                tensors[1] = weights;
            } else {
                ArrayList<Long> dimensions = new ArrayList<>(rank + 1);
                ArrayList<Long> strides = new ArrayList<>(rank + 1);
                dimensions.add((long) spec.groups());
                dimensions.add(channels / spec.groups());
                dimensions.addAll(weights.dimensions().subList(1, rank));
                strides.add(Math.multiplyExact(channels / spec.groups(), weights.strides().get(0)));
                strides.addAll(weights.strides());
                tensors[1] = new Tensor(weights.dataType(), dimensions, strides);
            }
            if (spec.bias() == CpuOneDnnKernelSpec.Bias.EXTERNAL_LEFT
                    || spec.bias() == CpuOneDnnKernelSpec.Bias.EXTERNAL_RIGHT) {
                throw new IllegalArgumentException("primitive convolution cannot absorb an external ADD");
            }
            if (inputCount == 3) tensors[2] = convolutionBias(spec.inputs().get(2), rank, channels);
        }
        tensors[inputCount] = output;
        return tensors;
    }

    private static Tensor leadingSingletons(Tensor tensor, int rank) {
        int missing = rank - tensor.dimensions().size();
        if (missing < 0) throw new IllegalArgumentException("operand rank exceeds output rank");
        if (missing == 0) return tensor;
        ArrayList<Long> dimensions = new ArrayList<>(rank), strides = new ArrayList<>(rank);
        long leadingStride = tensor.byteSize() / tensor.dataType().byteWidth();
        for (int axis = 0; axis < missing; axis++) {
            dimensions.add(1L);
            strides.add(leadingStride);
        }
        dimensions.addAll(tensor.dimensions());
        strides.addAll(tensor.strides());
        return new Tensor(tensor.dataType(), dimensions, strides);
    }

    private static Tensor convolutionBias(Tensor bias, int rank, long channels) {
        if (bias.dimensions().size() == 1 && bias.dimensions().get(0) == channels) return bias;
        Tensor expanded = leadingSingletons(bias, rank);
        for (int axis = 0; axis < rank; axis++) {
            long expected = axis == 1 ? channels : 1;
            if (expanded.dimensions().get(axis) != expected) {
                throw new IllegalArgumentException("convolution bias must contain one value per output channel");
            }
        }
        return new Tensor(bias.dataType(), List.of(channels), List.of(expanded.strides().get(1)));
    }

    private static final class Kernel implements CpuOneDnnAtomicKernel {
        private final CpuOneDnnLibrary library;
        private final CpuOneDnnKernelSpec spec;
        private final Api api;
        private final NativeScope scope;
        private final MemorySegment primitive;
        private final MemorySegment[] descriptors;
        private final long[] requiredBytes;
        private final MemorySegment scratchDescriptor;
        private final long scratchBytes;
        private final String implementation;
        private int references = 1;
        private boolean closed;

        private Kernel(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec, Api api, NativeScope scope,
                MemorySegment primitive, MemorySegment[] descriptors, long[] requiredBytes,
                MemorySegment scratchDescriptor, long scratchBytes, String implementation) {
            this.library = library;
            this.spec = spec;
            this.api = api;
            this.scope = scope;
            this.primitive = primitive;
            this.descriptors = descriptors;
            this.requiredBytes = requiredBytes;
            this.scratchDescriptor = scratchDescriptor;
            this.scratchBytes = scratchBytes;
            this.implementation = implementation;
        }

        @Override public CpuOneDnnKernelSpec spec() { return spec; }
        @Override public long scratchpadBytes() { return scratchBytes; }
        @Override public String implementation() { return implementation; }

        @Override public CpuOneDnnWorkspace createWorkspace() { return createWorkspace(null, null); }

        @Override public CpuOneDnnWorkspace createWorkspace(MemorySegment scratch, MemorySegment stream) {
            synchronized (this) {
                if (closed) throw new IllegalStateException("oneDNN primitive kernel is closed");
                references = Math.incrementExact(references);
            }
            try { return new Workspace(this, scratch, stream); }
            catch (Throwable problem) {
                try { release(); } catch (Throwable cleanup) {
                    if (cleanup != problem) problem.addSuppressed(cleanup);
                }
                throw propagate(problem);
            }
        }

        private void release() {
            synchronized (this) {
                if (--references != 0) return;
            }
            scope.close();
        }

        @Override public void close() {
            synchronized (this) {
                if (closed) return;
                closed = true;
            }
            release();
        }
    }

    private static final class Workspace extends CpuOneDnnWorkspace {
        private final Kernel owner;
        private NativeScope scope;
        private MemorySegment stream;
        private MemorySegment arguments;
        private MemorySegment[] memories;
        private MemorySegment[] boundSegments;
        private boolean leaseOwned;
        private boolean bound;

        private Workspace(Kernel kernel, MemorySegment scratch, MemorySegment sharedStream) {
            super(kernel, scratch);
            owner = kernel;
            try {
                scope = new NativeScope(kernel.library);
                Arena arena = scope.arena();
                MemorySegment result = arena.allocate(ADDRESS);
                stream = sharedStream != null ? sharedStream
                        : created((int) kernel.api.streamCreate.invokeExact(result, kernel.library.engine(),
                                STREAM_IN_ORDER), result, scope, kernel.api.streamDestroy, "create primitive stream");
                int tensorCount = kernel.descriptors.length;
                memories = new MemorySegment[tensorCount + 1];
                boundSegments = new MemorySegment[tensorCount];
                arguments = arena.allocate(Math.multiplyExact((long) memories.length, EXEC_ARG_BYTES), 8);
                for (int index = 0; index < memories.length; index++) {
                    MemorySegment descriptor = index == tensorCount
                            ? kernel.scratchDescriptor : kernel.descriptors[index];
                    result.set(ADDRESS, 0, MemorySegment.NULL);
                    memories[index] = created((int) kernel.api.memoryCreate.invokeExact(result, descriptor,
                            kernel.library.engine(), MemorySegment.NULL), result, scope,
                            kernel.api.memoryDestroy, "create primitive memory");
                    int argument = index == tensorCount ? SCRATCHPAD
                            : index == tensorCount - 1 ? DST : inputArgument(kernel.spec, index);
                    long offset = index * EXEC_ARG_BYTES;
                    arguments.set(JAVA_INT, offset, argument);
                    arguments.set(ADDRESS, offset + 8, memories[index]);
                }
                leaseOwned = true;
            } catch (Throwable problem) {
                try { close(); } catch (Throwable cleanup) {
                    if (cleanup != problem) problem.addSuppressed(cleanup);
                }
                throw propagate(problem);
            }
        }

        @Override public void bind(MemorySegment[] inputs, MemorySegment output) {
            requireOpen();
            validateBindings(inputs, output);
            Objects.requireNonNull(inputs, "inputs");
            int inputCount = owner.spec.inputs().size();
            if (inputs.length != inputCount) throw new IllegalArgumentException("primitive input count disagrees");
            for (int index = 0; index < inputCount; index++) {
                validateSegment(inputs[index], owner.requiredBytes[index],
                        owner.spec.inputs().get(index).dataType().byteWidth(), false);
            }
            validateSegment(output, owner.requiredBytes[inputCount], owner.spec.output().dataType().byteWidth(), true);
            MemorySegment scratch = writableSegment();
            validateSegment(scratch, owner.scratchBytes, 64, true);
            bound = false;
            try {
                for (int index = 0; index < inputCount; index++) {
                    check((int) owner.api.memorySetHandle.invokeExact(memories[index], inputs[index]),
                            "bind primitive input");
                }
                check((int) owner.api.memorySetHandle.invokeExact(memories[inputCount], output),
                        "bind primitive output");
                check((int) owner.api.memorySetHandle.invokeExact(memories[inputCount + 1], scratch),
                        "bind primitive scratchpad");
                for (int index = 0; index < inputCount; index++) boundSegments[index] = inputs[index];
                boundSegments[inputCount] = output;
                bound = true;
            } catch (Throwable problem) {
                throw propagate(problem);
            }
        }

        @Override public void execute() {
            requireOpen();
            if (!bound) throw new IllegalStateException("oneDNN primitive workspace is not bound");
            executeAndAwait();
        }

        @Override void enqueue() {
            requireOpen();
            if (!bound) throw new IllegalStateException("oneDNN primitive workspace is not bound");
            for (int index = 0; index < boundSegments.length; index++) {
                if (!boundSegments[index].scope().isAlive()
                        || !boundSegments[index].isAccessibleBy(Thread.currentThread())) {
                    throw new IllegalStateException("oneDNN primitive tensor is no longer accessible");
                }
            }
            try {
                int status = (int) owner.api.execute.invokeExact(owner.primitive, stream,
                        memories.length, arguments);
                check(status, "enqueue primitive");
            } catch (Throwable problem) {
                throw propagate(problem);
            }
        }

        @Override void await() {
            requireOpen();
            try { check((int) owner.api.streamWait.invokeExact(stream), "wait for primitive stream"); }
            catch (Throwable problem) { throw propagate(problem); }
        }

        @Override protected void closeNative() {
            Throwable failure = null;
            try { if (scope != null) scope.close(); } catch (Throwable problem) { failure = problem; }
            if (boundSegments != null) {
                for (int index = 0; index < boundSegments.length; index++) boundSegments[index] = null;
            }
            if (leaseOwned) {
                leaseOwned = false;
                try { owner.release(); } catch (Throwable problem) {
                    if (failure == null) failure = problem;
                    else if (problem != failure) failure.addSuppressed(problem);
                }
            }
            if (failure != null) throw propagate(failure);
        }
    }

    private static void validateSegment(MemorySegment segment, long requiredBytes, int alignment, boolean writable) {
        Objects.requireNonNull(segment, "tensor segment");
        if (!segment.isNative() || !segment.scope().isAlive()
                || !segment.isAccessibleBy(Thread.currentThread()) || segment.byteSize() < requiredBytes
                || (requiredBytes != 0 && segment.address() == 0)
                || (segment.address() & (alignment - 1)) != 0
                || (writable && segment.isReadOnly())) {
            throw new IllegalArgumentException("oneDNN requires accessible, sufficiently sized typed native storage"
                    + (writable ? " writable for output/scratchpad" : ""));
        }
    }

    /** All symbol lookup and ABI adaptation happens once, before any workspace exists. */
    private static final class Api {
        private final MethodHandle mdCreate, mdDestroy, mdSize;
        private final MethodHandle attrCreate, attrDestroy, attrFpmath, attrAccumulation, attrScratch, attrPostOps;
        private final MethodHandle postCreate, postDestroy, postEltwise;
        private final MethodHandle matmulCreate, convCreate, pdDestroy, queryMd, query;
        private final MethodHandle unaryCreate, binaryCreate, reductionCreate, reorderCreate, poolingCreate;
        private final MethodHandle primitiveCreate, primitiveDestroy, execute;
        private final MethodHandle memoryCreate, memoryDestroy, memorySetHandle;
        private final MethodHandle streamCreate, streamDestroy, streamWait;

        private Api(CpuOneDnnLibrary library) {
            mdCreate = library.bind("dnnl_memory_desc_create_with_strides",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
            mdDestroy = library.bind("dnnl_memory_desc_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            mdSize = library.bind("dnnl_memory_desc_get_size", FunctionDescriptor.of(JAVA_LONG, ADDRESS));
            attrCreate = library.bind("dnnl_primitive_attr_create", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            attrDestroy = library.bind("dnnl_primitive_attr_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            attrFpmath = library.bind("dnnl_primitive_attr_set_fpmath_mode",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
            attrAccumulation = library.bind("dnnl_primitive_attr_set_accumulation_mode",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
            attrScratch = library.bind("dnnl_primitive_attr_set_scratchpad_mode",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
            attrPostOps = library.bind("dnnl_primitive_attr_set_post_ops",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            postCreate = library.bind("dnnl_post_ops_create", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            postDestroy = library.bind("dnnl_post_ops_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            postEltwise = library.bind("dnnl_post_ops_append_eltwise",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_FLOAT, JAVA_FLOAT));
            matmulCreate = library.bind("dnnl_matmul_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            convCreate = library.bind("dnnl_convolution_forward_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS,
                            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            unaryCreate = library.bind("dnnl_eltwise_forward_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS,
                            JAVA_FLOAT, JAVA_FLOAT, ADDRESS));
            binaryCreate = library.bind("dnnl_binary_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            reductionCreate = library.bind("dnnl_reduction_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS,
                            JAVA_FLOAT, JAVA_FLOAT, ADDRESS));
            reorderCreate = library.bind("dnnl_reorder_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            poolingCreate = library.bind("dnnl_pooling_forward_primitive_desc_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS,
                            ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            pdDestroy = library.bind("dnnl_primitive_desc_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            queryMd = library.bind("dnnl_primitive_desc_query_md",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT));
            query = library.bind("dnnl_primitive_desc_query",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS));
            primitiveCreate = library.bind("dnnl_primitive_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            primitiveDestroy = library.bind("dnnl_primitive_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            execute = library.bind("dnnl_primitive_execute",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
            memoryCreate = library.bind("dnnl_memory_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            memoryDestroy = library.bind("dnnl_memory_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            memorySetHandle = library.bind("dnnl_memory_set_data_handle",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            streamCreate = library.bind("dnnl_stream_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
            streamDestroy = library.bind("dnnl_stream_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            streamWait = library.bind("dnnl_stream_wait", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        }
    }
}
