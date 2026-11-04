package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.check;
import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.propagate;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.NativeScope;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.ref.Reference;
import java.util.List;
import java.util.Objects;

/**
 * oneDNN 3.14 Graph C ABI emitter for an already-lowered typed F32-working numeric block.
 * This is not Synaptik automatic differentiation or independent graph lowering. External inputs,
 * including weights, are VARIABLE and retain canonical descriptors; opaque native-internal
 * layouts cannot replace public stores. Wider diagnostic descriptors are not production eligibility.
 */
public final class CpuOneDnnGraph {
    private static final long LOGICAL_TENSOR_BYTES = 224;
    private static final long OUTPUT_ID = 100;
    private static final int F32 = 3;
    private static final int VARIABLE = 1;
    private static final int STRIDED = 2;

    private CpuOneDnnGraph() { }

    /**
     * Compiles exactly one supported native partition and queries exact user scratch.
     * Every acquired handle and ABI allocation is scoped transactionally; partial failure closes
     * the scope. Analysis callers must close this temporary kernel after copying metadata, while
     * finalization transfers a freshly compiled kernel through prepared-resource ownership.
     *
     * @param library non-null open exact library whose lease the result retains
     * @param spec non-null immutable common numeric recipe; no tensor contents are read or retained
     * @return non-null caller-owned kernel; issued workspaces retain it until their own close
     * @throws NullPointerException if either argument is null
     * @throws CpuOneDnnCompiler.UnsupportedRealizationException if partition coverage, ports or external layouts cannot
     *     represent the exact recipe as one native block
     * @throws IllegalArgumentException if descriptors are invalid or unsupported
     * @throws IllegalStateException if the library is closed or native compilation/cleanup fails
     */
    public static CpuOneDnnKernel compile(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec) {
        Objects.requireNonNull(library, "library");
        Objects.requireNonNull(spec, "spec");
        requireSupportedAtomic(spec);
        NativeScope scope = new NativeScope(library);
        try {
            library.requireOpen();
            Api api = new Api(library);
            Arena arena = scope.arena();
            MemorySegment pointer = arena.allocate(ADDRESS);
            MemorySegment graph = created((int) api.graphCreate.invokeExact(pointer, 1, 0),
                    pointer, scope, api.graphDestroy, "create strict Graph");
            MemorySegment[] inputs = new MemorySegment[spec.inputs().size()];
            for (int index = 0; index < inputs.length; index++) {
                inputs[index] = logical(arena, index + 1L, spec.inputs().get(index), false);
            }
            MemorySegment output = logical(arena, OUTPUT_ID, spec.output(), false);
            int opCount;
            if (spec instanceof CpuOneDnnKernelSpec.Contraction contraction) {
                opCount = addContraction(scope, api, graph, inputs, output, contraction);
            } else {
                addAtomic(scope, api, graph, inputs, output, spec);
                opCount = 1;
            }
            check((int) api.graphFinalize.invokeExact(graph), "finalize Graph block");
            check((int) api.graphFilter.invokeExact(graph, 1), "filter Graph fusion partitions");
            MemorySegment count = arena.allocate(JAVA_LONG);
            check((int) api.partitionCount.invokeExact(graph, count), "query Graph partition count");
            long partitionCount = count.get(JAVA_LONG, 0);
            if (partitionCount <= 0 || partitionCount > opCount) {
                throw unsupported("invalid partition coverage count " + partitionCount);
            }
            MemorySegment partitions = arena.allocate(Math.multiplyExact(partitionCount, ADDRESS.byteSize()),
                    ADDRESS.byteAlignment());
            int partitionStatus = (int) api.getPartitions.invokeExact(graph, partitionCount, partitions);
            // Register even partial native outputs before checking the return status.
            ownPartitions(scope, api, partitions, partitionCount);
            check(partitionStatus, "get Graph partitions");
            boolean[] covered = new boolean[opCount];
            MemorySegment supported = arena.allocate(JAVA_BYTE);
            for (int index = 0; index < partitionCount; index++) {
                MemorySegment partition = partitions.getAtIndex(ADDRESS, index);
                if (partition.address() == 0) throw unsupported("null partition");
                check((int) api.isSupported.invokeExact(partition, supported), "query Graph partition support");
                if (supported.get(JAVA_BYTE, 0) == 0) throw unsupported("unsupported partition " + index);
                check((int) api.opCount.invokeExact(partition, count), "query Graph partition operation count");
                long size = count.get(JAVA_LONG, 0);
                if (size <= 0 || size > opCount) throw unsupported("invalid partition operation count");
                MemorySegment ids = arena.allocate(Math.multiplyExact(size, Long.BYTES), Long.BYTES);
                check((int) api.getOps.invokeExact(partition, size, ids), "query Graph partition operations");
                for (int item = 0; item < size; item++) {
                    long id = ids.getAtIndex(JAVA_LONG, item);
                    if (id < 1 || id > opCount || covered[(int) id - 1]) {
                        throw unsupported("unknown or repeated operation " + id);
                    }
                    covered[(int) id - 1] = true;
                }
            }
            for (boolean present : covered) {
                if (!present) throw unsupported("partitioning omitted a numeric-block operation");
            }
            // This contract admits one immutable primitive block, not a hidden inter-partition arena.
            if (partitionCount != 1) {
                throw unsupported("numeric block requires " + partitionCount + " partitions; single partition required");
            }
            MemorySegment partition = partitions.get(ADDRESS, 0);
            check((int) api.inputCount.invokeExact(partition, count), "query Graph input port count");
            if (count.get(JAVA_LONG, 0) != inputs.length) throw unsupported("unexpected external input ports");
            MemorySegment inputPorts = arena.allocate(LOGICAL_TENSOR_BYTES * inputs.length, Long.BYTES);
            check((int) api.inputPorts.invokeExact(partition, (long) inputs.length, inputPorts),
                    "query Graph input ports");
            int[] inputOrder = new int[inputs.length];
            boolean[] seen = new boolean[inputs.length];
            MemorySegment inputPointers = arena.allocate(ADDRESS, inputs.length);
            for (int port = 0; port < inputs.length; port++) {
                long id = inputPorts.get(JAVA_LONG, port * LOGICAL_TENSOR_BYTES);
                if (id < 1 || id > inputs.length || seen[(int) id - 1]) {
                    throw unsupported("unrecognized or repeated external input " + id);
                }
                int input = (int) id - 1;
                inputOrder[port] = input;
                seen[input] = true;
                inputPointers.setAtIndex(ADDRESS, port, inputs[input]);
            }
            check((int) api.outputCount.invokeExact(partition, count), "query Graph output port count");
            if (count.get(JAVA_LONG, 0) != 1) throw unsupported("unexpected external output ports");
            MemorySegment outputPort = arena.allocate(LOGICAL_TENSOR_BYTES, Long.BYTES);
            check((int) api.outputPorts.invokeExact(partition, 1L, outputPort), "query Graph output ports");
            if (outputPort.get(JAVA_LONG, 0) != OUTPUT_ID) throw unsupported("final output is not partition output");
            MemorySegment outputPointers = arena.allocate(ADDRESS);
            outputPointers.set(ADDRESS, 0, output);
            pointer.set(ADDRESS, 0, MemorySegment.NULL);
            MemorySegment compiled = created((int) api.compiledCreate.invokeExact(pointer, partition),
                    pointer, scope, api.compiledDestroy, "create compiled Graph partition");
            check((int) api.compile.invokeExact(partition, compiled, (long) inputs.length, inputPointers,
                    1L, outputPointers, library.engine()), "compile Graph numeric block");
            MemorySegment[] compiledInputs = new MemorySegment[inputs.length];
            long[] inputBytes = new long[inputs.length];
            for (int port = 0; port < inputs.length; port++) {
                int input = inputOrder[port];
                compiledInputs[port] = queryExternal(arena, api, compiled, input + 1L,
                        spec.inputs().get(input), count);
                inputBytes[port] = count.get(JAVA_LONG, 0);
            }
            MemorySegment compiledOutput = queryExternal(arena, api, compiled, OUTPUT_ID, spec.output(), count);
            long outputBytes = count.get(JAVA_LONG, 0);
            MemorySegment scratchDescriptor = arena.allocate(LOGICAL_TENSOR_BYTES, Long.BYTES);
            check((int) api.getScratch.invokeExact(compiled, scratchDescriptor), "query Graph user scratchpad descriptor");
            check((int) api.memorySize.invokeExact(scratchDescriptor, count), "query Graph user scratchpad bytes");
            long scratchBytes = count.get(JAVA_LONG, 0);
            if (scratchBytes < 0) throw unsupported("scratchpad size exceeds signed addressable memory");
            return new Kernel(library, spec, api, scope, compiled, compiledInputs, compiledOutput,
                    scratchDescriptor, inputOrder, inputBytes, outputBytes, scratchBytes);
        } catch (Throwable problem) {
            try { scope.close(); } catch (Throwable cleanup) {
                if (cleanup != problem) problem.addSuppressed(cleanup);
            }
            throw propagate(problem);
        }
    }

    private static MemorySegment created(int status, MemorySegment pointer, NativeScope scope,
            MethodHandle destroy, String operation) {
        MemorySegment handle = pointer.get(ADDRESS, 0);
        if (handle.address() != 0) scope.own(handle, destroy);
        check(status, operation);
        if (handle.address() == 0) throw new IllegalStateException("oneDNN returned null " + operation);
        return handle;
    }

    private static int physicalType(CpuOneDnnKernelSpec.Tensor tensor) {
        return switch (tensor.dataType()) {
            case FLOAT32 -> F32;
            case FLOAT16 -> 1;
            case BFLOAT16 -> 2;
            case BOOL -> 8; // Graph boolean is distinct from primitive u8.
            default -> throw unsupported("represented dtype is not F32, F16, BF16 or BOOL; Graph FP8 is unavailable");
        };
    }

    private static void requireSupportedAtomic(CpuOneDnnKernelSpec spec) {
        if (!(spec instanceof CpuOneDnnKernelSpec.Atomic)) throw unsupported("atomic recipe required");
        if (spec instanceof CpuOneDnnKernelSpec.Binary binary) {
            switch (binary.algorithm()) {
                case LT, LE, EQ, NE -> throw unsupported("3.14 Graph has no exact predicate operation");
                default -> { }
            }
        }
        if (spec instanceof CpuOneDnnKernelSpec.Pooling pooling
                && pooling.dilations().stream().anyMatch(value -> value != 1)) {
            throw unsupported("average pooling has no Graph dilation attribute");
        }
    }

    private static void addAtomic(NativeScope scope, Api api, MemorySegment graph, MemorySegment[] inputs,
            MemorySegment output, CpuOneDnnKernelSpec spec) throws Throwable {
        int kind = switch (spec) {
            case CpuOneDnnKernelSpec.Unary unary -> switch (unary.algorithm()) {
                case ABS -> 0;
                case SQRT -> 68;
                case TANH -> 75;
                case EXP -> 26;
            };
            case CpuOneDnnKernelSpec.Binary binary -> switch (binary.algorithm()) {
                case ADD -> 2;
                case SUB -> 74;
                case MUL -> 46;
                case DIV -> 20;
                case MIN -> 43;
                case MAX -> 40;
                case LT, LE, EQ, NE -> throw unsupported("predicate operation is unavailable");
            };
            case CpuOneDnnKernelSpec.Reduction reduction ->
                    reduction.algorithm() == CpuOneDnnKernelSpec.ReductionAlgorithm.SUM ? 57 : 56;
            case CpuOneDnnKernelSpec.Reorder ignored -> 60;
            case CpuOneDnnKernelSpec.Pooling ignored -> 3;
            default -> throw unsupported("not a standalone atomic operation");
        };
        MemorySegment operation = op(scope, api, 1, kind, "literal_site");
        for (MemorySegment input : inputs) addInput(api, operation, input);
        addOutput(api, operation, output);
        Arena arena = scope.arena();
        if (spec instanceof CpuOneDnnKernelSpec.Binary) {
            stringAttr(arena, api, operation, 128, "numpy");
        } else if (spec instanceof CpuOneDnnKernelSpec.Reduction reduction) {
            var axes = new java.util.ArrayList<Long>();
            for (int axis = 0; axis < reduction.input().dimensions().size(); axis++) {
                if (!reduction.input().dimensions().get(axis).equals(reduction.output().dimensions().get(axis))) {
                    axes.add((long) axis);
                }
            }
            if (axes.isEmpty()) throw unsupported("oneDNN 3.14 does not support identity reductions");
            longAttr(arena, api, operation, 64, axes);
            boolAttr(arena, api, operation, 97, true);
        } else if (spec instanceof CpuOneDnnKernelSpec.Pooling pooling) {
            stringAttr(arena, api, operation, 131, "NCX");
            stringAttr(arena, api, operation, 134, "floor");
            stringAttr(arena, api, operation, 129, "none");
            boolAttr(arena, api, operation, 96, false);
            longAttr(arena, api, operation, 67, pooling.kernel());
            longAttr(arena, api, operation, 75, pooling.strides());
            longAttr(arena, api, operation, 70, pooling.paddingLeft());
            longAttr(arena, api, operation, 71, pooling.paddingRight());
        }
        check((int) api.addOp.invokeExact(graph, operation), "add Graph literal site");
    }

    private static int addContraction(NativeScope scope, Api api, MemorySegment graph, MemorySegment[] inputs,
            MemorySegment output, CpuOneDnnKernelSpec.Contraction spec) throws Throwable {
        Arena arena = scope.arena();
        boolean externalBias = spec.bias() == CpuOneDnnKernelSpec.Bias.EXTERNAL_LEFT
                || spec.bias() == CpuOneDnnKernelSpec.Bias.EXTERNAL_RIGHT;
        boolean activation = spec.activation() != CpuOneDnnKernelSpec.Activation.NONE;
        int opCount = 1 + (externalBias ? 1 : 0) + (activation ? 1 : 0);
        long nextTensorId = inputs.length + 1L;
        MemorySegment contractionOutput = externalBias || activation
                ? logical(arena, nextTensorId++, spec.output(), true) : output;
        MemorySegment contraction = op(scope, api, 1,
                spec.operation() == CpuOneDnnKernelSpec.Operation.MATMUL ? 39 : 13, "contraction");
        addInput(api, contraction, inputs[0]);
        addInput(api, contraction, inputs[1]);
        if (spec.bias() == CpuOneDnnKernelSpec.Bias.INTRINSIC) addInput(api, contraction, inputs[2]);
        addOutput(api, contraction, contractionOutput);
        if (spec.operation() == CpuOneDnnKernelSpec.Operation.MATMUL) {
            boolAttr(arena, api, contraction, 101, false);
            boolAttr(arena, api, contraction, 102, false);
        } else {
            stringAttr(arena, api, contraction, 131, "NCX");
            stringAttr(arena, api, contraction, 135, "OIX");
            longAttr(arena, api, contraction, 50, List.of((long) spec.groups()));
            longAttr(arena, api, contraction, 75, spec.strides());
            longAttr(arena, api, contraction, 65, spec.dilations());
            longAttr(arena, api, contraction, 70, spec.paddingLeft());
            longAttr(arena, api, contraction, 71, spec.paddingRight());
        }
        check((int) api.addOp.invokeExact(graph, contraction), "add Graph contraction");
        MemorySegment current = contractionOutput;
        int nextOpId = 2;
        if (externalBias) {
            MemorySegment addOutput = activation ? logical(arena, nextTensorId, spec.output(), true) : output;
            MemorySegment add = op(scope, api, nextOpId++, 2, "external_add");
            if (spec.bias() == CpuOneDnnKernelSpec.Bias.EXTERNAL_LEFT) {
                addInput(api, add, inputs[2]);
                addInput(api, add, current);
            } else {
                addInput(api, add, current);
                addInput(api, add, inputs[2]);
            }
            stringAttr(arena, api, add, 128, "numpy");
            addOutput(api, add, addOutput);
            check((int) api.addOp.invokeExact(graph, add), "add Graph external Add");
            current = addOutput;
        }
        if (activation) {
            int kind = switch (spec.activation()) {
                case RELU -> 58;
                case SIGMOID -> 62;
                case TANH -> 75;
                case NONE -> throw new AssertionError("activation already checked");
            };
            MemorySegment unary = op(scope, api, nextOpId, kind, "activation");
            addInput(api, unary, current);
            addOutput(api, unary, output);
            check((int) api.addOp.invokeExact(graph, unary), "add Graph activation");
        }
        return opCount;
    }

    private static CpuOneDnnCompiler.UnsupportedRealizationException unsupported(String message) {
        return new CpuOneDnnCompiler.UnsupportedRealizationException(message);
    }

    private static MemorySegment logical(Arena arena, long id, CpuOneDnnKernelSpec.Tensor tensor,
            boolean anyLayout) {
        MemorySegment descriptor = arena.allocate(LOGICAL_TENSOR_BYTES, Long.BYTES);
        descriptor.set(JAVA_LONG, 0, id);
        descriptor.set(JAVA_INT, 8, tensor.dimensions().size());
        for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
            descriptor.set(JAVA_LONG, 16L + axis * Long.BYTES, tensor.dimensions().get(axis));
            descriptor.set(JAVA_LONG, 128L + axis * Long.BYTES, tensor.strides().get(axis));
        }
        descriptor.set(JAVA_INT, 112, physicalType(tensor));
        descriptor.set(JAVA_INT, 116, VARIABLE);
        descriptor.set(JAVA_INT, 120, anyLayout ? 1 : STRIDED);
        return descriptor;
    }

    private static MemorySegment queryExternal(Arena arena, Api api, MemorySegment compiled,
            long id, CpuOneDnnKernelSpec.Tensor expected, MemorySegment size) throws Throwable {
        MemorySegment descriptor = arena.allocate(LOGICAL_TENSOR_BYTES, Long.BYTES);
        check((int) api.queryLogical.invokeExact(compiled, id, descriptor), "query compiled Graph external descriptor");
        if (descriptor.get(JAVA_LONG, 0) != id || descriptor.get(JAVA_INT, 112) != physicalType(expected)
                || descriptor.get(JAVA_INT, 120) != STRIDED
                || descriptor.get(JAVA_INT, 8) != expected.dimensions().size()) {
            throw unsupported("compiled external descriptor changed type, rank or explicit layout for tensor " + id);
        }
        for (int axis = 0; axis < expected.dimensions().size(); axis++) {
            if (descriptor.get(JAVA_LONG, 16L + axis * Long.BYTES) != expected.dimensions().get(axis)
                    || descriptor.get(JAVA_LONG, 128L + axis * Long.BYTES) != expected.strides().get(axis)) {
                throw unsupported("compiled external descriptor changed shape or strides for tensor " + id);
            }
        }
        check((int) api.memorySize.invokeExact(descriptor, size), "query compiled Graph external bytes");
        if (size.get(JAVA_LONG, 0) != expected.byteSize()) {
            throw unsupported("compiled external byte span differs for tensor " + id);
        }
        return descriptor;
    }

    private static void ownPartitions(NativeScope scope, Api api, MemorySegment partitions, long count) {
        for (int index = 0; index < count; index++) {
            MemorySegment handle = partitions.getAtIndex(ADDRESS, index);
            if (handle.address() == 0) continue;
            try { scope.own(handle, api.partitionDestroy); }
            catch (Throwable problem) {
                // own() already destroyed the unregistered current handle; release the untouched suffix.
                for (int remaining = index + 1; remaining < count; remaining++) {
                    MemorySegment pending = partitions.getAtIndex(ADDRESS, remaining);
                    if (pending.address() == 0) continue;
                    try { check((int) api.partitionDestroy.invokeExact(pending), "destroy unregistered Graph partition"); }
                    catch (Throwable cleanup) { if (cleanup != problem) problem.addSuppressed(cleanup); }
                }
                throw propagate(problem);
            }
        }
    }

    private static MemorySegment op(NativeScope scope, Api api, long id, int kind, String name)
            throws Throwable {
        MemorySegment pointer = scope.arena().allocate(ADDRESS);
        MemorySegment nativeName = scope.arena().allocateFrom(name);
        return created((int) api.opCreate.invokeExact(pointer, id, kind, nativeName),
                pointer, scope, api.opDestroy, "create Graph operation");
    }

    private static void addInput(Api api, MemorySegment op, MemorySegment tensor) throws Throwable {
        check((int) api.addInput.invokeExact(op, tensor), "add Graph operation input");
    }

    private static void addOutput(Api api, MemorySegment op, MemorySegment tensor) throws Throwable {
        check((int) api.addOutput.invokeExact(op, tensor), "add Graph operation output");
    }

    private static void boolAttr(Arena arena, Api api, MemorySegment op, int attribute, boolean value)
            throws Throwable {
        MemorySegment data = arena.allocate(JAVA_BYTE);
        data.set(JAVA_BYTE, 0, (byte) (value ? 1 : 0));
        check((int) api.boolAttr.invokeExact(op, attribute, data, 1L), "set Graph boolean attribute");
    }

    private static void longAttr(Arena arena, Api api, MemorySegment op, int attribute, List<Long> values)
            throws Throwable {
        MemorySegment data = arena.allocate(JAVA_LONG, values.size());
        for (int index = 0; index < values.size(); index++) data.setAtIndex(JAVA_LONG, index, values.get(index));
        check((int) api.longAttr.invokeExact(op, attribute, data, (long) values.size()), "set Graph integer attribute");
    }

    private static void stringAttr(Arena arena, Api api, MemorySegment op, int attribute, String value)
            throws Throwable {
        MemorySegment data = arena.allocateFrom(value);
        check((int) api.stringAttr.invokeExact(op, attribute, data, (long) value.length()), "set Graph string attribute");
    }

    private static final class Kernel implements CpuOneDnnAtomicKernel {
        private final CpuOneDnnLibrary library;
        private final CpuOneDnnKernelSpec spec;
        private final Api api;
        private final NativeScope scope;
        private final MemorySegment compiled;
        private final MemorySegment[] inputDescriptors;
        private final MemorySegment outputDescriptor;
        private final MemorySegment scratchDescriptor;
        private final int[] inputOrder;
        private final long[] inputBytes;
        private final long outputBytes;
        private final long scratchBytes;
        private int references = 1;
        private boolean closed;

        private Kernel(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec, Api api, NativeScope scope,
                MemorySegment compiled, MemorySegment[] inputDescriptors, MemorySegment outputDescriptor,
                MemorySegment scratchDescriptor, int[] inputOrder, long[] inputBytes,
                long outputBytes, long scratchBytes) {
            this.library = library;
            this.spec = spec;
            this.api = api;
            this.scope = scope;
            this.compiled = compiled;
            this.inputDescriptors = inputDescriptors;
            this.outputDescriptor = outputDescriptor;
            this.scratchDescriptor = scratchDescriptor;
            this.inputOrder = inputOrder;
            this.inputBytes = inputBytes;
            this.outputBytes = outputBytes;
            this.scratchBytes = scratchBytes;
        }

        @Override public CpuOneDnnKernelSpec spec() { return spec; }
        @Override public long scratchpadBytes() { return scratchBytes; }
        @Override public String implementation() { return "onednn-3.14.0-graph:strict:single-partition:user-scratchpad"; }

        @Override public CpuOneDnnWorkspace createWorkspace() { return createWorkspace(null, null); }

        @Override public CpuOneDnnWorkspace createWorkspace(MemorySegment scratch, MemorySegment stream) {
            synchronized (this) {
                if (closed) throw new IllegalStateException("oneDNN Graph kernel is closed");
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
        private MemorySegment inputTable;
        private MemorySegment outputTable;
        private MemorySegment[] inputHandles;
        private MemorySegment outputHandle;
        private MemorySegment scratchHandle;
        private MemorySegment[] boundInputs;
        private MemorySegment boundOutput;
        private boolean ownsLease;
        private boolean bound;

        private Workspace(Kernel owner, MemorySegment scratch, MemorySegment sharedStream) {
            super(owner, scratch);
            this.owner = owner;
            try {
                scope = new NativeScope(owner.library);
                Arena arena = scope.arena();
                MemorySegment pointer = arena.allocate(ADDRESS);
                if (sharedStream == null) {
                    stream = created((int) owner.api.streamCreate.invokeExact(pointer, owner.library.engine(), 1),
                            pointer, scope, owner.api.streamDestroy, "create Graph execution stream");
                } else {
                    stream = sharedStream;
                }
                inputTable = arena.allocate(ADDRESS, owner.inputDescriptors.length);
                outputTable = arena.allocate(ADDRESS);
                inputHandles = new MemorySegment[owner.inputDescriptors.length];
                boundInputs = new MemorySegment[inputHandles.length];
                for (int port = 0; port < inputHandles.length; port++) {
                    inputHandles[port] = createTensor(owner.inputDescriptors[port], MemorySegment.NULL, pointer);
                    inputTable.setAtIndex(ADDRESS, port, inputHandles[port]);
                }
                outputHandle = createTensor(owner.outputDescriptor, MemorySegment.NULL, pointer);
                outputTable.set(ADDRESS, 0, outputHandle);
                scratchHandle = createTensor(owner.scratchDescriptor, writableSegment(), pointer);
                ownsLease = true;
            } catch (Throwable problem) {
                try { close(); } catch (Throwable cleanup) {
                    if (cleanup != problem) problem.addSuppressed(cleanup);
                }
                throw propagate(problem);
            }
        }

        private MemorySegment createTensor(MemorySegment descriptor, MemorySegment data, MemorySegment pointer)
                throws Throwable {
            pointer.set(ADDRESS, 0, MemorySegment.NULL);
            return created((int) owner.api.tensorCreate.invokeExact(pointer, descriptor, owner.library.engine(), data),
                    pointer, scope, owner.api.tensorDestroy, "create Graph execution tensor");
        }

        @Override public void bind(MemorySegment[] inputs, MemorySegment output) {
            requireOpen();
            validateBindings(inputs, output);
            Objects.requireNonNull(inputs, "inputs");
            if (inputs.length != inputHandles.length) throw new IllegalArgumentException("Graph input count differs");
            for (int port = 0; port < inputHandles.length; port++) {
                requireData(inputs[owner.inputOrder[port]], owner.inputBytes[port], false);
            }
            requireData(output, owner.outputBytes, true);
            bound = false;
            try {
                for (int port = 0; port < inputHandles.length; port++) {
                    MemorySegment data = inputs[owner.inputOrder[port]];
                    check((int) owner.api.setData.invokeExact(inputHandles[port], data), "bind Graph input data handle");
                    boundInputs[port] = data;
                }
                check((int) owner.api.setData.invokeExact(outputHandle, output), "bind Graph output data handle");
                boundOutput = output;
                bound = true;
            } catch (Throwable problem) { throw propagate(problem); }
        }

        private static void requireData(MemorySegment data, long bytes, boolean writable) {
            Objects.requireNonNull(data, "tensor data");
            if (!data.isNative() || !data.scope().isAlive() || !data.isAccessibleBy(Thread.currentThread())
                    || data.byteSize() < bytes || (writable && data.isReadOnly())) {
                throw new IllegalArgumentException("Graph tensor requires accessible native storage of its exact descriptor span");
            }
        }

        private static void requireLiveData(MemorySegment data) {
            if (!data.scope().isAlive() || !data.isAccessibleBy(Thread.currentThread())) {
                throw new IllegalStateException("bound Graph data is closed or inaccessible");
            }
        }

        @Override public void execute() {
            requireOpen();
            if (!bound) throw new IllegalStateException("Graph workspace has not been bound");
            try { enqueue(); } finally { await(); }
        }

        @Override void enqueue() {
            requireOpen();
            if (!bound) throw new IllegalStateException("Graph workspace has not been bound");
            for (int port = 0; port < inputHandles.length; port++) {
                requireLiveData(boundInputs[port]);
            }
            requireLiveData(boundOutput);
            try {
                int status = (int) owner.api.execute.invokeExact(owner.compiled, stream,
                        (long) inputHandles.length, inputTable, 1L, outputTable, scratchHandle);
                check(status, "enqueue Graph numeric block with user scratchpad");
            } catch (Throwable problem) { throw propagate(problem); }
            finally { Reference.reachabilityFence(this); }
        }

        @Override void await() {
            requireOpen();
            try { check((int) owner.api.streamWait.invokeExact(stream), "wait Graph execution stream"); }
            catch (Throwable problem) { throw propagate(problem); }
        }

        @Override protected void closeNative() {
            Throwable failure = null;
            try { if (scope != null) scope.close(); } catch (Throwable problem) { failure = problem; }
            bound = false;
            boundInputs = null;
            boundOutput = null;
            if (ownsLease) {
                ownsLease = false;
                try { owner.release(); } catch (Throwable problem) {
                    if (failure == null) failure = problem;
                    else if (problem != failure) failure.addSuppressed(problem);
                }
            }
            if (failure != null) throw propagate(failure);
        }
    }

    /** All symbols are resolved during compilation; invocation sites use exact C signatures. */
    private static final class Api {
        private final MethodHandle graphCreate;
        private final MethodHandle graphDestroy;
        private final MethodHandle graphFinalize;
        private final MethodHandle graphFilter;
        private final MethodHandle addOp;
        private final MethodHandle partitionCount;
        private final MethodHandle getPartitions;
        private final MethodHandle partitionDestroy;
        private final MethodHandle isSupported;
        private final MethodHandle opCount;
        private final MethodHandle getOps;
        private final MethodHandle inputCount;
        private final MethodHandle inputPorts;
        private final MethodHandle outputCount;
        private final MethodHandle outputPorts;
        private final MethodHandle opCreate;
        private final MethodHandle opDestroy;
        private final MethodHandle addInput;
        private final MethodHandle addOutput;
        private final MethodHandle boolAttr;
        private final MethodHandle longAttr;
        private final MethodHandle stringAttr;
        private final MethodHandle compiledCreate;
        private final MethodHandle compiledDestroy;
        private final MethodHandle compile;
        private final MethodHandle queryLogical;
        private final MethodHandle getScratch;
        private final MethodHandle memorySize;
        private final MethodHandle tensorCreate;
        private final MethodHandle tensorDestroy;
        private final MethodHandle setData;
        private final MethodHandle execute;
        private final MethodHandle streamCreate;
        private final MethodHandle streamDestroy;
        private final MethodHandle streamWait;

        private Api(CpuOneDnnLibrary library) {
            graphCreate = library.bind("dnnl_graph_graph_create_with_fpmath_mode",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_INT));
            graphDestroy = library.bind("dnnl_graph_graph_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            graphFinalize = library.bind("dnnl_graph_graph_finalize", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            graphFilter = library.bind("dnnl_graph_graph_filter", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT));
            addOp = library.bind("dnnl_graph_add_op", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            partitionCount = library.bind("dnnl_graph_graph_get_partition_num", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            getPartitions = library.bind("dnnl_graph_graph_get_partitions", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            partitionDestroy = library.bind("dnnl_graph_partition_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            isSupported = library.bind("dnnl_graph_partition_is_supported", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            opCount = library.bind("dnnl_graph_partition_get_op_num", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            getOps = library.bind("dnnl_graph_partition_get_ops", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            inputCount = library.bind("dnnl_graph_partition_get_input_ports_num", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            inputPorts = library.bind("dnnl_graph_partition_get_input_ports", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            outputCount = library.bind("dnnl_graph_partition_get_output_ports_num", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            outputPorts = library.bind("dnnl_graph_partition_get_output_ports", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            opCreate = library.bind("dnnl_graph_op_create", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS));
            opDestroy = library.bind("dnnl_graph_op_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            addInput = library.bind("dnnl_graph_op_add_input", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            addOutput = library.bind("dnnl_graph_op_add_output", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            boolAttr = library.bind("dnnl_graph_op_set_attr_bool", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG));
            longAttr = library.bind("dnnl_graph_op_set_attr_s64", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG));
            stringAttr = library.bind("dnnl_graph_op_set_attr_str", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_LONG));
            compiledCreate = library.bind("dnnl_graph_compiled_partition_create", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            compiledDestroy = library.bind("dnnl_graph_compiled_partition_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            compile = library.bind("dnnl_graph_partition_compile",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
            queryLogical = library.bind("dnnl_graph_compiled_partition_query_logical_tensor",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
            getScratch = library.bind("dnnl_graph_compiled_partition_get_scratchpad_logical_tensor",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            memorySize = library.bind("dnnl_graph_logical_tensor_get_mem_size", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            tensorCreate = library.bind("dnnl_graph_tensor_create", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, ADDRESS));
            tensorDestroy = library.bind("dnnl_graph_tensor_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            setData = library.bind("dnnl_graph_tensor_set_data_handle", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS));
            execute = library.bind("dnnl_graph_compiled_partition_execute_v2",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG, ADDRESS, ADDRESS));
            streamCreate = library.bind("dnnl_stream_create", FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
            streamDestroy = library.bind("dnnl_stream_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            streamWait = library.bind("dnnl_stream_wait", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        }
    }
}
