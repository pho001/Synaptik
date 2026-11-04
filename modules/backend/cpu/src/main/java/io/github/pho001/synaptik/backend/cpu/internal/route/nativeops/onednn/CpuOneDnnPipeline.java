package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.check;
import static io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.propagate;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn.CpuOneDnnLibrary.NativeScope;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.List;

/**
 * Native-ABI staging and ownership for a cold-fixed realization of one common unit. Each literal
 * site uses the existing atomic descriptor/compiler and invocation owner. This class only fixes
 * buffer slices, native controls and execute order; it never interprets semantic instructions.
 */
final class CpuOneDnnPipeline {
    private CpuOneDnnPipeline() { }

    static CpuOneDnnKernelMetadata describe(CpuOneDnnLibrary library, CpuOneDnnKernelSpec.Pipeline spec,
            CpuOneDnnRoutePlan.Api api) {
        var metadata = new ArrayList<CpuOneDnnKernelMetadata>(spec.steps().size());
        for (var step : spec.steps()) metadata.add(CpuOneDnnCompiler.describe(library, step.spec(), api));
        Layout layout = new Layout(spec, metadata, api);
        return new CpuOneDnnKernelMetadata(layout.bytes, layout.implementation);
    }

    static CpuOneDnnKernel compile(CpuOneDnnLibrary library, CpuOneDnnKernelSpec.Pipeline spec,
            CpuOneDnnRoutePlan.Api api) {
        var children = new ArrayList<CpuOneDnnAtomicKernel>(spec.steps().size());
        try {
            var metadata = new ArrayList<CpuOneDnnKernelMetadata>(spec.steps().size());
            for (var step : spec.steps()) {
                var child = (CpuOneDnnAtomicKernel) CpuOneDnnCompiler.compile(library, step.spec(), api);
                try { children.add(child); }
                catch (Throwable problem) {
                    try { child.close(); } catch (Throwable cleanup) { problem.addSuppressed(cleanup); }
                    throw problem;
                }
                metadata.add(new CpuOneDnnKernelMetadata(child.scratchpadBytes(), child.implementation()));
            }
            return new Kernel(library, spec, children.toArray(CpuOneDnnAtomicKernel[]::new),
                    new Layout(spec, metadata, api));
        } catch (Throwable problem) {
            for (int index = children.size() - 1; index >= 0; index--) {
                try { children.get(index).close(); } catch (Throwable cleanup) {
                    if (cleanup != problem) problem.addSuppressed(cleanup);
                }
            }
            throw propagate(problem);
        }
    }

    /** Exactly one charged allocation: aligned intermediates, immutable literals, atomic scratch. */
    private static final class Layout {
        private final long[] slotOffsets;
        private final long[] scratchOffsets;
        private final long bytes;
        private final String implementation;

        private Layout(CpuOneDnnKernelSpec.Pipeline spec, List<CpuOneDnnKernelMetadata> metadata,
                CpuOneDnnRoutePlan.Api api) {
            int internal = spec.intermediates().size() + spec.constants().size();
            slotOffsets = new long[internal];
            scratchOffsets = new long[metadata.size()];
            long next = 0;
            for (int index = 0; index < internal; index++) {
                next = aligned(next);
                slotOffsets[index] = next;
                long size = index < spec.intermediates().size() ? spec.intermediates().get(index).byteSize()
                        : spec.constants().get(index - spec.intermediates().size()).tensor().byteSize();
                next = Math.addExact(next, size);
            }
            StringBuilder identity = new StringBuilder("onednn-3.14.0-fixed-sites:").append(api);
            for (int index = 0; index < metadata.size(); index++) {
                next = aligned(next);
                scratchOffsets[index] = next;
                next = Math.addExact(next, metadata.get(index).scratchpadBytes());
                identity.append('|').append(metadata.get(index).implementation().length()).append(':')
                        .append(metadata.get(index).implementation());
            }
            bytes = next;
            implementation = identity.toString();
        }

        private static long aligned(long bytes) { return Math.addExact(bytes, 63) & ~63L; }
    }

    private static final class Kernel implements CpuOneDnnKernel {
        private final CpuOneDnnLibrary library;
        private final CpuOneDnnKernelSpec.Pipeline spec;
        private final CpuOneDnnAtomicKernel[] children;
        private final Layout layout;
        private final MethodHandle streamCreate, streamDestroy, streamWait;
        private int references = 1;
        private boolean closed;

        private Kernel(CpuOneDnnLibrary library, CpuOneDnnKernelSpec.Pipeline spec,
                CpuOneDnnAtomicKernel[] children, Layout layout) {
            this.library = library;
            this.spec = spec;
            this.children = children;
            this.layout = layout;
            streamCreate = library.bind("dnnl_stream_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT));
            streamDestroy = library.bind("dnnl_stream_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            streamWait = library.bind("dnnl_stream_wait", FunctionDescriptor.of(JAVA_INT, ADDRESS));
        }
        @Override public CpuOneDnnKernelSpec spec() { return spec; }
        @Override public long scratchpadBytes() { return layout.bytes; }
        @Override public String implementation() { return layout.implementation; }
        @Override public CpuOneDnnWorkspace createWorkspace() {
            synchronized (this) {
                if (closed) throw new IllegalStateException("oneDNN pipeline kernel is closed");
                references = Math.incrementExact(references);
            }
            try { return new Workspace(this); }
            catch (Throwable problem) {
                try { release(); } catch (Throwable cleanup) { if (cleanup != problem) problem.addSuppressed(cleanup); }
                throw propagate(problem);
            }
        }
        private void release() {
            synchronized (this) { if (--references != 0) return; }
            Throwable failure = null;
            for (int index = children.length - 1; index >= 0; index--) {
                try { children[index].close(); } catch (Throwable problem) {
                    if (failure == null) failure = problem;
                    else if (problem != failure) failure.addSuppressed(problem);
                }
            }
            if (failure != null) throw propagate(failure);
        }
        @Override public void close() {
            synchronized (this) { if (closed) return; closed = true; }
            release();
        }
    }

    private static final class Workspace extends CpuOneDnnWorkspace {
        private final Kernel owner;
        private NativeScope scope;
        private MemorySegment stream;
        private CpuOneDnnWorkspace[] children;
        private MemorySegment[] slots;
        private MemorySegment[][] stepInputs;
        private boolean ownsLease;
        private boolean bound;

        private Workspace(Kernel owner) {
            super(owner);
            this.owner = owner;
            try {
                var spec = owner.spec;
                scope = new NativeScope(owner.library);
                MemorySegment pointer = scope.arena().allocate(ADDRESS);
                int status = (int) owner.streamCreate.invokeExact(pointer, owner.library.engine(), 1);
                stream = pointer.get(ADDRESS, 0);
                if (stream.address() != 0) scope.own(stream, owner.streamDestroy);
                check(status, "create fixed-site pipeline stream");
                if (stream.address() == 0) throw new IllegalStateException("null fixed-site pipeline stream");
                int internalStart = spec.inputs().size() + 1;
                slots = new MemorySegment[internalStart + owner.layout.slotOffsets.length];
                for (int index = 0; index < owner.layout.slotOffsets.length; index++) {
                    long size = index < spec.intermediates().size() ? spec.intermediates().get(index).byteSize()
                            : spec.constants().get(index - spec.intermediates().size()).tensor().byteSize();
                    slots[internalStart + index] = writableSegment().asSlice(owner.layout.slotOffsets[index], size);
                }
                for (int index = 0; index < spec.constants().size(); index++) {
                    slots[internalStart + spec.intermediates().size() + index]
                            .set(JAVA_INT, 0, spec.constants().get(index).rawWords().getFirst());
                }
                children = new CpuOneDnnWorkspace[owner.children.length];
                stepInputs = new MemorySegment[children.length][];
                for (int index = 0; index < children.length; index++) {
                    children[index] = owner.children[index].createWorkspace(writableSegment().asSlice(
                            owner.layout.scratchOffsets[index], owner.children[index].scratchpadBytes()), stream);
                    stepInputs[index] = new MemorySegment[spec.steps().get(index).inputSlots().size()];
                }
                ownsLease = true;
            } catch (Throwable problem) {
                try { close(); } catch (Throwable cleanup) { if (cleanup != problem) problem.addSuppressed(cleanup); }
                throw propagate(problem);
            }
        }
        @Override public void bind(MemorySegment[] inputs, MemorySegment output) {
            requireOpen();
            bound = false;
            validateBindings(inputs, output);
            for (int index = 0; index < inputs.length; index++) slots[index] = inputs[index];
            slots[inputs.length] = output;
            for (int index = 0; index < children.length; index++) {
                var step = owner.spec.steps().get(index);
                for (int operand = 0; operand < stepInputs[index].length; operand++) {
                    stepInputs[index][operand] = slots[step.inputSlots().get(operand)];
                }
                children[index].bind(stepInputs[index], slots[step.outputSlot()]);
            }
            bound = true;
        }
        @Override public void execute() {
            requireOpen();
            if (!bound) throw new IllegalStateException("oneDNN pipeline workspace is not bound");
            try { enqueue(); } finally { await(); }
        }
        @Override void enqueue() {
            requireOpen();
            if (!bound) throw new IllegalStateException("oneDNN pipeline workspace is not bound");
            for (int index = 0; index < children.length; index++) children[index].enqueue();
        }
        @Override void await() {
            requireOpen();
            try { check((int) owner.streamWait.invokeExact(stream), "wait fixed-site pipeline stream"); }
            catch (Throwable problem) { throw propagate(problem); }
        }
        @Override protected void closeNative() {
            Throwable failure = null;
            if (children != null) {
                for (int index = children.length - 1; index >= 0; index--) {
                    try { if (children[index] != null) children[index].close(); } catch (Throwable problem) {
                        if (failure == null) failure = problem;
                        else if (problem != failure) failure.addSuppressed(problem);
                    }
                }
            }
            try { if (scope != null) scope.close(); } catch (Throwable problem) {
                if (failure == null) failure = problem;
                else if (problem != failure) failure.addSuppressed(problem);
            }
            slots = null;
            stepInputs = null;
            bound = false;
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
}
