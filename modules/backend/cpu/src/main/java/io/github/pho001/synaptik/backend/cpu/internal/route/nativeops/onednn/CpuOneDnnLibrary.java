package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Objects;
import java.util.UUID;

/**
 * Exact oneDNN native-kernel-library 3.14.0 little-endian LP64 C ABI and sequential (SEQ) CPU lifetime.
 * The integration owns the initial lease. Native scopes retain independent leases, so owner
 * closure prevents new admission without unloading code still needed by issued resources.
 * Foreign Function and Memory (FFM) symbol lookup and ABI adaptation are cold operations,
 * never invocation policy.
 */
public final class CpuOneDnnLibrary implements AutoCloseable {
    private final Arena arena;
    private final SymbolLookup symbols;
    private final Linker linker;
    private final MethodHandle destroyEngine;
    private final MemorySegment engine;
    private final Path path;
    private final String sessionIdentity = UUID.randomUUID().toString();
    private int references = 1;
    private boolean closed;

    /**
     * Loads the exact path and authenticates version/runtime before creating a CPU engine.
     *
     * @param path non-null absolute shared-library path; never discovered or substituted
     * @throws IllegalArgumentException if path, target ABI, version, runtime or symbols are invalid
     * @throws RuntimeException if loading or native engine construction/rollback fails
     */
    private CpuOneDnnLibrary(Path path) {
        this.path = path;
        if (!path.isAbsolute() || ADDRESS.byteSize() != Long.BYTES
                || ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
            throw new IllegalArgumentException("oneDNN requires an absolute library path and a little-endian LP64 target");
        }
        arena = Arena.ofShared();
        MemorySegment createdEngine = MemorySegment.NULL;
        MethodHandle engineDestroy = null;
        try {
            symbols = SymbolLookup.libraryLookup(path, arena);
            linker = Linker.nativeLinker();
            MethodHandle versionHandle = bind("dnnl_version", FunctionDescriptor.of(ADDRESS));
            MemorySegment version = ((MemorySegment) versionHandle.invokeExact()).reinterpret(32);
            int major = version.get(JAVA_INT, 0), minor = version.get(JAVA_INT, 4);
            int patch = version.get(JAVA_INT, 8), runtime = version.get(JAVA_INT, 24);
            if (major != 3 || minor != 14 || patch != 0 || runtime != 1) {
                throw new IllegalArgumentException("oneDNN requires ABI 3.14.0 and SEQ CPU runtime; found "
                        + major + '.' + minor + '.' + patch + " runtime=" + runtime);
            }
            engineDestroy = bind("dnnl_engine_destroy", FunctionDescriptor.of(JAVA_INT, ADDRESS));
            MethodHandle create = bind("dnnl_engine_create",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, JAVA_LONG));
            try (Arena outputs = Arena.ofConfined()) {
                MemorySegment pointer = outputs.allocate(ADDRESS);
                check((int) create.invokeExact(pointer, 1, 0L), "create CPU engine");
                createdEngine = pointer.get(ADDRESS, 0);
                if (createdEngine.address() == 0) throw new IllegalStateException("oneDNN returned a null CPU engine");
            }
            destroyEngine = engineDestroy;
            engine = createdEngine;
        } catch (Throwable problem) {
            if (createdEngine.address() != 0 && engineDestroy != null) {
                try { check((int) engineDestroy.invokeExact(createdEngine), "destroy partial CPU engine"); }
                catch (Throwable cleanup) { if (cleanup != problem) problem.addSuppressed(cleanup); }
            }
            try { arena.close(); } catch (Throwable cleanup) {
                if (cleanup != problem) problem.addSuppressed(cleanup);
            }
            throw propagate(problem);
        }
    }

    /**
     * Opens a required native library; failure never requests portable fallback.
     *
     * @param absolutePath non-null exact absolute path to oneDNN 3.14.0 LP64 SEQ
     * @return new non-null caller-owned library/engine owner; close releases its initial lease
     * @throws NullPointerException if absolutePath is null
     * @throws IllegalArgumentException if path, target ABI, version, runtime or symbols are invalid
     * @throws RuntimeException if required loading or native construction fails
     */
    public static CpuOneDnnLibrary open(Path absolutePath) {
        return new CpuOneDnnLibrary(Objects.requireNonNull(absolutePath, "absolutePath"));
    }

    /**
     * Returns the caller-supplied library identity without resolving another path.
     *
     * @return non-null immutable absolute path; no native lifetime transfers
     */
    public Path path() { return path; }
    /**
     * Identifies this exact load for analysis and tuning replay.
     *
     * @return non-null process-local opaque session identity, not a persistent binary identity
     */
    public String sessionIdentity() { return sessionIdentity; }
    /**
     * Borrows the shared CPU engine under an existing library or native-scope lease.
     *
     * @return non-null native engine handle; callers must not destroy it or outlive their lease
     */
    public MemorySegment engine() { return engine; }
    /**
     * Reports whether the integration owner still admits new work.
     *
     * @return true until owner close; issued scopes may remain alive after false is returned
     */
    public synchronized boolean isOpen() { return !closed; }

    /**
     * Binds one exact C symbol during cold preparation; hot paths retain the typed handle.
     *
     * @param symbol non-null required C symbol name
     * @param descriptor non-null exact function ABI descriptor
     * @return non-null downcall handle borrowing this library's native lifetime
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the symbol is missing or the descriptor is invalid
     * @throws IllegalStateException if the lookup lifetime has been released
     */
    public MethodHandle bind(String symbol, FunctionDescriptor descriptor) {
        MemorySegment address = symbols.find(symbol).orElseThrow(() ->
                new IllegalArgumentException("oneDNN 3.14.0 is missing required symbol " + symbol));
        return linker.downcallHandle(address, descriptor);
    }

    /**
     * Rejects new work after owner closure without changing issued native-scope leases.
     *
     * @throws IllegalStateException if this integration's library owner is closed
     */
    public synchronized void requireOpen() {
        if (closed) throw new IllegalStateException("oneDNN library is closed");
    }

    private synchronized void retain() {
        if (references == 0) throw new IllegalStateException("oneDNN library has been released");
        references = Math.incrementExact(references);
    }

    private void release() {
        synchronized (this) {
            if (--references != 0) return;
        }
        Throwable failure = null;
        try { check((int) destroyEngine.invokeExact(engine), "destroy CPU engine"); }
        catch (Throwable problem) { failure = problem; }
        try { arena.close(); } catch (Throwable problem) {
            if (failure == null) failure = problem;
            else if (problem != failure) failure.addSuppressed(problem);
        }
        if (failure != null) throw propagate(failure);
    }

    /**
     * Idempotently stops new owner admission and releases the initial library lease.
     * The engine and lookup arena close only after the last retained scope releases its lease.
     *
     * @throws IllegalStateException if final native destruction or arena cleanup fails
     */
    @Override public void close() {
        synchronized (this) {
            if (closed) return;
            closed = true;
        }
        release();
    }

    /**
     * Converts a non-success C status into a status-preserving native failure.
     *
     * @param status exact oneDNN status code; zero means success
     * @param operation diagnostic operation description, not a selection instruction
     * @throws NativeStatusException if status is nonzero
     */
    public static void check(int status, String operation) {
        if (status != 0) throw new NativeStatusException(operation, status);
    }

    /**
     * Preserves runtime/fatal failures and wraps checked foreign-call failures.
     *
     * @param problem original failure whose identity/cause must be retained
     * @return the same runtime exception or a new IllegalStateException wrapping a checked failure
     * @throws Error if problem is an Error, rethrown unchanged
     */
    public static RuntimeException propagate(Throwable problem) {
        if (problem instanceof RuntimeException failure) return failure;
        if (problem instanceof Error failure) throw failure;
        return new IllegalStateException("oneDNN foreign call failed", problem);
    }

    /** Native status failure retaining the exact C status for cold eligibility classification. */
    public static final class NativeStatusException extends IllegalStateException {
        private final int status;
        /**
         * Captures one failed native operation without retrying or translating its status.
         *
         * @param operation diagnostic operation description
         * @param status exact nonzero C status
         */
        private NativeStatusException(String operation, int status) {
            super("oneDNN " + operation + " failed with status " + status);
            this.status = status;
        }
        /**
         * Returns the original native status independently of diagnostic message text.
         *
         * @return exact C status code supplied by oneDNN
         */
        public int status() { return status; }
    }

    /** Transactional native objects and their ABI memory; closed in reverse acquisition order. */
    public static final class NativeScope implements AutoCloseable {
        private record OwnedHandle(MemorySegment handle, MethodHandle destroy) { }
        private final CpuOneDnnLibrary library;
        private final Arena arena;
        private final ArrayList<OwnedHandle> handles = new ArrayList<>();
        private boolean closed;

        /**
         * Acquires a library lease and an ABI-memory arena for one transactional native owner.
         *
         * @param library non-null library whose code/engine must outlive all registered handles
         * @throws NullPointerException if library is null
         * @throws IllegalStateException if the last library lease has already been released
         * @throws ArithmeticException if reference counting overflows
         */
        public NativeScope(CpuOneDnnLibrary library) {
            this.library = Objects.requireNonNull(library, "library");
            library.retain();
            try { arena = Arena.ofShared(); }
            catch (RuntimeException | Error problem) { library.release(); throw problem; }
        }
        /**
         * Borrows storage for native descriptors and argument tables.
         *
         * @return non-null shared arena owned and closed by this scope; never close it separately
         */
        public Arena arena() { return arena; }
        /**
         * Transfers one newly created native handle into reverse-order cleanup.
         *
         * @param handle non-null nonzero C handle whose ownership transfers to this scope
         * @param destroy non-null exact destructor accepting the handle and returning C status
         * @return the identical borrowed handle; no second ownership is created
         * @throws NullPointerException if an argument is null
         * @throws IllegalArgumentException if handle is the null native address
         */
        public MemorySegment own(MemorySegment handle, MethodHandle destroy) {
            Objects.requireNonNull(handle, "handle");
            Objects.requireNonNull(destroy, "destroy");
            if (handle.address() == 0) throw new IllegalArgumentException("cannot own a null oneDNN handle");
            try { handles.add(new OwnedHandle(handle, destroy)); }
            catch (RuntimeException | Error problem) {
                try { check((int) destroy.invokeExact(handle), "destroy unregistered handle"); }
                catch (Throwable cleanup) { if (cleanup != problem) problem.addSuppressed(cleanup); }
                throw problem;
            }
            return handle;
        }
        /**
         * Idempotently destroys all registered handles in reverse order, closes ABI memory and
         * releases the library lease, attempting every cleanup and suppressing secondary failures.
         *
         * @throws IllegalStateException if native destruction or memory cleanup fails
         */
        @Override public void close() {
            if (closed) return;
            closed = true;
            Throwable failure = null;
            for (int index = handles.size() - 1; index >= 0; index--) {
                OwnedHandle entry = handles.get(index);
                try { check((int) entry.destroy().invokeExact(entry.handle()), "destroy native object"); }
                catch (Throwable problem) {
                    if (failure == null) failure = problem;
                    else if (problem != failure) failure.addSuppressed(problem);
                }
            }
            handles.clear();
            try { arena.close(); } catch (Throwable problem) {
                if (failure == null) failure = problem;
                else if (problem != failure) failure.addSuppressed(problem);
            }
            try { library.release(); } catch (Throwable problem) {
                if (failure == null) failure = problem;
                else if (problem != failure) failure.addSuppressed(problem);
            }
            if (failure != null) throw propagate(failure);
        }
    }
}
