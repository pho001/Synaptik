package io.github.pho001.synaptik.backend.provider.openblas;

import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Owns one explicitly loaded OpenBLAS library lookup and its required native bindings.
 *
 * <p>Each successful {@link #open(String)} or {@link #open(Path)} call creates a distinct Java
 * lifetime owner. The owner does not discover a library, choose a platform name, select a CPU
 * route, or provide fallback behavior. Closing it ends only this owner's Foreign Function and
 * Memory lookup lifetime; the operating system may retain the underlying library.
 *
 * <p>The immutable bindings use a shared arena so they may be consumed concurrently by later
 * package-colocated provider operations while this owner remains open. Callers must not race
 * {@link #close()} with such an operation. Lifecycle observation and closing are thread-safe.
 */
public final class OpenBlasLibrary implements AutoCloseable {
    private static final OpenBlasNativeAccess NATIVE_ACCESS = new FfmOpenBlasNativeAccess();

    private final OpenBlasNativeBindings bindings;
    private final AtomicBoolean open = new AtomicBoolean(true);

    /**
     * Creates one open lifetime owner around a complete binding set.
     *
     * @param bindings the complete native bindings owned by this handle; must not be {@code null}
     * @throws NullPointerException if {@code bindings} is {@code null}
     */
    private OpenBlasLibrary(OpenBlasNativeBindings bindings) {
        this.bindings = Objects.requireNonNull(bindings, "bindings");
    }

    /**
     * Loads exactly the caller-specified operating-system library name and binds all required
     * OpenBLAS symbols.
     *
     * @param libraryName the nonblank library name passed unchanged to the operating-system
     *                    library lookup; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if {@code libraryName} is {@code null}
     * @throws IllegalArgumentException if {@code libraryName} is blank
     * @throws OpenBlasLoadException if loading, symbol resolution, or native handle binding fails
     */
    public static OpenBlasLibrary open(String libraryName) {
        return open(libraryName, NATIVE_ACCESS);
    }

    /**
     * Loads exactly the caller-specified absolute library path and binds all required OpenBLAS
     * symbols.
     *
     * @param absoluteLibraryPath the absolute library path passed unchanged to the operating-
     *                            system library lookup; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if {@code absoluteLibraryPath} is not absolute
     * @throws OpenBlasLoadException if loading, symbol resolution, or native handle binding fails
     */
    public static OpenBlasLibrary open(Path absoluteLibraryPath) {
        return open(absoluteLibraryPath, NATIVE_ACCESS);
    }

    /**
     * Loads a validated library name through the deterministic package-private native seam.
     *
     * @param libraryName the nonblank caller-selected library name; must not be {@code null}
     * @param nativeAccess the exact native access implementation; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if {@code libraryName} is blank
     * @throws OpenBlasLoadException if native access fails or returns no bindings
     */
    static OpenBlasLibrary open(String libraryName, OpenBlasNativeAccess nativeAccess) {
        Objects.requireNonNull(libraryName, "libraryName");
        Objects.requireNonNull(nativeAccess, "nativeAccess");
        if (libraryName.isBlank()) {
            throw new IllegalArgumentException("libraryName must not be blank");
        }
        return load("name '" + libraryName + "'", () -> nativeAccess.open(libraryName));
    }

    /**
     * Loads a validated absolute path through the deterministic package-private native seam.
     *
     * @param absoluteLibraryPath the absolute caller-selected path; must not be {@code null}
     * @param nativeAccess the exact native access implementation; must not be {@code null}
     * @return a new open caller-owned library handle; never {@code null}
     * @throws NullPointerException if either argument is {@code null}
     * @throws IllegalArgumentException if {@code absoluteLibraryPath} is not absolute
     * @throws OpenBlasLoadException if native access fails or returns no bindings
     */
    static OpenBlasLibrary open(Path absoluteLibraryPath, OpenBlasNativeAccess nativeAccess) {
        Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath");
        Objects.requireNonNull(nativeAccess, "nativeAccess");
        if (!absoluteLibraryPath.isAbsolute()) {
            throw new IllegalArgumentException("absoluteLibraryPath must be absolute");
        }
        return load("path '" + absoluteLibraryPath + "'", () -> nativeAccess.open(absoluteLibraryPath));
    }

    /**
     * Reports whether this Java lifetime owner has not yet been closed.
     *
     * <p>This local observation performs no symbol resolution or operating-system availability
     * check and cannot by itself coordinate a later native call with concurrent closure.
     *
     * @return {@code true} until the first close attempt claims this owner, otherwise {@code false}
     */
    public boolean isOpen() {
        return open.get();
    }

    /**
     * Ends this owner's native lookup lifetime once.
     *
     * <p>Repeated and concurrent calls are idempotent. If arena closure fails, this owner remains
     * observably closed and the unchecked cleanup failure is propagated to the winning caller.
     *
     * @throws IllegalStateException if the underlying shared arena cannot be closed
     */
    @Override
    public void close() {
        if (open.compareAndSet(true, false)) {
            bindings.close();
        }
    }

    /**
     * Returns the complete bindings for later package-colocated provider operations.
     *
     * @return the exact binding set retained by this owner; never {@code null}
     * @throws IllegalStateException if this owner has been closed
     */
    OpenBlasNativeBindings bindings() {
        if (!open.get()) {
            throw new IllegalStateException("OpenBLAS library is closed");
        }
        return bindings;
    }

    /**
     * Translates one already validated native loading attempt to the stable public failure type.
     *
     * @param description the caller selection included in failure diagnostics; must not be
     *                    {@code null}
     * @param loader the exact loading operation; must not be {@code null}
     * @return a new lifetime owner for the complete returned bindings; never {@code null}
     * @throws OpenBlasLoadException if loading fails or returns {@code null}
     */
    private static OpenBlasLibrary load(String description, BindingLoader loader) {
        Objects.requireNonNull(description, "description");
        Objects.requireNonNull(loader, "loader");
        try {
            OpenBlasNativeBindings bindings = loader.load();
            if (bindings == null) {
                throw new NullPointerException("nativeAccess returned null bindings");
            }
            return new OpenBlasLibrary(bindings);
        } catch (Throwable cause) {
            throw new OpenBlasLoadException("Failed to load OpenBLAS library " + description, cause);
        }
    }

    /** Loads one complete binding set for the shared public failure-translation boundary. */
    @FunctionalInterface
    private interface BindingLoader {
        /**
         * Loads one complete native binding set.
         *
         * @return the complete binding set; never {@code null}
         * @throws RuntimeException if native loading or binding fails
         */
        OpenBlasNativeBindings load();
    }
}
