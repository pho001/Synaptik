package io.github.pho001.synaptik.backend.provider.openblas;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Loads and binds OpenBLAS through the JDK Foreign Function and Memory API.
 *
 * <p>All integer parameters use the native C {@code int} layout, selecting the ordinary
 * 32-bit-{@code blasint} OpenBLAS C interface. This implementation binds symbols but never invokes
 * them. Any partial lookup lifetime is closed before a loading or binding failure escapes.
 */
final class FfmOpenBlasNativeAccess implements OpenBlasNativeAccess {
    private static final String SGEMM_SYMBOL = "cblas_sgemm";
    private static final String DGEMM_SYMBOL = "cblas_dgemm";
    private static final String SET_NUM_THREADS_SYMBOL = "openblas_set_num_threads";
    private static final String GET_NUM_THREADS_SYMBOL = "openblas_get_num_threads";
    private static final List<String> REQUIRED_SYMBOLS = List.of(
            SGEMM_SYMBOL, DGEMM_SYMBOL, SET_NUM_THREADS_SYMBOL, GET_NUM_THREADS_SYMBOL);

    private static final FunctionDescriptor SGEMM_DESCRIPTOR = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
            JAVA_FLOAT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_FLOAT, ADDRESS, JAVA_INT);
    private static final FunctionDescriptor DGEMM_DESCRIPTOR = FunctionDescriptor.ofVoid(
            JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT,
            JAVA_DOUBLE, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, JAVA_DOUBLE, ADDRESS, JAVA_INT);
    private static final FunctionDescriptor SET_NUM_THREADS_DESCRIPTOR =
            FunctionDescriptor.ofVoid(JAVA_INT);
    private static final FunctionDescriptor GET_NUM_THREADS_DESCRIPTOR =
            FunctionDescriptor.of(JAVA_INT);

    /** Creates a stateless production native-access implementation without loading a library. */
    FfmOpenBlasNativeAccess() {
    }

    /**
     * Loads and completely binds the exact supplied library name.
     *
     * @param libraryName the already validated nonblank library name; never {@code null}
     * @return the complete binding set owning its shared arena; never {@code null}
     * @throws RuntimeException if lookup, symbol resolution, or handle binding fails
     */
    @Override
    public OpenBlasNativeBindings open(String libraryName) {
        Objects.requireNonNull(libraryName, "libraryName");
        return open(arena -> SymbolLookup.libraryLookup(libraryName, arena));
    }

    /**
     * Loads and completely binds the exact supplied absolute library path.
     *
     * @param absoluteLibraryPath the already validated absolute library path; never {@code null}
     * @return the complete binding set owning its shared arena; never {@code null}
     * @throws RuntimeException if lookup, symbol resolution, or handle binding fails
     */
    @Override
    public OpenBlasNativeBindings open(Path absoluteLibraryPath) {
        Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath");
        return open(arena -> SymbolLookup.libraryLookup(absoluteLibraryPath, arena));
    }

    /**
     * Owns the partial shared-arena lifecycle while one lookup is resolved and bound.
     *
     * @param lookupFactory the exact caller-selected name or path lookup operation; must not be
     *                      {@code null}
     * @return the complete bindings that take ownership of the shared arena; never {@code null}
     * @throws RuntimeException if lookup, symbol resolution, or handle binding fails; a cleanup
     *                          failure is suppressed on the primary failure
     */
    private static OpenBlasNativeBindings open(Function<Arena, SymbolLookup> lookupFactory) {
        Objects.requireNonNull(lookupFactory, "lookupFactory");
        Arena arena = Arena.ofShared();
        try {
            SymbolLookup lookup = lookupFactory.apply(arena);
            Linker linker = Linker.nativeLinker();
            List<MemorySegment> addresses = resolveRequiredSymbols(lookup);
            MethodHandle sgemm = linker.downcallHandle(addresses.get(0), SGEMM_DESCRIPTOR);
            MethodHandle dgemm = linker.downcallHandle(addresses.get(1), DGEMM_DESCRIPTOR);
            MethodHandle setNumThreads =
                    linker.downcallHandle(addresses.get(2), SET_NUM_THREADS_DESCRIPTOR);
            MethodHandle getNumThreads =
                    linker.downcallHandle(addresses.get(3), GET_NUM_THREADS_DESCRIPTOR);
            return new OpenBlasNativeBindings(
                    arena, sgemm, dgemm, setNumThreads, getNumThreads);
        } catch (RuntimeException | Error failure) {
            try {
                arena.close();
            } catch (RuntimeException | Error cleanupFailure) {
                if (cleanupFailure != failure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Resolves every required symbol and reports all absences in required-list order.
     *
     * @param lookup the exact library lookup; must not be {@code null}
     * @return the four resolved addresses in required-symbol order; never {@code null}
     * @throws IllegalStateException if one or more required symbols are absent
     */
    private static List<MemorySegment> resolveRequiredSymbols(SymbolLookup lookup) {
        Objects.requireNonNull(lookup, "lookup");
        List<MemorySegment> addresses = new ArrayList<>(REQUIRED_SYMBOLS.size());
        List<String> missing = new ArrayList<>();
        for (String symbol : REQUIRED_SYMBOLS) {
            MemorySegment address = lookup.find(symbol).orElse(null);
            addresses.add(address);
            if (address == null) {
                missing.add(symbol);
            }
        }
        if (!missing.isEmpty()) {
            throw new IllegalStateException("Missing required OpenBLAS symbols: " + String.join(", ", missing));
        }
        return List.copyOf(addresses);
    }
}
