package backend.provider.blas.openblas;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * OpenBLAS CBLAS symbol table backed by Java FFM.
 */
final class OpenBlasSymbols {
    static final int CBLAS_ROW_MAJOR = 101;
    static final int CBLAS_NO_TRANS = 111;

    private static final OpenBlasSymbols INSTANCE = init();

    final boolean available;
    final String reason;
    final LookupSource source;
    @SuppressWarnings("unused")
    final Arena arenaRef;
    final MethodHandle sgemm;
    final MethodHandle dgemm;
    final MethodHandle sbgemm;
    final MethodHandle bgemm;
    final MethodHandle getNumThreads;
    final MethodHandle setNumThreads;
    final MethodHandle getParallel;

    private OpenBlasSymbols(
            boolean available,
            String reason,
            LookupSource source,
            Arena arenaRef,
            MethodHandle sgemm,
            MethodHandle dgemm,
            MethodHandle sbgemm,
            MethodHandle bgemm,
            MethodHandle getNumThreads,
            MethodHandle setNumThreads,
            MethodHandle getParallel
    ) {
        this.available = available;
        this.reason = reason;
        this.source = source;
        this.arenaRef = arenaRef;
        this.sgemm = sgemm;
        this.dgemm = dgemm;
        this.sbgemm = sbgemm;
        this.bgemm = bgemm;
        this.getNumThreads = getNumThreads;
        this.setNumThreads = setNumThreads;
        this.getParallel = getParallel;
    }

    static OpenBlasSymbols get() {
        return INSTANCE;
    }

    private static OpenBlasSymbols init() {
        try {
            Arena arena = Arena.ofShared();
            LookupResolution lookupResolution = resolveLookup(arena);
            SymbolLookup lookup = lookupResolution.lookup();
            Linker linker = Linker.nativeLinker();

            MethodHandle sgemm = linker.downcallHandle(
                    lookup.find("cblas_sgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT
                    )
            );

            MethodHandle dgemm = linker.downcallHandle(
                    lookup.find("cblas_dgemm").orElseThrow(),
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_DOUBLE,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_DOUBLE,
                            ADDRESS, JAVA_INT
                    )
            );

            MethodHandle sbgemm = optionalSbgemm(linker, lookup);
            MethodHandle bgemm = optionalBgemm(linker, lookup);
            MethodHandle getNumThreads = optionalIntReturn(linker, lookup, "openblas_get_num_threads");
            MethodHandle setNumThreads = optionalSetNumThreads(linker, lookup);
            MethodHandle getParallel = optionalIntReturn(linker, lookup, "openblas_get_parallel");

            return new OpenBlasSymbols(
                    true,
                    null,
                    lookupResolution.source(),
                    arena,
                    sgemm,
                    dgemm,
                    sbgemm,
                    bgemm,
                    getNumThreads,
                    setNumThreads,
                    getParallel
            );
        } catch (Throwable t) {
            return new OpenBlasSymbols(
                    false,
                    t.getClass().getSimpleName() + ": " + safeMessage(t),
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null
            );
        }
    }

    private static MethodHandle optionalSbgemm(Linker linker, SymbolLookup lookup) {
        try {
            MemorySegment sbgemmSym = lookup.find("cblas_sbgemm").orElse(null);
            if (sbgemmSym == null) {
                return null;
            }
            return linker.downcallHandle(
                    sbgemmSym,
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_FLOAT,
                            ADDRESS, JAVA_INT
                    )
            );
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static MethodHandle optionalBgemm(Linker linker, SymbolLookup lookup) {
        try {
            MemorySegment bgemmSym = lookup.find("cblas_bgemm").orElse(null);
            if (bgemmSym == null) {
                return null;
            }
            return linker.downcallHandle(
                    bgemmSym,
                    FunctionDescriptor.ofVoid(
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_INT, JAVA_INT, JAVA_INT,
                            JAVA_SHORT,
                            ADDRESS, JAVA_INT,
                            ADDRESS, JAVA_INT,
                            JAVA_SHORT,
                            ADDRESS, JAVA_INT
                    )
            );
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static MethodHandle optionalIntReturn(Linker linker, SymbolLookup lookup, String symbolName) {
        try {
            MemorySegment symbol = lookup.find(symbolName).orElse(null);
            if (symbol == null) {
                return null;
            }
            return linker.downcallHandle(symbol, FunctionDescriptor.of(JAVA_INT));
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static MethodHandle optionalSetNumThreads(Linker linker, SymbolLookup lookup) {
        try {
            MemorySegment symbol = lookup.find("openblas_set_num_threads").orElse(null);
            if (symbol == null) {
                return null;
            }
            return linker.downcallHandle(symbol, FunctionDescriptor.ofVoid(JAVA_INT));
        } catch (Throwable ignored) {
            return null;
        }
    }

    private static LookupResolution resolveLookup(Arena arena) {
        String explicit = System.getProperty("openblas.lib");
        if (explicit != null && !explicit.isBlank()) {
            return new LookupResolution(
                    SymbolLookup.libraryLookup(explicit.trim(), arena),
                    LookupSource.EXPLICIT_PROPERTY
            );
        }

        String environment = System.getenv("OPENBLAS_LIB");
        if (environment != null && !environment.isBlank()) {
            return new LookupResolution(
                    SymbolLookup.libraryLookup(environment.trim(), arena),
                    LookupSource.ENVIRONMENT
            );
        }

        return new LookupResolution(
                SymbolLookup.libraryLookup("openblas", arena),
                LookupSource.SYSTEM_LIBRARY
        );
    }

    private static String safeMessage(Throwable t) {
        String m = t.getMessage();
        return m == null ? "<no-message>" : m;
    }

    enum LookupSource {
        EXPLICIT_PROPERTY,
        ENVIRONMENT,
        SYSTEM_LIBRARY
    }

    private record LookupResolution(SymbolLookup lookup, LookupSource source) {
    }
}
