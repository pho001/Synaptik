package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLibrary;
import io.github.pho001.synaptik.backend.provider.openblas.OpenBlasLoadException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Field-free cold OpenBLAS discovery and exact-loader delegation owned by the CPU backend.
 *
 * <p>Discovery evaluates only the explicit request and the fixed automatic candidate table. A
 * successful load means only that the provider opened and bound its required symbols; this type
 * neither qualifies nor enables a CPU route and never queries or changes provider thread state.</p>
 */
final class CpuOpenBlasDiscovery {
    /** Prevents construction of the field-free discovery namespace. */
    private CpuOpenBlasDiscovery() { }

    /**
     * Performs one cold discovery using the production platform properties and exact provider
     * factories.
     *
     * @param request the immutable discovery intent
     * @return a caller-owned session containing immutable diagnostics and, on success, one
     *     borrowed invocation
     * @throws NullPointerException if {@code request} is {@code null}
     * @throws RuntimeException if platform lookup or an unrelated loading/adaptation defect fails
     * @throws Error if platform lookup, adaptation, cleanup, or a fatal provider cause fails
     */
    static CpuOpenBlasDiscoverySession discover(CpuOpenBlasDiscoveryRequest request) {
        return discover(request, new PlatformProperties() {
            @Override public String operatingSystem() {
                return System.getProperty("os.name", "");
            }

            @Override public String architecture() {
                return System.getProperty("os.arch", "");
            }
        }, CpuOpenBlasDiscovery::loadProductionResource);
    }

    /**
     * Performs deterministic discovery through injected cold platform and loader seams.
     *
     * @param request the immutable discovery intent
     * @param platformProperties the cold platform source, read only for automatic mode
     * @param loader the exact-selection loader
     * @return a caller-owned discovery session; never {@code null}
     * @throws NullPointerException if an argument or successful loader result is {@code null}
     * @throws RuntimeException if an unrelated platform, loader, adaptation, or result defect fails
     * @throws Error if the injected operations fail with an error
     */
    static CpuOpenBlasDiscoverySession discover(CpuOpenBlasDiscoveryRequest request,
            PlatformProperties platformProperties, Loader loader) {
        Objects.requireNonNull(request, "request");
        Objects.requireNonNull(platformProperties, "platformProperties");
        Objects.requireNonNull(loader, "loader");
        if (request.mode() == CpuOpenBlasDiscoveryRequest.Mode.DISABLED) {
            return new CpuOpenBlasDiscoverySession(new CpuOpenBlasDiscoveryResult(
                    request.mode(), Optional.empty(), List.of(), Optional.empty(),
                    CpuOpenBlasDiscoveryResult.Status.DISABLED), Optional.empty());
        }

        Optional<CpuOpenBlasDiscoveryResult.PlatformSnapshot> platformSnapshot = Optional.empty();
        List<CpuOpenBlasDiscoveryResult.Selection> candidates;
        switch (request.mode()) {
            case AUTOMATIC -> {
                String operatingSystem = platformProperties.operatingSystem();
                String architecture = platformProperties.architecture();
                var snapshot = CpuOpenBlasDiscoveryResult.PlatformSnapshot.from(
                        operatingSystem, architecture);
                platformSnapshot = Optional.of(snapshot);
                candidates = automaticCandidates(snapshot);
            }
            case EXACT_NAME -> candidates = List.of(new CpuOpenBlasDiscoveryResult.LibraryName(
                    request.exactName().orElseThrow()));
            case EXACT_ABSOLUTE_PATH -> candidates = List.of(
                    new CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath(
                            request.exactAbsolutePath().orElseThrow()));
            case DISABLED -> throw new AssertionError("disabled discovery returned above");
            default -> throw new AssertionError("unhandled discovery mode");
        }

        var attempts = new ArrayList<CpuOpenBlasDiscoveryResult.Attempt>();
        for (CpuOpenBlasDiscoveryResult.Selection candidate : candidates) {
            int position = attempts.size();
            CpuOpenBlasDiscoverySession.OwnedResource resource;
            try {
                resource = Objects.requireNonNull(loader.load(candidate),
                        "loader returned null owned resource");
            } catch (LoadFailure failure) {
                attempts.add(CpuOpenBlasDiscoveryResult.Attempt.failed(position, candidate,
                        failure.failureType(), failure.failureMessage()));
                continue;
            }
            attempts.add(CpuOpenBlasDiscoveryResult.Attempt.loaded(position, candidate));
            try {
                var result = new CpuOpenBlasDiscoveryResult(request.mode(), platformSnapshot,
                        attempts, Optional.of(candidate),
                        CpuOpenBlasDiscoveryResult.Status.LOADED);
                return new CpuOpenBlasDiscoverySession(result, Optional.of(resource));
            } catch (RuntimeException | Error primary) {
                closeAfterFailure(resource, primary);
                throw primary;
            }
        }
        var result = new CpuOpenBlasDiscoveryResult(request.mode(), platformSnapshot, attempts,
                Optional.empty(), CpuOpenBlasDiscoveryResult.Status.UNAVAILABLE);
        return new CpuOpenBlasDiscoverySession(result, Optional.empty());
    }

    /**
     * Returns the fixed ordered automatic candidates for one normalized platform snapshot.
     *
     * @param snapshot the immutable normalized platform facts
     * @return an immutable nonempty candidate list containing no duplicate typed selection
     * @throws NullPointerException if {@code snapshot} is {@code null}
     */
    static List<CpuOpenBlasDiscoveryResult.Selection> automaticCandidates(
            CpuOpenBlasDiscoveryResult.PlatformSnapshot snapshot) {
        Objects.requireNonNull(snapshot, "snapshot");
        var candidates = new ArrayList<CpuOpenBlasDiscoveryResult.Selection>();
        candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("openblas"));
        switch (snapshot.operatingSystemFamily()) {
            case MACOS -> {
                candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("libopenblas.dylib"));
                boolean arm = snapshot.normalizedArchitecture().equals("aarch64")
                        || snapshot.normalizedArchitecture().equals("arm64");
                candidates.add(new CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath(Path.of(arm
                        ? "/opt/homebrew/opt/openblas/lib/libopenblas.dylib"
                        : "/usr/local/opt/openblas/lib/libopenblas.dylib")));
                candidates.add(new CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath(Path.of(arm
                        ? "/usr/local/opt/openblas/lib/libopenblas.dylib"
                        : "/opt/homebrew/opt/openblas/lib/libopenblas.dylib")));
            }
            case LINUX -> {
                candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("libopenblas.so.0"));
                candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("libopenblas.so"));
            }
            case WINDOWS -> {
                candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("libopenblas.dll"));
                candidates.add(new CpuOpenBlasDiscoveryResult.LibraryName("openblas.dll"));
            }
            case OTHER -> { }
        }
        return List.copyOf(candidates);
    }

    /**
     * Delegates one exact selection to the provider and adapts the resulting lifetime.
     *
     * @param selection the exact non-null name or absolute path to pass unchanged
     * @return one invocation and its owning provider close action; never {@code null}
     * @throws LoadFailure if the provider reports an ordinary load or binding failure
     * @throws RuntimeException if invocation adaptation fails
     * @throws Error if loading, adaptation, or cleanup exposes a fatal failure
     */
    private static CpuOpenBlasDiscoverySession.OwnedResource loadProductionResource(
            CpuOpenBlasDiscoveryResult.Selection selection) throws LoadFailure {
        OpenBlasLibrary library;
        try {
            library = switch (selection) {
                case CpuOpenBlasDiscoveryResult.LibraryName name ->
                        OpenBlasLibrary.open(name.value());
                case CpuOpenBlasDiscoveryResult.AbsoluteLibraryPath path ->
                        OpenBlasLibrary.open(path.value());
            };
        } catch (OpenBlasLoadException failure) {
            rethrowFatalCause(failure);
            throw new LoadFailure(failure.getClass().getName(), failure.getMessage());
        }
        try {
            CpuOpenBlasInvocation invocation = CpuOpenBlasRouteSelector.borrowedInvocation(library);
            return new CpuOpenBlasDiscoverySession.OwnedResource(invocation, library::close);
        } catch (RuntimeException | Error primary) {
            closeLibraryAfterFailure(library, primary);
            throw primary;
        }
    }

    /**
     * Preserves fatal failures hidden in the provider's translated cause chain.
     *
     * @param failure the provider load failure whose cause chain is inspected by identity
     * @throws VirtualMachineError if the chain contains a virtual-machine failure
     * @throws ThreadDeath if the chain contains thread termination
     */
    @SuppressWarnings("removal")
    private static void rethrowFatalCause(OpenBlasLoadException failure) {
        Set<Throwable> visited = Collections.newSetFromMap(new IdentityHashMap<>());
        for (Throwable current = failure; current != null && visited.add(current);
                current = current.getCause()) {
            if (current instanceof VirtualMachineError fatal) throw fatal;
            if (current instanceof ThreadDeath fatal) throw fatal;
        }
    }

    /**
     * Closes a provider opened before invocation adaptation failed and suppresses a distinct
     * cleanup failure on the primary failure.
     *
     * @param library the non-null provider lifetime to close
     * @param primary the non-null adaptation failure that will be propagated
     */
    private static void closeLibraryAfterFailure(OpenBlasLibrary library, Throwable primary) {
        try {
            library.close();
        } catch (RuntimeException | Error cleanupFailure) {
            suppressDistinct(primary, cleanupFailure);
        }
    }

    /**
     * Closes a loaded resource when result or session construction fails.
     *
     * @param resource the non-null owned invocation and close action
     * @param primary the non-null construction failure that will be propagated
     */
    private static void closeAfterFailure(CpuOpenBlasDiscoverySession.OwnedResource resource,
            Throwable primary) {
        try {
            resource.closeAction().close();
        } catch (RuntimeException | Error cleanupFailure) {
            suppressDistinct(primary, cleanupFailure);
        }
    }

    /**
     * Attaches one cleanup failure without attempting prohibited self-suppression.
     *
     * @param primary the non-null failure that remains authoritative
     * @param cleanupFailure the non-null cleanup failure to suppress when distinct by identity
     */
    private static void suppressDistinct(Throwable primary, Throwable cleanupFailure) {
        if (cleanupFailure != primary) primary.addSuppressed(cleanupFailure);
    }

    /** Cold platform-property access used only by automatic discovery. */
    interface PlatformProperties {
        /**
         * Reads the operating-system property once.
         *
         * @return the raw value, or {@code null} to represent absence
         */
        String operatingSystem();

        /**
         * Reads the architecture property once.
         *
         * @return the raw value, or {@code null} to represent absence
         */
        String architecture();
    }

    /** Exact typed selection loader used by cold native-free tests and production adaptation. */
    @FunctionalInterface
    interface Loader {
        /**
         * Loads and adapts one exact selection.
         *
         * @param selection the exact typed selection to load unchanged
         * @return one owned invocation/close pair; never {@code null}
         * @throws LoadFailure if the provider reports an ordinary load or binding failure
         */
        CpuOpenBlasDiscoverySession.OwnedResource load(
                CpuOpenBlasDiscoveryResult.Selection selection) throws LoadFailure;
    }

    /** Checked ordinary loader failure containing immutable provider diagnostic strings only. */
    static final class LoadFailure extends Exception {
        private final String failureType;
        private final String failureMessage;

        /**
         * Creates one ordinary expected loader failure.
         *
         * @param failureType the nonblank thrown provider type name
         * @param failureMessage the nullable thrown provider message
         * @throws NullPointerException if {@code failureType} is {@code null}
         * @throws IllegalArgumentException if {@code failureType} is blank
         */
        LoadFailure(String failureType, String failureMessage) {
            super(null, null, false, false);
            this.failureType = Objects.requireNonNull(failureType, "failureType");
            if (failureType.isBlank()) {
                throw new IllegalArgumentException("failureType must not be blank");
            }
            this.failureMessage = failureMessage;
        }

        /**
         * Returns the original provider failure type name.
         *
         * @return the nonblank immutable type name
         */
        String failureType() {
            return failureType;
        }

        /**
         * Returns the original provider failure message.
         *
         * @return the immutable message, or {@code null} when the provider supplied none
         */
        String failureMessage() {
            return failureMessage;
        }
    }
}
