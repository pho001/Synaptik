package io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas;

import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * Immutable CPU-private intent for one explicit cold OpenBLAS discovery call.
 *
 * <p>The request contains no configuration lookup or fallback policy beyond its selected mode.
 * Exact values are passed unchanged to discovery; automatic candidates are not appended to an
 * exact request.</p>
 *
 * @param mode the required discovery mode
 * @param exactName the exact nonblank loader name, present only for {@link Mode#EXACT_NAME}
 * @param exactAbsolutePath the exact absolute loader path, present only for
 *     {@link Mode#EXACT_ABSOLUTE_PATH}
 */
record CpuOpenBlasDiscoveryRequest(Mode mode, Optional<String> exactName,
        Optional<Path> exactAbsolutePath) {
    /** Closed discovery-mode vocabulary. */
    enum Mode {
        /** Perform no platform lookup and no provider load. */
        DISABLED,
        /** Try the bounded candidates for one snapshotted platform. */
        AUTOMATIC,
        /** Try one exact operating-system loader name. */
        EXACT_NAME,
        /** Try one exact absolute filesystem path. */
        EXACT_ABSOLUTE_PATH
    }

    /**
     * Validates one immutable request and its mode-specific value relationship.
     *
     * @param mode the required discovery mode
     * @param exactName the optional exact name; must be present and nonblank only in exact-name
     *     mode
     * @param exactAbsolutePath the optional exact path; must be present and absolute only in
     *     exact-path mode
     * @throws NullPointerException if a required reference is {@code null}
     * @throws IllegalArgumentException if the values disagree with the mode, the exact name is
     *     blank, or the exact path is relative
     */
    CpuOpenBlasDiscoveryRequest {
        Objects.requireNonNull(mode, "mode");
        exactName = Objects.requireNonNull(exactName, "exactName");
        exactAbsolutePath = Objects.requireNonNull(exactAbsolutePath, "exactAbsolutePath");
        switch (mode) {
            case DISABLED, AUTOMATIC -> {
                if (exactName.isPresent() || exactAbsolutePath.isPresent()) {
                    throw new IllegalArgumentException(
                            "disabled and automatic discovery accept no exact selection");
                }
            }
            case EXACT_NAME -> {
                if (exactName.isEmpty() || exactAbsolutePath.isPresent()) {
                    throw new IllegalArgumentException(
                            "exact-name discovery requires only an exact name");
                }
                if (exactName.orElseThrow().isBlank()) {
                    throw new IllegalArgumentException("exact OpenBLAS name must not be blank");
                }
            }
            case EXACT_ABSOLUTE_PATH -> {
                if (exactName.isPresent() || exactAbsolutePath.isEmpty()) {
                    throw new IllegalArgumentException(
                            "exact-path discovery requires only an exact absolute path");
                }
                if (!exactAbsolutePath.orElseThrow().isAbsolute()) {
                    throw new IllegalArgumentException("exact OpenBLAS path must be absolute");
                }
            }
        }
    }

    /**
     * Creates a request that performs no discovery.
     *
     * @return an immutable disabled request
     */
    static CpuOpenBlasDiscoveryRequest disabled() {
        return new CpuOpenBlasDiscoveryRequest(Mode.DISABLED, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a request for the bounded platform candidate table.
     *
     * @return an immutable automatic request
     */
    static CpuOpenBlasDiscoveryRequest automatic() {
        return new CpuOpenBlasDiscoveryRequest(Mode.AUTOMATIC, Optional.empty(), Optional.empty());
    }

    /**
     * Creates a request for one exact loader name.
     *
     * @param name the nonblank loader name to preserve unchanged
     * @return an immutable exact-name request
     * @throws NullPointerException if {@code name} is {@code null}
     * @throws IllegalArgumentException if {@code name} is blank
     */
    static CpuOpenBlasDiscoveryRequest exactName(String name) {
        return new CpuOpenBlasDiscoveryRequest(Mode.EXACT_NAME,
                Optional.of(Objects.requireNonNull(name, "name")), Optional.empty());
    }

    /**
     * Creates a request for one exact absolute path.
     *
     * @param path the absolute path to preserve unchanged
     * @return an immutable exact-path request
     * @throws NullPointerException if {@code path} is {@code null}
     * @throws IllegalArgumentException if {@code path} is relative
     */
    static CpuOpenBlasDiscoveryRequest exactAbsolutePath(Path path) {
        return new CpuOpenBlasDiscoveryRequest(Mode.EXACT_ABSOLUTE_PATH, Optional.empty(),
                Optional.of(Objects.requireNonNull(path, "path")));
    }
}
