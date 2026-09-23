package io.github.pho001.synaptik.backend.metal;

import java.nio.file.Path;
import java.util.Objects;

/**
 * Immutable caller-selected configuration for opening one Metal backend integration.
 *
 * <p>The native bridge is intentionally explicit: Synaptik performs no classpath, environment,
 * working-directory, or platform-table discovery for Metal. Construction validates and snapshots
 * the normalized absolute path without opening the library. {@link MetalBackendIntegration#open}
 * owns native loading, context creation, and rollback.</p>
 *
 * @param nativeLibraryPath non-null absolute path to the Synaptik Metal native bridge; normalized
 *     structurally and retained without filesystem lookup
 */
public record MetalBackendConfiguration(Path nativeLibraryPath) {
    /**
     * Validates and snapshots one Metal configuration.
     *
     * @param nativeLibraryPath non-null absolute native bridge path
     * @throws NullPointerException if {@code nativeLibraryPath} is {@code null}
     * @throws IllegalArgumentException if the path is not absolute
     */
    public MetalBackendConfiguration {
        Objects.requireNonNull(nativeLibraryPath, "nativeLibraryPath");
        if (!nativeLibraryPath.isAbsolute()) {
            throw new IllegalArgumentException("nativeLibraryPath must be absolute");
        }
        nativeLibraryPath = nativeLibraryPath.normalize();
    }
}
