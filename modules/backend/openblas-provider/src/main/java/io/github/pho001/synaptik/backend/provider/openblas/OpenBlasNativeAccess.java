package io.github.pho001.synaptik.backend.provider.openblas;

import java.nio.file.Path;

/**
 * Opens one explicit library and returns its complete required native binding set.
 *
 * <p>This package-private seam exists only to isolate deterministic unit tests from installed
 * native libraries. It owns no discovery, fallback, route, configuration, or cache policy.
 */
interface OpenBlasNativeAccess {
    /**
     * Opens exactly the supplied operating-system library name.
     *
     * @param libraryName the already validated nonblank name; never {@code null}
     * @return the complete caller-owned binding set; never {@code null}
     * @throws RuntimeException if loading or complete binding fails
     */
    OpenBlasNativeBindings open(String libraryName);

    /**
     * Opens exactly the supplied absolute library path.
     *
     * @param absoluteLibraryPath the already validated absolute path; never {@code null}
     * @return the complete caller-owned binding set; never {@code null}
     * @throws RuntimeException if loading or complete binding fails
     */
    OpenBlasNativeBindings open(Path absoluteLibraryPath);
}
