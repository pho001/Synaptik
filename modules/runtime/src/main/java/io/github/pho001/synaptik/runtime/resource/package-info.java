/**
 * Defines nominal lifecycle roles for physical Runtime representations implemented by concrete
 * backends.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.resource.BufferRepresentation} represents a
 * physical form of one logical buffer, while
 * {@link io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation} represents
 * backend-local per-run scratch. The roles are intentionally distinct and share only unchecked,
 * non-throw-declared cleanup through {@link java.lang.AutoCloseable#close()}.
 *
 * <p>This package provides no physical storage implementation and no allocation, access,
 * transfer, backend/device key, validity, residency, coherence, publication, pooling, or
 * discovery mechanism. Concrete backend modules own those physical implementations and
 * mechanics. Runtime owns only the per-run logical association and cleanup orchestration defined
 * by the {@code runtime.run} package.
 */
package io.github.pho001.synaptik.runtime.resource;
