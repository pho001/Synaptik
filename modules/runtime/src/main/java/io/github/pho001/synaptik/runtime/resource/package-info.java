/**
 * Defines nominal lifecycle roles for physical Runtime representations implemented by concrete
 * backends, plus immutable prepared descriptions of how each run obtains them.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.resource.BufferRepresentation} represents a
 * physical form of one logical buffer, while
 * {@link io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation} represents
 * backend-local per-run scratch. The roles are intentionally distinct and share only unchecked,
 * non-throw-declared cleanup through {@link java.lang.AutoCloseable#close()}.
 * {@link io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan} associates dense
 * prepared buffer positions with borrowed caller-input occurrences or backend-owned creation
 * callbacks and associates each workspace position with a backend-owned creator. The immutable
 * plan invokes no callback and owns no physical representation.
 *
 * <p>This package provides no physical storage implementation and no allocation, access,
 * transfer, backend/device key, coherence, publication, pooling, or discovery mechanism.
 * Concrete backend modules own physical implementations and allocation, release, transfer, and
 * access mechanics. Runtime owns cold creation orchestration, structural per-run residency,
 * explicit buffer-copy validity, ownership, and cleanup through the {@code runtime.run} package.
 */
package io.github.pho001.synaptik.runtime.resource;
