package io.github.pho001.synaptik.model.operation;

/**
 * Marks an immutable, typed value containing the semantic parameters of an operation.
 *
 * <p>Implementations expose family-specific parameters through typed fields and accessors rather
 * than string-keyed maps. They must provide structural equality and hashing, and they must
 * defensively isolate any mutable constructor input so their observable state cannot change after
 * construction. Immutable records and enums are the intended implementation forms.</p>
 *
 * <p>Attribute values contain semantic parameters only. They must not retain mutable tensors,
 * live services, backend or device handles, compiler or runtime state, storage, execution routes,
 * or other mutable infrastructure. This marker deliberately declares no common fields or methods;
 * focused operation-family contracts define their own immutable attribute shapes.</p>
 */
public interface OperationAttrs {}
