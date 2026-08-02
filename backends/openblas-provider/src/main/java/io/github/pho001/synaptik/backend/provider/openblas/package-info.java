/**
 * Provides explicit caller-directed OpenBLAS loading, thread control, and low-level matrix
 * multiplication.
 *
 * <p>The package is a JDK-only leaf below the CPU backend. It owns library lookup lifetime and
 * exact OpenBLAS C symbol binding, while CPU policy owns route selection, fallback, configuration,
 * thread choices, batching, broadcasting, layout conversion, packing, and executable storage.
 * Its current public calls perform only one already-normalized FLOAT32 or FLOAT64 dense row-major,
 * non-transposed general matrix multiplication (GEMM):
 * {@code C[m,n] = alpha * (A[m,k] x B[k,n]) + beta * C[m,n]}.
 *
 * <p>All matrix segments begin at byte offset zero and remain caller-owned. Inputs may be
 * read-only and may overlap each other; output must be writable and its required range must not
 * overlap either required input range. Calls retain, allocate, copy, reinterpret, or close no
 * caller memory. Complete validation precedes an output-empty no-op, while a positive-output
 * zero contraction still invokes GEMM. Scalars are forwarded unchanged, and the package makes no
 * OpenBLAS numerical, determinism, or performance guarantee.
 *
 * <p>Thread query and setter calls expose the already-bound OpenBLAS utility functions directly.
 * The positive count is conservatively treated as mutable library/process state that owners of
 * one loaded binary may observe in common. The package does not claim shared coordination across
 * independently loaded copies, loader namespaces, or arbitrary native consumers. It defines no
 * deterministic winner for concurrent setters, no atomic query/set sequence, no serialization
 * with GEMM, and no automatic or close-time restoration. A caller owns coordination and any
 * restoration through a still-open owner.
 *
 * <p>Concurrent calls are permitted while the library remains open, subject to the caller's
 * coordination of thread state and nonconflicting matrix access. Callers must keep every segment
 * scope alive and accessible for the complete GEMM and must not race library closure with GEMM or
 * thread-control invocation. The package adds no native-call synchronization, active-call
 * tracking, or close-race success guarantee.
 */
package io.github.pho001.synaptik.backend.provider.openblas;
