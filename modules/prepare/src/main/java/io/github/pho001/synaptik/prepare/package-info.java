/**
 * Defines complete shared graph preparation, assignment, typed backend finalization, and
 * schedule-validation contracts.
 *
 * <p>The {@code analysis} subpackage supplies validated backend analysis inputs and exact resource
 * declarations. This root package projects complete partition contexts, associates declarations
 * with Runtime-owned slots, hands each opaque typed plan back to its owning backend for immutable
 * executable construction, and validates one explicitly assembled complete schedule. The
 * stateless public facade creates reusable recipes only; physical allocation, mutable per-run
 * state, execution, backend discovery, and Engine composition remain outside this package.</p>
 */
package io.github.pho001.synaptik.prepare;
