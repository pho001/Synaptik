/**
 * Defines shared assignment and typed backend-finalization contracts for prepared partitions.
 *
 * <p>The {@code analysis} subpackage supplies validated backend analysis inputs and exact resource
 * declarations. This root package associates those declarations with Runtime-owned slots,
 * constructs one shared memory plan, and hands each opaque typed plan back to its owning backend
 * for immutable executable construction. Public prepare orchestration, schedules, physical
 * allocation, and per-run execution remain outside this package's current surface.</p>
 */
package io.github.pho001.synaptik.prepare;
