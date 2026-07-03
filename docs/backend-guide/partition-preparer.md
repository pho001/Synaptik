# Prepare an owned partition (planned contract)

## Outcome and status

This guide explains the concrete backend's handoff from a `PlannedPartition` to prepared executable state. None of these contracts is implemented yet, so names and code shapes are conceptual.

## Ownership

Shared `modules/prepare` will define `BackendPartitionPreparer`, preparation context, prepared-partition contracts, and validation. Each concrete backend implements lowering, fusion, specialization, route selection, executable creation, storage, and workspace decisions for its own partitions.

```text
PlannedPartition(owner = CPU)
  -> CPU partition preparer
  -> CPU lowering and route choice
  -> PreparedPartition + PreparedExecutable
```

## Conceptual preparation path

For a CPU-owned `[2, 3] × [3, 4]` matrix multiplication, lowering sees 24 multiply contributions (`2 × 3 × 4`) and eight outputs. CPU prepare may choose a scalar route for a small region or OpenBLAS for a larger compatible region. That choice is backend-local and may depend on prepare configuration and workspace needs.

The preparer must declare input/output slots, resource lifetime, workspace requirements, and typed diagnostic contributions. The resulting executable computes only its prepared region; it does not perform runtime fallback.

## Failures and resources

- Reject a partition whose owner does not match the backend.
- Fail preparation when capability claims cannot be realized; do not emit a partially valid executable.
- Release native handles and backend resources according to the eventual prepared-execution lifetime contract.
- Keep per-run mutable bindings out of reusable prepared state.
- Record enough typed diagnostics to identify lowering and route decisions without leaking business logic into trace DTOs.

## Validation expectations

Unit-test lowering and failure paths, run architecture tests, and use backend-conformance tests to verify that declared capabilities can prepare and execute. Native routes additionally require resource-cleanup and platform validation.

See [Runtime/prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md) and [Kernel routes](kernel-routes.md).
