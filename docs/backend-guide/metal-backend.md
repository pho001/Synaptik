# Metal backend (planned implementation)

## Outcome and status

This guide records the intended Metal boundary. No Metal Java or native implementation exists yet.

## Planned scope

`backends/metal` will own Metal capability reporting, MPSGraph and custom-kernel lowering, fusion and specialization, executable creation, Metal storage and workspaces, materialization, native bridge integration, and typed traces.

Planning will select `owner = Metal`; Metal prepare will select MPSGraph or a custom kernel. Runtime will receive only prepared contracts. Global autograd and public tensor semantics remain outside Metal.

## Native and resource prerequisites

Future implementation requires an Apple platform with supported Metal tooling. Tasks must define native handle ownership, command-buffer and resource lifetime, error translation, synchronization, concurrent-run behavior, and cleanup. This page does not claim an operating-system version or device support matrix before tests establish one.

## Training boundary

A fused Metal optimizer implementation belongs to Metal prepare or kernels. The training extension owns the optimizer algorithm and must not add a `MetalOptimizerBridge` or depend on `backends/metal`.

## Validation

Future validation includes architecture tests, backend-conformance tests for declared capabilities, integration tests on supported hardware, native failure and cleanup tests, and benchmarks for route decisions.

See the [Metal master plan](../planning/backends/metal/master-plan.md) and [Metal strategy note](../design/notes/metal-backend-strategy.md).
