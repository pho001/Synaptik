# Metal backend foundation

## Outcome and supported scope

This guide explains the current Metal backend foundation and the boundary a later executable
route must preserve. Today the module provides an explicitly constructed capability provider that
rejects every operation plus package-private macOS arm64 native and storage mechanics. It does
not provide supported tensor execution.

```text
Planning query -> fail-closed provider -> no Metal ownership

explicit dylib path -> native context -> run-owned buffer/workspace -> deterministic cleanup
                                      (no prepared Runtime integration yet)
```

The two lines are deliberately separate. A working device, command queue, or buffer does not make
an operation eligible for Metal ownership.

## Prerequisites

The Java foundation uses JDK 26 Foreign Function and Memory (FFM) APIs. Building and exercising
the native bridge requires an Apple-silicon macOS host, Xcode Command Line Tools, and the system
Foundation and Metal frameworks. The caller supplies the absolute dylib path; the backend does
not scan, discover, extract, package, sign, or cache a library.

See the [native build and ABI guide](../../native/metal-macos-arm64/README.md) for the local build
command and opt-in round trip.

## Contracts and ownership

| Concern | Current owner and behavior |
|---|---|
| Capability truth | `MetalCapabilityProvider` reports backend identity `metal`, rejects null queries, and returns `false` for every operation. |
| Native lifetime | One package-private context owns the default Metal device, one command queue, and the FFM lookup lifetime. |
| Buffer storage | Each buffer wrapper owns one fresh shared-storage native buffer and one context child lease. |
| Workspace storage | Each workspace wrapper owns scratch storage and a child lease but exposes no host byte access. |
| Run association | Runtime defines the nominal representation roles, but task 0001 does not create a prepared representation plan or `RunState`. |
| Future lowering | Metal prepare will own route selection and lowering only after a later task defines a complete executable route. |

The capability provider is immutable and thread-safe. Buffer access and close are serialized per
resource. Context access admission is atomic with context close, but an admitted native action
runs outside the context monitor. Distinct open resources may therefore overlap. An action
admitted before context close may finish using its retained child lease; a later action is
rejected before native invocation. Closing a context defers native release until the final child
wrapper closes. Repeated close calls do not retry native release.

## Current foundation lifecycle

The current concrete scenario is a resource round trip, not an operation execution example.

1. Build the native dylib and supply its absolute path.
2. Java resolves the ABI version first, requires version `1`, then resolves the six remaining
   symbols before creating a context.
3. Context creation retains one default device and one command queue.
4. Buffer creation returns a fresh opaque handle. A logical size of zero still has private
   one-byte physical backing, while its permitted logical range remains zero.
5. Upload and download validate segment liveness, thread access, writability, native provenance,
   and both segment and logical-buffer ranges before crossing the native boundary.
6. Closing a buffer consumes its native handle and releases its context lease. The context and
   symbol lookup close after owner close and the last child release.

For the opt-in test input, an eight-byte buffer receives five bytes from source offset `1` at
buffer offset `2`. Downloading those five bytes to destination offset `2` produces
`[0, 0, 2, 3, 4, 5, 6, 0]`. This proves bounded byte preservation and ownership cleanup. It does
not prove tensor semantics, an executable route, command submission, or synchronization.

## Registration and future composition

The provider is passed explicitly to Planning. It is not discovered through `ServiceLoader`, a
registry, or a runtime service locator. No current Engine composition registers Metal, and no
prepared Runtime recipe creates these representations. A future route must first let Planning
select Metal ownership, then let Metal analyze and finalize that partition against shared assigned
slots. Only the resulting prepared executable may reach Runtime.

Conceptual future lifecycle (not a current API):

```text
truthful capability -> Planning ownership -> Metal analysis -> shared slot assignment
                    -> Metal finalization -> prepared executable -> Runtime invocation
```

## Failures and diagnostics

Null capability queries fail before inspection. Library loading, ABI mismatch, missing symbols,
no default device, no command queue, allocation failure, invalid native ranges, copy failure, and
unknown status values all fail closed. The package-private native exception retains the operation
name and exact status integer. Cleanup preserves the primary failure and suppresses distinct
later cleanup failures in deterministic order. Public Engine exception translation does not yet
exist for Metal.

## Conformance and validation

Task 0001 validates provider behavior, exact handle ownership, allocation and byte geometry,
concurrent/idempotent close, close-versus-access ordering, partial cleanup failures, and the real
seven-symbol arm64 ABI. Backend-conformance and end-to-end suites are deferred because Metal
advertises and executes no Model operation. Performance evidence is also deferred because there
is no executable route to measure.

## Limitations and related documentation

There is no supported operation, prepared Runtime integration, Engine composition, MPSGraph or
custom-kernel route, command submission, synchronization guarantee, FLOAT16 or BFLOAT16 support
claim, automatic library discovery, or performance claim. Model task 0026 must define FLOAT16
semantics before any backend can advertise it; two-byte storage alone proves neither FLOAT16 nor
BFLOAT16 execution support.

Related documentation:

- [Architecture contract](../../ARCHITECTURE.md#metal-backend)
- [Runtime / Prepare / Backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Metal master plan](../planning/backends/metal/master-plan.md)
- [Metal task 0001](../planning/backends/metal/tasks/0001-metal-capability-storage-and-native-foundation.md)
- [Metal strategy note](../design/notes/metal-backend-strategy.md)
