# Metal foundation for macOS arm64

This directory contains the version-one, non-executing Metal native foundation. It creates one
default Metal device and command queue per context and allocates shared-storage buffers, but it
submits no commands and contains no Metal Performance Shaders Graph (MPSGraph) integration or
tensor operation.

The Java owner supplies one absolute dynamic-library path, validates the ABI, creates a context,
and releases each opaque context or buffer handle exactly once. A logical zero-byte buffer has
one private physical backing byte but still accepts only zero-byte logical access. These handles
are process-local ownership tokens, not addresses or serialization values for callers.

## Build prerequisites and output

Build the local dynamic library on an Apple-silicon macOS host with Xcode Command Line Tools:

```bash
./native/metal-macos-arm64/build.sh
```

The script invokes the macOS SDK compiler for arm64, enables automatic reference counting (ARC),
and links the system Foundation and Metal frameworks. It writes only
`native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib`. The `build/` directory is
ignored and the binary is not a packaged, signed, notarized, or published project artifact.

## ABI surface

The dylib exports exactly seven `synaptik_metal_*` symbols: one unsigned version query plus
context create/release, buffer create/release, and buffer upload/download. ABI version `1` uses
opaque `void *` handles, signed 32-bit status results, unsigned 64-bit sizes and offsets, and
caller-supplied output cells for created handles. The complete normative signatures, status
values, output-cell rules, and cleanup order are recorded in
[Metal task 0001](../../docs/planning/backends/metal/tasks/0001-metal-capability-storage-and-native-foundation.md#exact-native-abi-contract).

## Native round trip

Run the opt-in Java round trip against that exact output:

```bash
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :backends:metal:test --tests '*MetalFoundationTest.nativeFoundationRoundTrip'
```

The Java test skips unless the environment variable is present. A successful round trip proves
context/buffer ownership and bounded shared-memory copies only; it does not establish operation
capability, command submission, synchronization, route support, or performance.

## Ownership and failure boundary

The Objective-C implementation contains exceptions and returns stable status codes to Java. A
successful create transfers one retained handle to Java; the matching release consumes it. Java
buffer and workspace wrappers retain context leases. Context close atomically prevents new access
admission, while an already admitted action may finish outside the context monitor and distinct
resources may overlap. Native context release waits for the last child. This foundation does not
discover the library, select a device beyond the system default, package a binary, expose
workspace bytes, or define public Engine error translation.
