# Build oneDNN with Apple Accelerate

## Goal and boundary

Build an **explicit optional** [oneDNN](../glossary.md#term-onednn) 3.14.0 library for Apple
[arm64/AArch64](../glossary.md#term-aarch64) using the operating system's
[Accelerate](../glossary.md#term-accelerate) [BLAS](../glossary.md#term-blas) implementation.
This does not introduce another Synaptik backend. CPU remains the backend identity; persistent
prepared resources and per-invocation native controls retain their existing separate lifetimes.
The application still selects an absolute library path.

The [existing loading contract](../backend-guide/cpu-backend.md#external-binary-and-loading-requirements)
remains exact: 3.14.0, little-endian [LP64](../glossary.md#term-lp64), CPU runtime
[SEQ](../glossary.md#term-seq-native-runtime), Graph enabled, no GPU runtime.
No Homebrew library is overwritten, no native [ABI](../glossary.md#term-application-binary-interface-abi) changes, and no runtime
fallback or automatic selection is added. This recipe is not an instruction to change the sealed
private release or to publish third-party binaries.

## Why these options matter

oneDNN can use different internal matrix-multiplication implementations. The ordinary local
library selected [KleidiAI](../glossary.md#term-kleidiai)'s Arm ASIMD (Neon vector-instruction) kernel. PyTorch's CPU wheel
used Accelerate, not oneDNN. Merely calling both paths “CPU” would hide that distinction.
Upstream 3.14 already supports
[`ONEDNN_BLAS_VENDOR=ACCELERATE`](https://github.com/uxlfoundation/oneDNN/blob/v3.14/cmake/blas.cmake).
Disabling `ONEDNN_AARCH64_USE_KAI` lets the existing GEMM implementation reach that BLAS path
rather than selecting KleidiAI first. Neither option changes Synaptik's numerical ownership.

The current Apple [SDK](../glossary.md#term-sdk) exposes a global `bfloat16_t` name through Accelerate. oneDNN 3.14's
unqualified C-wrapper definition then conflicts with its own same-named internal type.
The [minimal compatibility patch](../../tools/benchmarks/onednn-3.14-accelerate-bfloat16.patch)
qualifies **oneDNN's original type** in two parameters. It changes no C symbol, data representation,
arithmetic, or exported ABI; it does not replace the type with Apple's `__bf16`.

## Prerequisites

- Apple arm64 macOS with Xcode Command Line Tools, including its C/C++ compiler and SDK.
- [CMake](../glossary.md#term-cmake), `curl`, `tar`, `patch` and `shasum` on PATH; the exercised CMake version is 4.4.5.
- The Synaptik checkout, run from its root. Native compilation does not require a JDK;
  using Synaptik afterward requires the [ordinary JDK/native-access setup](../getting-started.md).

## Download, authenticate and build

The source [SHA-256](../glossary.md#term-sha-256) is the same one pinned by the
[existing qualification workflow](../../.github/workflows/onednn-cpu.yml).
Upstream tag `v3.14` resolves to commit `18378c8475643c4172fe059fefba855b14c07345`;
its library version is 3.14.0, not a newer patch release.

```sh
# Run at the checkout root. Keep this output separate from system/Homebrew packages.
root="$PWD"
work="$root/build/onednn-3.14-accelerate"
mkdir -p "$work/source"

# Refuse an unexpected source archive before executing its build configuration.
curl --fail --location \
  https://github.com/uxlfoundation/oneDNN/archive/refs/tags/v3.14.tar.gz \
  --output "$work/onednn-v3.14.tar.gz"
printf '%s  %s\n' \
  063eca4137678ba79b2da7030cbe2b41b1cc7b15a76ac13728a531a710e7392d \
  "$work/onednn-v3.14.tar.gz" | shasum --algorithm 256 --check

tar -xzf "$work/onednn-v3.14.tar.gz" --strip-components=1 -C "$work/source"
# One namespace qualification fixes the Apple SDK name collision, not its ABI.
patch --strip=1 --directory="$work/source" \
  --input="$root/tools/benchmarks/onednn-3.14-accelerate-bfloat16.patch"

# Keep the existing required version/runtime/Graph contract; opt into Apple's BLAS.
cmake -S "$work/source" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$work/install" -DCMAKE_INSTALL_LIBDIR=lib \
  -DONEDNN_CPU_RUNTIME=SEQ -DONEDNN_GPU_RUNTIME=NONE \
  -DONEDNN_BUILD_GRAPH=ON -DONEDNN_LIBRARY_TYPE=SHARED \
  -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF \
  -DONEDNN_VERBOSE=ON -DONEDNN_AARCH64_USE_KAI=OFF \
  -DONEDNN_BLAS_VENDOR=ACCELERATE
cmake --build "$work/build" --parallel 4
cmake --install "$work/build"
```

The resulting library is
`build/onednn-3.14-accelerate/install/lib/libdnnl.3.14.dylib`. Upstream license and
third-party notices are installed under `install/share/doc/dnnl/`; do not discard them when
handling that external installation. Deprecation warnings from Apple's legacy LP64 CBLAS
interface are not suppressed or presented as errors in Synaptik.

## Select and observe it explicitly

```sh
# The ordinary CLI/TUI application must first be built as described in its guide.
# These variables limit this process's numerical thread demand, not all machine activity.
VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  tools/cli/build/install/synaptik/bin/synaptik train \
  --data train.csv --output line.synaptik --backend onednn \
  --onednn-library "$PWD/build/onednn-3.14-accelerate/install/lib/libdnnl.3.14.dylib" \
  --onednn-api auto
```

Supply real data in the [CLI/TUI numeric format](train-with-cli.md#dataset-and-static-batch-rules).
In the TUI, choose the oneDNN backend and enter this absolute library path. `AUTO`,
`PRIMITIVES` and `GRAPH` remain API preferences, not commands forcing native eligibility.
Use [cold CPU diagnostics](../backend-guide/cpu-backend.md#explicit-onednn-composition) for the
actual selected route/API. An upstream verbose name such as `gemm:jit:f32` is not by itself
proof of the underlying BLAS vendor; retain configuration and native-stack evidence too.

The same-version build and all five declared dense FLOAT32 correctness shapes were actually
executed through AUTO, PRIMITIVES and GRAPH. This does not qualify every operation, layout,
precision, host or future SDK. Unsupported native occurrences still follow the ordinary cold
CPU selection rules; this recipe never broadens the support matrix. Native arithmetic and
storage semantics remain those in the [CPU guide](../backend-guide/cpu-backend.md).

### Precision availability is separate from library loading

The exercised Accelerate/KAI-disabled build does **not** provide native
[BFLOAT16](../glossary.md#term-bfloat16) matmul through either native API. Some
[FP8](../glossary.md#term-fp8) Primitive cases are also unavailable on this arm64 host.
Loading a compatible oneDNN library therefore does not force those occurrences onto oneDNN.
The existing CPU planner retains eligible portable implementations before execution; this is
not a runtime fallback after a failed native call. Actual public BF16 computations passed with
each of AUTO, PRIMITIVES and GRAPH preferences.

See the [matched performance and qualification report](../planning/native-performance-comparison.md)
for actual native routes, tested scopes, all timing controls, remaining gaps and architecture-only
proposals. Do not make this optional F32-oriented build a universal precision default.
