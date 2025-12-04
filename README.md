# ComputationalGraph

Lightweight Java computational graph with:
- Tensors (double[]) with shapes/strides
- Autograd (reverse-mode) for element-wise ops
- Graph optimizer that fuses contiguous element-wise operations
- Bytecode generation (ASM) to JIT fused forward/backward kernels
- Pluggable backends (CPU implemented; CUDA/OpenCL placeholders)

## Requirements
- JDK 21 (toolchain configured in Gradle)
- Gradle (or use the Gradle Wrapper once generated)
- macOS/Linux/Windows
- Optional: GitHub CLI (gh) if you plan to publish to GitHub

Vector API note:
- Project references `jdk.incubator.vector`. Gradle adds `--add-modules=jdk.incubator.vector` for compile and runtime.

## Build & Run (Gradle)
Using local Gradle:
- Run: `gradle run`
- Build: `gradle build`
- Tests (if added): `gradle test`

Using Gradle Wrapper (recommended):
- Generate wrapper (once): `gradle wrapper`
- Run: `./gradlew run` (macOS/Linux) or `.\gradlew.bat run` (Windows)
- Build: `./gradlew build`

## Project Layout
- `src/`
  - `Tensor/*`: Tensor model and graph node
  - `Operations/*`: Element-wise ops (add, sub, mul, div, log, exp, pow, …)
  - `Backend/*`: Backend dispatch (CPU implemented; CUDA/OpenCL stubs)
  - `Graph/*`: Graph optimizer, compiled graph, ASM fused codegen
  - `Utils/*`: ASM codegen utilities (slot management, node/operator info)
- `build.gradle`, `settings.gradle`: Gradle build files

## Quick Start
The `Main` class constructs a small element-wise graph, runs forward/backward multiple times,
and compares timings with and without the optimizer. Adjust shapes/repeats in `Main` to benchmark.

## Notes
- ASM dependencies are declared in `build.gradle`:
  - `org.ow2.asm:asm` and `asm-commons`
  - `net.bytebuddy:byte-buddy` for ByteBuddy’s shaded ASM references in experimental generators
- Fused kernels are generated at runtime and loaded via a custom class loader.
- GPU backends are placeholders and can be implemented later.

## License
Not specified. Add a LICENSE file if you plan to open-source.
