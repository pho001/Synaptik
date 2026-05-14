# Changelog

All notable project-level release changes are tracked here.

This project uses public-preview semantic versioning while the API is still
pre-1.0. Until `1.0.0`, minor releases may include source or binary breaking
changes when they cleanly advance the graph/runtime architecture.

## 0.1.0-alpha.2 - 2026-05-14

### Added

- Gradle publication metadata for GitHub/JitPack consumption.
- Source JAR publication through the `maven-publish` Gradle plugin.
- JitPack build configuration that publishes the library to Maven Local during
  the JitPack install step.

### Changed

- Public dependency coordinates for GitHub consumption are now documented as
  `com.github.pho001:Synaptik:v0.1.0-alpha.2`.

## 0.1.0-alpha.1 - 2026-05-14

### Release Status

- First public technical preview baseline.
- Intended for code review, experimentation, architecture feedback, and local
  benchmarking.
- Not a production-stability claim.

### Added

- Public `Tensor` graph API with dense tensor storage, dtype-aware values,
  shape/stride metadata, view-like layout transforms, and reverse-mode autodiff.
- Compiled graph lifecycle split into semantic graph construction, compile,
  prepare, and execute.
- Compile configuration split into semantic canonicalization, graph
  optimization, backend planning, region optimization, and memory planning.
- Runtime configuration for CPU kernels, fused execution, BLAS, approximation,
  accelerator availability, buffer binding, and publication policy.
- CPU backend coverage for the primary dense tensor families:
  - elementwise and scalar elementwise;
  - broadcasting and `where`;
  - reductions;
  - indexing and scatter/gather families;
  - matmul and linear;
  - conv2d and pool2d;
  - softmax/log-softmax;
  - loss-adjacent operations;
  - selected backward operations.
- CPU fused execution paths for selected fused elementwise families.
- ONNX static dense inference import/export subset with checked coverage docs.
- ONNX runtime `INT64` index tensor support on CPU for relevant import/export
  rows.
- Metal and CUDA accelerator coverage matrices with explicit support,
  partial/fallback, and unsupported rows.
- Scoped Metal execution paths with dtype/layout/index/NN/loss/training
  coverage evidence and trace-visible fallback.
- Scoped CUDA runtime and coverage truth with capability-gated native support
  and explicit unsupported rows.
- Benchmark, graph autotune, platform calibration, profile persistence, and
  report/tracing infrastructure.
- Documentation for architecture, compute flow, optimizer stages, backend
  planning, ONNX coverage, calibration/autotune, testing, troubleshooting, and
  extension workflows.

### Current Limitations

- Public API and internal package boundaries are still pre-1.0 and may change.
- ONNX support targets a static dense inference subset, not the full ONNX
  ecosystem.
- Runtime dynamic shapes, general multi-output runtime ops, control-flow ops,
  sparse tensors, quantized tensors, string tensors, sequence/map/optional
  values, and external ONNX data files are not supported.
- Layer-aware ONNX import/export is future scope.
- CPU remains the correctness oracle. Accelerator support is capability-gated
  and backend-specific.
- Local benchmark and calibration outputs are not release artifacts unless
  explicitly promoted.

### Verification Baseline

The release-hardening verification target is:

```bash
./gradlew classes
./gradlew test --tests 'onnx.*'
./gradlew test --tests SourceTreeHygieneTest
```

Optional hardware-specific gates:

```bash
./gradlew metalTest
./gradlew cudaTest
```
