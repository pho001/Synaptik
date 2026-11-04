# Task 0024: Versioned local production [distribution](../../../../glossary.md#term-distribution)

> **Editorial historical reading key.**
>
> This document records a local distribution mechanism and one exercised source snapshot, not
> authorization to publish a public release. Gradle Java/Maven components carry implemented products
> and their actual API dependencies across the checkout-to-consumer boundary; the reviewed inventory
> and provenance identify which dirty source bytes produced those artifacts. Packaging owns selection
> and delivery, not numerical semantics, backend discovery, or execution ownership.
>
> The [current composition
> contract](../../../../reference/engine/composition.md#current-explicit-cpu-and-metal-composition)
> explains why public backend construction types require API-visible dependencies and why callers
> transfer integrations to Engine. The [current compilation
> port](../../../../reference/compiler/artifacts.md#current-complete-artifact-compilation-and-integration-port)
> distinguishes a cross-module SPI from an ordinary-user facade. Shipping either surface does not
> change cleanup responsibilities. Preserve the archive omission, correction, consumer results, and
> legal caveats as historical evidence; later source changes require a new reviewed inventory and
> freeze.

## Status

Complete — MAIN integrated the exact reviewed source after actual archive/[publication](../../../../glossary.md#term-publication), uncached
extracted-source [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names), native installation and external [Maven](../../../../glossary.md#term-maven)/[Gradle](../../../../glossary.md#term-gradle) [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) consumers.
Final independent source/evidence/documentation review APPROVE0. This is a local packaging
mechanism, not an [owner](../../../../glossary.md#term-owner)-authorized public [release](../../../../glossary.md#term-release) or final all-[PURE](../../../../glossary.md#term-pure) release candidate.

## Goal, class and authority

Class C: make the real product closure installable by external consumers using standard [Gradle](../../../../glossary.md#term-gradle)
Java/[Maven](../../../../glossary.md#term-maven) components, and freeze the exact dirty source that produced a versioned [distribution](../../../../glossary.md#term-distribution).
Use the [root dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>, existing [Engine](../../../../glossary.md#term-engine)/
[backend ownership](../../../../glossary.md#backend-ownership) and [production goal](../../../production-release-goal.md#release-acceptance-checklist).
No new [module](../../../../glossary.md#term-module), executor, [shading](../../../../glossary.md#term-shading), [fat JAR](../../../../glossary.md#term-fat-jar), compatibility wrapper or remote publishing service.

## [Distribution](../../../../glossary.md#term-distribution) contract

- Retain group io.github.pho001.synaptik and default 0.1.0-SNAPSHOT for ordinary development;
  accept an explicit releaseVersion property for the 0.1.0 production candidate. Do not invent 1.0.<span data-glossary-navigation="true"> ([SNAPSHOT glossary](../../../../glossary.md#term-snapshot))</span>
- Publish only explicit implemented products: Model, Backend Contract, Trace, Config, Planning,
  Compiler, Runtime, Prepare, CPU, Metal, OpenBLAS provider, Tuning, Training, Checkpoint, Engine,
  NN, Data and Text. Preserve logical project mappings and unique existing leaf artifact names.
  Exclude grouping, test, benchmark, CUDA, ONNX and CLI marker projects from supported publication.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Backend glossary](../../../../glossary.md#backend); [Trace glossary](../../../../glossary.md#trace); [Planning glossary](../../../../glossary.md#term-planning); [Compiler glossary](../../../../glossary.md#term-compiler); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [leaf glossary](../../../../glossary.md#term-leaf); [artifact glossary](../../../../glossary.md#term-artifact); [CUDA glossary](../../../../glossary.md#term-cuda); [ONNX glossary](../../../../glossary.md#term-onnx); [CLI glossary](../../../../glossary.md#term-cli); [publication glossary](../../../../glossary.md#term-publication))</span>
- Use maven-publish/from(java), sources and Javadoc JARs and a versioned local file Maven repository.
  No external credentials, uploads or invented POM licence/copyright-owner/signing metadata.<span data-glossary-navigation="true"> ([maven glossary](../../../../glossary.md#term-maven); [Javadoc glossary](../../../../glossary.md#term-javadoc); [POM glossary](../../../../glossary.md#term-pom); [owner glossary](../../../../glossary.md#term-owner); [metadata glossary](../../../../glossary.md#term-metadata))</span>
- Correct existing implementation edges whose types appear in public supported signatures to api,
  based on actual source. Preserve allowed dependency direction; do not re-export unrelated internals.
  Tuning is real Engine functionality and remains. Remove Engine's unused CUDA marker edge after
  checking source; retain the unsupported CUDA source/module for future development.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [module glossary](../../../../glossary.md#term-module))</span>
- Include the current complete THIRD_PARTY_NOTICES in applicable CPU binary/source artifacts and
  outer distributions. Preserve embedded Gradle-wrapper Apache materials. Do not change generated
  CPU cache/class identities or the verified three-entry inner Metal native package format.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [distributions glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [Gradle glossary](../../../../glossary.md#term-gradle); [cache glossary](../../../../glossary.md#term-cache))</span>
- Produce a versioned binary/local-repository distribution and a deterministic complete buildable
  source snapshot. Existing verified Metal local ZIP may be included through its existing explicit
  package input/verification task, not library discovery, a verifier bypass or copied build logic.
- Source selection must include all required build/wrapper/module/native/test/doc inputs and dirty
  accepted source, while excluding .git, ignored build/cache/temp/IDE files, unrelated working
  directories and secrets. Do not delete user files. Do not use HEAD-only git archive or a broad
  unchecked recursive project copy. Keep source selection and provenance one shared owner.<span data-glossary-navigation="true"> ([provenance glossary](../../../../glossary.md#provenance))</span>
- Record base revision, product version, each selected relative source path and SHA256, plus artifact
  hashes outside the manifest being hashed. Reproducible entry order/timestamps; no absolute developer
  paths or environment-dependent source claims. Do not claim bit-identical native rebuilds/signatures.<span data-glossary-navigation="true"> ([hashes glossary](../../../../glossary.md#term-hash))</span><span data-glossary-navigation="true"> ([manifest glossary](../../../../glossary.md#term-manifest))</span>
- Respect existing configuration-cache/build conventions. Prefer ordinary Gradle archive/publication
  tasks and one focused existing gradle/ script if needed, not a custom release framework/buildSrc.<span data-glossary-navigation="true"> ([release glossary](../../../../glossary.md#term-release))</span>

## Files, integration and [validation](../../../../glossary.md#term-validation)

- Root build/settings and gradle/ build scripts; affected implemented-product build.gradle.kts;
  existing architecture checks only where the obsolete CUDA edge/source assertions become invalid.
  Do not repin source/dependency-copy tests; delete obsolete wiring assertions, retain real behavior.<span data-glossary-navigation="true"> ([gradle glossary](../../../../glossary.md#term-gradle); [CUDA glossary](../../../../glossary.md#term-cuda); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- Depends on: Complete Engine0023, Metal0080/81/82 and current ordinary product closure. MAIN verifies
  exact dirty source and independent Ready review. The final release source freezes only AFTER the
  outstanding all-PURE work; this task implements and actually exercises the packaging mechanism.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [PURE glossary](../../../../glossary.md#term-pure))</span>
- Ready authority clarification is integrated and independently APPROVE0 in the root dependency
  diagram, scoped public composition and explanatory dependency guide: CPU/Metal are supported,
  CUDA's unused Engine edge is not required, marker source/registration remain and are not published.
  Base HEAD07a281edd3086efde2853211b71b48288db027e1 plus accepted dirty manifest
  ordinary-gradient-distribution-base.json; MAIN owns the final release freeze.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [Engine glossary](../../../../glossary.md#term-engine); [manifest glossary](../../../../glossary.md#term-manifest); [gradient glossary](../../../../glossary.md#term-gradient); [distribution glossary](../../../../glossary.md#term-distribution))</span>
- Conflicts: all listed build/config files. MAIN alone owns root/scoped authority, docs, versioned
  changelog, integration and final source freeze. Parallel group RELEASE-ORDINARY-GRADIENT-DISTRIBUTION;
  ordinary Compiler Java factoring is disjoint. Use a clean implementation context, absolute paths.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Order: independent contract review/Ready -> clean exact-source context -> complete build changes ->
  independent source/API-closure review -> MAIN publication/archive gate -> clean external Gradle/Maven
  consumer resolution and actual CPU/Metal compute/train/persist workflows -> final docs/evidence review.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [publication glossary](../../../../glossary.md#term-publication); [Maven glossary](../../../../glossary.md#term-maven))</span>
- Workers run no commands/builds/tests/lint/formatters/native loads. Report exact changed-file
  preimage/final SHAs, selected modules/tasks/properties and any genuinely missing prerequisite.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [modules glossary](../../../../glossary.md#term-module))</span>
- MAIN validates artifact/POM dependency closure, notice delivery, deterministic source entries and
  checksums, source extracted outside checkout with a clean build, and documented examples compiled
  against publications without adding hidden transitive dependencies or developer-local classpaths.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [POM glossary](../../../../glossary.md#term-pom))</span>

Legal boundary: project-wide licence/redistribution grant and copyright [ownership](../../../../glossary.md#term-ownership-meanings) are not supplied.
Do not fabricate them or label an internal local candidate an authorized public [release](../../../../glossary.md#term-release). Finish
all reachable technical packaging; MAIN obtains the [owner](../../../../glossary.md#term-owner) decision separately.

## Result

MAIN integrated twelve exact reviewed build/architecture-check paths, including one focused Kotlin
[release](../../../../glossary.md#term-release) script and its literal source inventory. Standard Java/[Maven](../../../../glossary.md#term-maven) publications expose eighteen
implemented products with their real [API](../../../../glossary.md#term-application-programming-interface-api) dependencies; [Engine](../../../../glossary.md#term-engine) no longer carries the unused [CUDA](../../../../glossary.md#term-cuda)
marker [dependency](../../../../glossary.md#term-dependency). [CPU](../../../../glossary.md#term-cpu) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata)/source/[Javadoc](../../../../glossary.md#term-javadoc) JARs contain the complete unmodified root notices.
No public upload, project licence, trusted signing, [fat JAR](../../../../glossary.md#term-fat-jar) or native-format change was invented.

The [publication](../../../../glossary.md#term-publication)/architecture gate passed with configuration-[cache](../../../../glossary.md#term-cache) support and 25 architecture
tests, no skips. The first successful build still produced an invalid source ZIP: Ant's default
excludes omitted the explicitly admitted `.gitignore`. That failure is retained. The reviewed
settings-phase correction uses the same tracked allowlist and standard Ant matching/removal,
removing only default patterns matching an explicitly admitted path. Recovered actual archives
contain every selected source path and exact byte [hash](../../../../glossary.md#term-hash), including `.gitignore`, with executable
wrapper/native scripts and deterministic timestamps. No custom ZIP writer or broad discovery.

The exercised [snapshot](../../../../glossary.md#term-snapshot) contained 2,497 selected source files plus generated [provenance](../../../../glossary.md#provenance); 544
[artifact](../../../../glossary.md#term-artifact) [checksum](../../../../glossary.md#term-checksum) entries passed. All eighteen POMs and the supported closure were inspected.
Two forced source/provenance rebuilds produced the same ZIP SHA256; the second reused the
configuration cache. The source ZIP was extracted outside any Git checkout and compiled with
`--no-build-cache --rerun-tasks`: all 29 selected [compile](../../../../glossary.md#compile)/consumer/architecture tasks executed.
This qualifies the mechanism and that exact snapshot, not source changes accepted afterward.

A fresh external Maven repository resolved exactly eighteen [runtime](../../../../glossary.md#term-runtime) JARs, each byte-identical
to the extracted publications, without checkout classes. Unchanged independent public [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)
programs passed CPU [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[F64](../../../../glossary.md#floating-point-precision-and-rounding) and actual-[GPU](../../../../glossary.md#term-gpu) [Metal](../../../../glossary.md#term-metal) F32, including every [parameter](../../../../glossary.md#term-parameter) role, repeated
updates, save/close/load and newly configured training. The unchanged Metal [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) program passed
all-parameter updates and [persistence](../../../../glossary.md#term-persistence)/new training. The exact documented F32 runtime-length [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)
program passed changed lengths/padded NaNs, cold and trained saves, load, fresh compute and a new
[optimizer step](../../../../glossary.md#optimizer-and-stochastic-gradient-descent). This last example is a workflow consumer, not a new independent [gradient](../../../../glossary.md#term-gradient) [oracle](../../../../glossary.md#term-test-oracle).

The same source corpus also compiled and ran through standalone [Gradle](../../../../glossary.md#term-gradle) against only the extracted
repository, using the shipped extracted-source wrapper: CPU F32/F64 MLP, actual Metal CNN and the
documented RNN passed. The binary archive's own inner Metal ZIP was extracted and independently
verified; the installed library SHA256 is
`4c2d36d2a546e1db05000604a27fd06348fbd19b3aa946e3e0cee9415d3dd2d2`.
These are ad-hoc-signed local source-matched artifacts, not an officially trusted prebuilt package.

Receipts retained in the production-[run](../../../../glossary.md#term-run) evidence directory: `engine0024-main-integration.json`,
`engine0024-recovered-artifact-evidence.json`, `engine0024-source-reproducibility.json`,
`engine0024-clean-source-javac.json`, and both `engine0024-external-*-runtime-evidence.json`.
[Installation instructions](../../../../getting-started.md#install-a-versioned-local-candidate)
use the exercised commands and coordinates. Later accepted source additions/removals must update
the reviewed inventory before the final freeze. Final all-[PURE](../../../../glossary.md#term-pure) [qualification](../../../../glossary.md#term-qualification), root [checkpoint](../../../../glossary.md#term-checkpoint),
final versioned release-source/artifact freeze and the [owner](../../../../glossary.md#term-owner)-only legal grant remain open.
