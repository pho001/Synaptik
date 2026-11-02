# Task 0024: Versioned local production distribution

## Status

Complete — MAIN integrated the exact reviewed source after actual archive/publication, uncached
extracted-source compilation, native installation and external Maven/Gradle CPU/Metal consumers.
Final independent source/evidence/documentation review APPROVE0. This is a local packaging
mechanism, not an owner-authorized public release or final all-PURE release candidate.

## Goal, class and authority

Class C: make the real product closure installable by external consumers using standard Gradle
Java/Maven components, and freeze the exact dirty source that produced a versioned distribution.
Use the [root dependency rules](../../../../../ARCHITECTURE.md#dependency-rules), existing Engine/
backend ownership and [production goal](../../../production-release-goal.md#release-acceptance-checklist).
No new module, executor, shading, fat JAR, compatibility wrapper or remote publishing service.

## Distribution contract

- Retain group io.github.pho001.synaptik and default 0.1.0-SNAPSHOT for ordinary development;
  accept an explicit releaseVersion property for the 0.1.0 production candidate. Do not invent 1.0.
- Publish only explicit implemented products: Model, Backend Contract, Trace, Config, Planning,
  Compiler, Runtime, Prepare, CPU, Metal, OpenBLAS provider, Tuning, Training, Checkpoint, Engine,
  NN, Data and Text. Preserve logical project mappings and unique existing leaf artifact names.
  Exclude grouping, test, benchmark, CUDA, ONNX and CLI marker projects from supported publication.
- Use maven-publish/from(java), sources and Javadoc JARs and a versioned local file Maven repository.
  No external credentials, uploads or invented POM licence/copyright-owner/signing metadata.
- Correct existing implementation edges whose types appear in public supported signatures to api,
  based on actual source. Preserve allowed dependency direction; do not re-export unrelated internals.
  Tuning is real Engine functionality and remains. Remove Engine's unused CUDA marker edge after
  checking source; retain the unsupported CUDA source/module for future development.
- Include the current complete THIRD_PARTY_NOTICES in applicable CPU binary/source artifacts and
  outer distributions. Preserve embedded Gradle-wrapper Apache materials. Do not change generated
  CPU cache/class identities or the verified three-entry inner Metal native package format.
- Produce a versioned binary/local-repository distribution and a deterministic complete buildable
  source snapshot. Existing verified Metal local ZIP may be included through its existing explicit
  package input/verification task, not library discovery, a verifier bypass or copied build logic.
- Source selection must include all required build/wrapper/module/native/test/doc inputs and dirty
  accepted source, while excluding .git, ignored build/cache/temp/IDE files, unrelated working
  directories and secrets. Do not delete user files. Do not use HEAD-only git archive or a broad
  unchecked recursive project copy. Keep source selection and provenance one shared owner.
- Record base revision, product version, each selected relative source path and SHA256, plus artifact
  hashes outside the manifest being hashed. Reproducible entry order/timestamps; no absolute developer
  paths or environment-dependent source claims. Do not claim bit-identical native rebuilds/signatures.
- Respect existing configuration-cache/build conventions. Prefer ordinary Gradle archive/publication
  tasks and one focused existing gradle/ script if needed, not a custom release framework/buildSrc.

## Files, integration and validation

- Root build/settings and gradle/ build scripts; affected implemented-product build.gradle.kts;
  existing architecture checks only where the obsolete CUDA edge/source assertions become invalid.
  Do not repin source/dependency-copy tests; delete obsolete wiring assertions, retain real behavior.
- Depends on: Complete Engine0023, Metal0080/81/82 and current ordinary product closure. MAIN verifies
  exact dirty source and independent Ready review. The final release source freezes only AFTER the
  outstanding all-PURE work; this task implements and actually exercises the packaging mechanism.
- Ready authority clarification is integrated and independently APPROVE0 in the root dependency
  diagram, scoped public composition and explanatory dependency guide: CPU/Metal are supported,
  CUDA's unused Engine edge is not required, marker source/registration remain and are not published.
  Base HEAD07a281edd3086efde2853211b71b48288db027e1 plus accepted dirty manifest
  ordinary-gradient-distribution-base.json; MAIN owns the final release freeze.
- Conflicts: all listed build/config files. MAIN alone owns root/scoped authority, docs, versioned
  changelog, integration and final source freeze. Parallel group RELEASE-ORDINARY-GRADIENT-DISTRIBUTION;
  ordinary Compiler Java factoring is disjoint. Use a clean implementation context, absolute paths.
- Order: independent contract review/Ready -> clean exact-source context -> complete build changes ->
  independent source/API-closure review -> MAIN publication/archive gate -> clean external Gradle/Maven
  consumer resolution and actual CPU/Metal compute/train/persist workflows -> final docs/evidence review.
- Workers run no commands/builds/tests/lint/formatters/native loads. Report exact changed-file
  preimage/final SHAs, selected modules/tasks/properties and any genuinely missing prerequisite.
- MAIN validates artifact/POM dependency closure, notice delivery, deterministic source entries and
  checksums, source extracted outside checkout with a clean build, and documented examples compiled
  against publications without adding hidden transitive dependencies or developer-local classpaths.

Legal boundary: project-wide licence/redistribution grant and copyright ownership are not supplied.
Do not fabricate them or label an internal local candidate an authorized public release. Finish
all reachable technical packaging; MAIN obtains the owner decision separately.

## Result

MAIN integrated twelve exact reviewed build/architecture-check paths, including one focused Kotlin
release script and its literal source inventory. Standard Java/Maven publications expose eighteen
implemented products with their real API dependencies; Engine no longer carries the unused CUDA
marker dependency. CPU binary/source/Javadoc JARs contain the complete unmodified root notices.
No public upload, project licence, trusted signing, fat JAR or native-format change was invented.

The publication/architecture gate passed with configuration-cache support and 25 architecture
tests, no skips. The first successful build still produced an invalid source ZIP: Ant's default
excludes omitted the explicitly admitted `.gitignore`. That failure is retained. The reviewed
settings-phase correction uses the same tracked allowlist and standard Ant matching/removal,
removing only default patterns matching an explicitly admitted path. Recovered actual archives
contain every selected source path and exact byte hash, including `.gitignore`, with executable
wrapper/native scripts and deterministic timestamps. No custom ZIP writer or broad discovery.

The exercised snapshot contained 2,497 selected source files plus generated provenance; 544
artifact checksum entries passed. All eighteen POMs and the supported closure were inspected.
Two forced source/provenance rebuilds produced the same ZIP SHA256; the second reused the
configuration cache. The source ZIP was extracted outside any Git checkout and compiled with
`--no-build-cache --rerun-tasks`: all 29 selected compile/consumer/architecture tasks executed.
This qualifies the mechanism and that exact snapshot, not source changes accepted afterward.

A fresh external Maven repository resolved exactly eighteen runtime JARs, each byte-identical
to the extracted publications, without checkout classes. Unchanged independent public MLP
programs passed CPU F32/F64 and actual-GPU Metal F32, including every parameter role, repeated
updates, save/close/load and newly configured training. The unchanged Metal CNN program passed
all-parameter updates and persistence/new training. The exact documented F32 runtime-length RNN
program passed changed lengths/padded NaNs, cold and trained saves, load, fresh compute and a new
optimizer step. This last example is a workflow consumer, not a new independent gradient oracle.

The same source corpus also compiled and ran through standalone Gradle against only the extracted
repository, using the shipped extracted-source wrapper: CPU F32/F64 MLP, actual Metal CNN and the
documented RNN passed. The binary archive's own inner Metal ZIP was extracted and independently
verified; the installed library SHA256 is
`4c2d36d2a546e1db05000604a27fd06348fbd19b3aa946e3e0cee9415d3dd2d2`.
These are ad-hoc-signed local source-matched artifacts, not an officially trusted prebuilt package.

Receipts retained in the production-run evidence directory: `engine0024-main-integration.json`,
`engine0024-recovered-artifact-evidence.json`, `engine0024-source-reproducibility.json`,
`engine0024-clean-source-javac.json`, and both `engine0024-external-*-runtime-evidence.json`.
[Installation instructions](../../../../getting-started.md#install-a-versioned-local-candidate)
use the exercised commands and coordinates. Later accepted source additions/removals must update
the reviewed inventory before the final freeze. Final all-PURE qualification, root checkpoint,
final versioned release-source/artifact freeze and the owner-only legal grant remain open.
